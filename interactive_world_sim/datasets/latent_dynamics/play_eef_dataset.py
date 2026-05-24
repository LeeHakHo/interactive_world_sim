"""PlayEEFDataset: LeRobot v2.1 dataset reader for `play_robot_*_eef` (robot)
and `human_play_eef_data/play_human_eef_*` (human) datasets.

This dataset class is the robot/human side of the data pipeline for
HUMAN_ROBOT_ALIGN_PLAN-2 Phase 0 + Phase A1. It returns a unified 8-dim EEF
action chunk: pos(3) + quat_xyzw(4) + gripper(1) — see Plan-2 §4 for the
schema decision. The returned dict matches the schema consumed by
LatentWorldModel.training_step:

    {
        "obs": {<camera_key>: (T, 3, H, W) float32 in [0,1], ...},
        "goal": {<camera_key>: (3, H, W) float32 in [0,1], ...},
        "action": (T, 8) float32,
        "is_early_stop": (1,) bool,
        "rel_stop_idx": (1,) int64,
    }

Frame convention is NOT unified at this layer — `frame_convention="world"` for
robot (Trossen world frame) and `frame_convention="camera"` for human (cam_high
optical frame). Phase A2 will decide how to unify. The align module's
embodiment-specific input MLP can absorb the frame difference in A1 since the
sanity training is robot-only.

Note on AV1 video decoding: parquet uses AV1 codec (verified in info.json).
PyAV is required (decord doesn't support AV1). Random access is slow with AV1
so we lazily decode each episode's needed frame indices in one sequential pass
and cache them as a numpy memmap on disk to keep RAM footprint bounded.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

import av
import cv2
import numpy as np
import torch
import torch.utils.data
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as ScipyRot

from interactive_world_sim.utils.normalizer import (
    LinearNormalizer,
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)

from .base_dataset import BaseImageDataset


_ROBOT_COLUMNS = {
    "action_pos": "action_right_ee_position",
    "action_quat": "action_right_ee_quat_xyzw",
    "action_gripper": "action_right_gripper",
    "obs_pos": "obs_right_ee_position",
    "obs_quat": "obs_right_ee_quat_xyzw",
}

_HUMAN_COLUMNS = {
    "action_pos": "observation.eef.position_right",
    "action_rot_matrix": "observation.eef.rotation_matrix_right",
    "action_width": "observation.eef.width_right",
    "action_detected": "observation.eef.detected_right",
}


def _load_episode_parquet(episode_path: Path):
    """Load a parquet episode and return columns as numpy arrays."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(episode_path))
    cols: dict[str, np.ndarray] = {}
    for name in table.column_names:
        cols[name] = np.asarray(table.column(name).to_pylist())
    return cols


def _matrix9_to_quat_xyzw(rot9: np.ndarray) -> np.ndarray:
    """(N, 9) row-major rotation matrix → (N, 4) xyzw quaternion.

    NaNs propagate to NaN quaternions (filtered out before training).
    """
    out = np.full((rot9.shape[0], 4), np.nan, dtype=np.float32)
    valid = ~np.isnan(rot9).any(axis=1)
    if valid.any():
        mats = rot9[valid].reshape(-1, 3, 3).astype(np.float64)
        # Some hand-pose matrices may be slightly non-orthogonal; orthogonalize
        # via SVD-based projection to the nearest rotation before quat.
        u, _, vt = np.linalg.svd(mats)
        det = np.linalg.det(u @ vt)
        s = np.ones_like(det)
        s[det < 0] = -1.0
        mats_proj = u @ (s[:, None, None] * vt)
        try:
            quats = ScipyRot.from_matrix(mats_proj).as_quat()  # xyzw
        except Exception:
            quats = np.zeros((mats.shape[0], 4))
            quats[:, -1] = 1.0
        out[valid] = quats.astype(np.float32)
    return out


def _decode_video_to_memmap(
    video_path: Path,
    cache_path: Path,
    resolution: int,
    expected_frames: int,
    crop: Optional[tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Decode the entire MP4 to a uint8 memmap of shape (N, 3, H, W).

    The memmap is created on first call and re-used on subsequent calls. AV1
    random access is slow so we do one sequential pass.

    Args:
        crop: optional (x, y, w, h) pixel crop applied *before* resize. With
            cam_high (480×640), the default training crop is (195, 195, 256,
            256) to keep aspect ratio and focus on the workspace, then resize
            to `resolution`.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        try:
            arr = np.memmap(
                cache_path,
                mode="r",
                dtype=np.uint8,
                shape=(expected_frames, 3, resolution, resolution),
            )
            return arr
        except ValueError:
            cache_path.unlink(missing_ok=True)

    print(f"[PlayEEFDataset] decoding {video_path} → memmap {cache_path}")
    arr = np.memmap(
        cache_path,
        mode="w+",
        dtype=np.uint8,
        shape=(expected_frames, 3, resolution, resolution),
    )
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        i = 0
        for frame in container.decode(stream):
            if i >= expected_frames:
                break
            img = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
            if crop is not None:
                x, y, w, h = crop
                img = img[y : y + h, x : x + w]
            if img.shape[0] != resolution or img.shape[1] != resolution:
                img = cv2.resize(
                    img, (resolution, resolution), interpolation=cv2.INTER_AREA
                )
            arr[i] = img.transpose(2, 0, 1)  # (3, H, W)
            i += 1
    finally:
        container.close()
    if i != expected_frames:
        raise RuntimeError(
            f"expected {expected_frames} frames from {video_path}, decoded {i}"
        )
    arr.flush()
    return np.memmap(
        cache_path,
        mode="r",
        dtype=np.uint8,
        shape=(expected_frames, 3, resolution, resolution),
    )


def _crop_tag(crop: Optional[tuple[int, int, int, int]]) -> str:
    if crop is None:
        return "nocrop"
    x, y, w, h = crop
    return f"crop{x}-{y}-{w}-{h}"


class PlayEEFDataset(BaseImageDataset):
    """Robot or human play data with 8-dim EEF action chunks."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_dirs: List[str] = [str(d) for d in cfg.dataset_dirs]
        self.domain: str = cfg.get("domain", "robot")
        assert self.domain in ("robot", "human"), self.domain
        self.frame_convention: str = cfg.get(
            "frame_convention", "world" if self.domain == "robot" else "camera"
        )
        self.video_keys: List[str] = list(cfg.video_keys)
        # obs_keys aliases the LatentWorldModel-facing camera keys; they match
        # the order of video_keys 1:1. e.g.
        #   video_keys  = ["observation.images.cam_high", "observation.images.cam_right_wrist"]
        #   obs_keys    = ["camera_0_color", "camera_1_color"]
        self.obs_keys: List[str] = list(cfg.obs_keys)
        assert len(self.obs_keys) == len(self.video_keys)
        self.resolution: int = int(cfg.resolution)
        self.horizon: int = int(cfg.horizon)
        self.val_horizon: int = int(cfg.get("val_horizon", self.horizon))
        self.skip_frame: int = int(cfg.get("skip_frame", 1))
        self.pad_before: int = int(cfg.get("pad_before", 0))
        self.pad_after: int = int(cfg.get("pad_after", 0))
        self.val_ratio: float = float(cfg.get("val_ratio", 0.1))
        self.seed: int = int(cfg.get("seed", 42))
        self.goal_sample: str = cfg.get("goal_sample", "intermediate")
        self.skip_idx: int = int(cfg.get("skip_idx", 1))
        self.cache_root: Path = Path(cfg.get("cache_root", "/tmp/play_eef_cache"))
        # Per-camera crop list, aligned 1:1 with video_keys. Each element is
        # either null or [x, y, w, h]. cam_high default focuses on the table.
        crops_cfg = cfg.get("crops", None)
        self.crops: List[Optional[tuple[int, int, int, int]]] = []
        if crops_cfg is None:
            self.crops = [None] * len(self.video_keys)
        else:
            for c in crops_cfg:
                if c is None:
                    self.crops.append(None)
                else:
                    assert len(c) == 4, f"crop must be (x,y,w,h), got {c}"
                    self.crops.append((int(c[0]), int(c[1]), int(c[2]), int(c[3])))
            assert len(self.crops) == len(self.video_keys)

        # Per-episode metadata
        self._episodes: list[dict] = []
        # Optional per-dataset episode filter for Step 3 sweep.
        # None → use all episodes; [] → load nothing; list[int] → keep only those.
        subset = cfg.get("episode_subset", None)
        if subset is not None:
            subset = list(int(i) for i in subset)
            self._episode_subset: Optional[set[int]] = set(subset)
        else:
            self._episode_subset = None
        for ds_dir in self.dataset_dirs:
            ds_path = Path(ds_dir)
            info_path = ds_path / "meta" / "info.json"
            eps_jsonl = ds_path / "meta" / "episodes.jsonl"
            if not info_path.exists():
                raise FileNotFoundError(
                    f"Missing meta/info.json under {ds_dir}. Did you download the dataset?"
                )
            with info_path.open() as f:
                info = json.load(f)
            chunks_size = info.get("chunks_size", 1000)
            with eps_jsonl.open() as f:
                for line in f:
                    rec = json.loads(line)
                    ep_idx = rec["episode_index"]
                    if (
                        self._episode_subset is not None
                        and ep_idx not in self._episode_subset
                    ):
                        continue
                    n_frames = rec["length"]
                    chunk_idx = ep_idx // chunks_size
                    pq_path = (
                        ds_path
                        / "data"
                        / f"chunk-{chunk_idx:03d}"
                        / f"episode_{ep_idx:06d}.parquet"
                    )
                    if not pq_path.exists():
                        raise FileNotFoundError(f"Missing {pq_path}")
                    self._episodes.append({
                        "dataset_dir": ds_path,
                        "episode_index": ep_idx,
                        "chunk_idx": chunk_idx,
                        "n_frames": n_frames,
                        "parquet_path": pq_path,
                    })

        # Split episodes deterministically (last val_ratio episodes are val)
        n_eps = len(self._episodes)
        n_val = max(1, int(round(n_eps * self.val_ratio))) if n_eps > 1 else 0
        n_val = min(n_val, n_eps - 1) if n_eps > 1 else 0
        self.train_mask = np.zeros(n_eps, dtype=bool)
        self.val_mask = np.zeros(n_eps, dtype=bool)
        self.train_mask[: n_eps - n_val] = True
        self.val_mask[n_eps - n_val :] = True

        # Build the (action_chunk, video memmaps) for each episode lazily.
        for ep in self._episodes:
            ep["cache_dir"] = (
                self.cache_root
                / f"{self.domain}__{ep['dataset_dir'].name}__ep{ep['episode_index']:06d}"
            )
            ep["actions"] = None  # populated on first __getitem__ touch
            ep["video_arrays"] = None
            ep["valid_mask"] = None

        # Action normalizer is fit lazily from the union of training episodes
        # the first time get_normalizer is called.
        self._action_stats_cache: Optional[dict] = None
        self._gripper_range: Optional[tuple[float, float]] = None

        # Sample indices: list of (ep_idx, start_frame) for training
        self._train_indices = self._build_indices(self.train_mask, self.horizon)
        self._val_indices = self._build_indices(self.val_mask, self.val_horizon)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def _build_indices(
        self,
        ep_mask: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """Build (N, 2) array of (ep_idx, start_frame) sample windows."""
        span = horizon * self.skip_frame
        indices: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(self._episodes):
            if not ep_mask[ep_idx]:
                continue
            n_frames = ep["n_frames"]
            max_start = n_frames - span
            if max_start < 0:
                continue
            for start in range(0, max_start + 1):
                indices.append((ep_idx, start))
        if not indices:
            return np.zeros((0, 2), dtype=np.int64)
        return np.asarray(indices, dtype=np.int64)

    # ------------------------------------------------------------------
    # Lazy episode loading
    # ------------------------------------------------------------------
    def _ensure_episode_loaded(self, ep_idx: int) -> None:
        ep = self._episodes[ep_idx]
        if ep["actions"] is not None:
            return
        cols = _load_episode_parquet(ep["parquet_path"])
        n_frames = ep["n_frames"]
        action_chunk, valid_mask = self._build_action_chunk(cols, n_frames)
        ep["actions"] = action_chunk
        ep["valid_mask"] = valid_mask

        video_arrays: list[np.memmap] = []
        for vk, crop in zip(self.video_keys, self.crops):
            mp4_path = (
                ep["dataset_dir"]
                / "videos"
                / f"chunk-{ep['chunk_idx']:03d}"
                / vk
                / f"episode_{ep['episode_index']:06d}.mp4"
            )
            cache_path = (
                ep["cache_dir"]
                / f"{vk.replace('/', '_')}_res{self.resolution}_{_crop_tag(crop)}.npy"
            )
            arr = _decode_video_to_memmap(
                mp4_path, cache_path, self.resolution, n_frames, crop=crop
            )
            video_arrays.append(arr)
        ep["video_arrays"] = video_arrays

    def _build_action_chunk(
        self,
        cols: dict[str, np.ndarray],
        n_frames: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assemble an (N, 8) float32 action array + (N,) bool valid mask."""
        if self.domain == "robot":
            pos = np.stack(cols[_ROBOT_COLUMNS["action_pos"]]).astype(np.float32)
            quat = np.stack(cols[_ROBOT_COLUMNS["action_quat"]]).astype(np.float32)
            gripper = np.stack(cols[_ROBOT_COLUMNS["action_gripper"]]).astype(np.float32)
            if gripper.ndim == 1:
                gripper = gripper[:, None]
            valid = np.ones(n_frames, dtype=bool)
        else:  # human
            pos = np.stack(cols[_HUMAN_COLUMNS["action_pos"]]).astype(np.float32)
            rot9 = np.stack(cols[_HUMAN_COLUMNS["action_rot_matrix"]]).astype(np.float32)
            quat = _matrix9_to_quat_xyzw(rot9)
            width = np.stack(cols[_HUMAN_COLUMNS["action_width"]]).astype(np.float32)
            if width.ndim == 1:
                width = width[:, None]
            detected = np.stack(cols[_HUMAN_COLUMNS["action_detected"]]).astype(bool).reshape(-1)
            valid = detected & ~np.isnan(quat).any(axis=1) & ~np.isnan(pos).any(axis=1)
            gripper = width
            # Forward-fill invalid frames with last valid pose so the dataloader
            # doesn't crash; we expose the mask to the caller via `valid_mask`
            # for downstream filtering decisions.
            pos = _ffill(pos, valid)
            quat = _ffill(quat, valid)
            gripper = _ffill(gripper, valid)
        action = np.concatenate([pos, quat, gripper], axis=1).astype(np.float32)
        assert action.shape == (n_frames, 8), action.shape
        return action, valid

    # ------------------------------------------------------------------
    # Normalizer (Min-Max for pos & gripper, identity for quat)
    # ------------------------------------------------------------------
    def get_normalizer(
        self, mode: str = "none", **kwargs: dict
    ) -> LinearNormalizer:
        if self._action_stats_cache is None:
            self._fit_action_stats()
        normalizer = LinearNormalizer()
        normalizer["action"] = get_range_normalizer_from_stat(
            self._action_stats_cache
        )
        for key in self.obs_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def _fit_action_stats(self) -> None:
        """Compute per-dim min/max over all training-split valid frames."""
        actions_concat: list[np.ndarray] = []
        for ep_idx, ep in enumerate(self._episodes):
            if not self.train_mask[ep_idx]:
                continue
            self._ensure_episode_loaded(ep_idx)
            valid = ep["valid_mask"]
            actions_concat.append(ep["actions"][valid])
        all_actions = np.concatenate(actions_concat, axis=0)
        # For quat we want identity normalization (range = [-1,1] is fine for
        # unit quats). For pos/gripper we want range normalization to [-1, 1].
        stat = array_to_stats(all_actions)
        # Override quat (cols 3..7) to be identity-like: min=-1, max=1.
        # `get_range_normalizer_from_stat` maps min→-1, max→+1, so we just
        # clamp the stat min/max to (-1, 1) for those four dims.
        stat["min"][3:7] = -1.0
        stat["max"][3:7] = 1.0
        self._action_stats_cache = stat
        self._gripper_range = (float(stat["min"][7]), float(stat["max"][7]))

    def get_all_actions(self) -> torch.Tensor:
        actions_concat: list[np.ndarray] = []
        for ep_idx, ep in enumerate(self._episodes):
            if not self.train_mask[ep_idx]:
                continue
            self._ensure_episode_loaded(ep_idx)
            actions_concat.append(ep["actions"])
        return torch.from_numpy(np.concatenate(actions_concat, axis=0))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        if self.is_val:
            return max(len(self._val_indices) // self.skip_idx, len(self._episodes))
        return len(self._train_indices)

    def get_validation_dataset(self) -> "PlayEEFDataset":
        val_set = copy.copy(self)
        val_set.is_val = True
        return val_set

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_val:
            indices = self._val_indices
            horizon = self.val_horizon
            if len(indices) == 0:
                # No val indices (e.g., single-episode dataset). Fall back to
                # train indices on val_mask=False with a short horizon.
                return self._dummy_item(horizon)
            ep_idx, start = indices[(idx * self.skip_idx) % len(indices)]
        else:
            indices = self._train_indices
            horizon = self.horizon
            ep_idx, start = indices[idx % len(indices)]
        return self._sample_window(int(ep_idx), int(start), horizon)

    def _dummy_item(self, horizon: int) -> Dict[str, torch.Tensor]:
        obs_dict = {
            k: torch.zeros((horizon, 3, self.resolution, self.resolution), dtype=torch.float32)
            for k in self.obs_keys
        }
        goal_dict = {
            k: torch.zeros((3, self.resolution, self.resolution), dtype=torch.float32)
            for k in self.obs_keys
        }
        return {
            "obs": obs_dict,
            "goal": goal_dict,
            "action": torch.zeros((horizon, 8), dtype=torch.float32),
            "is_early_stop": torch.tensor([False]),
            "rel_stop_idx": torch.tensor([horizon - 1], dtype=torch.long),
        }

    def _sample_window(
        self, ep_idx: int, start: int, horizon: int
    ) -> Dict[str, torch.Tensor]:
        self._ensure_episode_loaded(ep_idx)
        ep = self._episodes[ep_idx]
        skip = self.skip_frame
        idxs = start + np.arange(horizon) * skip
        # Clamp to episode length
        idxs = np.minimum(idxs, ep["n_frames"] - 1)

        obs_dict: dict[str, torch.Tensor] = {}
        goal_dict: dict[str, torch.Tensor] = {}
        for ok, varr in zip(self.obs_keys, ep["video_arrays"]):
            seq = np.asarray(varr[idxs])  # (T, 3, H, W) uint8
            obs_dict[ok] = torch.from_numpy(seq.astype(np.float32) / 255.0)
            # Goal frame for IWS pipeline: random intermediate after window
            if self.goal_sample == "final":
                gframe = ep["n_frames"] - 1
            elif self.goal_sample == "intermediate":
                window_end = int(idxs[-1])
                gframe = (
                    np.random.randint(window_end, ep["n_frames"])
                    if window_end < ep["n_frames"] - 1
                    else window_end
                )
            else:
                gframe = int(idxs[-1])
            goal_dict[ok] = torch.from_numpy(
                np.asarray(varr[gframe]).astype(np.float32) / 255.0
            )
        actions = ep["actions"][idxs].astype(np.float32)  # (T, 8)
        return {
            "obs": obs_dict,
            "goal": goal_dict,
            "action": torch.from_numpy(actions),
            "is_early_stop": torch.tensor([False]),
            "rel_stop_idx": torch.tensor([horizon - 1], dtype=torch.long),
        }


class MixedPlayEEFDataset(BaseImageDataset):
    """Concat-wrapper that exposes a combined robot+human PlayEEFDataset.

    For HUMAN_ROBOT_ALIGN_PLAN-2 Phase 0 "mixed baseline": train IWS Stage 1
    on robot+human pixel reconstruction to *reproduce* the encoder-disjoint
    failure described in Plan-2 §1.4. Each sample carries an `embodiment`
    field ("robot" | "human") for downstream consumers; LatentWorldModel
    Stage 1 ignores it.

    Config schema (DictConfig):
        robot:   sub-DictConfig with PlayEEFDataset fields (domain=robot)
        human:   sub-DictConfig with PlayEEFDataset fields (domain=human)
        sample_ratio (optional): probability of drawing from the robot half
            on each __getitem__. One of:
              - float in (0, 1): explicit P(robot). 0.5 means 1:1 embodiment-
                balanced (oversamples the smaller half).
              - "auto" (default): time-proportional, i.e. P(robot) =
                |robot_train_samples| / (|robot| + |human|). Each individual
                frame is drawn with equal probability across both halves.
            A2 sweep will be in the align module training, not here.

    The wrapper presents `obs_keys` / `resolution` / `horizon` from the robot
    half (assumed to match the human half for cross-embodiment training).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.robot = PlayEEFDataset(cfg.robot)
        self.human = PlayEEFDataset(cfg.human)
        assert self.robot.obs_keys == self.human.obs_keys, (
            f"robot.obs_keys={self.robot.obs_keys} != human.obs_keys={self.human.obs_keys}"
        )
        assert self.robot.resolution == self.human.resolution
        assert self.robot.horizon == self.human.horizon
        # sample_ratio: "auto" (time-proportional) or explicit float P(robot).
        raw_ratio = cfg.get("sample_ratio", "auto")
        if isinstance(raw_ratio, str) and raw_ratio.lower() == "auto":
            n_r, n_h = len(self.robot), len(self.human)
            self.sample_ratio: float = n_r / max(n_r + n_h, 1)
        else:
            self.sample_ratio = float(raw_ratio)
        print(
            f"[MixedPlayEEFDataset] |robot|={len(self.robot)} |human|={len(self.human)}  "
            f"P(robot)={self.sample_ratio:.3f}"
        )
        # Surface a few fields the IWS pipeline reads off the dataset.
        self.obs_keys = self.robot.obs_keys
        self.resolution = self.robot.resolution
        self.horizon = self.robot.horizon
        self.val_horizon = self.robot.val_horizon
        # Use deterministic length: enough to cover ratio · robot + (1-ratio) · human
        # without truncation. Sampling weights are applied per __getitem__.
        self._train_len = len(self.robot) + len(self.human)
        # rng for sample-side draw (separate from dataloader workers)
        self._rng = np.random.RandomState(int(cfg.get("seed", 0)))

    def get_normalizer(self, mode: str = "none", **kwargs) -> LinearNormalizer:
        """Fit action min-max over the union of robot + human valid frames so
        both domains land in [-1, 1]. Image normalizer is the canonical
        get_image_range_normalizer (identical to robot/human individually)."""
        # Concatenate stats by computing once over union.
        # We piggy-back on each sub-dataset's lazy-loading; concatenate all
        # valid action frames from both then fit one min-max.
        actions_concat: list[np.ndarray] = []
        for sub in (self.robot, self.human):
            for ep_idx, ep in enumerate(sub._episodes):
                if not sub.train_mask[ep_idx]:
                    continue
                sub._ensure_episode_loaded(ep_idx)
                valid = ep["valid_mask"]
                actions_concat.append(ep["actions"][valid])
        all_actions = np.concatenate(actions_concat, axis=0)
        stat = array_to_stats(all_actions)
        stat["min"][3:7] = -1.0
        stat["max"][3:7] = 1.0

        normalizer = LinearNormalizer()
        normalizer["action"] = get_range_normalizer_from_stat(stat)
        for key in self.obs_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.cat(
            [self.robot.get_all_actions(), self.human.get_all_actions()],
            dim=0,
        )

    def __len__(self) -> int:
        if self.is_val:
            return max(1, len(self.robot) // 100) + max(1, len(self.human) // 100)
        return self._train_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_val:
            half = len(self) // 2
            if idx < half:
                rob_val = self._robot_val()
                item = rob_val[idx % len(rob_val)]
                emb = "robot"
            else:
                hum_val = self._human_val()
                item = hum_val[(idx - half) % len(hum_val)]
                emb = "human"
        else:
            if self._rng.random() < self.sample_ratio:
                item = self.robot[idx % len(self.robot)]
                emb = "robot"
            else:
                item = self.human[idx % len(self.human)]
                emb = "human"
        item["embodiment"] = emb
        item["domain_label"] = torch.tensor(
            1 if emb == "robot" else 0, dtype=torch.long,
        )
        return item

    def get_validation_dataset(self) -> "MixedPlayEEFDataset":
        val = copy.copy(self)
        val.is_val = True
        return val

    def _robot_val(self):
        if not hasattr(self, "_robot_val_cache"):
            self._robot_val_cache = self.robot.get_validation_dataset()
        return self._robot_val_cache

    def _human_val(self):
        if not hasattr(self, "_human_val_cache"):
            self._human_val_cache = self.human.get_validation_dataset()
        return self._human_val_cache


def _ffill(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Forward-fill invalid rows with the last valid row; backfill from first."""
    out = arr.copy()
    last = None
    for i in range(arr.shape[0]):
        if valid[i]:
            last = arr[i]
        elif last is not None:
            out[i] = last
    # Backfill leading invalids
    first = None
    for i in range(arr.shape[0]):
        if valid[i]:
            first = arr[i]
            break
    if first is not None:
        for i in range(arr.shape[0]):
            if valid[i]:
                break
            out[i] = first
    return out
