import concurrent.futures
import copy
import io
import json
import multiprocessing
import os
import shutil
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import zarr
import zarr.storage
from filelock import FileLock
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from interactive_world_sim.utils.imagecodecs_numcodecs import Jpeg2k, register_codecs
from interactive_world_sim.utils.normalizer import (
    LinearNormalizer,
    array_to_stats,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from interactive_world_sim.utils.pytorch_util import dict_apply
from interactive_world_sim.utils.replay_buffer import ReplayBuffer
from interactive_world_sim.utils.sampler import SequenceSampler

from .base_dataset import BaseImageDataset

register_codecs()


def _decode_image(img_dict: dict, h: int, w: int) -> np.ndarray:
    """Decode JPEG bytes from Parquet to uint8 numpy array (H, W, C)."""
    img = Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB")
    img_np = np.array(img)
    if img_np.shape[:2] != (h, w):
        img_np = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_AREA)
    return img_np


_FLOW_STORAGE_SIZE = (256, 256)  # default spatial resolution stored in zarr


def _load_flow_frame(flow_dir: str, episode_idx: int, frame_idx: int, split: str = "train") -> Optional[np.ndarray]:
    """Load one frame of optical flow from a per-episode .pt file.

    File format: {flow_dir}/{split}/{episode_idx}/0.pt  shape (T, 2, H, W)
    Returns (2, H, W) float32 numpy array at native resolution, or None if file missing.
    """
    flow_path = os.path.join(flow_dir, split, str(episode_idx), "0.pt")
    if not os.path.exists(flow_path):
        return None
    episode_flow = torch.load(flow_path, map_location="cpu", weights_only=True)  # (T, 2, H, W)
    if not isinstance(episode_flow, torch.Tensor) or episode_flow.ndim != 4:
        return None
    frame_idx = min(frame_idx, episode_flow.shape[0] - 1)
    return episode_flow[frame_idx].float().numpy()  # (2, H, W)


def _load_episode_flow(flow_dir: str, episode_idx: int, episode_length: int, split: str = "train") -> Optional[np.ndarray]:
    """Load full episode flow at native resolution. Returns (T, 2, H, W) or None."""
    flow_path = os.path.join(flow_dir, split, str(episode_idx), "0.pt")
    if not os.path.exists(flow_path):
        return None
    episode_flow = torch.load(flow_path, map_location="cpu", weights_only=True)  # (T, 2, H, W)
    if not isinstance(episode_flow, torch.Tensor) or episode_flow.ndim != 4:
        return None
    # clip/pad to episode_length
    T = episode_flow.shape[0]
    if T > episode_length:
        episode_flow = episode_flow[:episode_length]
    elif T < episode_length:
        pad = episode_flow[-1:].expand(episode_length - T, -1, -1, -1)
        episode_flow = torch.cat([episode_flow, pad], dim=0)
    return episode_flow.float().numpy()


def _detect_flow_available(flow_dir: str, episode_indices: List[int], split: str = "train") -> bool:
    """Check if flow data exists for at least one episode."""
    for epi_idx in episode_indices:
        flow_path = os.path.join(flow_dir, split, str(epi_idx), "0.pt")
        if os.path.exists(flow_path):
            return True
    return False


def _convert_libero_to_replay(
    store: zarr.storage.Store,
    shape_meta: dict,
    dataset_dir: str,
    episode_indices: List[int],
    flow_dir: Optional[str] = None,
    flow_split: str = "train",
    flow_storage_size: Optional[tuple] = None,
    n_workers: Optional[int] = None,
    max_inflight_tasks: Optional[int] = None,
) -> ReplayBuffer:
    """Convert LIBERO Parquet files to a zarr ReplayBuffer."""
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    rgb_keys: list = []
    lowdim_keys: list = []
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        type_ = attr.get("type", "low_dim")
        if type_ == "rgb":
            rgb_keys.append(key)
        elif type_ == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    info_path = os.path.join(dataset_dir, "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)
    chunks_size = info["chunks_size"]
    data_path_template = info["data_path"]

    # resolve flow storage size
    fh, fw = flow_storage_size if flow_storage_size is not None else _FLOW_STORAGE_SIZE

    # detect flow availability before the main loop
    has_flow = False
    if flow_dir is not None:
        has_flow = _detect_flow_available(flow_dir, episode_indices, split=flow_split)
        if has_flow:
            print(f"Flow features detected: storing at ({fh}, {fw})")
        else:
            print("Warning: flow_dir provided but no flow files found. Skipping flow.")

    # first pass: collect episode lengths to know total steps
    episode_lengths: list = []
    for epi_idx in tqdm(episode_indices, desc="Scanning episodes"):
        chunk_idx = epi_idx // chunks_size
        parquet_path = os.path.join(
            dataset_dir,
            data_path_template.format(episode_chunk=chunk_idx, episode_index=epi_idx),
        )
        df = pd.read_parquet(parquet_path, columns=["actions"])
        episode_lengths.append(len(df))

    episode_ends = list(np.cumsum(episode_lengths))
    n_steps = episode_ends[-1]
    meta_group.array(
        "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )

    # pre-allocate zarr arrays for lowdim and flow
    lowdim_zarr: dict = {}
    action_shape = (n_steps, 7)  # placeholder; real shape set after first episode
    for key in ["action"] + lowdim_keys:
        lowdim_zarr[key] = None  # lazily initialized after first episode

    flow_zarr = None
    if has_flow:
        flow_zarr = data_group.zeros(
            name="flow",
            shape=(n_steps, 2, fh, fw),
            chunks=(min(64, n_steps), 2, fh, fw),
            dtype=np.float32,
            overwrite=True,
        )

    lowdim_data_dict: dict = {"action": []}
    for key in lowdim_keys:
        lowdim_data_dict[key] = []
    rgb_data_dict: dict = {k: [] for k in rgb_keys}

    write_ptr = 0
    for epi_idx, episode_length in tqdm(
        zip(episode_indices, episode_lengths), desc="Loading episodes", total=len(episode_indices)
    ):
        chunk_idx = epi_idx // chunks_size
        parquet_path = os.path.join(
            dataset_dir,
            data_path_template.format(episode_chunk=chunk_idx, episode_index=epi_idx),
        )
        df = pd.read_parquet(parquet_path)

        lowdim_data_dict["action"].append(
            np.stack(df["actions"].values).astype(np.float32)
        )
        for key in lowdim_keys:
            lowdim_data_dict[key].append(
                np.stack(df[key].values).astype(np.float32)
            )
        for key in rgb_keys:
            c, h, w = tuple(obs_shape_meta[key]["shape"])
            imgs = np.stack(
                [_decode_image(row, h, w) for row in df[key]], axis=0
            )  # (T, H, W, C)
            rgb_data_dict[key].append(imgs)

        # write flow incrementally to avoid large RAM spike
        if flow_zarr is not None:
            epi_flow = _load_episode_flow(flow_dir, epi_idx, episode_length, split=flow_split)
            if epi_flow is None:
                epi_flow = np.zeros((episode_length, 2, fh, fw), dtype=np.float32)
            elif epi_flow.shape[-2:] != (fh, fw):
                # resize flow to target storage size
                flow_tensor = torch.from_numpy(epi_flow)  # (T, 2, H, W)
                flow_tensor = torch.nn.functional.interpolate(
                    flow_tensor, size=(fh, fw), mode="bilinear", align_corners=False
                )
                # scale flow values proportionally to new resolution
                scale_h = fh / epi_flow.shape[-2]
                scale_w = fw / epi_flow.shape[-1]
                flow_tensor[:, 0] *= scale_w
                flow_tensor[:, 1] *= scale_h
                epi_flow = flow_tensor.numpy()
            flow_zarr[write_ptr:write_ptr + episode_length] = epi_flow

        write_ptr += episode_length

    for key, data in lowdim_data_dict.items():
        arr = np.concatenate(data, axis=0)
        data_group.array(
            name=key,
            data=arr,
            shape=arr.shape,
            chunks=arr.shape,
            compressor=None,
            dtype=arr.dtype,
        )

    def img_copy(
        zarr_arr: zarr.Array, zarr_idx: int, src: np.ndarray, src_idx: int
    ) -> bool:
        try:
            zarr_arr[zarr_idx] = src[src_idx]
            _ = zarr_arr[zarr_idx]
            return True
        except Exception:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures: set = set()
        for key, data in rgb_data_dict.items():
            arr = np.concatenate(data, axis=0)
            c, h, w = tuple(obs_shape_meta[key]["shape"])
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps, h, w, c),
                chunks=(1, h, w, c),
                compressor=Jpeg2k(level=50),
                dtype=np.uint8,
            )
            for idx in tqdm(range(arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in completed:
                        if not f.result():
                            raise RuntimeError("Failed to encode image!")
                futures.add(executor.submit(img_copy, img_arr, idx, arr, idx))
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    return ReplayBuffer(root)


def load_replay_buffer(
    dataset_dir: str,
    use_cache: bool,
    shape_meta: dict,
    episode_indices: List[int],
    cache_name: str = "cache",
    cache_dir: Optional[str] = None,
    flow_dir: Optional[str] = None,
    flow_split: str = "train",
    flow_storage_size: Optional[tuple] = None,
) -> ReplayBuffer:
    if cache_dir is None:
        cache_dir = dataset_dir
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_zarr_path = os.path.join(cache_dir, f"{cache_name}.zarr.zip")
        cache_lock_path = cache_zarr_path + ".lock"
        print("Acquiring lock on cache.")
        with FileLock(cache_lock_path):
            if not os.path.exists(cache_zarr_path):
                try:
                    print("Cache does not exist. Creating!")
                    replay_buffer = _convert_libero_to_replay(
                        store=zarr.MemoryStore(),
                        shape_meta=shape_meta,
                        dataset_dir=dataset_dir,
                        episode_indices=episode_indices,
                        flow_dir=flow_dir,
                        flow_split=flow_split,
                        flow_storage_size=flow_storage_size,
                    )
                    print("Saving cache to disk.")
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        replay_buffer.save_to_store(store=zip_store)
                except Exception as e:
                    if os.path.exists(cache_zarr_path):
                        os.remove(cache_zarr_path)
                    raise e
            else:
                print("Loading cached ReplayBuffer from Disk.")
                with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                    replay_buffer = ReplayBuffer.copy_from_store(
                        src_store=zip_store, store=zarr.MemoryStore()
                    )
                print("Loaded!")
    else:
        replay_buffer = _convert_libero_to_replay(
            store=zarr.MemoryStore(),
            shape_meta=shape_meta,
            dataset_dir=dataset_dir,
            episode_indices=episode_indices,
            flow_dir=flow_dir,
            flow_split=flow_split,
            flow_storage_size=flow_storage_size,
        )
    return replay_buffer


class LiberoDataset(BaseImageDataset):
    """Dataset for LIBERO in LeRobot (Parquet) format.

    Data layout on disk:
        {dataset_dir}/
            meta/info.json
            data/chunk-{chunk:03d}/episode_{index:06d}.parquet
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        shape_meta = cfg.shape_meta
        dataset_dir = cfg.dataset_dir
        horizon = cfg.horizon * cfg.skip_frame
        pad_before = cfg.pad_before
        pad_after = cfg.pad_after
        use_cache = cfg.use_cache
        val_ratio = cfg.val_ratio if "val_ratio" in cfg else 0.1
        self.val_horizon = (
            cfg.val_horizon * cfg.skip_frame if "val_horizon" in cfg else horizon
        )
        self.skip_idx = cfg.skip_idx if "skip_idx" in cfg else 1
        self.aug_mode = "none"
        self.aug = None

        info_path = os.path.join(dataset_dir, "meta", "info.json")
        with open(info_path) as f:
            info = json.load(f)
        total_episodes = info["total_episodes"]
        self._val_episode_indices = list(range(99, total_episodes, 100))
        val_set = set(self._val_episode_indices)
        train_episode_indices = [i for i in range(total_episodes) if i not in val_set]

        cache_dir = cfg.cache_dir if "cache_dir" in cfg else None
        self._cache_dir = cache_dir
        flow_dir = cfg.flow_dir if "flow_dir" in cfg else None
        self._flow_dir = flow_dir
        resolution = cfg.resolution if "resolution" in cfg else 256
        flow_storage_size = (resolution, resolution)
        self._flow_storage_size = flow_storage_size

        self.replay_buffer = load_replay_buffer(
            dataset_dir=dataset_dir,
            use_cache=use_cache,
            shape_meta=shape_meta,
            episode_indices=train_episode_indices,
            cache_name="cache_train",
            cache_dir=cache_dir,
            flow_dir=flow_dir,
            flow_split="train",
            flow_storage_size=flow_storage_size,
        )

        rgb_keys: list = []
        lowdim_keys: list = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type_ = attr.get("type", "low_dim")
            if type_ == "rgb":
                rgb_keys.append(key)
            elif type_ == "low_dim":
                lowdim_keys.append(key)

        train_mask = np.ones((self.replay_buffer.n_episodes,), dtype=bool)
        all_keys = list(self.replay_buffer.keys())

        intermediate_keys = ["action"]

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            goal_sample=cfg.goal_sample,
            keys=all_keys,
            skip_frame=cfg.skip_frame,
            keys_to_keep_intermediate=intermediate_keys,
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.depth_keys = []
        self.mask_keys = []
        self.train_mask = train_mask
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.dataset_dir = dataset_dir
        self.skip_frame = cfg.skip_frame
        self.goal_sample = cfg.goal_sample
        self.use_cache = use_cache
        self.resolution = cfg.resolution

    def get_normalizer(self, mode: str = "none", **kwargs: dict) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        stat = array_to_stats(self.replay_buffer["action"])
        normalizer["action"] = get_range_normalizer_from_stat(stat)

        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            normalizer[key] = get_range_normalizer_from_stat(stat)

        return normalizer

    def __len__(self) -> int:
        if self.is_val:
            return self.replay_buffer.n_episodes // self.skip_idx
        return len(self.sampler)

    def get_validation_dataset(self) -> "BaseImageDataset":
        val_set = copy.copy(self)
        val_set.is_val = True
        val_set.replay_buffer = load_replay_buffer(
            dataset_dir=self.dataset_dir,
            use_cache=self.use_cache,
            shape_meta=self.shape_meta,
            episode_indices=self._val_episode_indices,
            cache_name="cache_val",
            cache_dir=self._cache_dir,
            flow_dir=self._flow_dir,
            flow_split="val",
            flow_storage_size=self._flow_storage_size,
        )
        val_mask = np.ones((val_set.replay_buffer.n_episodes,), dtype=bool)
        val_keys = list(val_set.replay_buffer.keys())
        val_set.sampler = SequenceSampler(
            replay_buffer=val_set.replay_buffer,
            sequence_length=self.val_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
            skip_idx=self.skip_idx,
            goal_sample=self.goal_sample,
            skip_frame=self.skip_frame,
            keys=val_keys,
            keys_to_keep_intermediate=["action"],
        )
        val_set.train_mask = val_mask
        return val_set

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        obs_dict = dict()
        final_dict = dict()

        for key in self.rgb_keys:
            obs_images = sample[key].astype(np.uint8)
            final_images = sample[f"{key}_final"].astype(np.uint8)
            obs_dict[key] = np.moveaxis(obs_images, -1, 1).astype(np.float32) / 255.0
            final_dict[key] = (
                np.moveaxis(final_images, -1, 0).astype(np.float32) / 255.0
            )
            del sample[f"{key}_final"]
            del sample[key]

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key].astype(np.float32)
            final_dict[key] = sample[f"{key}_final"].astype(np.float32)
            del sample[f"{key}_final"]
            del sample[key]

        actions = sample["action"].astype(np.float32)
        data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "goal": dict_apply(final_dict, torch.from_numpy),
            "action": torch.from_numpy(actions),
            "is_early_stop": torch.from_numpy(np.array([sample["is_early_stop"]])),
            "rel_stop_idx": torch.from_numpy(np.array([sample["rel_stop_idx"]])),
        }
        if "flow" in sample:
            data["flow"] = torch.from_numpy(sample["flow"].astype(np.float32))
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_val:
            epi_idx = idx * self.skip_idx
            epi_start = (
                self.replay_buffer.episode_ends[epi_idx - 1] if epi_idx > 0 else 0
            )
            epi_end = self.replay_buffer.episode_ends[epi_idx]
            val_horizon = self.val_horizon
            epi_len = epi_end - epi_start
            offset = (idx % 4) * (epi_len // 4)  # 0%, 25%, 50%, 75% of episode
            epi_start = epi_start + offset
            seq_end = min(epi_end, epi_start + val_horizon)
            sample = dict()
            for key in self.sampler.keys:
                sample[key] = self.replay_buffer[key][epi_start:seq_end]
                if sample[key].shape[0] < val_horizon:
                    pad_len = val_horizon - sample[key].shape[0]
                    pad_shape = (pad_len, *np.ones_like(sample[key].shape[1:]).tolist())
                    sample_pad = np.tile(sample[key][-1:], pad_shape)
                    sample[key] = np.concatenate([sample[key], sample_pad], axis=0)
                if key in self.sampler.keys_to_keep_intermediate:
                    inter_frames = sample[key].shape[0] // self.skip_frame
                    sample_shape = list(sample[key].shape[1:])
                    sample_shape[0] = sample_shape[0] * self.skip_frame
                    sample[key] = sample[key].reshape(
                        inter_frames, self.skip_frame, *sample[key].shape[1:]
                    )
                    sample[key] = sample[key].reshape(-1, *sample_shape)
                else:
                    sample[key] = sample[key][:: self.skip_frame]
                sample[f"{key}_final"] = sample[key][-1]
                sample["is_early_stop"] = False
                sample["rel_stop_idx"] = val_horizon - 1
        else:
            sample = self.sampler.sample_sequence(idx)
        return self._sample_to_data(sample)
