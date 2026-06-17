#!/usr/bin/env python3
"""Convert LeRobot-format play data (parquet + mp4) to IWS HDF5 format.

Source layout (play_data):
  data/chunk-000/episode_000000[_tag].parquet
  videos/chunk-000/observation.images.cam_high/episode_000000[_tag].mp4
  videos/chunk-000/observation.images.cam_low/episode_000000[_tag].mp4

Output layout:
  <out_dir>/train/episode_0.hdf5 ... episode_N.hdf5
  <out_dir>/val/episode_0.hdf5   ... episode_M.hdf5

HDF5 structure (matches real_custom_dataset.py expectations):
  action:                   (T, 14) float32  -- joint positions
  joint_action:             (T, 14) float32  -- same
  obs/joint_pos:            (T, 14) float32  -- observation.state
  obs/full_joint_pos:       (T, 14) float32  -- same
  obs/world_t_robot_base:   (T, 2, 4, 4) float32  -- identity (unused with ctrl_mode=joint)
  obs/images/camera_0_color:(T, H, W, 3) uint8    -- cam_high
  obs/images/camera_1_color:(T, H, W, 3) uint8    -- cam_low
  timestamp:                (T,) float64

Usage:
  python scripts/convert_play_lerobot_to_hdf5.py \
      --src data/play_data \
      --dst data/play_custom

Requirements (already in requirements.txt):
  pyarrow, av, h5py, numpy, tqdm
"""

import argparse
import random
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from interactive_world_sim.utils.aloha_conts import PUPPET_GRIPPER_JOINT_NORMALIZE_FN


CAM_HIGH_KEY = "observation.images.cam_high"
CAM_LOW_KEY = "observation.images.cam_low"
CAM_WRIST_KEY = "observation.images.cam_right_wrist"

# Camera pose in robot base frame (from obs_head_camera_position/quat, constant across all frames)
_CAM_POS_IN_BASE  = np.array([-0.32467514,  0.009,       1.0477746 ], dtype=np.float32)
_CAM_QUAT_XYZW    = np.array([ 0.6903455,  -0.6903455,   0.15304592, -0.15304592], dtype=np.float32)
_R_BASE_CAM       = Rotation.from_quat(_CAM_QUAT_XYZW).as_matrix().astype(np.float32)  # (3,3)


def decode_video_frames(
    video_path: Path,
    crop: tuple[int, int, int, int] | None = None,
    size: int | None = None,
) -> np.ndarray:
    """Decode all frames from an mp4 file. Returns (T, H, W, 3) uint8.

    crop: (x, y, w, h) in pixels, applied before resize
    size: output square size after crop (e.g. 128)
    """
    import cv2
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            img = frame.to_ndarray(format="rgb24")
            if crop is not None:
                x, y, w, h = crop
                img = img[y:y+h, x:x+w]
            if size is not None:
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            frames.append(img)
    return np.stack(frames, axis=0)


def read_parquet(parquet_path: Path):
    """Read parquet and return action, state, timestamp arrays.

    Human demo parquets have no action/state columns — handled with zeros.
    """
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()

    T = len(df)
    timestamp = df["timestamp"].to_numpy().astype(np.float64)

    if "action" in df.columns:
        joint_action = np.stack(df["action"].tolist(), axis=0).astype(np.float32)
    else:
        joint_action = np.zeros((T, 7), dtype=np.float32)

    if "observation.state" in df.columns:
        state = np.stack(df["observation.state"].tolist(), axis=0).astype(np.float32)
    else:
        state = np.zeros((T, 7), dtype=np.float32)

    if "obs_right_ee_position" in df.columns:
        eef_pos  = np.stack(df["obs_right_ee_position"].tolist(), axis=0).astype(np.float32)
        eef_quat = np.stack(df["obs_right_ee_quat_xyzw"].tolist(), axis=0).astype(np.float32)
        act_ee_pos   = np.stack(df["action_right_ee_position"].tolist(), axis=0).astype(np.float32)
        act_ee_euler = np.stack(df["action_right_ee_euler_xyz"].tolist(), axis=0).astype(np.float32)
        if "action_right_gripper" in df.columns:
            gripper = df["action_right_gripper"].to_numpy().astype(np.float32).reshape(-1, 1)
        else:
            gripper = PUPPET_GRIPPER_JOINT_NORMALIZE_FN(joint_action[:, 6:7])
        eef_action = np.concatenate([act_ee_pos, act_ee_euler, gripper], axis=1)
    else:
        eef_pos    = np.zeros((T, 3), dtype=np.float32)
        eef_quat   = np.zeros((T, 4), dtype=np.float32)
        eef_action = np.zeros((T, 7), dtype=np.float32)

    return joint_action, eef_action, state, timestamp, eef_pos, eef_quat


def write_hdf5(
    out_path: Path,
    joint_action: np.ndarray,
    eef_action: np.ndarray,
    state: np.ndarray,
    timestamp: np.ndarray,
    cam_high: np.ndarray,
    cam_wrist: np.ndarray,
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    is_human: bool = False,
    action_mask: np.ndarray | None = None,
) -> None:
    """Write one episode to HDF5.

    is_human: zeros out action and wrist view, sets action_mask=0 (unless action_mask provided).
    action_mask: per-frame float32 array (T,). Overrides is_human mask logic when provided.
    """
    T = joint_action.shape[0]
    assert cam_high.shape[0] == T, f"cam_high frame count {cam_high.shape[0]} != parquet rows {T}"
    assert cam_wrist.shape[0] == T, f"cam_wrist frame count {cam_wrist.shape[0]} != parquet rows {T}"

    eye = np.eye(4, dtype=np.float32)
    world_t_robot_base = np.stack([eye, eye], axis=0)[None].repeat(T, axis=0)  # (T, 2, 4, 4)

    if is_human:
        eef_action = np.zeros_like(eef_action)
        cam_wrist  = np.zeros_like(cam_wrist)
    if action_mask is None:
        action_mask = np.zeros(T, dtype=np.float32) if is_human else np.ones(T, dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["is_human"] = int(is_human)
        f.create_dataset("action", data=eef_action, dtype="float32")
        f.create_dataset("joint_action", data=joint_action, dtype="float32")
        f.create_dataset("action_mask", data=action_mask, dtype="float32")
        f.create_dataset("timestamp", data=timestamp, dtype="float64")

        obs = f.create_group("obs")
        obs.create_dataset("joint_pos", data=state, dtype="float32")
        obs.create_dataset("full_joint_pos", data=state, dtype="float32")
        obs.create_dataset("world_t_robot_base", data=world_t_robot_base, dtype="float32")
        obs.create_dataset("eef_pos", data=eef_pos, dtype="float32")
        obs.create_dataset("eef_quat", data=eef_quat, dtype="float32")

        imgs = obs.create_group("images")
        imgs.create_dataset("camera_0_color", data=cam_high, dtype="uint8")
        imgs.create_dataset("camera_1_color", data=cam_wrist, dtype="uint8")


def _interp_nan(arr: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Linear interpolation (and edge extrapolation) of undetected frames per dimension."""
    x_valid = np.where(detected)[0]
    if len(x_valid) == 0:
        return arr
    x_all = np.arange(len(arr))
    result = arr.copy()
    for d in range(arr.shape[1]):
        result[:, d] = np.interp(x_all, x_valid, arr[x_valid, d])
    return result


def read_parquet_human_eef(parquet_path: Path):
    """Read human_eef parquet (yusenluo9z/play_human_eef_*).

    EEF position/rotation are in cam_high camera frame.
    Converts to robot base frame using hardcoded extrinsics.
    NaN frames (detected_right=False) are linearly interpolated from neighboring valid frames.
    action_mask=1 for all frames after interpolation.
    """
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()

    T = len(df)
    timestamp = df["timestamp"].to_numpy().astype(np.float64)
    joint_action = np.zeros((T, 7), dtype=np.float32)
    state        = np.zeros((T, 7), dtype=np.float32)

    detected = np.array([bool(v[0]) if hasattr(v, "__len__") else bool(v)
                         for v in df["observation.eef.detected_right"]], dtype=bool)

    pos_cam      = np.stack(df["observation.eef.position_right"].tolist(),        axis=0).astype(np.float32)  # (T,3)
    rot_cam_flat = np.stack(df["observation.eef.rotation_matrix_right"].tolist(), axis=0).astype(np.float32)  # (T,9)
    width        = np.stack(df["observation.eef.width_right"].tolist(),            axis=0).astype(np.float32).reshape(T, 1)  # (T,1)

    # zero out NaN before interpolation so interp works cleanly
    pos_cam[~detected]      = 0.0
    rot_cam_flat[~detected] = 0.0
    width[~detected]        = 0.0

    # interpolate NaN frames
    pos_cam      = _interp_nan(pos_cam,      detected)
    rot_cam_flat = _interp_nan(rot_cam_flat, detected)
    width        = _interp_nan(width,        detected)

    # position: camera frame → robot base frame
    pos_base = (pos_cam @ _R_BASE_CAM.T) + _CAM_POS_IN_BASE  # (T,3)

    # rotation: camera frame → robot base frame → euler_xyz (all frames now valid)
    R_cam_eef  = rot_cam_flat.reshape(T, 3, 3)
    R_base_eef = np.einsum("ij,njk->nik", _R_BASE_CAM, R_cam_eef)  # (T,3,3)
    euler_base = Rotation.from_matrix(R_base_eef).as_euler("xyz").astype(np.float32)

    eef_action  = np.concatenate([pos_base, euler_base, width], axis=1)  # (T,7)
    eef_pos     = pos_base
    eef_quat    = np.zeros((T, 4), dtype=np.float32)
    action_mask = np.ones(T, dtype=np.float32)  # all frames valid after interpolation

    return joint_action, eef_action, state, timestamp, eef_pos, eef_quat, action_mask


def convert_episode_human_eef(
    src_dir: Path,
    stem: str,
    out_path: Path,
    crop: tuple[int, int, int, int] | None = None,
    size: int | None = None,
) -> None:
    parquet_path   = src_dir / "data"   / "chunk-000" / f"{stem}.parquet"
    cam_high_path  = src_dir / "videos" / "chunk-000" / CAM_HIGH_KEY  / f"{stem}.mp4"

    joint_action, eef_action, state, timestamp, eef_pos, eef_quat, action_mask = \
        read_parquet_human_eef(parquet_path)

    cam_high = decode_video_frames(cam_high_path, crop=crop, size=size)

    T = min(len(joint_action), len(cam_high))
    h = w = size if size else 128
    cam_wrist = np.zeros((T, h, w, 3), dtype=np.uint8)  # no wrist cam for human

    write_hdf5(
        out_path,
        joint_action[:T], eef_action[:T], state[:T], timestamp[:T],
        cam_high[:T], cam_wrist,
        eef_pos[:T], eef_quat[:T],
        is_human=True,
        action_mask=action_mask[:T],
    )


def collect_episode_stems(data_dir: Path) -> list[str]:
    """Return sorted list of episode stems (filename without .parquet)."""
    chunk = data_dir / "data" / "chunk-000"
    stems = sorted(p.stem for p in chunk.glob("episode_*.parquet"))
    return stems


def convert_episode(
    src_dir: Path,
    stem: str,
    out_path: Path,
    crop: tuple[int, int, int, int] | None = None,
    size: int | None = None,
    mask_human: bool = False,
) -> None:
    parquet_path = src_dir / "data" / "chunk-000" / f"{stem}.parquet"
    cam_high_path  = src_dir / "videos" / "chunk-000" / CAM_HIGH_KEY  / f"{stem}.mp4"
    cam_wrist_path = src_dir / "videos" / "chunk-000" / CAM_WRIST_KEY / f"{stem}.mp4"

    joint_action, eef_action, state, timestamp, eef_pos, eef_quat = read_parquet(parquet_path)
    cam_high  = decode_video_frames(cam_high_path, crop=crop, size=size)

    if cam_wrist_path.exists():
        cam_wrist = decode_video_frames(cam_wrist_path, crop=None, size=size)
    else:
        T = len(cam_high)
        h = w = size if size else 128
        cam_wrist = np.zeros((T, h, w, 3), dtype=np.uint8)

    is_human = mask_human
    T = min(len(joint_action), len(cam_high), len(cam_wrist))
    write_hdf5(out_path, joint_action[:T], eef_action[:T], state[:T], timestamp[:T],
               cam_high[:T], cam_wrist[:T], eef_pos[:T], eef_quat[:T], is_human=is_human)


def main():
    parser = argparse.ArgumentParser(description="Convert play_data LeRobot → IWS HDF5")
    parser.add_argument("--src-robot", required=True, help="Robot train source directory (e.g. data/play_v2/robot_train)")
    parser.add_argument("--src-robot-val", default=None, help="Robot val source directory (e.g. data/play_v2/robot_val). All episodes go to val/. If omitted, last robot episode from --src-robot is used.")
    parser.add_argument("--src-human", default=None, help="Human source directory (e.g. data/play_v2/human_train). Episodes included with action/wrist masked.")
    parser.add_argument("--src-human-eef", default=None, help="Human EEF source directory (e.g. data/play_v2/human_eef_train). EEF converted from camera frame to robot base frame; per-frame action_mask based on detection.")
    parser.add_argument("--dst", required=True, help="Output directory for HDF5 dataset")
    parser.add_argument("--val-stems", nargs="+", default=None,
                        help="Episode stems from --src-robot to use as val (e.g. episode_000000). "
                             "Remaining robot episodes go to train. Ignored if --src-robot-val is set.")
    parser.add_argument("--val-human", action="store_true", help="Use last human episode as val (only applies when --src-robot-val is not set)")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        default=None, help="Crop region for cam_high: x y w h (pixels)")
    parser.add_argument("--size", type=int, default=128, help="Output square size after crop (default: 128)")
    args = parser.parse_args()

    dst_dir = Path(args.dst)
    crop = tuple(args.crop) if args.crop else None

    # collect robot episodes
    robot_dir = Path(args.src_robot)
    robot_stems = collect_episode_stems(robot_dir)
    print(f"Found {len(robot_stems)} robot episodes in {robot_dir}")

    # collect human episodes
    human_stems = []
    if args.src_human:
        human_dir = Path(args.src_human)
        human_stems = collect_episode_stems(human_dir)
        print(f"Found {len(human_stems)} human episodes in {human_dir}")

    # split: robot_val dir (--src-robot-val) → val, all robot_train → train
    # fallback: last robot episode → val if --src-robot-val not given
    if args.src_robot_val:
        robot_val_dir = Path(args.src_robot_val)
        robot_val_stems = collect_episode_stems(robot_val_dir)
        print(f"Found {len(robot_val_stems)} val episodes in {robot_val_dir}")
        robot_train = robot_stems  # all train-source episodes → train
    elif args.val_stems:
        val_set = set(args.val_stems)
        missing = val_set - set(robot_stems)
        if missing:
            print(f"WARNING: --val-stems not found in {robot_dir}: {sorted(missing)}")
        robot_val_dir = robot_dir
        robot_val_stems = [s for s in robot_stems if s in val_set]
        robot_train     = [s for s in robot_stems if s not in val_set]
        print(f"Split by --val-stems: {len(robot_train)} train, {len(robot_val_stems)} val")
    else:
        robot_val_dir = robot_dir
        robot_val_stems = [robot_stems[-1]] if robot_stems else []
        robot_train = robot_stems[:-1]

    human_val   = [human_stems[-1]] if (human_stems and args.val_human) else []
    human_train = human_stems[:-1] if (human_stems and args.val_human) else human_stems

    # human_eef episodes (per-frame action_mask, no wrist cam)
    human_eef_stems = []
    if args.src_human_eef:
        human_eef_dir = Path(args.src_human_eef)
        human_eef_stems = collect_episode_stems(human_eef_dir)
        print(f"Found {len(human_eef_stems)} human_eef episodes in {human_eef_dir}")

    # train_items: (stem, src_dir, is_human, is_human_eef)
    train_items = [(s, robot_dir,    False, False) for s in robot_train] + \
                  [(s, human_dir if args.src_human else robot_dir, True, False) for s in human_train] + \
                  [(s, human_eef_dir if args.src_human_eef else robot_dir, False, True) for s in human_eef_stems]
    val_items   = [(s, robot_val_dir, False, False) for s in robot_val_stems] + \
                  [(s, human_dir if args.src_human else robot_dir, True, False) for s in human_val]

    print(f"Train: {len(train_items)} (robot={len(robot_train)}, human={len(human_train)}, human_eef={len(human_eef_stems)})")
    print(f"Val:   {len(val_items)}   (robot={len(robot_val_stems)}, human={len(human_val)})")

    for split, items in [("train", train_items), ("val", val_items)]:
        print(f"\n--- Converting {split} ({len(items)} episodes) ---")
        for i, (stem, src_dir, is_human, is_human_eef) in enumerate(tqdm(items, desc=split)):
            out_path = dst_dir / split / f"episode_{i}.hdf5"
            if is_human_eef:
                convert_episode_human_eef(src_dir, stem, out_path, crop=crop, size=args.size)
            else:
                convert_episode(src_dir, stem, out_path, crop=crop, size=args.size, mask_human=is_human)

    print(f"\nDone. Dataset written to: {dst_dir}")
    print(f"  train: {len(train_items)} episodes")
    print(f"  val:   {len(val_items)} episodes")
    print(f"\nRun with:")
    print(f"  dataset=play_custom_dataset")
    print(f"  dataset.dataset_dir={dst_dir}")


if __name__ == "__main__":
    main()
