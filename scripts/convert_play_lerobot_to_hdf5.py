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
from tqdm import tqdm

from interactive_world_sim.utils.aloha_conts import PUPPET_GRIPPER_JOINT_NORMALIZE_FN


CAM_HIGH_KEY = "observation.images.cam_high"
CAM_LOW_KEY = "observation.images.cam_low"
CAM_WRIST_KEY = "observation.images.cam_right_wrist"


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
    """Read parquet and return action, state, timestamp arrays."""
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()

    joint_action = np.stack(df["action"].tolist(), axis=0).astype(np.float32)      # (T, 7)
    state = np.stack(df["observation.state"].tolist(), axis=0).astype(np.float32)  # (T, 7)
    timestamp = df["timestamp"].to_numpy().astype(np.float64)                      # (T,)

    T = len(joint_action)
    if "obs_right_ee_position" in df.columns:
        eef_pos  = np.stack(df["obs_right_ee_position"].tolist(), axis=0).astype(np.float32)
        eef_quat = np.stack(df["obs_right_ee_quat_xyzw"].tolist(), axis=0).astype(np.float32)
        act_ee_pos   = np.stack(df["action_right_ee_position"].tolist(), axis=0).astype(np.float32)
        act_ee_euler = np.stack(df["action_right_ee_euler_xyz"].tolist(), axis=0).astype(np.float32)
        gripper      = PUPPET_GRIPPER_JOINT_NORMALIZE_FN(joint_action[:, 6:7])
        eef_action   = np.concatenate([act_ee_pos, act_ee_euler, gripper], axis=1)
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
) -> None:
    """Write one episode to HDF5.

    is_human: if True, zeros out action and wrist view, sets action_mask=0.
    """
    T = joint_action.shape[0]
    assert cam_high.shape[0] == T, f"cam_high frame count {cam_high.shape[0]} != parquet rows {T}"
    assert cam_wrist.shape[0] == T, f"cam_wrist frame count {cam_wrist.shape[0]} != parquet rows {T}"

    eye = np.eye(4, dtype=np.float32)
    world_t_robot_base = np.stack([eye, eye], axis=0)[None].repeat(T, axis=0)  # (T, 2, 4, 4)

    if is_human:
        eef_action = np.zeros_like(eef_action)
        cam_wrist  = np.zeros_like(cam_wrist)
    action_mask = np.zeros(T, dtype=np.float32) if is_human else np.ones(T, dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
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

    is_human = mask_human and ("_human_" in stem)
    T = min(len(joint_action), len(cam_high), len(cam_wrist))
    write_hdf5(out_path, joint_action[:T], eef_action[:T], state[:T], timestamp[:T],
               cam_high[:T], cam_wrist[:T], eef_pos[:T], eef_quat[:T], is_human=is_human)


def main():
    parser = argparse.ArgumentParser(description="Convert play_data LeRobot → IWS HDF5")
    parser.add_argument("--src-robot", required=True, help="Robot source directory (e.g. data/play_single/robot)")
    parser.add_argument("--src-human", default=None, help="Human source directory (e.g. data/play_single/human). If provided, human episodes are included with action/wrist masked.")
    parser.add_argument("--dst", required=True, help="Output directory for HDF5 dataset")
    parser.add_argument("--val-human", action="store_true", help="Use first human episode as val (default: use first robot episode)")
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

    # split: first robot episode = val, rest = train
    # (or first human episode = val if --val-human)
    robot_val   = [robot_stems[-1]] if robot_stems else []
    robot_train = robot_stems[:-1]
    human_val   = [human_stems[-1]] if (human_stems and args.val_human) else []
    human_train = human_stems[:-1] if (human_stems and args.val_human) else human_stems

    train_items = [(s, robot_dir, False) for s in robot_train] + \
                  [(s, human_dir if args.src_human else robot_dir, True) for s in human_train]
    val_items   = [(s, robot_dir, False) for s in robot_val] + \
                  [(s, human_dir if args.src_human else robot_dir, True) for s in human_val]

    print(f"Train: {len(train_items)} (robot={len(robot_train)}, human={len(human_train)})")
    print(f"Val:   {len(val_items)}   (robot={len(robot_val)}, human={len(human_val)})")

    for split, items in [("train", train_items), ("val", val_items)]:
        print(f"\n--- Converting {split} ({len(items)} episodes) ---")
        for i, (stem, src_dir, is_human) in enumerate(tqdm(items, desc=split)):
            out_path = dst_dir / split / f"episode_{i}.hdf5"
            convert_episode(src_dir, stem, out_path, crop=crop, size=args.size, mask_human=is_human)

    print(f"\nDone. Dataset written to: {dst_dir}")
    print(f"  train: {len(train_items)} episodes")
    print(f"  val:   {len(val_items)} episodes")
    print(f"\nRun with:")
    print(f"  dataset=play_custom_dataset")
    print(f"  dataset.dataset_dir={dst_dir}")


if __name__ == "__main__":
    main()
