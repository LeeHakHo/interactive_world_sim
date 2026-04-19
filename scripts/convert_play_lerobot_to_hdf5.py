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


CAM_HIGH_KEY = "observation.images.cam_high"
CAM_LOW_KEY = "observation.images.cam_low"


def decode_video_frames(video_path: Path) -> np.ndarray:
    """Decode all frames from an mp4 file. Returns (T, H, W, 3) uint8."""
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames, axis=0)


def read_parquet(parquet_path: Path):
    """Read parquet and return action, state, timestamp arrays."""
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()

    action = np.stack(df["action"].tolist(), axis=0).astype(np.float32)        # (T, 14)
    state = np.stack(df["observation.state"].tolist(), axis=0).astype(np.float32)  # (T, 14)
    timestamp = df["timestamp"].to_numpy().astype(np.float64)                  # (T,)

    return action, state, timestamp


def write_hdf5(
    out_path: Path,
    action: np.ndarray,
    state: np.ndarray,
    timestamp: np.ndarray,
    cam_high: np.ndarray,
    cam_low: np.ndarray,
) -> None:
    """Write one episode to HDF5."""
    T = action.shape[0]
    assert cam_high.shape[0] == T, f"cam_high frame count {cam_high.shape[0]} != parquet rows {T}"
    assert cam_low.shape[0] == T, f"cam_low frame count {cam_low.shape[0]} != parquet rows {T}"

    # world_t_robot_base: identity for both arms (unused by ctrl_mode=joint)
    eye = np.eye(4, dtype=np.float32)
    world_t_robot_base = np.stack([eye, eye], axis=0)[None].repeat(T, axis=0)  # (T, 2, 4, 4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("action", data=action, dtype="float32")
        f.create_dataset("joint_action", data=action, dtype="float32")
        f.create_dataset("timestamp", data=timestamp, dtype="float64")

        obs = f.create_group("obs")
        obs.create_dataset("joint_pos", data=state, dtype="float32")
        obs.create_dataset("full_joint_pos", data=state, dtype="float32")
        obs.create_dataset("world_t_robot_base", data=world_t_robot_base, dtype="float32")

        imgs = obs.create_group("images")
        imgs.create_dataset("camera_0_color", data=cam_high, dtype="uint8")
        imgs.create_dataset("camera_1_color", data=cam_low, dtype="uint8")


def collect_episode_stems(data_dir: Path) -> list[str]:
    """Return sorted list of episode stems (filename without .parquet)."""
    chunk = data_dir / "data" / "chunk-000"
    stems = sorted(p.stem for p in chunk.glob("episode_*.parquet"))
    return stems


def convert_episode(src_dir: Path, stem: str, out_path: Path) -> None:
    parquet_path = src_dir / "data" / "chunk-000" / f"{stem}.parquet"
    cam_high_path = src_dir / "videos" / "chunk-000" / CAM_HIGH_KEY / f"{stem}.mp4"
    cam_low_path = src_dir / "videos" / "chunk-000" / CAM_LOW_KEY / f"{stem}.mp4"

    action, state, timestamp = read_parquet(parquet_path)
    cam_high = decode_video_frames(cam_high_path)
    cam_low = decode_video_frames(cam_low_path)

    # Trim to shortest length in case of minor off-by-one between video and parquet
    T = min(len(action), len(cam_high), len(cam_low))
    write_hdf5(out_path, action[:T], state[:T], timestamp[:T], cam_high[:T], cam_low[:T])


def main():
    parser = argparse.ArgumentParser(description="Convert play_data LeRobot → IWS HDF5")
    parser.add_argument("--src", required=True, help="Source play_data directory")
    parser.add_argument("--dst", required=True, help="Output directory for HDF5 dataset")
    parser.add_argument("--robot_only", action="store_true", help="Exclude human episodes, use only robot data")
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    stems = collect_episode_stems(src_dir)
    print(f"Found {len(stems)} episodes in {src_dir}")

    if args.robot_only:
        stems = [s for s in stems if "_human_" not in s]
        print(f"Robot-only mode: {len(stems)} episodes after filtering")
        VAL_STEMS = {"episode_000000"}
    else:
        VAL_STEMS = {"episode_000000", "episode_000000_human_0"}

    val_stems_list = [s for s in stems if s in VAL_STEMS]
    train_stems = [s for s in stems if s not in VAL_STEMS]

    print(f"Train: {len(train_stems)}  Val: {len(val_stems_list)}")

    for split, split_stems in [("train", train_stems), ("val", val_stems_list)]:
        print(f"\n--- Converting {split} ({len(split_stems)} episodes) ---")
        for i, stem in enumerate(tqdm(split_stems, desc=split)):
            out_path = dst_dir / split / f"episode_{i}.hdf5"
            convert_episode(src_dir, stem, out_path)

    print(f"\nDone. Dataset written to: {dst_dir}")
    print(f"  train: {len(train_stems)} episodes")
    print(f"  val:   {len(val_stems_list)} episodes")
    print(f"\nRun with:")
    print(f"  dataset=play_custom_dataset")
    print(f"  dataset.dataset_dir={dst_dir}")
    print(f"  dataset.action_mode=joint")


if __name__ == "__main__":
    main()
