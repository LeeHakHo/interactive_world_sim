#!/usr/bin/env python3
"""Download play_robot_v3 datasets (one episode per HF repo) into a single
LeRobot-style directory, renaming each repo's episode by its repo index.

Each HF repo holds exactly one episode (episode_000000), e.g.
  aytsaiusc/play_robot_v3_0_eef, aytsaiusc/play_robot_v3_1_eef, ...
Repo index 0 is conventionally the validation episode; 1~N are training.

Output layout (consumable by scripts/convert_play_lerobot_to_hdf5.py):
  <dst>/data/chunk-000/episode_{i:06d}.parquet
  <dst>/videos/chunk-000/observation.images.cam_high/episode_{i:06d}.mp4
  <dst>/videos/chunk-000/observation.images.cam_right_wrist/episode_{i:06d}.mp4

Usage:
  python scripts/download_play_v3.py --dst data/play_robot_v3 --start 0 --end 18

Then convert (episode_000000 = val, rest = train):
  python scripts/convert_play_lerobot_to_hdf5.py \
      --src-robot data/play_robot_v3 \
      --val-stems episode_000000 \
      --dst data/play_robot_v3_hdf5 \
      --crop 195 195 256 256 --size 128
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

CAMS = [
    "observation.images.cam_high",
    "observation.images.cam_right_wrist",
]


def main():
    ap = argparse.ArgumentParser(description="Download play_robot_v3 (1 episode/repo) into one dir")
    ap.add_argument("--dst", required=True, help="Merged output directory")
    ap.add_argument("--prefix", default="aytsaiusc/play_robot_v3_", help="HF repo id prefix")
    ap.add_argument("--suffix", default="_eef", help="HF repo id suffix")
    ap.add_argument("--start", type=int, default=0, help="Start repo index (inclusive)")
    ap.add_argument("--end", type=int, default=18, help="End repo index (inclusive)")
    ap.add_argument("--tmp", default=None, help="Temp download dir (default <dst>/.tmp_downloads)")
    ap.add_argument("--resume", action="store_true", help="Skip repos already downloaded to tmp")
    args = ap.parse_args()

    dst = Path(args.dst)
    tmp = Path(args.tmp) if args.tmp else dst / ".tmp_downloads"
    tmp.mkdir(parents=True, exist_ok=True)

    n = 0
    for i in range(args.start, args.end + 1):
        repo = f"{args.prefix}{i}{args.suffix}"
        repo_tmp = tmp / f"robot_{i}"

        if args.resume and repo_tmp.exists() and any(repo_tmp.iterdir()):
            print(f"[SKIP] {repo} (already in {repo_tmp})")
        else:
            print(f"Downloading {repo} ...")
            snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(repo_tmp))

        # each repo holds exactly one episode parquet
        src_parquets = sorted((repo_tmp / "data" / "chunk-000").glob("episode_*.parquet"))
        assert len(src_parquets) == 1, f"{repo}: expected 1 episode, found {len(src_parquets)}"
        src_stem = src_parquets[0].stem
        new_stem = f"episode_{i:06d}"

        dst_pq = dst / "data" / "chunk-000" / f"{new_stem}.parquet"
        dst_pq.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_parquets[0], dst_pq)

        for cam in CAMS:
            src_mp4 = repo_tmp / "videos" / "chunk-000" / cam / f"{src_stem}.mp4"
            if src_mp4.exists():
                dst_mp4 = dst / "videos" / "chunk-000" / cam / f"{new_stem}.mp4"
                dst_mp4.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_mp4, dst_mp4)

        print(f"  {repo} -> {new_stem}")
        n += 1

    print(f"\nDone. {n} episodes merged into {dst}")
    print(f"  val  : episode_{args.start:06d}")
    print(f"  train: episode_{args.start+1:06d} ... episode_{args.end:06d}")


if __name__ == "__main__":
    main()
