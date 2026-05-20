#!/usr/bin/env python3
"""Download and merge robot + human play demo datasets from Hugging Face.

Downloads ykorkmaz/play_robot_demo_0~20 and ykorkmaz/play_human_demo_0~20,
then merges them into a single local directory.

Usage:
    python scripts/download_and_merge_play_data.py --local_dir data/play_data
    python scripts/download_and_merge_play_data.py --local_dir data/play_data --start 0 --end 20

    # Resume interrupted download (skips already-downloaded repos)
    python scripts/download_and_merge_play_data.py --local_dir data/play_data --resume
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


ROBOT_REPO_PREFIX = "aytsaiusc/play_robot_"
ROBOT_REPO_SUFFIX = "_eef"
HUMAN_REPO_PREFIX = "aytsaiusc/play_human_"
HUMAN_REPO_SUFFIX = ""
HUMAN_EEF_REPO_PREFIX = "yusenluo9z/play_human_eef_"
HUMAN_EEF_REPO_SUFFIX = ""


def download_repo(repo_id: str, local_dir: Path, resume: bool) -> bool:
    """Download a single HF dataset repo. Returns True on success."""
    if resume and local_dir.exists() and any(local_dir.iterdir()):
        print(f"  [SKIP] {repo_id} — already exists at {local_dir}")
        return True

    print(f"  Downloading {repo_id} → {local_dir} ...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
        )
        print(f"  Done: {repo_id}")
        return True
    except Exception as e:
        print(f"  ERROR downloading {repo_id}: {e}")
        return False


def merge_into(src_dir: Path, dst_dir: Path, tag: str, index: int):
    """Copy/move files from src_dir into dst_dir, namespacing by tag and index."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(src_dir.iterdir()):
        # Skip HF metadata files
        if item.name.startswith(".") or item.name == "README.md":
            continue
        dest = dst_dir / item.name
        if item.is_dir():
            if dest.exists():
                # Merge subdirectory contents
                merge_into(item, dest, tag, index)
            else:
                shutil.move(str(item), dest)
        else:
            if dest.exists():
                # Rename to avoid collision: e.g. episode_0.zarr → episode_0_robot_3.zarr
                stem = item.stem
                suffix = "".join(item.suffixes)
                dest = dst_dir / f"{stem}_{tag}_{index}{suffix}"
            shutil.move(str(item), dest)


def main():
    parser = argparse.ArgumentParser(description="Download and merge play demo datasets")
    parser.add_argument("--local_dir", required=True, help="Output directory for merged dataset")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive, default 0)")
    parser.add_argument("--end", type=int, default=20, help="End index (inclusive, default 20)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip repos whose local download dir already exists",
    )
    parser.add_argument(
        "--tmp_dir",
        default=None,
        help="Temporary directory for per-repo downloads (default: <local_dir>/.tmp_downloads)",
    )
    parser.add_argument(
        "--robot_only",
        action="store_true",
        help="Download only robot repos, skip human",
    )
    parser.add_argument(
        "--human_only",
        action="store_true",
        help="Download only human repos, skip robot",
    )
    parser.add_argument(
        "--human_eef_only",
        action="store_true",
        help="Download only human_eef repos (yusenluo9z/play_human_eef_*)",
    )
    parser.add_argument(
        "--human_end",
        type=int,
        default=None,
        help="End index for human repos (default: same as --end)",
    )
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else local_dir / ".tmp_downloads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(args.start, args.end + 1))
    human_end = args.human_end if args.human_end is not None else args.end
    human_indices = list(range(args.start, human_end + 1))
    repos = []
    only_flags = [args.robot_only, args.human_only, args.human_eef_only]
    any_only = any(only_flags)
    if not any_only or args.robot_only:
        repos += [(ROBOT_REPO_PREFIX + str(i) + ROBOT_REPO_SUFFIX, "robot", i) for i in indices]
    if not any_only or args.human_only:
        repos += [(HUMAN_REPO_PREFIX + str(i) + HUMAN_REPO_SUFFIX, "human", i) for i in human_indices]
    if not any_only or args.human_eef_only:
        repos += [(HUMAN_EEF_REPO_PREFIX + str(i) + HUMAN_EEF_REPO_SUFFIX, "human_eef", i) for i in human_indices]

    print(f"\n{'='*60}")
    print(f"Downloading {len(repos)} repos ({len(indices)} robot + {len(indices)} human)")
    print(f"Tmp dir  : {tmp_dir}")
    print(f"Output   : {local_dir}")
    print(f"{'='*60}\n")

    failed = []
    for repo_id, tag, idx in repos:
        repo_tmp = tmp_dir / f"{tag}_{idx}"
        ok = download_repo(repo_id, repo_tmp, resume=args.resume)
        if not ok:
            failed.append(repo_id)
            continue

        print(f"  Merging {tag}_{idx} into {local_dir} ...")
        merge_into(repo_tmp, local_dir, tag, idx)

    print(f"\n{'='*60}")
    if failed:
        print(f"Completed with {len(failed)} failure(s):")
        for r in failed:
            print(f"  - {r}")
    else:
        print(f"All {len(repos)} repos downloaded and merged successfully.")
    print(f"Merged dataset at: {local_dir}")
    print(f"{'='*60}\n")

    # Optionally show merged directory tree (top level only)
    print("Top-level contents of merged dataset:")
    for p in sorted(local_dir.iterdir()):
        if p.name == ".tmp_downloads":
            continue
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
        print(f"  {p.name:40s}  {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
