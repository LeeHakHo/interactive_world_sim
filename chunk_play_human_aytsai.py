"""Pre-slice each full play_human (active version) cam_high video into 5-min
sub-clips for the phantom pipeline. Chunk dir names are pure integers
ep*1000+chunk_idx (phantom int()-parses them)."""
import os
import subprocess
from pathlib import Path

import aytsai_pipeline_config as cfg

c = cfg.get()
SRC_BASE = Path(c["repo_root"]) / "human_play_data"
RAW_BASE = Path(c["repo_root"]) / "phantom" / "data" / "raw"
EPS = c["human_eps"]
FPS = 30
CHUNK_SEC = 300
CHUNK_FRAMES = FPS * CHUNK_SEC  # 9000


def total_frames(video: Path) -> int:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "csv=p=0", str(video)], text=True).strip()
    return int(out)


def chunk_video(src: Path, ep: int, dst_root: Path, n_chunks: int) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    for i in range(n_chunks):
        demo_num = cfg.demo_num(ep, i)
        dst = dst_root / str(demo_num) / "video_L.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f"[skip] {dst} exists")
            continue
        start = i * CHUNK_SEC
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-i", str(src),
             "-t", str(CHUNK_SEC), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "fast", str(dst)], check=True)
        print(f"[ok] chunk {i:03d} -> {dst}")


def main() -> None:
    for ep in EPS:
        repo_leaf = c["human_repo_pattern"].format(n=ep).split("/")[-1]
        src = SRC_BASE / repo_leaf / "videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
        nframes = total_frames(src)
        n_chunks = (nframes + CHUNK_FRAMES - 1) // CHUNK_FRAMES
        print(f"ep {ep} ({repo_leaf}): {nframes} frames -> {n_chunks} chunks")
        chunk_video(src, ep, RAW_BASE / c["raw_subdir"], n_chunks)

        full_dir = RAW_BASE / f"play_human_{c['version']}" / str(ep)
        full_dir.mkdir(parents=True, exist_ok=True)
        link = full_dir / "video_L.mp4"
        if not link.exists():
            link.symlink_to(src)
            print(f"[ok] symlink full ep{ep} -> {link}")


if __name__ == "__main__":
    main()
