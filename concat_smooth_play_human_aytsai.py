"""Concatenate per-chunk action npz outputs back to a full-episode trajectory,
run phantom's smoothing (GP for pts/widths, gaussian-SLERP for oris) once, and
ALSO carry the 3 downstream hand keypoints (wrist=kpt0, thumb=kpt4, index=kpt8)
to full-ep, aligned 1:1 with the eef detections (union_indices order).

Version-aware via aytsai_pipeline_config (AYTSAI_VERSION, default v3).

Writes under:
  phantom/data/<processed_full_subdir>/play_human_<version>/{ep}/
    action_processor/actions_right_single_arm.npz
        union_indices, ee_pts, ee_oris, ee_widths, kpts3_2d  (kpts3_2d: (N,3,2) full-frame px)
    smoothing_processor/smoothed_actions_right_single_arm.npz  (smoothed ee_pts/oris/widths)
    chunk_slices.npz   (chunk_ids, chunk_detect_counts)
"""
import argparse
import sys
from pathlib import Path
import numpy as np

import aytsai_pipeline_config as cfg

c = cfg.get()
sys.path.insert(0, str(Path(c["repo_root"]) / "phantom"))
from phantom.processors.smoothing_processor import SmoothingProcessor  # type: ignore

CHUNKS_BASE = Path(c["repo_root"]) / "phantom" / "data" / c["processed_chunks_subdir"] / c["raw_subdir"]
OUT_BASE = Path(c["repo_root"]) / "phantom" / "data" / c["processed_full_subdir"] / f"play_human_{c['version']}"
EPS = c["human_eps"]
CHUNKS_PER_EP = c["chunks_per_ep"]
CHUNK_FRAMES = 9000  # frames per chunk (5min @ 30fps)
KPT_IDS = [0, 4, 8]  # wrist, thumb tip, index tip


def concat_ep(ep: int) -> None:
    union_all, pts_all, oris_all, widths_all, kpts3_all = [], [], [], [], []
    chunk_ids, chunk_detect_counts = [], []
    for i in range(CHUNKS_PER_EP):
        chunk_dir = CHUNKS_BASE / str(ep * 1000 + i)
        npz_path = chunk_dir / "action_processor" / "actions_right_single_arm.npz"
        if not npz_path.exists():
            print(f"  [skip] chunk {i:03d} npz missing: {npz_path}")
            continue
        a = np.load(npz_path)
        union_local = a["union_indices"].astype(np.int64)
        union_all.append(union_local + i * CHUNK_FRAMES)
        pts_all.append(a["ee_pts"])
        oris_all.append(a["ee_oris"])
        widths_all.append(a["ee_widths"])

        # 3 downstream keypoints at the eef-detected frames (full-frame px, raw)
        hand_npz = chunk_dir / "hand_processor" / "hand_data_right.npz"
        h = np.load(hand_npz)
        k3 = h["kpts_2d"][union_local][:, KPT_IDS, :]  # (n_det,3,2)
        assert len(k3) == len(union_local), f"chunk {i}: kpts {len(k3)} != union {len(union_local)}"
        kpts3_all.append(k3.astype(np.float64))

        chunk_ids.append(i)
        chunk_detect_counts.append(len(a["ee_pts"]))
    if not pts_all:
        print(f"  ep {ep}: no chunks found, skipping")
        return

    union = np.concatenate(union_all)
    pts = np.concatenate(pts_all, axis=0)
    oris = np.concatenate(oris_all, axis=0)
    widths = np.concatenate(widths_all, axis=0)
    kpts3 = np.concatenate(kpts3_all, axis=0)
    print(f"  ep {ep}: concatenated {len(pts)} detections from {len(pts_all)} chunks")

    out_dir = OUT_BASE / str(ep)
    (out_dir / "action_processor").mkdir(parents=True, exist_ok=True)
    (out_dir / "smoothing_processor").mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / "action_processor" / "actions_right_single_arm.npz",
             union_indices=union, ee_pts=pts, ee_oris=oris, ee_widths=widths, kpts3_2d=kpts3)

    smoothed_pts = SmoothingProcessor.gaussian_process_smoothing(pts)
    smoothed_oris = SmoothingProcessor.gaussian_slerp_smoothing(oris, sigma=10.0)
    smoothed_widths = SmoothingProcessor.gaussian_process_smoothing(widths)
    np.savez(out_dir / "smoothing_processor" / "smoothed_actions_right_single_arm.npz",
             ee_pts=smoothed_pts, ee_oris=smoothed_oris, ee_widths=smoothed_widths)
    print(f"  ep {ep}: smoothed -> {out_dir}/smoothing_processor/")

    np.savez(out_dir / "chunk_slices.npz",
             chunk_ids=np.array(chunk_ids, dtype=np.int64),
             chunk_detect_counts=np.array(chunk_detect_counts, dtype=np.int64))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=int, nargs="+", default=EPS)
    args = parser.parse_args()
    for ep in args.eps:
        print(f"=== ep {ep} ===")
        concat_ep(ep)


if __name__ == "__main__":
    main()
