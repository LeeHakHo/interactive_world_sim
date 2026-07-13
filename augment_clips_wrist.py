"""Wrist side-car for can human/robot clips: unclipped HaMeR wrist2d + valid mask + rot3d.

Why: gen_flow_render_dataset_caneef.py::human_loader synthesizes a fake "wrist" (eef
slot0 = thumb/index pinch-center midpoint, ~0.1px offset -- degenerate) and discards
the real HaMeR wrist landmark + hand rotation that ARE present in the source parquet
(human_play_eef_data/play_human_can_eef_*, columns kpt2d_crop128_wrist and
rotation_matrix_right; see build_eef_dataset_can.py + .superpowers/sdd/
audit-annotation-layer.md -- annotation layer is clean, degeneracy is introduced only
by the clips builder). This script does NOT touch the existing clips npz; it writes
SIDE-CAR files aligned back to the clips via vid/fidx (same pattern as
augment_clips_tracks3d.py).

Frame convention: kpt2d_crop128_wrist is stored in the exact downstream crop
(60,60,390,390)->128 pixel frame that gen_flow_render_dataset_caneef.CROP also uses
for the clips' own eef/tracks points (build_eef_dataset_can.py CROP_XYWH + _CROP_S),
so wrist2d_high = kpt2d_crop128_wrist / 128 lands in the SAME [0,1]-normalized crop
frame as the clips' `eef` field -- directly overlay-comparable, UNCLIPPED (may be
<0 or >1 since the workspace crop is not hand-centered). NaN in the parquet already
means "undetected" for both kpt2d_crop128_wrist and rotation_matrix_right (verified:
both are exactly NaN iff detected_right is False), so no extra detected-flag load is
needed -- it propagates through indexing automatically.

Robot side gets rot3d only (no analogous "wrist" concept): sourced from
action_right_ee_quat_xyzw, the exact column gen_flow_render_dataset_caneef.py's
robot_loader uses to build the commanded EE rotation for the robot eef points.

iws env:
  python augment_clips_wrist.py human   # -> wrist_sidecar_human.npz (needs clips_human_L24.npz)
  python augment_clips_wrist.py robot   # -> wrist_sidecar_robot.npz (needs clips_robot.npz, L48)
  python augment_clips_wrist.py both    # (default)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

OUTDIR = "outputs/flow_render_dataset_can_dual"
HUMAN_DIRS = [f"human_play_eef_data/play_human_can_eef_{i}" for i in range(1, 13)]
ROBOT_DIRS = [f"human_play_data/play_robot_can_{i}_eef" for i in range(1, 19)]


def human_vid_to_dir(vid: int) -> str:
    return HUMAN_DIRS[vid]


def robot_vid_to_dir(vid: int) -> str:
    return ROBOT_DIRS[vid - 100]


def load_human_parquet(d: str):
    df = pd.read_parquet(
        f"{d}/data/chunk-000/episode_000000.parquet",
        columns=["observation.eef.kpt2d_crop128_wrist", "observation.eef.rotation_matrix_right"])
    kw = np.stack(df["observation.eef.kpt2d_crop128_wrist"].values).astype(np.float64)   # (M,2) crop128 px
    rot = np.stack(df["observation.eef.rotation_matrix_right"].values).astype(np.float64).reshape(-1, 3, 3)
    return kw, rot


def load_robot_rot(d: str):
    df = pd.read_parquet(f"{d}/data/chunk-000/episode_000000.parquet",
                         columns=["action_right_ee_quat_xyzw"])
    quat = np.stack(df["action_right_ee_quat_xyzw"].values).astype(np.float64)  # (M,4) xyzw
    return Rotation.from_quat(quat).as_matrix()   # (M,3,3)


def gather(src: np.ndarray, fidx: np.ndarray) -> np.ndarray:
    """src: (M, ...); fidx: (n,L) int -> (n,L,...) with out-of-range rows -> NaN."""
    M = len(src)
    valid = fidx < M
    fi_c = np.clip(fidx, 0, M - 1)
    out = src[fi_c].copy()
    out[~valid] = np.nan
    return out


def augment_human(clips_path: str, out_path: str) -> None:
    z = np.load(clips_path)
    VID, FIDX = z["vid"], z["fidx"]
    N, L = FIDX.shape
    wrist_px = np.full((N, L, 2), np.nan, np.float64)
    rot3d = np.full((N, L, 3, 3), np.nan, np.float64)
    for vid in np.unique(VID):
        d = human_vid_to_dir(int(vid))
        kw, rot = load_human_parquet(d)
        rows = np.where(VID == vid)[0]
        wrist_px[rows] = gather(kw, FIDX[rows])
        rot3d[rows] = gather(rot, FIDX[rows])
        print(f"  human vid {vid} ({d}): {len(rows)} clips", flush=True)
    wrist2d_high = (wrist_px / 128.0).astype(np.float32)   # crop-norm [0,1], UNCLIPPED
    finite = np.isfinite(wrist2d_high).all(-1)
    inbounds = (wrist2d_high[..., 0] >= 0) & (wrist2d_high[..., 0] <= 1) & \
               (wrist2d_high[..., 1] >= 0) & (wrist2d_high[..., 1] <= 1)
    wrist_valid = finite & inbounds
    os.makedirs(OUTDIR, exist_ok=True)
    np.savez(out_path, wrist2d_high=wrist2d_high, wrist_valid=wrist_valid,
             rot3d=rot3d.astype(np.float32))
    print(f"saved {out_path}: detected(finite) {finite.mean()*100:.1f}%, "
          f"wrist_valid(finite&inbounds) {wrist_valid.mean()*100:.1f}%", flush=True)


def augment_robot(clips_path: str, out_path: str) -> None:
    z = np.load(clips_path)
    VID, FIDX = z["vid"], z["fidx"]
    N, L = FIDX.shape
    rot3d = np.full((N, L, 3, 3), np.nan, np.float64)
    for vid in np.unique(VID):
        d = robot_vid_to_dir(int(vid))
        rot = load_robot_rot(d)
        rows = np.where(VID == vid)[0]
        rot3d[rows] = gather(rot, FIDX[rows])
        print(f"  robot vid {vid} ({d}): {len(rows)} clips", flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    np.savez(out_path, rot3d=rot3d.astype(np.float32))
    print(f"saved {out_path}: finite(rot3d) {np.isfinite(rot3d[..., 0, 0]).mean()*100:.1f}%",
          flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("human", "both"):
        augment_human(f"{OUTDIR}/clips_human_L24.npz", f"{OUTDIR}/wrist_sidecar_human.npz")
    if which in ("robot", "both"):
        augment_robot(f"{OUTDIR}/clips_robot.npz", f"{OUTDIR}/wrist_sidecar_robot.npz")
    print("DONE", flush=True)
