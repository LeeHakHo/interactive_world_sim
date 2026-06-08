"""Build LeRobot-formatted EEF-annotated datasets for HF upload (version-aware).

For each human ep of the active version (AYTSAI_VERSION, default v3):
  src LeRobot:  human_play_data/<human_repo_leaf>/
  src phantom:  phantom/data/<processed_full_subdir>/play_human_<ver>/{ep}/
                  action_processor/actions_right_single_arm.npz   (union_indices, kpts3_2d)
                  smoothing_processor/smoothed_actions_right_single_arm.npz  (smoothed eef)
  dst:          human_play_eef_data/<eef_local_subdir>/

Adds per-frame columns aligned to the original 30fps frame_index:
  observation.eef.position_right        float32 (3,)  ROBOT-WORLD frame, z==z_plane; NaN if undetected
  observation.eef.rotation_matrix_right float32 (9,)  EE orientation in robot-world (row-major); NaN
  observation.eef.width_right           float32 (1,)  thumb-index distance (m); NaN
  observation.eef.detected_right        bool    (1,)
  observation.eef.kpt2d_crop128_wrist   float32 (2,)  HaMeR kpt0 projected into crop(190,225,210,205)->128; NaN
  observation.eef.kpt2d_crop128_thumb   float32 (2,)  HaMeR kpt4 (thumb tip) in crop128; NaN
  observation.eef.kpt2d_crop128_index   float32 (2,)  HaMeR kpt8 (index tip) in crop128; NaN

EEF 3D is plane-pinned to the robot's locked plane (one coordinate system shared
with the robot datasets). The 2D crop128 points match the downstream flow-WM crop;
points can fall outside [0,128) (the hand enters from the frame edge) -> stored as-is.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import aytsai_pipeline_config as cfg
from pin_eef_to_plane import intrinsics_matrix, pin_points_to_plane, rotate_oris_to_world

c = cfg.get()
sys.path.insert(0, str(Path(c["repo_root"]) / "interactive_world_sim" / "algorithms" / "latent_decompose"))
from ot_align import robot_world_to_cam  # type: ignore

SRC_LEROBOT_ROOT = Path(c["repo_root"]) / "human_play_data"
SRC_PHANTOM_ROOT = Path(c["repo_root"]) / "phantom" / "data" / c["processed_full_subdir"] / f"play_human_{c['version']}"
DST_ROOT = Path(c["repo_root"]) / "human_play_eef_data"

_KJ = json.load(open(Path(c["repo_root"]) / "phantom_human_play" / "intrinsics_cam_high.json"))["left"]
K = intrinsics_matrix(_KJ["fx"], _KJ["fy"], _KJ["cx"], _KJ["cy"])
T_CAM_WORLD = robot_world_to_cam()
Z_PLANE = json.load(open(Path(c["repo_root"]) / f"z_plane_{c['version']}.json"))["z_plane"]

# downstream training crop (same as the bbox detect crop) -> resize 128
CROP_XYWH = (190, 225, 210, 205)
_CROP_O = np.array([CROP_XYWH[0], CROP_XYWH[1]], dtype=np.float64)
_CROP_S = np.array([128.0 / CROP_XYWH[2], 128.0 / CROP_XYWH[3]], dtype=np.float64)

EEF_FEATURES = {
    "observation.eef.position_right": {"dtype": "float32", "shape": [3], "names": ["x", "y", "z"]},
    "observation.eef.rotation_matrix_right": {"dtype": "float32", "shape": [9], "names": None},
    "observation.eef.width_right": {"dtype": "float32", "shape": [1], "names": None},
    "observation.eef.detected_right": {"dtype": "bool", "shape": [1], "names": None},
    "observation.eef.kpt2d_crop128_wrist": {"dtype": "float32", "shape": [2], "names": ["u", "v"]},
    "observation.eef.kpt2d_crop128_thumb": {"dtype": "float32", "shape": [2], "names": ["u", "v"]},
    "observation.eef.kpt2d_crop128_index": {"dtype": "float32", "shape": [2], "names": ["u", "v"]},
}


def copy_tree_concrete(src: Path, dst: Path) -> None:
    """Mirror src into dst, materialising actual bytes (follow symlinks) so HF
    upload doesn't push broken links."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for src_path in src.rglob("*"):
        out = dst / src_path.relative_to(src)
        if src_path.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, out)


def build_eef_arrays(n_frames: int, phantom_ep_dir: Path) -> dict:
    raw = np.load(phantom_ep_dir / "action_processor" / "actions_right_single_arm.npz")
    sm = np.load(phantom_ep_dir / "smoothing_processor" / "smoothed_actions_right_single_arm.npz")
    idx = raw["union_indices"].astype(np.int64)
    ee_pts = sm["ee_pts"].astype(np.float64)           # (M,3) cam frame, smoothed
    ee_oris = sm["ee_oris"].astype(np.float64)         # (M,3,3) cam frame
    ee_widths = sm["ee_widths"].astype(np.float32)     # (M,)
    kpts3 = raw["kpts3_2d"].astype(np.float64)         # (M,3,2) full-frame px (wrist,thumb,index)

    assert idx.max() < n_frames, f"union_indices.max={idx.max()} >= n_frames={n_frames}"
    assert len(ee_pts) == len(idx) == len(kpts3)

    # 3D eef: pin cam-frame eef onto robot's locked plane -> robot-world frame
    p_world, _ = pin_points_to_plane(ee_pts, K, T_CAM_WORLD, Z_PLANE)
    oris_world = rotate_oris_to_world(ee_oris, T_CAM_WORLD)
    # 3 keypoints -> crop128 px
    k3c = (kpts3 - _CROP_O) * _CROP_S                  # (M,3,2)

    pos = np.full((n_frames, 3), np.nan, dtype=np.float32)
    rot = np.full((n_frames, 9), np.nan, dtype=np.float32)
    width = np.full((n_frames, 1), np.nan, dtype=np.float32)
    det = np.zeros((n_frames, 1), dtype=bool)
    kw = np.full((n_frames, 2), np.nan, dtype=np.float32)
    kt = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ki = np.full((n_frames, 2), np.nan, dtype=np.float32)

    pos[idx] = p_world.astype(np.float32)
    rot[idx] = oris_world.reshape(-1, 9).astype(np.float32)
    width[idx, 0] = ee_widths
    det[idx, 0] = True
    kw[idx] = k3c[:, 0, :].astype(np.float32)
    kt[idx] = k3c[:, 1, :].astype(np.float32)
    ki[idx] = k3c[:, 2, :].astype(np.float32)
    return {
        "observation.eef.position_right": pos,
        "observation.eef.rotation_matrix_right": rot,
        "observation.eef.width_right": width,
        "observation.eef.detected_right": det,
        "observation.eef.kpt2d_crop128_wrist": kw,
        "observation.eef.kpt2d_crop128_thumb": kt,
        "observation.eef.kpt2d_crop128_index": ki,
    }


def rewrite_parquet(parquet_path: Path, eef_arrays: dict) -> None:
    df = pd.read_parquet(parquet_path)
    n = len(df)
    for col, arr in eef_arrays.items():
        assert len(arr) == n, f"{col} len {len(arr)} != parquet len {n}"
        df[col] = list(arr)
    df.to_parquet(parquet_path, index=False)


def update_info_json(info_path: Path) -> None:
    info = json.loads(info_path.read_text())
    for k, v in EEF_FEATURES.items():
        info["features"][k] = v
    info_path.write_text(json.dumps(info, indent=4))


def update_readme(readme_path: Path, ep: int, rate: float) -> None:
    repo = c["human_repo_pattern"].format(n=ep)
    wrist_inb = "~90%"
    extra = f"""# {c['eef_local_subdir'].format(n=ep)}

LeRobot re-release of `{repo}` with right-hand end-effector annotations from
phantom (HaMeR) + smoothing.

**3D EEF is in the ROBOT-WORLD frame** (one shared coordinate system with the
robot datasets), pinned onto the robot's locked tabletop plane: x/y from the
reliable 2D hand detection, z fixed to z_plane = {Z_PLANE:.5f} m (mean of robot
v3 action z).

**3 keypoints (wrist/thumb/index)** are also given as 2D pixels in the downstream
crop(190,225,210,205)->resize 128 frame (matches the robot base+2-fingertip
3-point convention). Points may fall outside [0,128) when the hand enters from
the frame edge (wrist {wrist_inb} in-frame); stored as-is.

| column | dtype | shape | meaning |
|---|---|---|---|
| `observation.eef.position_right` | float32 | (3,) | EE position, robot-world frame (m); z==z_plane; NaN if none |
| `observation.eef.rotation_matrix_right` | float32 | (9,) | EE orientation 3x3 row-major, robot-world; NaN |
| `observation.eef.width_right` | float32 | (1,) | thumb-index distance (m), gripper-opening proxy; NaN |
| `observation.eef.detected_right` | bool | (1,) | True iff HaMeR detected the right hand |
| `observation.eef.kpt2d_crop128_wrist` | float32 | (2,) | HaMeR kpt0 in crop128 px; NaN |
| `observation.eef.kpt2d_crop128_thumb` | float32 | (2,) | HaMeR kpt4 (thumb tip) in crop128 px; NaN |
| `observation.eef.kpt2d_crop128_index` | float32 | (2,) | HaMeR kpt8 (index tip) in crop128 px; NaN |

Detection rate: {rate:.1f}% of frames. Original videos and parquet columns are
unchanged.
"""
    readme_path.write_text(extra)


def process_episode(ep: int) -> None:
    print(f"=== ep {ep} ===")
    src = SRC_LEROBOT_ROOT / c["human_repo_pattern"].format(n=ep).split("/")[-1]
    dst = DST_ROOT / c["eef_local_subdir"].format(n=ep)
    print(f"  copy {src} -> {dst}")
    copy_tree_concrete(src, dst)

    info_path = dst / "meta" / "info.json"
    n_frames = int(json.loads(info_path.read_text())["total_frames"])

    eef_arrays = build_eef_arrays(n_frames, SRC_PHANTOM_ROOT / str(ep))
    n_det = int(eef_arrays["observation.eef.detected_right"].sum())
    rate = 100.0 * n_det / n_frames
    print(f"  detected {n_det}/{n_frames} ({rate:.1f}%)")
    if rate < 70.0:
        print(f"  WARNING: low detection rate {rate:.1f}% for ep {ep}")

    rewrite_parquet(dst / "data" / "chunk-000" / "episode_000000.parquet", eef_arrays)
    update_info_json(info_path)
    update_readme(dst / "README.md", ep, rate)
    print(f"  done ep {ep}")


def main() -> None:
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    for ep in c["human_eps"]:
        process_episode(ep)
    print(f"\nAll {len(c['human_eps'])} datasets built under {DST_ROOT}")


if __name__ == "__main__":
    main()
