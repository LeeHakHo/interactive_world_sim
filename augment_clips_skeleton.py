"""Skeleton side-car: per-frame 2D OSCAR-style agent skeleton keypoints for both camera
views (cam_high top-down, cam_low oblique), for the can dual-view dataset.

Why: FLOW_WARP_REPR_LOG.md 用户指示四/candidate#8 -- draw the agent as a skeleton line
chain (wrist/joints -> lines) instead of a learned mask, as a condition for the renderer.
Deterministic from robot joints (pinocchio FK) / human hand keypoints -- NO information
from future frames (hard no-leakage requirement): robot side uses only `joint` (the
commanded action/proprioception already in clips_robot.npz, a legal WM input); human side
uses only the CURRENT frame's own hand annotation (kpts21_3d_world / kpt2d_crop128_wrist,
already computed per-frame by the existing annotation pipeline, same as eef3d/wrist_sidecar).
Nothing here peeks at t' > t.

Does NOT touch the existing clips_*.npz or wrist_sidecar_*.npz -- writes NEW side-car
files aligned back via vid/fidx (same side-car pattern as augment_clips_wrist.py /
augment_clips_tracks3d.py).

--- Robot: FK chain (ported from train_maskgen_caneef.py) ---
fk_points() is a verbatim port of train_maskgen_caneef.py's FK: pinocchio model built from
Trossen_Analysis/stationary_ai.urdf, joints6 = the first 6 of the `joint` (2700,48,7) array
already in clips_robot.npz (no need to re-read the source parquet). Chain =
[link_1..link_6, carriage_left, carriage_right, ee_gripper_link] (9 points, same SKEL list
as train_maskgen_caneef.py) -- the "URDF chain the maskgen/QC scripts use"
(skeleton_silhouette_qc.py's insight: joint ORIGINS project correctly onto the real arm;
URDF mesh vertices had an offset bug -- so we project FK frame origins, never mesh verts).
Projection = the exact same recipe as gen_flow_render_dataset_caneef.py / train_maskgen_
caneef.py: world point -> T_cw (REAL rig calib, calib/rgb_cam_calib_can_REAL.json for
cam_high, calib/rgb_cam_calib_can_low_REAL.json for cam_low) -> pinhole K -> crop-normalize
by the SAME crop box each view's clips were built with (CROP_HIGH=(60,60,390,390),
CROP_LOW=(0,0,640,480)) -- so skeleton coords land in the exact same [0,1] crop-norm frame
as `eef`/`eef_low`/`tracks`, UNCLIPPED (may exit [0,1]).

Caveat (verified numerically, see task-skel-sidecar-report.md): fk_points() does NOT drive
the two prismatic finger-carriage joints (ported as-is from train_maskgen_caneef.py), so
the two finger endpoints (carriage_left/right) always land at a fixed nominal separation
regardless of the actual commanded gripper opening -- by design, per the deliverable spec,
`grip` is carried alongside for the renderer to use as a LINE-WIDTH modulator instead
(OSCAR-style visual gripper-state cue), not as a finger-position driver. Also verified:
the chain's literal last point (ee_gripper_link) is NOT the same physical point as any of
the 3 clips `eef` slots -- it is exactly base3d + R@LINK6_TO_EE (0.0mm 3D residual over 8
random samples), i.e. the un-pulled-back nominal EE mount, ~156mm/~25-50px away from
`eef[...,0]` (=`base`, which IS exactly `link_6`, 0.0-0.1px residual) by construction. The
true "gripper tip" visual match is carriage_left/right vs eef fingertip slots 1/2
(~1.5-2.8px, well inside the 10px gate) -- that is the pair the acceptance check reports.

--- Human: wrist + fingertips + forearm stub ---
wrist = wrist_sidecar_human.npz's wrist2d_high (real HaMeR wrist, unclipped crop-norm,
already verified against the annotation pipeline). fingertip1/2 = clips eef slots 1/2
(finger points, NOT degenerate -- only slot 0 "pinch-center" was flagged degenerate in
augment_clips_wrist.py's docstring). forearm_stub direction: the task doc suggested picking
a rot3d column empirically; we tried that first (see task-skel-sidecar-report.md) and found
NO single rotation-matrix axis/sign was visually consistent across samples (the wrist
rotation encodes hand twist, not a fixed skeletal bone direction). Instead we use the
already-annotated 21-keypoint hand skeleton (`observation.eef.kpts21_3d_world`, same
per-frame annotation source as eef3d/wrist_sidecar, verified: projecting kpt0 through the
cam_high calib matches the trusted wrist2d_high to ~1-7px) -- direction = wrist -
mean(4 MCP knuckles [5,9,13,17]), i.e. "away from the hand mass, through the wrist,
into the forearm". Eyeballed on 6 random frames with visible forearm: hugs the real arm
(forearm_axis_diag.png / forearm_kpt_diag.png in the QC dump). d=0.15 crop-norm units
(~19px @128) forearm stub length, matching the task spec.

Bonus over the literal spec: for cam_low we do NOT fall back to NaN for wrist/forearm --
kpts21_3d_world is a full WORLD 3D point (unlike wrist2d_high which is only a cam_high
pixel), so it re-projects into cam_low exactly like eef3d does for eef_low. Documented
here as a deviation from the task's literal fallback instruction, justified because it
turned out to be derivable and is strictly current-frame (no leakage).

Env:
  robot mode needs pinocchio -> phantom env:
    /scr/yusenluo/anaconda3/envs/phantom/bin/python augment_clips_skeleton.py robot
  human mode needs only numpy/pandas/cv2/scipy -> iws (or phantom) env:
    /scr/yusenluo/anaconda3/envs/iws/bin/python augment_clips_skeleton.py human
  (bare `env` is broken on this cluster; call the env's python directly as above.)

Output:
  outputs/flow_render_dataset_can_dual/skel_sidecar_robot.npz
    skel2d_high (2700,48,9,2) f32, skel2d_low (2700,48,9,2) f32 [crop-norm, unclipped],
    joint_names (9,) str, segments (8,2) int32 [chain connectivity for drawing], grip (2700,48) f32.
  outputs/flow_render_dataset_can_dual/skel_sidecar_human.npz
    skel2d_high (1800,24,4,2) f32, skel2d_low (1800,24,4,2) f32 [crop-norm, unclipped, NaN
    where the frame's hand wasn't detected], joint_names (4,) str
    = ["wrist","fingertip1","fingertip2","forearm_stub"].

--- ISOMORPHIC V2 (env SKEL_V2=1, robot mode only) ---
Motivation: make the robot skeleton topology structurally identical (same J=8, same "chain
into a terminal 2-way spread" shape) to the human skeleton above, and make gripper aperture
*geometrically* visible in the keypoints themselves (v1's carriage_left/right, per Concerns
note 2, do NOT move with true grip state -- `grip` was only ever a side-channel for line
width). V2 does not replace v1 (v1 stays in production, untouched); it is written to a
SEPARATE file skel_sidecar_robot_v2.npz only when SKEL_V2=1 is set, and v1's own output path
is not written to in that mode.

Topology (8 points, isomorphic to human's wrist->fin1/fin2 + forearm_stub):
  [link_1, link_2, link_3, link_4, link_5, link_6, fingertipL', fingertipR']
  segments = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[5,7]] -- the same arm chain as v1 for
  link_1..6, then link_6 spreads to the two fingertips (mirrors human wrist->fin1/fin2), and
  the arm chain into link_6 mirrors the human forearm-stub bone. `ee_gripper_link` (v1's
  9th/last point, the un-pulled-back nominal EE mount -- see v1 Concerns note 1) is DROPPED:
  keeping it would add a second point converging on link_6 from a different direction (a
  forward "V" the human topology does not have).

Grip geometric closure (fixes v1 Concerns note 2): computed in 3D, BEFORE projection, using
the SAME per-frame pinocchio FK carriage_left/right points as v1 (`fk_points()` unmodified)
-- so this is still 100% derived from `joint`/`grip` (both already legal WM inputs in
clips_robot.npz), current-frame only, no leakage. Let car_l, car_r = the two carriage 3D
points, mid = their 3D midpoint, t = clip(grip / GRIP_MAX, 0, 1) (GRIP_MAX=0.04, the dataset's
observed max, verified via clips_robot.npz['grip'].max() ~= 0.039996; grip=0 -> fully closed
per the can-pipeline convention, e.g. grip~=0.029 = holding the can). Then:
  fingertipL' = mid + (car_l - mid) * t
  fingertipR' = mid + (car_r - mid) * t
i.e. at t=0 (fully closed) both fingertips collapse onto the 3D midpoint; at t=1 (fully open,
grip>=GRIP_MAX) they sit exactly at v1's nominal carriage_left/right. Both views are then
projected from these adjusted 3D points with the identical calibration/crop recipe as v1
(`project_batch` + `to_norm_batch`, same T_HIGH/K_HIGH/T_LOW/K_LOW/CROP_HIGH/CROP_LOW).

Output (SKEL_V2=1 only):
  outputs/flow_render_dataset_can_dual/skel_sidecar_robot_v2.npz
    skel2d_high (2700,48,8,2) f32, skel2d_low (2700,48,8,2) f32 [crop-norm, unclipped],
    joint_names (8,) str, segments (7,2) int32, grip (2700,48) f32 (verbatim copy, same as
    v1, for any downstream consumer that still wants the raw scalar alongside the now-visible
    geometric aperture).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

OUTDIR = "outputs/flow_render_dataset_can_dual"
HUMAN_DIRS = [f"human_play_eef_data/play_human_can_eef_{i}" for i in range(1, 13)]
ROBOT_DIRS = [f"human_play_data/play_robot_can_{i}_eef" for i in range(1, 19)]

CROP_HIGH = (60, 60, 390, 390)   # gen_flow_render_dataset_caneef.py CROP, VIEW=high
CROP_LOW = (0, 0, 640, 480)      # gen_flow_render_dataset_caneef.py CROP, VIEW=low
_CAL_DIR = "calib"


def can_rig_KT(view: str):
    """T_cw (world->cam, 4x4) + K (3x3), REAL rig calib. Same formula as
    gen_flow_render_dataset_caneef.can_rig_KT / train_maskgen_caneef.py's project_crop128
    (verified algebraically identical: pc = R_wc^T @ (p_world - t_wc))."""
    cal_file = "rgb_cam_calib_can_REAL.json" if view == "high" else "rgb_cam_calib_can_low_REAL.json"
    import json
    r = json.load(open(f"{_CAL_DIR}/{cal_file}"))
    R_wc = Rotation.from_quat(r["quat_xyzw"]).as_matrix()
    t_wc = np.asarray(r["t_world"])
    T = np.eye(4)
    T[:3, :3] = R_wc.T
    T[:3, 3] = -R_wc.T @ t_wc
    K = np.array([[r["f"], 0, r["cx"]], [0, r["f"], r["cy"]], [0, 0, 1.0]])
    return T, K


T_HIGH, K_HIGH = can_rig_KT("high")
T_LOW, K_LOW = can_rig_KT("low")


def project_batch(P: np.ndarray, T_cw: np.ndarray, K: np.ndarray) -> np.ndarray:
    """P: (...,3) world points -> (...,2) pixel uv; NaN where behind the camera."""
    pc = np.einsum("ij,...j->...i", T_cw[:3, :3], P) + T_cw[:3, 3]
    z = pc[..., 2]
    safe_z = np.where(z > 1e-3, z, 1.0)
    u = K[0, 0] * pc[..., 0] / safe_z + K[0, 2]
    v = K[1, 1] * pc[..., 1] / safe_z + K[1, 2]
    uv = np.stack([u, v], -1)
    uv[z <= 1e-3] = np.nan
    return uv


def to_norm_batch(uv: np.ndarray, crop) -> np.ndarray:
    x, y, w, h = crop
    out = uv.copy()
    out[..., 0] = (uv[..., 0] - x) / w
    out[..., 1] = (uv[..., 1] - y) / h
    return out


def gather(src: np.ndarray, fidx: np.ndarray) -> np.ndarray:
    """src: (M, ...); fidx: (n,L) int -> (n,L,...) with out-of-range rows -> NaN.
    Same helper as augment_clips_wrist.py::gather."""
    M = len(src)
    valid = fidx < M
    fi_c = np.clip(fidx, 0, M - 1)
    out = src[fi_c].copy()
    out[~valid] = np.nan
    return out


# ---------------------------------------------------------------------------
# Robot: FK chain (pinocchio; phantom env only)
# ---------------------------------------------------------------------------
ROBOT_SKEL = [f"follower_right_link_{i}" for i in range(1, 7)] + [
    "follower_right_carriage_left", "follower_right_carriage_right",
    "follower_right_ee_gripper_link"]
ROBOT_SEGS = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 8), (6, 8), (7, 8)], np.int32)

# --- ISOMORPHIC V2 (see module docstring) ---
SKEL_V2 = os.environ.get("SKEL_V2", "0") == "1"
GRIP_MAX = 0.04  # dataset observed max (clips_robot.npz['grip'].max() ~= 0.039996)
ROBOT_SKEL_V2 = [f"follower_right_link_{i}" for i in range(1, 7)] + [
    "follower_right_fingertipL_closed", "follower_right_fingertipR_closed"]
ROBOT_SEGS_V2 = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7)], np.int32)


def augment_robot(clips_path: str, out_path: str) -> None:
    import pinocchio as pin  # deferred: only robot mode needs the phantom env

    MODEL = pin.buildModelFromUrdf("Trossen_Analysis/stationary_ai.urdf")
    DATA = MODEL.createData()
    FIDS = [MODEL.getFrameId(n) for n in ROBOT_SKEL]
    JIDX = [MODEL.joints[MODEL.getJointId(f"follower_right_joint_{j}")].idx_q for j in range(6)]

    def fk_points(joints6: np.ndarray) -> np.ndarray:
        q = pin.neutral(MODEL)
        for k, idxq in enumerate(JIDX):
            q[idxq] = joints6[k]
        pin.forwardKinematics(MODEL, DATA, q)
        pin.updateFramePlacements(MODEL, DATA)
        return np.stack([DATA.oMf[f].translation for f in FIDS])

    z = np.load(clips_path, mmap_mode="r")
    J = np.asarray(z["joint"])       # (N,L,7)
    GRIP = np.asarray(z["grip"])     # (N,L)
    N, L = J.shape[:2]
    P9 = np.empty((N, L, 9, 3), np.float64)
    for n in range(N):
        for t in range(L):
            P9[n, t] = fk_points(J[n, t, :6].astype(np.float64))
        if n % 300 == 0:
            print(f"  robot FK {n}/{N}", flush=True)

    if not SKEL_V2:
        # v1 behavior, UNCHANGED.
        skel2d_high = to_norm_batch(project_batch(P9, T_HIGH, K_HIGH), CROP_HIGH).astype(np.float32)
        skel2d_low = to_norm_batch(project_batch(P9, T_LOW, K_LOW), CROP_LOW).astype(np.float32)
        os.makedirs(OUTDIR, exist_ok=True)
        np.savez(out_path, skel2d_high=skel2d_high, skel2d_low=skel2d_low,
                 joint_names=np.array(ROBOT_SKEL), segments=ROBOT_SEGS,
                 grip=GRIP.astype(np.float32))
        print(f"saved {out_path}: skel2d_high finite {np.isfinite(skel2d_high).all(-1).mean()*100:.1f}%, "
              f"skel2d_low finite {np.isfinite(skel2d_low).all(-1).mean()*100:.1f}%", flush=True)
        return

    # --- ISOMORPHIC V2: geometric grip closure in 3D, before projection ---
    car_l = P9[:, :, 6, :]   # (N,L,3) carriage_left, same FK point as v1
    car_r = P9[:, :, 7, :]   # (N,L,3) carriage_right
    mid = 0.5 * (car_l + car_r)
    t_close = np.clip(GRIP.astype(np.float64) / GRIP_MAX, 0.0, 1.0)[..., None]  # (N,L,1)
    fin_l = mid + (car_l - mid) * t_close
    fin_r = mid + (car_r - mid) * t_close
    P8 = np.concatenate(
        [P9[:, :, :6, :], fin_l[:, :, None, :], fin_r[:, :, None, :]], axis=2)  # (N,L,8,3)

    skel2d_high_v2 = to_norm_batch(project_batch(P8, T_HIGH, K_HIGH), CROP_HIGH).astype(np.float32)
    skel2d_low_v2 = to_norm_batch(project_batch(P8, T_LOW, K_LOW), CROP_LOW).astype(np.float32)
    v2_path = out_path.replace("skel_sidecar_robot.npz", "skel_sidecar_robot_v2.npz")
    assert v2_path != out_path, f"refusing to overwrite v1 output: {out_path}"
    os.makedirs(OUTDIR, exist_ok=True)
    np.savez(v2_path, skel2d_high=skel2d_high_v2, skel2d_low=skel2d_low_v2,
             joint_names=np.array(ROBOT_SKEL_V2), segments=ROBOT_SEGS_V2,
             grip=GRIP.astype(np.float32))
    print(f"saved {v2_path}: skel2d_high finite {np.isfinite(skel2d_high_v2).all(-1).mean()*100:.1f}%, "
          f"skel2d_low finite {np.isfinite(skel2d_low_v2).all(-1).mean()*100:.1f}%", flush=True)


# ---------------------------------------------------------------------------
# Human: wrist (sidecar) + fingertips (clips eef) + forearm stub (kpts21_3d_world)
# ---------------------------------------------------------------------------
HUMAN_JOINT_NAMES = ["wrist", "fingertip1", "fingertip2", "forearm_stub"]
MCP_IDX = [5, 9, 13, 17]           # index/middle/ring/pinky MCP knuckles (HaMeR/MediaPipe 21kp)
FOREARM_PROBE_M = 0.08             # 3D probe offset (m) used only to sense a 2D direction
FOREARM_STUB_D = 0.15              # stub length, crop-norm units (task spec)


def augment_human(clips_path: str, wrist_sidecar_path: str, out_path: str) -> None:
    z = np.load(clips_path, mmap_mode="r")
    ws = np.load(wrist_sidecar_path)
    VID, FIDX = np.asarray(z["vid"]), np.asarray(z["fidx"])
    eef_high = np.asarray(z["eef"], np.float64)       # (N,L,3,2) crop-norm cam_high
    eef_low = np.asarray(z["eef_low"], np.float64)    # (N,L,3,2) crop-norm cam_low
    wrist2d_high = np.asarray(ws["wrist2d_high"], np.float64)   # (N,L,2) crop-norm, unclipped
    N, L = FIDX.shape

    kpts_wrist = np.full((N, L, 3), np.nan)
    kpts_mcp = np.full((N, L, 3), np.nan)
    for vid in np.unique(VID):
        d = HUMAN_DIRS[int(vid)]
        df = pd.read_parquet(f"{d}/data/chunk-000/episode_000000.parquet",
                              columns=["observation.eef.kpts21_3d_world"])
        kp = np.stack(df["observation.eef.kpts21_3d_world"].values).reshape(-1, 21, 3)
        rows = np.where(VID == vid)[0]
        g = gather(kp, FIDX[rows])          # (n,L,21,3)
        kpts_wrist[rows] = g[:, :, 0]
        kpts_mcp[rows] = g[:, :, MCP_IDX].mean(2)
        print(f"  human vid {vid} ({d}): {len(rows)} clips", flush=True)

    dir3d = kpts_wrist - kpts_mcp
    dir3d = dir3d / (np.linalg.norm(dir3d, axis=-1, keepdims=True) + 1e-9)
    probe3d = kpts_wrist + FOREARM_PROBE_M * dir3d

    def stub_2d(anchor2d, T, K, crop):
        u0 = to_norm_batch(project_batch(kpts_wrist, T, K), crop)
        u1 = to_norm_batch(project_batch(probe3d, T, K), crop)
        d2 = u1 - u0
        d2 = d2 / (np.linalg.norm(d2, axis=-1, keepdims=True) + 1e-9)
        return anchor2d + FOREARM_STUB_D * d2

    forearm_high = stub_2d(wrist2d_high, T_HIGH, K_HIGH, CROP_HIGH)
    wrist2d_low = to_norm_batch(project_batch(kpts_wrist, T_LOW, K_LOW), CROP_LOW)
    forearm_low = stub_2d(wrist2d_low, T_LOW, K_LOW, CROP_LOW)

    skel2d_high = np.stack(
        [wrist2d_high, eef_high[..., 1, :], eef_high[..., 2, :], forearm_high], 2).astype(np.float32)
    skel2d_low = np.stack(
        [wrist2d_low, eef_low[..., 1, :], eef_low[..., 2, :], forearm_low], 2).astype(np.float32)

    os.makedirs(OUTDIR, exist_ok=True)
    np.savez(out_path, skel2d_high=skel2d_high, skel2d_low=skel2d_low,
             joint_names=np.array(HUMAN_JOINT_NAMES))
    print(f"saved {out_path}: high finite {np.isfinite(skel2d_high).all(-1).mean()*100:.1f}%, "
          f"low finite {np.isfinite(skel2d_low).all(-1).mean()*100:.1f}%", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("robot", "both"):
        augment_robot(f"{OUTDIR}/clips_robot.npz", f"{OUTDIR}/skel_sidecar_robot.npz")
    if which in ("human", "both"):
        augment_human(f"{OUTDIR}/clips_human_L24.npz", f"{OUTDIR}/wrist_sidecar_human.npz",
                       f"{OUTDIR}/skel_sidecar_human.npz")
    print("DONE", flush=True)
