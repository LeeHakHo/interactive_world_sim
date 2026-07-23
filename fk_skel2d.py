"""机器人关节角(joint,7)-> pinocchio FK -> 9关键点2D骨架(cam_high/cam_low, crop-norm)。

Task4 FK半份:被 keyboard demo 渲染阶段调用,把预测/合成的 joint 轨迹变成
agent-condition 骨架通道。FK+ROBOT_SKEL/ROBOT_SEGS 逐字照搬
augment_clips_skeleton.py::augment_robot 的 v1 9点路径;投影复用 Task1 已验证的
keyboard_3d_control.can_KT/project_norm/CROP(与 augment_clips_skeleton.py 的
can_rig_KT/project_batch/to_norm_batch 代数等价,同一份 REAL 标定+crop)。

只在 conda `phantom` env 跑(真 pinocchio 3.9.0 + buildModelFromUrdf);
.venv_wan/iws 的 pinocchio 是假 0.1 桩,没有 buildModelFromUrdf。
"""
import glob
import os

import numpy as np

from keyboard_3d_control import CROP, can_KT, project_norm

URDF_PATH = "Trossen_Analysis/stationary_ai.urdf"
PRECOMPUTE_DIR = "outputs/video_arch_wm/keyboard_3d/precompute"
SKEL2D_OUTDIR = "outputs/video_arch_wm/keyboard_3d/skel2d"

# --- 逐字照搬 augment_clips_skeleton.py 的 v1 9点骨架常量 ---
ROBOT_SKEL = [f"follower_right_link_{i}" for i in range(1, 7)] + [
    "follower_right_carriage_left", "follower_right_carriage_right",
    "follower_right_ee_gripper_link"]
ROBOT_SEGS = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 8), (6, 8), (7, 8)], np.int32)

_MODEL = None
_DATA = None
_FIDS = None
_JIDX = None


def _load_model():
    """Lazy: pinocchio import + URDF build only when FK is actually needed
    (keeps module importable for the shape test / non-FK callers if pinocchio
    were ever missing, though in the phantom env it is always present)."""
    global _MODEL, _DATA, _FIDS, _JIDX
    if _MODEL is not None:
        return
    import pinocchio as pin

    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    fids = [model.getFrameId(n) for n in ROBOT_SKEL]
    jidx = [model.joints[model.getJointId(f"follower_right_joint_{j}")].idx_q for j in range(6)]
    _MODEL, _DATA, _FIDS, _JIDX = model, data, fids, jidx


def fk_points(joints6: np.ndarray) -> np.ndarray:
    """joints6 (6,) -> (9,3) world-frame FK origins for ROBOT_SKEL. EXACT port of
    augment_clips_skeleton.py::augment_robot's fk_points closure."""
    import pinocchio as pin

    _load_model()
    q = pin.neutral(_MODEL)
    for k, idxq in enumerate(_JIDX):
        q[idxq] = joints6[k]
    pin.forwardKinematics(_MODEL, _DATA, q)
    pin.updateFramePlacements(_MODEL, _DATA)
    return np.stack([_DATA.oMf[f].translation for f in _FIDS])


GRIP_MAX = 0.04   # 夹爪全开宽度 (can grip 值域 0=闭..0.04=开)


def fk_skel2d_dual(joint: np.ndarray, grip: np.ndarray = None):
    """joint (H,7) -> (skel2d_high (H,9,2), skel2d_low (H,9,2), segments (8,2)).

    Uses joint[:, :6] for FK (the first 6 joint angles; JIDX are indices 0-5).
    Projects each of the 9 FK world points into cam_high and cam_low crop-norm
    2D using the SAME REAL calibration + crop as the rest of the can_dual
    pipeline (keyboard_3d_control.can_KT / project_norm / CROP).

    grip (H,) optional: 若给, 按 grip 几何闭合两指点(carriage_left/right, idx 6/7)——
    在 3D 里向两指中点收拢 t_close=clip(grip/GRIP_MAX,0,1)(照 augment v2 的闭合式,
    但保持 v1 9点/8段拓扑不变, 只挪两指点)。grip=0(闭)->两指并拢, grip=GRIP_MAX(开)->原位。
    让渲染骨架的夹爪随 grip 张合(视觉 grip 响应)。
    """
    joint = np.asarray(joint, np.float64)
    H = joint.shape[0]
    P9 = np.empty((H, 9, 3), np.float64)
    for t in range(H):
        P9[t] = fk_points(joint[t, :6])

    if grip is not None:
        t_close = np.clip(np.asarray(grip, np.float64) / GRIP_MAX, 0.0, 1.0)[:, None]  # (H,1)
        mid = 0.5 * (P9[:, 6] + P9[:, 7])                                              # (H,3) 两指中点
        P9[:, 6] = mid + (P9[:, 6] - mid) * t_close
        P9[:, 7] = mid + (P9[:, 7] - mid) * t_close

    T_high, K_high = can_KT("high")
    T_low, K_low = can_KT("low")
    skel2d_high = np.empty((H, 9, 2), np.float32)
    skel2d_low = np.empty((H, 9, 2), np.float32)
    for t in range(H):
        for p in range(9):
            skel2d_high[t, p] = project_norm(P9[t, p], T_high, K_high, CROP["high"])
            skel2d_low[t, p] = project_norm(P9[t, p], T_low, K_low, CROP["low"])

    return skel2d_high, skel2d_low, ROBOT_SEGS


def main():
    files = sorted(glob.glob(os.path.join(PRECOMPUTE_DIR, "*.npz")))
    if not files:
        print(f"no precompute files found under {PRECOMPUTE_DIR}, nothing to do")
        return
    os.makedirs(SKEL2D_OUTDIR, exist_ok=True)
    for fp in files:
        name = os.path.basename(fp)
        z = np.load(fp)
        joint = np.asarray(z["joint"])  # (H,7)
        grip = np.asarray(z["grip"]) if "grip" in z.files else None   # (H,) 几何闭合两指点
        skel2d_high, skel2d_low, segments = fk_skel2d_dual(joint, grip)
        out_path = os.path.join(SKEL2D_OUTDIR, name)
        np.savez(out_path, skel2d_high=skel2d_high, skel2d_low=skel2d_low, segments=segments)
        gsp = "" if grip is None else f" grip_close[{grip.min():.3f}..{grip.max():.3f}]"
        print(f"saved {out_path}: joint {joint.shape} -> skel2d_high {skel2d_high.shape}{gsp}", flush=True)


if __name__ == "__main__":
    main()
