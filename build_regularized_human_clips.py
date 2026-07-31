"""建 regularized human clips(双视角eef都canon化)供 dummy5 训练测human-help。"""
import numpy as np
from regularize_human_eef import robot_canon_params, regularize_eef
RB = "outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz"
HM = "outputs/flow_render_dataset_can_dual_L48/clips_human_L48_retrack_realwrist.npz"
OUT = "outputs/flow_render_dataset_can_dual_L48/clips_human_L48_retrack_realwrist_REG.npz"
zr = np.load(RB); zh = np.load(HM)
d = dict(zh)
for efk, view in [("eef", "high"), ("eef_low", "low")]:
    # robot canon 参数按对应视角
    R0, Lw = robot_canon_params({"eef": zr[efk]})
    hef = np.nan_to_num(zh[efk].astype(np.float64))
    d[efk] = regularize_eef(hef, R0, Lw).astype(np.float32)
    print(f"{efk}: R0={R0:.4f} Lw={Lw:.4f} regularized {d[efk].shape}", flush=True)
np.savez(OUT, **d)
print(f"[save] {OUT}", flush=True)
