"""解人类演示的 robot 动作(BPP/Do-As-I-Do): 3D IDM 吃 human(3D物体流+3D手轨迹)→ robot Δjoint
→ 从 robot canonical joint_0 积分 → FK robot 骨架 → 叠在人类画面(看机器人是否复现人类动作)。
env: phantom(FK)。CK=模型目录(默认C1); HSEQS=人类clip。"""
import os, sys, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "."); os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ["ACTION"] = "mp"
import exp_scel_dualview_wm as W
import idm_data as D, idm_model as M
from fk_skel2d import fk_skel2d_dual
setattr(sys.modules["__main__"], "DualLWC", W.DualLWC)

CK = os.environ.get("CK", "outputs/idm_derisk/C1")
OUT = os.environ.get("OUT_DIR", "outputs/idm_derisk/human_solve")
os.makedirs(OUT, exist_ok=True)
KP, FF = 4, 4
zr = np.load("outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz")
zh = np.load("outputs/flow_render_dataset_can_dual/clips_human_L24_retrack.npz")   # 有 tracks3d+eef3d
ck = torch.load(f"{CK}/idm.pt", map_location="cpu", weights_only=False)
XF = ck.get("model") == "xf"
if XF:
    m = M.IDMTransformer(C=ck["C"], N=ck["N"], W=ck["W"], ptype=ck["ptype"])
else:
    m = M.IDM(din=ck["din"], hidden=ck.get("hid", 512), n_layers=ck.get("nlayer", 3))
m.load_state_dict(ck["state"]); m.eval()
spec = ck["spec"]; xm, xs, ym, ys = ck["x_mean"], ck["x_std"], ck["y_mean"], ck["y_std"]
J0 = zr["joint"][:, 0, :6].mean(0)                                    # robot canonical 起始位姿(锚)
HSEQS = [int(x) for x in os.environ.get("HSEQS", "0,1,2,3").split(",")]


def solve(ci):
    if XF:
        Xt, _, _, _ = D.build_windows_tokens(zh, spec, KP, FF, clips=[ci])
        with torch.no_grad(): Yp = m(torch.tensor((Xt - xm) / xs).float()).numpy() * ys + ym
    else:
        X, _, _ = D.build_windows(zh, spec, KP, FF, clips=[ci])
        with torch.no_grad(): Yp = m(torch.tensor((X - xm) / xs).float()).numpy() * ys + ym
    dj, gp = Yp[:, :6], Yp[:, 6]
    jt = np.zeros((len(dj) + 1, 6)); jt[0] = J0
    for t in range(len(dj)): jt[t + 1] = jt[t] + dj[t]
    grip = np.concatenate([gp, gp[-1:]])
    sh, sl, _ = fk_skel2d_dual(np.concatenate([jt, np.zeros((len(jt), 1))], 1), grip)   # (T,9,2)x2
    return sh, grip


BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (6, 8), (7, 8)]   # 9点骨架连线
S = 128
ncol = 6
for ci in HSEQS:
    sh, grip = solve(ci)
    fr = zh["frames"][ci]; heef = np.nan_to_num(zh["eef"][ci].astype(np.float32))   # (L,3,2) human 手
    L = len(fr); idxs = np.linspace(0, min(L, len(sh)) - 1, ncol).astype(int)
    fig, axes = plt.subplots(1, ncol, figsize=(3 * ncol, 3.2))
    for ax, t in zip(axes, idxs):
        ax.imshow(fr[t].astype(np.uint8)); ax.axis("off"); ax.set_title(f"t={t}", fontsize=8)
        sk = sh[t] * S
        for a, b in BONES:
            ax.plot([sk[a, 0], sk[b, 0]], [sk[a, 1], sk[b, 1]], "-", c="cyan", lw=1.6)
        ax.scatter(sk[:, 0], sk[:, 1], c="cyan", s=10, zorder=3)
        hp = heef[t] * S
        ax.scatter(hp[:, 0], hp[:, 1], c="yellow", s=40, marker="*", edgecolors="k", linewidths=0.5, zorder=4)
    fig.suptitle(f"human seq{ci}: 青=IDM解出的robot骨架 叠在人类画面 | 黄=人类手(参照). 看机器人是否复现人类动作", fontsize=10)
    p = f"{OUT}/human_solve_seq{ci}.png"; plt.savefig(p, dpi=110, bbox_inches="tight"); plt.close()
    print(f"[fig] {p} grip[min,max]={grip.min():.3f},{grip.max():.3f}", flush=True)
print("DONE", flush=True)
