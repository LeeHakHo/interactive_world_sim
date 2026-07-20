"""human-helps 二维图: x=robot 数据量 N, y=human-helps, 每个 action 表示一条曲线。
回答"不同 robot 数据量下, 各 action 表示的 human-helps 一样吗"。修好混训, H=20。
输出 outputs/cross_embodiment_wm/humanhelps_2d.png + .txt
"""
import os
os.environ.setdefault("SPLIT", "okfirst")
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import __main__ as _m, exp_scel_dualview_wm as W
_m.DualLWC = W.DualLWC; _m.RasterAct = W.RasterAct
p = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(p); plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
dev = "cuda"; K = W.K; DS = "outputs/flow_render_dataset_can_dual"
z = np.load(f"{DS}/clips_robot.npz"); ras = np.load(f"{DS}/raster_sidecar_robot.npz")
trA = z["tracks"].astype(np.float32); trB = np.nan_to_num(z["tracks_low"].astype(np.float32), nan=0.5)
trD_all = np.concatenate([trA, trB], 2)
ok = np.isfinite(trA).all((1, 2, 3)); ho, _ = W.split_okfirst(ok, W.HELDOUT)
B = "outputs/cross_embodiment_wm/dualview_wm_skelact"


def ev(path, act, Hn=20):
    if not os.path.exists(path):
        return None
    m = torch.load(path, map_location=dev, weights_only=False).eval()
    if act in ("raster", "rasterg"):
        efA, efB = W.load_action_tokens(act, "r", z, ras)
    else:
        efA, efB = W.load_action_tokens("dummy5", "r", z)
    e = []
    for i in range(0, len(ho), 32):
        b = ho[i:i + 32]
        trD = torch.from_numpy(trD_all[b]).float().to(dev)
        ea = torch.from_numpy(efA[b]).float().to(dev); eb = torch.from_numpy(efB[b]).float().to(dev)
        with torch.no_grad():
            pr = W.rollout_dual(m, trD, ea, eb, Hn).cpu().numpy()
        gt = np.concatenate([trA[b, K:K + Hn], trB[b, K:K + Hn]], 2)
        e.append(np.linalg.norm(pr - gt, axis=-1).mean((1, 2)) * 128)
    return float(np.concatenate(e).mean())


# (act, N) -> (r_path, rh_path)
PATHS = {
    ("dummy5", 100): (f"{B}/PILOT_dummy5_r_n100/wm_dual.pt", f"{B}/PILOT_dummy5_rh_n100/wm_dual.pt"),
    ("dummy5", 300): (f"{B}/FIX_dummy5_r_n300/wm_dual.pt", f"{B}/FIX_dummy5_rh_n300/wm_dual.pt"),
    ("dummy5", 2460): ("outputs/cross_embodiment_wm/dualview_wm/wm_dual.pt", f"{B}/SWEEP_dummy5_rh_nall/wm_dual.pt"),
    ("raster", 100): (f"{B}/PILOT_raster_r_n100/wm_dual.pt", f"{B}/PILOT_raster_rh_n100/wm_dual.pt"),
    ("raster", 300): (f"{B}/SWEEP_raster_r_n300/wm_dual.pt", f"{B}/SWEEP_raster_rh_n300/wm_dual.pt"),
    ("raster", 2460): (f"{B}/FULL_raster_r_nall/wm_dual.pt", f"{B}/SWEEP_raster_rh_nall/wm_dual.pt"),
    ("rasterg", 100): (f"{B}/PILOT_rasterg_r_n100/wm_dual.pt", f"{B}/PILOT_rasterg_rh_n100/wm_dual.pt"),
    ("rasterg", 300): (f"{B}/SWEEP_rasterg_r_n300/wm_dual.pt", f"{B}/SWEEP_rasterg_rh_n300/wm_dual.pt"),
    ("rasterg", 2460): (f"{B}/FULL_rasterg_r_nall/wm_dual.pt", f"{B}/SWEEP_rasterg_rh_nall/wm_dual.pt"),
}
COL = {"dummy5": "#1f77b4", "raster": "#2ca02c", "rasterg": "#d62728"}
Ns = [100, 300, 2460]
data = {}
lines = ["human-helps by (action 表示 × robot 数据量), 修好混训, H=20 drift px",
         f"{'表示':9s} {'N':>5s} {'ro':>7s} {'rh':>7s} {'Δ':>7s} {'相对%':>7s}"]
for act in ("dummy5", "raster", "rasterg"):
    data[act] = {"N": [], "ro": [], "rh": [], "delta": [], "pct": []}
    for N in Ns:
        rp, rhp = PATHS[(act, N)]
        ro = ev(rp, act); rh = ev(rhp, act)
        if ro and rh:
            data[act]["N"].append(N); data[act]["ro"].append(ro); data[act]["rh"].append(rh)
            data[act]["delta"].append(ro - rh); data[act]["pct"].append((ro - rh) / ro * 100)
            lines.append(f"{act:9s} {N:5d} {ro:7.2f} {rh:7.2f} {ro - rh:7.2f} {(ro - rh) / ro * 100:6.0f}%")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for act in ("dummy5", "raster", "rasterg"):
    d = data[act]
    if d["N"]:
        ax[0].plot(d["N"], d["delta"], "o-", lw=2, color=COL[act], label=act)
        ax[1].plot(d["N"], d["pct"], "o-", lw=2, color=COL[act], label=act)
for a in ax:
    a.set_xscale("log"); a.axhline(0, c="k", lw=.6); a.grid(alpha=.3, which="both"); a.legend()
    a.set_xlabel("robot 数据量 N (log)")
ax[0].set_ylabel("绝对 Δ = ro − rh (px, ↑帮更多)")
ax[0].set_title("human-helps 绝对量: 随 N 衰减\n(光栅表示 Δ 更大=基线烂假象)", fontsize=10)
ax[1].set_ylabel("相对改善 % = Δ/ro (↑帮更多)")
ax[1].set_title("★human-helps 相对量: 三表示曲线接近\n(human 作用与表示基本无关, 稀缺大→全量耗尽)", fontsize=10)
fig.suptitle("human-helps by (action 表示 × robot 数据量) | 修好混训, can_dual, H=20", fontsize=12)
fig.tight_layout()
OUT = "outputs/cross_embodiment_wm/humanhelps_2d.png"
fig.savefig(OUT, dpi=150, bbox_inches="tight")
open("outputs/cross_embodiment_wm/humanhelps_2d.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nsaved {OUT}\n=== DONE ===")
