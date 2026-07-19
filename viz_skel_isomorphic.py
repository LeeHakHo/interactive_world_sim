"""同构骨架(skel v2)中间产物核验图 + gif.

回答: robot 的 4 个动作点和 human 的 4 个动作点, 语义上真的一一对应吗? 抓握开合反映得出来吗?

对应关系(exp_scel_dualview_wm.SKEL_ROBOT_IDX = [5,6,7,4]):
  槽位0  robot link_6        <-> human wrist        (腕/手掌根)
  槽位1  robot fingertipL'   <-> human fingertip1   (指尖1)
  槽位2  robot fingertipR'   <-> human fingertip2   (指尖2)
  槽位3  robot link_5        <-> human forearm_stub (前臂根/接近方向)

输出 outputs/cross_embodiment_wm/skel_isomorphic_qc/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import imageio.v2 as imageio

for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

DS = "outputs/flow_render_dataset_can_dual"
OUT = "outputs/cross_embodiment_wm/skel_isomorphic_qc"
os.makedirs(OUT, exist_ok=True)
SKEL_ROBOT_IDX = [5, 6, 7, 4]
SLOT = ["槽位0 腕/掌根", "槽位1 指尖A", "槽位2 指尖B", "槽位3 前臂根"]
RNAME = ["link_6", "fingertipL'", "fingertipR'", "link_5"]
HNAME = ["wrist", "fingertip1", "fingertip2", "forearm_stub"]
COL = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]      # 槽位配色, 两域一致
SEG = [(3, 0), (0, 1), (0, 2)]                           # 前臂根->腕->两指尖


def draw(ax, img, pts, title, note=""):
    ax.imshow(img)
    for a, b in SEG:
        if np.isfinite(pts[[a, b]]).all():
            ax.plot(pts[[a, b], 0], pts[[a, b], 1], "-", c="w", lw=2.6, zorder=2)
            ax.plot(pts[[a, b], 0], pts[[a, b], 1], "-", c="k", lw=1.2, zorder=3)
    for i, p in enumerate(pts):
        if np.isfinite(p).all():
            ax.scatter(*p, s=95, c=COL[i], edgecolors="w", linewidths=1.6, zorder=4)
            ax.annotate(str(i), p, c="w", fontsize=8, ha="center", va="center", zorder=5,
                        fontweight="bold")
    ax.set_title(title + (f"\n{note}" if note else ""), fontsize=9)
    ax.axis("off")


def grip_width(pts):
    """指尖A-指尖B 的像素距离 = 抓握开合的直接读数."""
    return float(np.linalg.norm(pts[1] - pts[2])) if np.isfinite(pts[[1, 2]]).all() else np.nan


def main():
    zr = np.load(f"{DS}/clips_robot.npz"); zh = np.load(f"{DS}/clips_human_L24.npz")
    skr = np.load(f"{DS}/skel_sidecar_robot_v2.npz"); skh = np.load(f"{DS}/skel_sidecar_human.npz")
    rf, hf = zr["frames"], zh["frames"]
    rs = skr["skel2d_high"][:, :, SKEL_ROBOT_IDX] * 128.0          # (N,L,4,2) 归一化->像素
    hs = skh["skel2d_high"][:, :, :4] * 128.0
    rg = zr["grip"]                                                 # robot 真实夹爪开合(ground truth)

    # 挑开合差异最大的 robot clip: 用真实 grip 信号
    gspan = np.nan_to_num(rg.max(1) - rg.min(1))
    ri = int(np.argsort(-gspan)[0])
    ropen = int(np.nanargmax(rg[ri])); rclose = int(np.nanargmin(rg[ri]))
    # human 挑指尖距离变化最大的
    hw = np.array([[grip_width(hs[i, t]) for t in range(hs.shape[1])] for i in range(min(300, len(hs)))])
    hspan = np.nanmax(hw, 1) - np.nanmin(hw, 1)
    hi = int(np.nanargmax(hspan)); hopen = int(np.nanargmax(hw[hi])); hclose = int(np.nanargmin(hw[hi]))

    # ---- 图1: 同构对应 + 开合 ----
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.8))
    draw(axes[0, 0], rf[ri, ropen], rs[ri, ropen], f"ROBOT 张开  clip{ri} f{ropen}",
         f"指尖间距 {grip_width(rs[ri, ropen]):.1f}px | 真实grip {rg[ri, ropen]:.3f}")
    draw(axes[0, 1], rf[ri, rclose], rs[ri, rclose], f"ROBOT 闭合  clip{ri} f{rclose}",
         f"指尖间距 {grip_width(rs[ri, rclose]):.1f}px | 真实grip {rg[ri, rclose]:.3f}")
    draw(axes[1, 0], hf[hi, hopen], hs[hi, hopen], f"HUMAN 张开  clip{hi} f{hopen}",
         f"指尖间距 {grip_width(hs[hi, hopen]):.1f}px")
    draw(axes[1, 1], hf[hi, hclose], hs[hi, hclose], f"HUMAN 闭合  clip{hi} f{hclose}",
         f"指尖间距 {grip_width(hs[hi, hclose]):.1f}px")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=9, mfc=COL[i], mec="w",
                          label=f"{i}  {SLOT[i]}   robot={RNAME[i]}  human={HNAME[i]}") for i in range(4)]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("同构骨架 skel-v2 核验:robot 4 点 ←→ human 4 点(同色=同槽位)\n"
                 "上排 robot,下排 human;线段 = 前臂根→腕→两指尖", fontsize=10.5)
    fig.tight_layout(rect=[0, 0.14, 1, 0.95])
    fig.savefig(f"{OUT}/01_同构对应与开合.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图2: 指尖间距时间曲线 vs robot 真实 grip(骨架能否反映抓握) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    w = [grip_width(rs[ri, t]) for t in range(rs.shape[1])]
    ax2 = axes[0]; ax2.plot(w, c="#d62728", lw=2, label="骨架指尖间距(px)")
    ax2b = ax2.twinx(); ax2b.plot(rg[ri], c="#1f77b4", lw=2, ls="--", label="真实夹爪 grip")
    ax2.set_title(f"ROBOT clip{ri}:骨架读数 vs 真实夹爪信号", fontsize=10)
    ax2.set_xlabel("帧"); ax2.set_ylabel("指尖间距 px", color="#d62728"); ax2b.set_ylabel("grip", color="#1f77b4")
    cc = np.corrcoef(np.nan_to_num(w), np.nan_to_num(rg[ri]))[0, 1]
    ax2.text(0.03, 0.06, f"相关系数 r = {cc:.3f}", transform=ax2.transAxes,
             fontsize=10, bbox=dict(fc="w", alpha=.85))
    axes[1].plot(hw[hi], c="#d62728", lw=2)
    axes[1].set_title(f"HUMAN clip{hi}:骨架指尖间距(无真实 grip 可对照)", fontsize=10)
    axes[1].set_xlabel("帧"); axes[1].set_ylabel("指尖间距 px")
    for a in (ax2, axes[1]): a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/02_开合信号对照.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图3: 全体统计 — 两域指尖间距分布(同构后量纲是否可比) ----
    rw_all = np.array([[grip_width(rs[i, t]) for t in range(rs.shape[1])] for i in range(min(300, len(rs)))])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.hist(rw_all[np.isfinite(rw_all)], bins=60, alpha=.6, label=f"robot (n={np.isfinite(rw_all).sum()})",
            color="#1f77b4", density=True)
    ax.hist(hw[np.isfinite(hw)], bins=60, alpha=.6, label=f"human (n={np.isfinite(hw).sum()})",
            color="#d62728", density=True)
    ax.set_xlabel("指尖间距 px"); ax.set_ylabel("密度"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("两域指尖间距分布:同构后量纲是否可比", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/03_两域量纲对比.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- gif: 骨架跟随 ----
    for tag, frames, pts, idx in (("robot", rf, rs, ri), ("human", hf, hs, hi)):
        fr = []
        for t in range(frames.shape[1]):
            f, a = plt.subplots(figsize=(3.1, 3.3))
            draw(a, frames[idx, t], pts[idx, t], f"{tag.upper()} clip{idx} f{t}",
                 f"指尖间距 {grip_width(pts[idx, t]):.1f}px")
            f.tight_layout(); f.canvas.draw()
            fr.append(np.asarray(f.canvas.buffer_rgba())[..., :3].copy()); plt.close(f)
        imageio.mimsave(f"{OUT}/04_骨架跟随_{tag}.gif", fr, duration=0.12, loop=0)

    stats = (f"robot 展示 clip={ri} (grip 跨度最大, 张开f{ropen} 闭合f{rclose})\n"
             f"human 展示 clip={hi} (指尖间距跨度最大, 张开f{hopen} 闭合f{hclose})\n"
             f"robot 骨架指尖间距 vs 真实 grip 相关系数 r={cc:.3f}\n"
             f"robot 指尖间距 中位数 {np.nanmedian(rw_all):.1f}px  范围 [{np.nanmin(rw_all):.1f},{np.nanmax(rw_all):.1f}]\n"
             f"human 指尖间距 中位数 {np.nanmedian(hw):.1f}px  范围 [{np.nanmin(hw):.1f},{np.nanmax(hw):.1f}]\n")
    open(f"{OUT}/summary.txt", "w").write(stats)
    print(stats + f"\n输出 {OUT}\n=== DONE ===")


if __name__ == "__main__":
    main()
