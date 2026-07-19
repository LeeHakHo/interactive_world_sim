"""完整运动链光栅化(OSCAR 式)中间产物核验:原帧 | 关节叠加 | 光栅化线画图。

核心要看的:human 用 21 个关节、robot 用 8 个 link,**点数不同也不需要配对**,
光栅化后都是同样大小的一张线画图 —— 这是两域共享的唯一格式。

输出 outputs/cross_embodiment_wm/fullchain_raster_qc/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import imageio.v2 as imageio

for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
plt.rcParams["axes.unicode_minus"] = False

from augment_clips_fullchain import rasterize_chain

DS = "outputs/flow_render_dataset_can_dual"
OUT = "outputs/cross_embodiment_wm/fullchain_raster_qc"
os.makedirs(OUT, exist_ok=True)
SIZE = int(os.environ.get("RSIZE", "32"))


def panel(ax, img, pts, segs, title):
    ax.imshow(img)
    uv = pts * 128.0
    for a, b in segs:
        if np.isfinite(uv[[a, b]]).all():
            ax.plot(uv[[a, b], 0], uv[[a, b], 1], "-", c="#00e5ff", lw=1.6, zorder=2)
    ok = np.isfinite(uv).all(-1)
    ax.scatter(uv[ok, 0], uv[ok, 1], s=14, c="#ffea00", edgecolors="k", linewidths=0.4, zorder=3)
    ax.set_title(title, fontsize=8.5); ax.axis("off")


def main():
    zr = np.load(f"{DS}/clips_robot.npz"); zh = np.load(f"{DS}/clips_human_L24.npz")
    cr = np.load(f"{DS}/chain_sidecar_robot.npz"); ch = np.load(f"{DS}/chain_sidecar_human.npz")
    rows = []
    for tag, z, c, n in (("ROBOT", zr, cr, 8), ("HUMAN", zh, ch, 21)):
        f = z["frames"]; P = c["chain2d_high"]; segs = c["segments"]
        mot = np.linalg.norm(np.diff(np.nanmean(P[:400], 2), axis=1), axis=-1).sum(1)
        idx = int(np.nanargmax(np.nan_to_num(mot)))
        ts = np.linspace(0, P.shape[1] - 1, 4).astype(int)
        rows.append((tag, f[idx], P[idx], segs, idx, ts, n))

    fig, axes = plt.subplots(4, 4, figsize=(11, 11.6))
    for r, (tag, fr, P, segs, idx, ts, n) in enumerate(rows):
        for c_, t in enumerate(ts):
            panel(axes[r * 2, c_], fr[t], P[t], segs, f"{tag} clip{idx} f{t}\n{n} 关节 / {len(segs)} 边")
            R = rasterize_chain(P[t][None], segs, size=SIZE)[0]
            axes[r * 2 + 1, c_].imshow(R, cmap="gray", vmin=0, vmax=1)
            axes[r * 2 + 1, c_].set_title(f"光栅化 {SIZE}×{SIZE}", fontsize=8.5)
            axes[r * 2 + 1, c_].axis("off")
    fig.suptitle(f"完整运动链光栅化(OSCAR 式)| 上两行 robot({rows[0][6]} link),下两行 human({rows[1][6]} 关节)\n"
                 f"★ 点数不同、不做任何配对;光栅化后同为 {SIZE}×{SIZE} 线画图 = 两域共享的唯一格式",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(f"{OUT}/01_完整链光栅化对照.png", dpi=145, bbox_inches="tight")
    plt.close(fig)

    # 与旧 skel-v2 的 4 点削减版对比
    sv = np.load(f"{DS}/skel_sidecar_robot_v2.npz"); sh = np.load(f"{DS}/skel_sidecar_human.npz")
    SEG4 = np.array([(3, 0), (0, 1), (0, 2)], np.int32)
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for r, (tag, fr, P4, idx, ts) in enumerate([
            ("ROBOT", zr["frames"], sv["skel2d_high"][:, :, [5, 6, 7, 4]], rows[0][4], rows[0][5]),
            ("HUMAN", zh["frames"], sh["skel2d_high"][:, :, :4], rows[1][4], rows[1][5])]):
        for c_, t in enumerate(ts):
            panel(axes[r, c_], fr[idx, t], P4[idx, t], SEG4, f"{tag} f{t} — 旧 skel-v2(削成 4 点)")
    fig.suptitle("对照:旧 skel-v2 把两域都削成 4 点强行配对\n"
                 "(已量化证明失配:两指对称比 1.04 vs 1.27、槽位3 human 是常数占位符)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT}/02_对照_旧4点削减版.png", dpi=145, bbox_inches="tight")
    plt.close(fig)

    for tag, fr, P, segs, idx, _, n in rows:
        frames = []
        for t in range(P.shape[0]):
            f_, ax = plt.subplots(1, 2, figsize=(5.4, 2.9))
            panel(ax[0], fr[t], P[t], segs, f"{tag} clip{idx} f{t} ({n} 关节)")
            ax[1].imshow(rasterize_chain(P[t][None], segs, size=SIZE)[0], cmap="gray", vmin=0, vmax=1)
            ax[1].set_title(f"光栅化 {SIZE}×{SIZE}", fontsize=8.5); ax[1].axis("off")
            f_.tight_layout(); f_.canvas.draw()
            frames.append(np.asarray(f_.canvas.buffer_rgba())[..., :3].copy()); plt.close(f_)
        imageio.mimsave(f"{OUT}/03_链跟随_{tag.lower()}.gif", frames, duration=0.12, loop=0)

    open(f"{OUT}/summary.txt", "w").write(
        f"完整运动链光栅化 QC | 光栅图 {SIZE}x{SIZE}\n"
        f"robot: {rows[0][6]} link / {len(rows[0][3])} 边 (clip {rows[0][4]})\n"
        f"human: {rows[1][6]} 关节 / {len(rows[1][3])} 边 (clip {rows[1][4]})\n"
        f"★ 两域点数不同且不做配对, 格式统一在光栅图\n")
    print(f"输出 {OUT}\n=== DONE ===")


if __name__ == "__main__":
    main()
