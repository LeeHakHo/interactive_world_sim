"""物体位置 heatmap aux target(2026-07-31 spec §6.5 aux=域共享object-centric)。
每视角在物体2D质心放高斯(16×16), 与 latents 对齐(同clip序, tL帧)。decoder 的 aux 头从带噪latent预测它,
逼 trunk 定位共享物体(同一罐子, 域共享), human 在此目标同域可迁移 → 塑trunk。
输出格式对齐 train_multihead_wm.load_aux: feat (N,2,K=1,tL,16,16) f16。SRC=robot|human。.venv_wan。
用法: SRC=robot .venv_wan/bin/python build_object_heatmap.py
"""
import os, numpy as np

GRID = 16
RB = "outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz"
HM = "outputs/flow_render_dataset_can_dual_L48/clips_human_L48_retrack_realwrist.npz"
OUTDIR = "outputs/video_arch_wm/aux_targets"; os.makedirs(OUTDIR, exist_ok=True)


def heatmap_at(cx_norm, cy_norm, sigma=1.5):
    """在 16×16 的 (cx,cy)(crop-norm[0,1]) 放高斯 -> (16,16) [0,1]。"""
    cx, cy = cx_norm * GRID, cy_norm * GRID
    yy, xx = np.ogrid[:GRID, :GRID]
    h = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return h.astype(np.float32)


def build_clip_heatmap(tr_hi, tr_lo, tL, L):
    """(L,K,2)x2 -> (2,1,tL,16,16) 物体质心高斯。"""
    out = np.zeros((2, 1, tL, GRID, GRID), np.float32)
    trs = [np.nan_to_num(tr_hi.astype(np.float32)), np.nan_to_num(tr_lo.astype(np.float32))]
    for k in range(tL):
        rf = 0 if k == 0 else min(4 * k, L - 1)
        for v in range(2):
            c = trs[v][rf].mean(0)                            # 物体2D质心
            out[v, 0, k] = heatmap_at(float(c[0]), float(c[1]))
    return out


def main():
    src = os.environ.get("SRC", "robot")
    z = np.load(RB if src == "robot" else HM)
    tr, trl = z["tracks"], z["tracks_low"]; N, L = tr.shape[:2]; tL = 6
    rows = range(4) if os.environ.get("SMOKE") == "1" else range(N)
    feat = np.zeros((N, 2, 1, tL, GRID, GRID), np.float16)
    for i, n in enumerate(rows):
        feat[n] = build_clip_heatmap(tr[n], trl[n], tL, L).astype(np.float16)
        if i % 200 == 0: print(f"objheat {i}/{N} (clip {n})", flush=True)
    np.savez(f"{OUTDIR}/objheat_{src}.npz", feat=feat)
    print(f"saved {OUTDIR}/objheat_{src}.npz {feat.shape}", flush=True)


if __name__ == "__main__":
    main()
