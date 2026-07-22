"""M4 条件构建: can dual clips -> 视频 DiT 的逐(latent帧)条件, 复用现有 flow_cond/gmask/warp。
7ch = flow(3: dx,dy,footprint) + agent gmask(1) + warp(3), 与 DualViewDiTG 一致。
时序对齐: Wan 因果 VAE latent 帧 k <-> 原始帧 rf(0->0, k>=1 -> min(4k, L-1))。128 建 cond -> avg-pool 到 16。
输出: outputs/video_arch_wm/cond_can_dual/cond_{tag}.npz : cond(N,V=2,7,tL,16,16) f16 (与 latents 同 N 序)。
在 iws env 跑 (gmask net + cv2)。env: SMOKE=1 只前 4 clip; VIDS 限定。
"""
import os, sys, numpy as np, torch, cv2
os.environ.setdefault("RES", "128"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MASKGEN", "outputs/flow_wm/maskgen_caneef/maskgen_caneef.pt")
os.environ.setdefault("GMASK", "1"); os.environ.setdefault("GMASK_LOW", "1"); os.environ.setdefault("WARP", "1")
sys.path.insert(0, ".")
import exp_scel_dualview_dit as DIT              # flow_cond
import exp_scel_dualview_gmaskcond as G          # gmask_low_imgs, warp_preview, load_gmask_low
import exp_v3_human_helps_pixels as HP           # gmask_imgs, load_gmask

DS = "outputs/flow_render_dataset_can_dual/clips_robot.npz"
SKELF = "outputs/flow_render_dataset_can_dual/skel_sidecar_robot.npz"
OUTDIR = "outputs/video_arch_wm/cond_can_dual"; os.makedirs(OUTDIR, exist_ok=True)
GRID = 16; POOL = 128 // GRID
SKEL = os.environ.get("SKEL", "0") == "1"        # ablation B: agent 通道用骨架线画替代 gmask 剪影
_SK = np.load(SKELF) if SKEL else None
_SEG = _SK["segments"] if SKEL else None


def skel_chan(pts2d):                             # (9,2) crop-norm -> (128,128) 线画 [0,1]
    img = np.zeros((128, 128), np.float32); P = (pts2d * 128).astype(np.int32)
    for a, b in _SEG:
        if np.all(np.abs(pts2d[a]) < 3) and np.all(np.abs(pts2d[b]) < 3):
            cv2.line(img, tuple(P[a]), tuple(P[b]), 1.0, 3, cv2.LINE_AA)
    for i, p in enumerate(P):
        if np.all(np.abs(pts2d[i]) < 3): cv2.circle(img, tuple(p), 3, 1.0, -1)
    return img


def pool16(x):                                    # (C,128,128) -> (C,16,16) avg-pool
    t = torch.from_numpy(x[None]).float()
    return torch.nn.functional.avg_pool2d(t, POOL)[0].numpy()


def build_clip(A, n, tL, L, low_ok):
    """clip n -> cond (V=2, 7, tL, 16,16)。A=已读进内存的数组 dict(避免 npz 反复解压)。"""
    f32 = lambda a: np.nan_to_num(np.asarray(a, np.float32), nan=0.0)
    tr = [f32(A["tracks"][n]), f32(A["tracks_low"][n])]; ef = [f32(A["eef"][n]), f32(A["eef_low"][n])]
    vs = [f32(A["vis"][n]), f32(A["vis_low"][n])]; fr = [A["frames"][n], A["frames_low"][n]]
    jt = f32(A["joint"][n])                        # (L,7)
    out = np.zeros((2, 7, tL, GRID, GRID), np.float32)
    for k in range(tL):
        rf = 0 if k == 0 else min(4 * k, L - 1)
        for v in range(2):
            if v == 1 and not low_ok: continue     # low 无效 -> 该视角条件全 0
            tr0, trt = tr[v][0], tr[v][rf]; ef0, eft = ef[v][0], ef[v][rf]; vis = vs[v][rf]
            fc = DIT.flow_cond(tr0, trt, ef0, eft, vis)                    # (3,128,128)
            if SKEL:
                g = skel_chan(A["skel2d_high" if v == 0 else "skel2d_low"][n, rf])   # 骨架线画替代剪影
            else:
                g = (HP.gmask_imgs(jt[rf][None], eft[None])[0] if v == 0
                     else G.gmask_low_imgs(jt[rf][None], eft[None])[0])    # (128,128) or (H,W)
                if g.shape[0] != 128: g = cv2.resize(g, (128, 128))
            wp = G.warp_preview(fr[v][0], tr0, trt, vis)                   # (3,128,128)
            c7 = np.concatenate([fc, g[None], wp], 0)                      # (7,128,128)
            out[v, :, k] = pool16(c7)
    return out.astype(np.float16)


def main():
    z = np.load(DS)
    L = z["tracks"].shape[1]; tL = 1 + (L - 1) // 4; N = z["tracks"].shape[0]
    if not HP._G: HP.load_gmask()
    G.load_gmask_low()
    rows = range(N)
    if os.environ.get("SMOKE") == "1": rows = range(4)
    elif os.environ.get("VIDS"):
        vids = set(int(x) for x in os.environ["VIDS"].split(","))
        rows = [n for n in range(N) if z["vid"][n] in vids]
    low_valid = z["low_valid"] if "low_valid" in z.files else np.ones(N, bool)
    # ★一次性把数组读进内存(否则 npz 每次 z[key][n] 都全量解压 -> 0.6s/clip)
    print("loading arrays into RAM ...", flush=True)
    A = {k: z[k][:] for k in ["tracks", "tracks_low", "eef", "eef_low", "vis", "vis_low",
                              "frames", "frames_low", "joint"]}
    if SKEL: A["skel2d_high"] = _SK["skel2d_high"][:]; A["skel2d_low"] = _SK["skel2d_low"][:]
    cond = np.zeros((N, 2, 7, tL, GRID, GRID), np.float16)
    rows = list(rows)
    import time; t0 = time.time()
    for i, n in enumerate(rows):
        cond[n] = build_clip(A, n, tL, L, bool(low_valid[n]))
        if i % 100 == 0: print(f"cond {i}/{len(rows)} (clip {n})  {(time.time()-t0)/max(i,1):.3f}s/clip", flush=True)
    tag = "smoke" if os.environ.get("SMOKE") == "1" else (os.environ.get("VIDS", "all").replace(",", "_"))
    if SKEL: tag = "skel_" + tag
    np.savez(f"{OUTDIR}/cond_{tag}.npz", cond=cond, tL=np.array(tL))
    print(f"saved {OUTDIR}/cond_{tag}.npz  cond{cond.shape} tL={tL}", flush=True)
    print("=== cond DONE ===", flush=True)


if __name__ == "__main__":
    main()
