"""M5 ablation 在真实 e2e(②pred-flow→③)下比较, 非只GT-flow天花板。
MODE=warp: with-warp(7ch) vs warp-off(4ch), 都pred-flow, agent=gmask。
MODE=skel: mask(gmask) vs skel(骨架), 都pred-flow 7ch。
② rollout_dual(dummy5, GT eef驱动)→pred tracks→pred-flow cond(flow用pred, agent/warp相应)。
标准 save_combined_gif(GT|变体1|变体2)+ 全帧&agent区 LPIPS。用 .venv_wan/bin/python。
"""
import os, sys, numpy as np, torch, cv2, lpips
sys.path.insert(0, "."); os.environ.setdefault("RES", "128"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MASKGEN", "outputs/flow_wm/maskgen_caneef/maskgen_caneef.pt")
os.environ.setdefault("GMASK", "1"); os.environ.setdefault("GMASK_LOW", "1")
import exp_scel_dualview_gmaskcond as G
import exp_v3_human_helps_pixels as HP
import exp_scel_dualview_wm as W
import exp_scel_dualview_comb as DC
import exp_scel_dualview_dit as DIT
from video_dit import VideoDiT
from wan_vae import WanVAE
from viz_combined import save_combined_gif, build_flow_cols
for _c in [W.DualLWC, DC.DualCombLWC, DIT.DualViewDiT, G.DualViewDiTG, VideoDiT]:
    setattr(sys.modules["__main__"], _c.__name__, _c)

MODE = os.environ.get("MODE", "warp")                       # warp | skel
DS = "outputs/flow_render_dataset_can_dual/clips_robot.npz"
SK = "outputs/flow_render_dataset_can_dual/skel_sidecar_robot.npz"
WM = "outputs/cross_embodiment_wm/abs_vs_rel_humanhelps/wm_dummy5_rh_N3000.pt"
CK = {"warp-with": "outputs/video_arch_wm/m4_video_dit/video_dit_ema.pt",
      "warp-off": "outputs/video_arch_wm/m4_ablA_nowarp/video_dit_ema.pt",
      "mask": "outputs/video_arch_wm/m4_video_dit/video_dit_ema.pt",
      "skel": "outputs/video_arch_wm/m4_ablB_skel/video_dit_ema.pt"}
OUT = f"outputs/video_arch_wm/m5_e2e_{MODE}"; os.makedirs(f"{OUT}/gifs", exist_ok=True)
SEQS = [int(x) for x in os.environ.get("SEQS", "332,59,418,442").split(",")]
NS = 20; dev = "cuda"; GRID = 16; POOL = 8; tL = 12; L = 48
CROPS = {0: (60, 60, 390, 390), 1: (0, 0, 640, 480)}
VIDP = {v: f"human_play_data/play_robot_can_{{}}_eef/videos/chunk-000/observation.images.cam_{n}/episode_000000.mp4"
        for v, n in [(0, "high"), (1, "low")]}
_SK = np.load(SK); _SEG = _SK["segments"]


def pool16(x): return torch.nn.functional.avg_pool2d(torch.from_numpy(x[None]).float(), POOL)[0].numpy()


def skel_chan(pts2d):
    img = np.zeros((128, 128), np.float32); P = (pts2d * 128).astype(np.int32)
    for a, b in _SEG:
        if np.all(np.abs(pts2d[a]) < 3) and np.all(np.abs(pts2d[b]) < 3): cv2.line(img, tuple(P[a]), tuple(P[b]), 1.0, 3, cv2.LINE_AA)
    for i, p in enumerate(P):
        if np.all(np.abs(pts2d[i]) < 3): cv2.circle(img, tuple(p), 3, 1.0, -1)
    return img


def cond_pred(v, tr0, trt, ef0, eft, vis, jt, fr0, sk2d, agent, warp):
    fc = DIT.flow_cond(tr0, trt, ef0, eft, vis)             # pred-flow (3,128,128)
    if agent == "skel": g = skel_chan(sk2d)
    else:
        g = (HP.gmask_imgs(jt[None], eft[None])[0] if v == 0 else G.gmask_low_imgs(jt[None], eft[None])[0])
        if g.shape[0] != 128: g = cv2.resize(g, (128, 128))
    ch = [fc, g[None]]
    if warp: ch.append(G.warp_preview(fr0, tr0, trt, vis))
    return pool16(np.concatenate(ch, 0).astype(np.float32))


@torch.no_grad()
def sample(model, za, cond, steps=NS):
    B, V, C, _, g, _ = za.shape; T = cond.shape[3]
    x = torch.randn(B, V, C, T, g, g, device=dev); x[:, :, :, :1] = za
    for i in range(steps):
        t = torch.full((B, T), i / steps, device=dev); t[:, 0] = 1.0
        x = x + model(x, t, cond) / steps; x[:, :, :, :1] = za
    return x


def gt256(vid, fidx, v):
    import av
    x, y, w, h = CROPS[v]; need = set(int(i) for i in fidx); got = {}
    c = av.open(VIDP[v].format(vid - 100 + 1))
    for i, fr in enumerate(c.decode(video=0)):
        if i in need: got[i] = cv2.resize(fr.to_ndarray(format="rgb24")[y:y + h, x:x + w], (256, 256), interpolation=cv2.INTER_AREA)
        if i > max(need): break
    c.close()
    return np.stack([got[int(i)] for i in fidx])


def agent_lpips(lp, gt, rend, gm):
    vals = []
    for t in range(len(gt)):
        ys, xs = np.where(gm[t] > 0.4)
        if len(ys) < 20: continue
        y0, y1, x0, x1 = max(ys.min()-8, 0), min(ys.max()+8, 255), max(xs.min()-8, 0), min(xs.max()+8, 255)
        a = torch.from_numpy(cv2.resize(gt[t][y0:y1, x0:x1], (128, 128))).permute(2, 0, 1)[None].to(dev)*2-1
        b = torch.from_numpy(cv2.resize(rend[t][y0:y1, x0:x1], (128, 128))).permute(2, 0, 1)[None].to(dev)*2-1
        vals.append(lp(a, b).item())
    return float(np.mean(vals)) if vals else -1


def main():
    z = np.load(DS); zl = np.load("outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz")
    lat, lat_low = zl["lat"], zl["lat_low"]
    wm = torch.load(WM, map_location=dev, weights_only=False).eval()
    if MODE == "warp": variants = [("with-warp", CK["warp-with"], "gmask", True, 7), ("warp-off", CK["warp-off"], "gmask", False, 4)]
    else:              variants = [("mask", CK["mask"], "gmask", True, 7), ("skel", CK["skel"], "skel", True, 7)]
    models = {name: torch.load(ck, map_location=dev, weights_only=False).eval() for name, ck, *_ in variants}
    vae = WanVAE(device=dev); lp = lpips.LPIPS(net="alex").to(dev).eval()
    if not HP._G: HP.load_gmask()
    G.load_gmask_low()
    K = W.K; P = 48
    u8 = lambda a: (np.clip(a, 0, 1)*255).astype(np.uint8); r128 = lambda im: cv2.resize(im, (128, 128))
    lines = []
    for si in SEQS:
        trg = [np.nan_to_num(z["tracks"][si].astype(np.float32)), np.nan_to_num(z["tracks_low"][si].astype(np.float32))]
        efg = [np.nan_to_num(z["eef"][si].astype(np.float32)), np.nan_to_num(z["eef_low"][si].astype(np.float32))]
        vsg = [np.nan_to_num(z["vis"][si].astype(np.float32)), np.nan_to_num(z["vis_low"][si].astype(np.float32))]
        jt = np.nan_to_num(z["joint"][si].astype(np.float32)); frs = [z["frames"][si], z["frames_low"][si]]
        sk = [_SK["skel2d_high"][si], _SK["skel2d_low"][si]]
        fidx = z["fidx"][si]; vid = int(z["vid"][si])
        # ② rollout -> pred tracks (GT首帧上下文 + GT eef驱动)
        trD = np.concatenate([trg[0][:K], trg[1][:K]], 1)[None]
        pr = W.rollout_dual(wm, torch.from_numpy(trD).float().to(dev), torch.from_numpy(efg[0][None]).float().to(dev),
                            torch.from_numpy(efg[1][None]).float().to(dev), L - K)[0].cpu().numpy()
        predtr = [np.concatenate([trg[v][:K], pr[:, v*P:(v+1)*P]], 0) for v in range(2)]
        f0 = [gt256(vid, fidx[:1], v)[0] for v in range(2)]
        za = torch.stack([vae.encode(torch.from_numpy(f0[v].astype(np.float32).transpose(2, 0, 1)[None, :, None]/255.))[0, :, :1] for v in range(2)])[None]
        outs = {}
        for name, ck, agent, warp, cc in variants:
            cond = np.zeros((2, cc, tL, GRID, GRID), np.float32)
            for k in range(tL):
                rf = 0 if k == 0 else min(4*k, L-1)
                for v in range(2):
                    cond[v, :, k] = cond_pred(v, trg[v][0], predtr[v][rf], efg[v][0], efg[v][rf], vsg[v][rf], jt[rf], frs[v][0], sk[v][rf], agent, warp)
            xs = sample(models[name], za, torch.from_numpy(cond[None]).float().to(dev))
            outs[name] = [vae.decode(xs[:, v])[0].permute(1, 2, 3, 0).cpu().numpy() for v in range(2)]
        names = [vv[0] for vv in variants]
        for v in range(2):
            Tp = min(outs[names[0]][v].shape[0], outs[names[1]][v].shape[0])
            gtpx = gt256(vid, fidx[:Tp], v).astype(np.float32)/255.
            ef = efg[v]
            gm = np.stack([cv2.resize((HP.gmask_imgs(jt[min(t, 47)][None], ef[min(t, 47)][None])[0] if v == 0 else G.gmask_low_imgs(jt[min(t, 47)][None], ef[min(t, 47)][None])[0]), (256, 256)) for t in range(Tp)])
            for name in names:
                a = torch.from_numpy(gtpx).permute(0, 3, 1, 2).to(dev)*2-1; b = torch.from_numpy(outs[name][v][:Tp]).permute(0, 3, 1, 2).to(dev)*2-1
                full = lp(a, b).mean().item(); ag = agent_lpips(lp, gtpx, outs[name][v][:Tp], gm)
                lines.append(f"seq{si} cam{'high' if v==0 else 'low'} {name:10s}: full {full:.4f}  agent区 {ag:.4f}  [pred-flow e2e]")
                print(lines[-1], flush=True)
            rend = np.stack([np.stack([r128(u8(gtpx[t])) for t in range(Tp)])] +
                            [np.stack([r128(u8(outs[nm][v][t])) for t in range(Tp)]) for nm in names])
            gtobj = trg[v][:Tp]
            fl = np.stack([build_flow_cols(rend[0], gtobj, [None], ef[:Tp])[0]] +
                          [build_flow_cols(rend[i+1], gtobj, [predtr[v][:Tp]], ef[:Tp])[0] for i in range(2)])   # 变体列画红②pred
            save_combined_gif(f"{OUT}/gifs/seq{si}_cam{'high' if v==0 else 'low'}.gif", rend, fl,
                              ["GT"] + names, [None, None, None], 0,
                              caption=f"seq{si} cam{'high' if v==0 else 'low'} | ablation {MODE} @ ②pred-flow e2e (红=②pred)")
    open(f"{OUT}/summary.txt", "w").write(f"M5 ablation {MODE} @ ②pred-flow e2e (非GT-flow天花板)\ngif列: GT | {names[0]} | {names[1]}\n\n" + "\n".join(lines) + "\n")
    print(f"=== e2e ablation {MODE} DONE ===", flush=True)


if __name__ == "__main__":
    main()
