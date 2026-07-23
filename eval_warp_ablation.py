"""M5 ablation A 直接对比: 条件带 warp vs warp-off。标准 save_combined_gif(列 GT | with-warp | warp-off)。
with-warp = base ckpt + cond_all(7ch); warp-off = ablA ckpt + cond_all[:4](flow3+gmask1)。
+ agent 区 LPIPS。用 .venv_wan/bin/python。
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

DS = "outputs/flow_render_dataset_can_dual/clips_robot.npz"
CK_WARP = "outputs/video_arch_wm/m4_video_dit/video_dit_ema.pt"
CK_WOFF = "outputs/video_arch_wm/m4_ablA_nowarp/video_dit_ema.pt"
CD = "outputs/video_arch_wm/cond_can_dual/cond_all.npz"
LAT = "outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz"
OUT = "outputs/video_arch_wm/m5_warp_ablation"; os.makedirs(f"{OUT}/gifs", exist_ok=True)
SEQS = [int(x) for x in os.environ.get("SEQS", "332,59,418,442").split(",")]
NS = 20; dev = "cuda"
CROPS = {0: (60, 60, 390, 390), 1: (0, 0, 640, 480)}
VIDP = {v: f"human_play_data/play_robot_can_{{}}_eef/videos/chunk-000/observation.images.cam_{n}/episode_000000.mp4"
        for v, n in [(0, "high"), (1, "low")]}


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
        y0, y1, x0, x1 = max(ys.min() - 8, 0), min(ys.max() + 8, 255), max(xs.min() - 8, 0), min(xs.max() + 8, 255)
        ga = cv2.resize(gt[t][y0:y1, x0:x1], (128, 128)); ra = cv2.resize(rend[t][y0:y1, x0:x1], (128, 128))
        a = torch.from_numpy(ga).permute(2, 0, 1)[None].to(dev) * 2 - 1
        b = torch.from_numpy(ra).permute(2, 0, 1)[None].to(dev) * 2 - 1
        vals.append(lp(a, b).item())
    return float(np.mean(vals)) if vals else -1


def main():
    z = np.load(DS); zl = np.load(LAT); lat, lat_low = zl["lat"], zl["lat_low"]
    cond = np.load(CD)["cond"]
    mW = torch.load(CK_WARP, map_location=dev, weights_only=False).eval()
    mO = torch.load(CK_WOFF, map_location=dev, weights_only=False).eval()
    vae = WanVAE(device=dev); lp = lpips.LPIPS(net="alex").to(dev).eval()
    if not HP._G: HP.load_gmask()
    G.load_gmask_low()
    u8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8); r128 = lambda im: cv2.resize(im, (128, 128))
    lines = []
    for si in SEQS:
        za = torch.from_numpy(np.stack([lat[si], lat_low[si]])[None, :, :, :1].astype(np.float32)).to(dev)
        fidx = z["fidx"][si]; vid = int(z["vid"][si])
        c7 = torch.from_numpy(cond[si][None].astype(np.float32)).to(dev)
        outs = {"with-warp": [vae.decode(sample(mW, za, c7)[:, v])[0].permute(1, 2, 3, 0).cpu().numpy() for v in range(2)],
                "warp-off": [vae.decode(sample(mO, za, c7[:, :, :4])[:, v])[0].permute(1, 2, 3, 0).cpu().numpy() for v in range(2)]}
        for v in range(2):
            Tp = min(outs["with-warp"][v].shape[0], outs["warp-off"][v].shape[0])
            gtpx = gt256(vid, fidx[:Tp], v).astype(np.float32) / 255.
            jt = np.nan_to_num(z["joint"][si].astype(np.float32)); ef = np.nan_to_num((z["eef"] if v == 0 else z["eef_low"])[si].astype(np.float32))
            gm = np.stack([cv2.resize((HP.gmask_imgs(jt[min(t, 47)][None], ef[min(t, 47)][None])[0] if v == 0
                                       else G.gmask_low_imgs(jt[min(t, 47)][None], ef[min(t, 47)][None])[0]), (256, 256)) for t in range(Tp)])
            for tag in ["with-warp", "warp-off"]:
                a = torch.from_numpy(gtpx).permute(0, 3, 1, 2).to(dev) * 2 - 1
                b = torch.from_numpy(outs[tag][v][:Tp]).permute(0, 3, 1, 2).to(dev) * 2 - 1
                full = lp(a, b).mean().item(); ag = agent_lpips(lp, gtpx, outs[tag][v][:Tp], gm)
                lines.append(f"seq{si} cam{'high' if v==0 else 'low'} {tag:9s}: full {full:.4f}  agent区 {ag:.4f}")
                print(lines[-1], flush=True)
            rend = np.stack([np.stack([r128(u8(gtpx[t])) for t in range(Tp)]),
                             np.stack([r128(u8(outs["with-warp"][v][t])) for t in range(Tp)]),
                             np.stack([r128(u8(outs["warp-off"][v][t])) for t in range(Tp)])])
            gtobj = np.nan_to_num((z["tracks"] if v == 0 else z["tracks_low"])[si].astype(np.float32))[:Tp]
            fl = np.stack([build_flow_cols(rend[i], gtobj, [None], ef[:Tp])[0] for i in range(3)])
            save_combined_gif(f"{OUT}/gifs/seq{si}_cam{'high' if v==0 else 'low'}.gif", rend, fl,
                              ["GT", "with-warp", "warp-off"], [None, None, None], 0,
                              caption=f"seq{si} cam{'high' if v==0 else 'low'} | ablation A: warp条件 vs 去warp")
    open(f"{OUT}/summary.txt", "w").write(
        "M5 ablation A: 条件带warp vs warp-off (标准protocol)\ngif列: GT | with-warp | warp-off\n\n"
        + "\n".join(lines) + "\n")
    print("=== warp_ablation DONE ===", flush=True)


if __name__ == "__main__":
    main()
