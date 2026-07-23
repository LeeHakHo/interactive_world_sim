"""M5 ablation B 直接对比: agent 用 mask(gmask剪影) vs skel(骨架线画)。
同 seq 渲两版(base ckpt+cond_all / ablB ckpt+cond_skel_all)-> 一行并列 gif: 列= GT | mask | skel。
+ agent 区 LPIPS(用 gmask bbox 裁 agent 区, 全帧被相同场景冲淡, agent 区才真正分 mask vs skel)。
用 .venv_wan/bin/python。standard save_combined_gif(Rendered行+Flow行 绿GT锚)。
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
CK_MASK = "outputs/video_arch_wm/m4_video_dit/video_dit_ema.pt"
CK_SKEL = "outputs/video_arch_wm/m4_ablB_skel/video_dit_ema.pt"
CD_MASK = "outputs/video_arch_wm/cond_can_dual/cond_all.npz"
CD_SKEL = "outputs/video_arch_wm/cond_can_dual/cond_skel_all.npz"
LAT = "outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz"
OUT = "outputs/video_arch_wm/m5_mask_vs_skel"; os.makedirs(f"{OUT}/gifs", exist_ok=True)
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
    """gt/rend (T,256,256,3)[0,1]; gm (T,256,256) agent剪影. -> agent区 LPIPS(bbox裁+resize128)。"""
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
    z = np.load(DS); zl = np.load(LAT); tL = int(zl["tL"])
    lat, lat_low = zl["lat"], zl["lat_low"]
    cmask = np.load(CD_MASK)["cond"]; cskel = np.load(CD_SKEL)["cond"]
    mM = torch.load(CK_MASK, map_location=dev, weights_only=False).eval()
    mS = torch.load(CK_SKEL, map_location=dev, weights_only=False).eval()
    vae = WanVAE(device=dev); lp = lpips.LPIPS(net="alex").to(dev).eval()
    if not HP._G: HP.load_gmask()
    G.load_gmask_low()
    u8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8); r128 = lambda im: cv2.resize(im, (128, 128))
    lines = []
    for si in SEQS:
        za = torch.from_numpy(np.stack([lat[si], lat_low[si]])[None, :, :, :1].astype(np.float32)).to(dev)
        fidx = z["fidx"][si]; vid = int(z["vid"][si])
        outs = {}
        for tag, model, cond in [("mask", mM, cmask), ("skel", mS, cskel)]:
            c = torch.from_numpy(cond[si][None].astype(np.float32)).to(dev)
            xs = sample(model, za, c)
            outs[tag] = [vae.decode(xs[:, v])[0].permute(1, 2, 3, 0).cpu().numpy() for v in range(2)]
        for v in range(2):
            Tp = min(outs["mask"][v].shape[0], outs["skel"][v].shape[0])
            gtpx = gt256(vid, fidx[:Tp], v).astype(np.float32) / 255.
            jt = np.nan_to_num(z["joint"][si].astype(np.float32)); ef = np.nan_to_num((z["eef"] if v == 0 else z["eef_low"])[si].astype(np.float32))
            # agent 剪影 @256 (每帧, rf 对应)
            gm = np.stack([cv2.resize((HP.gmask_imgs(jt[min(4*0+t,47)][None], ef[min(t,47)][None])[0] if v == 0
                                       else G.gmask_low_imgs(jt[min(t,47)][None], ef[min(t,47)][None])[0]), (256, 256)) for t in range(Tp)])
            for tag in ["mask", "skel"]:
                a = torch.from_numpy(gtpx).permute(0, 3, 1, 2).to(dev) * 2 - 1
                b = torch.from_numpy(outs[tag][v][:Tp]).permute(0, 3, 1, 2).to(dev) * 2 - 1
                full = lp(a, b).mean().item(); ag = agent_lpips(lp, gtpx, outs[tag][v][:Tp], gm)
                lines.append(f"seq{si} cam{'high' if v==0 else 'low'} {tag}: full {full:.4f}  agent区 {ag:.4f}")
                print(lines[-1], flush=True)
            # gif @128: 列 GT | mask | skel (一行内并列)
            rend = np.stack([np.stack([r128(u8(gtpx[t])) for t in range(Tp)]),
                             np.stack([r128(u8(outs["mask"][v][t])) for t in range(Tp)]),
                             np.stack([r128(u8(outs["skel"][v][t])) for t in range(Tp)])])
            gtobj = np.nan_to_num((z["tracks"] if v == 0 else z["tracks_low"])[si].astype(np.float32))[:Tp]
            eefv = ef[:Tp]
            fl = np.stack([build_flow_cols(rend[i], gtobj, [None], eefv)[0] for i in range(3)])
            save_combined_gif(f"{OUT}/gifs/seq{si}_cam{'high' if v==0 else 'low'}.gif", rend, fl,
                              ["GT", "mask(gmask)", "skel(骨架)"], [None, None, None], 0,
                              caption=f"seq{si} cam{'high' if v==0 else 'low'} | agent条件: mask vs skel")
    open(f"{OUT}/summary.txt", "w").write(
        "M5 ablation B: agent mask vs skel 直接对比\ngif列: GT | mask | skel (一行并列)\n"
        "★agent区LPIPS = 真正判据(全帧被相同场景冲淡)\n\n" + "\n".join(lines) + "\n")
    print("=== mask_vs_skel DONE ===", flush=True)


if __name__ == "__main__":
    main()
