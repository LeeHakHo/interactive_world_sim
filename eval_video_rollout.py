"""M5: 训好的视频 DiT ③ replay eval -> 真渲染质量(测真交付物 = decode 到像素的 LPIPS 含 agent + gif)。
流程: 每 clip 取 Wan latent 首帧当 I0 锚 -> 采样整段(GT flow cond) -> Wan decode 到像素 -> vs GT。
用 .venv_wan/bin/python(需 Wan VAE decode + video_dit + lpips)。
产物 outputs/video_arch_wm/m5_replay/: 每 seq gif(cam_high/low 两列, GT行|Rendered行) + summary(LPIPS)。
env: CKPT=video_dit_ema.pt SEQS=... STEPS_SAMPLE=20。
"""
import os, sys, numpy as np, torch, cv2, imageio, lpips
sys.path.insert(0, "."); os.environ.setdefault("HF_HUB_OFFLINE", "1")
from video_dit import VideoDiT
from wan_vae import WanVAE

CKPT = os.environ.get("CKPT", "outputs/video_arch_wm/m4_video_dit/video_dit_ema.pt")
LAT = os.environ.get("LAT", "outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz")
COND = os.environ.get("COND", "outputs/video_arch_wm/cond_can_dual/cond_all.npz")
DS = "outputs/flow_render_dataset_can_dual/clips_robot.npz"
OUT = os.environ.get("OUT", "outputs/video_arch_wm/m5_replay"); os.makedirs(OUT, exist_ok=True)
SEQS = [int(x) for x in os.environ.get("SEQS", "332,59,418,442").split(",")]
NS = int(os.environ.get("STEPS_SAMPLE", "20")); CCOND = int(os.environ.get("CCOND", "0")); dev = "cuda"
CROPS = {0: (60, 60, 390, 390), 1: (0, 0, 640, 480)}
VID = {v: f"human_play_data/play_robot_can_{{}}_eef/videos/chunk-000/observation.images.cam_{n}/episode_000000.mp4"
       for v, n in [(0, "high"), (1, "low")]}


@torch.no_grad()
def sample(model, z_anchor, cond, steps):
    B, V, C, _, g, _ = z_anchor.shape; T = cond.shape[3]
    x = torch.randn(B, V, C, T, g, g, device=dev); x[:, :, :, :1] = z_anchor
    for i in range(steps):
        t = torch.full((B, T), i / steps, device=dev); t[:, 0] = 1.0
        x = x + model(x, t, cond) / steps; x[:, :, :, :1] = z_anchor
    return x


def gt_frames_256(vid, fidx, view):
    """取真帧 256(与训练同 crop)-> (len,256,256,3)。"""
    import av
    x, y, w, h = CROPS[view]; need = set(int(i) for i in fidx); got = {}
    c = av.open(VID[view].format(vid - 100 + 1))
    for i, fr in enumerate(c.decode(video=0)):
        if i in need:
            got[i] = cv2.resize(fr.to_ndarray(format="rgb24")[y:y + h, x:x + w], (256, 256), interpolation=cv2.INTER_AREA)
        if i > max(need): break
    c.close()
    return np.stack([got[int(i)] for i in fidx])


def main():
    zl = np.load(LAT); zc = np.load(COND); zd = np.load(DS)
    lat, lat_low, tL = zl["lat"], zl["lat_low"], int(zl["tL"]); cond = zc["cond"]
    model = torch.load(CKPT, map_location=dev, weights_only=False).eval()
    vae = WanVAE(device=dev); lp = lpips.LPIPS(net="alex").to(dev).eval()
    from viz_combined import save_combined_gif, build_flow_cols
    C = lat.shape[1]; P = None
    u8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
    lines = []
    for si in SEQS:
        z = torch.from_numpy(np.stack([lat[si], lat_low[si]])[None].astype(np.float32)).to(dev)  # (1,2,48,tL,16,16)
        c = torch.from_numpy(cond[si][None].astype(np.float32)).to(dev)
        if CCOND: c = c[:, :, :CCOND]                                   # ablation A: warp-off 4ch
        xs = sample(model, z[:, :, :, :1], c, NS)                        # (1,2,48,tL,16,16)
        # decode 两视角
        rend = []; gts = []
        fidx = zd["fidx"][si]; vid = int(zd["vid"][si])
        for v in range(2):
            px = vae.decode(xs[:, v])[0].permute(1, 2, 3, 0).cpu().numpy()  # (Tpix,256,256,3)[0,1]
            rend.append(px)
            gt = gt_frames_256(vid, fidx[:px.shape[0]], v).astype(np.float32) / 255.
            gts.append(gt)
            with torch.no_grad():
                a = torch.from_numpy(gt).permute(0, 3, 1, 2).to(dev) * 2 - 1
                b = torch.from_numpy(px[:len(gt)]).permute(0, 3, 1, 2).to(dev) * 2 - 1
                lpv = lp(a, b).mean().item()
            lines.append(f"seq{si} cam{'high' if v==0 else 'low'}: render LPIPS {lpv:.4f} (含agent)")
            print(lines[-1], flush=True)
        # gif: 两列(high|low), GT行 + Rendered行
        Tp = min(rend[0].shape[0], rend[1].shape[0], gts[0].shape[0], gts[1].shape[0])
        top = [np.concatenate([u8(gts[0][t]), u8(gts[1][t])], 1) for t in range(Tp)]
        bot = [np.concatenate([u8(rend[0][t]), u8(rend[1][t])], 1) for t in range(Tp)]
        frames = [np.concatenate([a, b], 0) for a, b in zip(top, bot)]
        imageio.mimsave(f"{OUT}/seq{si}_replay.gif", frames, fps=8, loop=0)
    open(f"{OUT}/summary.txt", "w").write(
        "M5 视频 DiT ③ replay eval(Wan decode 到像素, GT flow 条件, 首帧 I0 锚)。\n"
        f"ckpt={CKPT}\ngif: 上=GT(high|low) 下=Rendered。\n\n" + "\n".join(lines) + "\n")
    print("=== M5 replay DONE ===", flush=True)


if __name__ == "__main__":
    main()
