"""渲染器 ablation: 同 seq + 同 GT-flow(retrack) + 256全分辨率(不缩), 只换③, 看锐度。
隔离渲染器质量(②不影响锐度)。列 = GT | 老m4(VideoDiT+grip+warp) | ro_80k(多头,无grip/warp) | grip③(多头+grip+warp)。"""
import os, sys, numpy as np, torch, cv2, imageio
os.environ.setdefault("HF_HUB_OFFLINE", "1"); sys.path.insert(0, ".")
from eval_mh_render import sample, gt256          # 复用 sample + GT帧(256, retrack同crop)
from wan_vae import WanVAE
dev = "cuda"; TLCAP = 6; NS = 20
LAT = np.load("outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz")
lat, lat_low = LAT["lat"][:, :, :TLCAP], LAT["lat_low"][:, :, :TLCAP]
DS = np.load("outputs/flow_render_dataset_can_dual/clips_robot.npz")   # vid/fidx 取GT帧(与eval_mh_render一致; retrack只改tracks不改vid/fidx)
vae = WanVAE(device=dev)
SEQS = [int(x) for x in os.environ.get("SEQS", "332,418").split(",")]
# (label, ckpt, cond文件, CCOND)
VARIANTS = [
    ("old_m4_VideoDiT",  "outputs/video_arch_wm/m4_skel_gripaware_60k/video_dit_ema.pt",
        "outputs/video_arch_wm/cond_can_dual/cond_skel_grip_retrack.npz", 7),
    ("ro_80k_multihead", "outputs/video_arch_wm/epsplit_L48_mh/ro_80k/mh_ema.pt",
        "outputs/video_arch_wm/cond_can_dual/cond_skel_all_retrack.npz", 4),
    ("grip3_multihead",  "outputs/video_arch_wm/epsplit_L48_mh/grip_ro/mh_ema.pt",
        "outputs/video_arch_wm/cond_can_dual/cond_skel_grip_retrack.npz", 7),
]
u8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
OUT = os.environ.get("OUT", "outputs/video_arch_wm/render_ablation"); os.makedirs(f"{OUT}/gifs", exist_ok=True)

for si in SEQS:
    vid = int(DS["vid"][si]); fidx = DS["fidx"][si]
    za = torch.from_numpy(np.stack([lat[si], lat_low[si]])[None, :, :, :1].astype(np.float32)).to(dev)
    renders = {}   # label -> (Tp,256,256,3) cam_high
    for label, ckpt, condp, ccond in VARIANTS:
        m = torch.load(ckpt, map_location=dev, weights_only=False).eval()
        cond = np.load(condp)["cond"][si:si+1, :, :ccond, :TLCAP]
        c = torch.from_numpy(cond.astype(np.float32)).to(dev)
        with torch.no_grad():
            xs = sample(m, za, c, NS)
            dec = vae.decode(xs[:, 0])[0].permute(1, 2, 3, 0).cpu().numpy()   # cam_high (Tp,256,256,3)
        renders[label] = dec
        del m; torch.cuda.empty_cache()
        print(f"seq{si} {label}: rendered {dec.shape}", flush=True)
    Tp = min(min(r.shape[0] for r in renders.values()), 40)
    gt = gt256(vid, fidx[:Tp], 0).astype(np.float32) / 255.   # cam_high GT
    # 拼: GT | old_m4 | ro_80k | grip3, 每帧 256宽 x N列
    cols = [("GT", gt)] + [(lab, renders[lab]) for lab, *_ in VARIANTS]
    frames = []
    for t in range(Tp):
        row = np.concatenate([u8(c[1][t]) for c in cols], 1)   # (256, 256*4, 3)
        # 顶部标签条
        bar = np.ones((18, row.shape[1], 3), np.uint8) * 255
        for i, (lab, _) in enumerate(cols):
            cv2.putText(bar, lab[:18], (i*256+3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,150), 1)
        frames.append(np.concatenate([bar, row], 0))
    imageio.mimsave(f"{OUT}/gifs/ablation_seq{si}.gif", frames, fps=6)
    print(f"=== saved {OUT}/gifs/ablation_seq{si}.gif ({Tp}帧, 256全分辨率) ===", flush=True)
print("DONE")
