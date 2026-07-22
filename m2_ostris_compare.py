"""M2a: ostris-16ch VAE roundtrip 对标 (在 iws env 跑; Wan 那边 m1 已有 256 数字)。
同一批 can 帧, ostris 在 128(原生工作分辨率) 与 256 各 roundtrip, LPIPS/PSNR vs GT(同分辨率)。
产物: outputs/video_arch_wm/m2_vae_compare/ostris_{res}_{view}_triple.png + ostris_metrics.txt。
判读与 Wan(m1: cam_high 0.037/cam_low 0.030 @256) 对比 -> 决定迁移是否掉质量。
"""
import os, numpy as np, av, cv2, torch, lpips
os.environ.setdefault("VAE_NAME", "ostris/vae-kl-f8-d16"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch.nn.functional as F
from diffusers import AutoencoderKL

OUT = "outputs/video_arch_wm/m2_vae_compare"; os.makedirs(OUT, exist_ok=True)
CROPS = {0: (60, 60, 390, 390), 1: (0, 0, 640, 480)}
VID = {v: f"human_play_data/play_robot_can_1_eef/videos/chunk-000/observation.images.cam_{n}/episode_000000.mp4"
       for v, n in [(0, "high"), (1, "low")]}
dev = "cuda"


def load_frames(view, idxs, res):
    x, y, w, h = CROPS[view]; want = set(idxs); got = {}
    c = av.open(VID[view])
    for i, fr in enumerate(c.decode(video=0)):
        if i in want:
            img = fr.to_ndarray(format="rgb24")[y:y + h, x:x + w]
            got[i] = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
        if i > max(want): break
    c.close()
    return np.stack([got[i] for i in idxs])


def main():
    vae = AutoencoderKL.from_pretrained(os.environ["VAE_NAME"]).to(dev).eval()
    sf = float(vae.config.scaling_factor)
    lp = lpips.LPIPS(net="alex").to(dev).eval()
    idxs = list(range(400, 417))
    lines = ["M2a ostris-16ch VAE roundtrip (对标 Wan2.2@256: cam_high 0.0373 / cam_low 0.0296)", ""]
    for res in [128, 256]:
        for view, vname in [(0, "cam_high"), (1, "cam_low")]:
            fr = load_frames(view, idxs, res).astype(np.float32) / 255.
            x = torch.from_numpy(fr).permute(0, 3, 1, 2).to(dev)          # (N,3,res,res)[0,1]
            with torch.no_grad():
                z = vae.encode(x * 2 - 1).latent_dist.mode() * sf
                xr = ((vae.decode(z / sf).sample + 1) / 2).clamp(0, 1)
                lpv = lp(x * 2 - 1, xr * 2 - 1).mean().item()
            xr_np = xr.permute(0, 2, 3, 1).cpu().numpy()
            mse = np.mean((fr - xr_np) ** 2); psnr = 10 * np.log10(1 / max(mse, 1e-10))
            lines.append(f"ostris @{res} {vname}: LPIPS {lpv:.4f}  PSNR {psnr:.2f}dB  (latent {res//8}x{res//8}x16)")
            k = 8; u8 = lambda z: (np.clip(z, 0, 1) * 255).astype(np.uint8)
            diff = np.clip(np.abs(fr[k] - xr_np[k]) * 5, 0, 1)
            trip = np.concatenate([u8(fr[k]), u8(xr_np[k]), u8(diff)], 1)
            cv2.imwrite(f"{OUT}/ostris_{res}_{vname}_triple.png", cv2.cvtColor(trip, cv2.COLOR_RGB2BGR))
            print(lines[-1], flush=True)
    lines += ["", "对比结论: Wan2.2@256 latent=16x16x48(时序4×) vs ostris@256 latent=32x32x16(无时序).",
              "ostris 空间 latent 格子多4× 单帧可能更保真; Wan 卖点=时序压缩+视频先验(逐帧略逊可接受).",
              "迁移理由是视频建模而非逐帧保真; 只要 Wan 重建眼检 OK(罐/夹爪清晰)即通过 M2 gate."]
    open(f"{OUT}/ostris_metrics.txt", "w").write("\n".join(lines) + "\n")
    print("=== M2a DONE ===", flush=True)


if __name__ == "__main__":
    main()
