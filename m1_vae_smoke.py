"""M1: Wan2.2 VAE 真 can 帧 roundtrip 可视化 + 指标。
用官方 wan_vae.WanVAE (需 .venv_wan/bin/python)。crop 与 exp_scel_dualview_wm.CROPS 一致 -> resize 256。
产物: outputs/video_arch_wm/m1_vae_smoke/ : 单帧三联图(原|重建|差×放大) + T=17 clip gif(原/重建两行) + summary.txt。
"""
import os, numpy as np, av, cv2, torch, imageio, lpips
from wan_vae import WanVAE

OUT = "outputs/video_arch_wm/m1_vae_smoke"; os.makedirs(OUT, exist_ok=True)
CROPS = {0: (60, 60, 390, 390), 1: (0, 0, 640, 480)}          # view0=cam_high, view1=cam_low (同现有管线)
VID = {v: f"human_play_data/play_robot_can_1_eef/videos/chunk-000/observation.images.cam_{n}/episode_000000.mp4"
       for v, n in [(0, "high"), (1, "low")]}
RES = 256


def load_frames(view, idxs):
    """读指定帧号 -> crop -> resize RES -> (N,RES,RES,3) uint8。"""
    x, y, w, h = CROPS[view]; want = set(idxs); got = {}
    c = av.open(VID[view])
    for i, fr in enumerate(c.decode(video=0)):
        if i in want:
            img = fr.to_ndarray(format="rgb24")[y:y + h, x:x + w]
            got[i] = cv2.resize(img, (RES, RES), interpolation=cv2.INTER_AREA)
        if i > max(want): break
    c.close()
    return np.stack([got[i] for i in idxs])


def main():
    dev = "cuda"
    v = WanVAE(device=dev)
    lp = lpips.LPIPS(net="alex").to(dev).eval()
    to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.).permute(0, 3, 1, 2)  # (N,3,H,W)[0,1]

    # 取两视角各一段连续 17 帧 + 单帧样本
    idxs = list(range(400, 417))
    lines = []
    for view, vname in [(0, "cam_high"), (1, "cam_low")]:
        fr = load_frames(view, idxs)                                  # (17,RES,RES,3)
        x = to_t(fr)[:, :, None].permute(2, 1, 0, 3, 4).contiguous()  # -> (1,3,T,H,W)
        x = x.reshape(1, 3, len(idxs), RES, RES)
        xr = v.roundtrip(x)[0].permute(1, 2, 3, 0).cpu().numpy()      # (T,H,W,3)[0,1]
        gt = fr.astype(np.float32) / 255.
        # 指标 (逐帧)
        with torch.no_grad():
            a = torch.from_numpy(gt).permute(0, 3, 1, 2).to(dev) * 2 - 1
            b = torch.from_numpy(xr).permute(0, 3, 1, 2).to(dev) * 2 - 1
            lpv = lp(a, b).mean().item()
        mse = np.mean((gt - xr) ** 2); psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
        lines.append(f"{vname}: roundtrip LPIPS(alex) {lpv:.4f}  PSNR {psnr:.2f}dB  (256 native, T=17)")
        # 单帧三联图: 原 | 重建 | 差×5
        k = 8; u8 = lambda z: (np.clip(z, 0, 1) * 255).astype(np.uint8)
        diff = np.clip(np.abs(gt[k] - xr[k]) * 5, 0, 1)
        trip = np.concatenate([u8(gt[k]), u8(xr[k]), u8(diff)], 1)
        cv2.imwrite(f"{OUT}/{vname}_triple.png", cv2.cvtColor(trip, cv2.COLOR_RGB2BGR))
        # clip gif: 上原 下重建
        frames = [np.concatenate([u8(gt[t]), u8(xr[t])], 0) for t in range(len(idxs))]
        imageio.mimsave(f"{OUT}/{vname}_clip.gif", frames, fps=6, loop=0)
        print(lines[-1], flush=True)
    open(f"{OUT}/summary.txt", "w").write(
        "M1 Wan2.2 VAE roundtrip smoke (官方 diffusers AutoencoderKLWan, 256 native can 帧)\n"
        "z_dim=48 空间16× 时序4× | T=17 -> 5 latent 帧 | 官方 latents_mean/std 归一\n\n"
        + "\n".join(lines) + "\n\n三联图=原|重建|差×5; clip gif=上原/下重建。\n"
        "判读: LPIPS 越低越好; 眼检重建糊不糊(尤其罐子/夹爪)。M2 再与 ostris 对标。\n")
    print("=== M1 DONE ===", flush=True)


if __name__ == "__main__":
    main()
