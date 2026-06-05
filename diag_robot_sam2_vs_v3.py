"""User: "mask_subtract 里面看着还可以". Clarify: in mask_subtract, SAM2 is used for HUMAN
(SAM2∪dark-sleeve); ROBOT uses deterministic v3 *because SAM2 failed on robot*. g learns the
ROBOT mask, so the relevant question is: how does SAM2 look ON ROBOT? Eyeball the cached
sam2raw vs v3(final) overlaid on real robot_3 frames. Pick frames where SAM2 coverage is
highest (most likely table-balloon) + uniform frames. Output: outputs/diag_robot_sam2_vs_v3/
"""
import os, numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "outputs/flow_wm/diag_robot_sam2_vs_v3"; os.makedirs(OUT, exist_ok=True)
D = "play_robot_eef/_video_cache/robot__play_robot_3_eef__ep000000/"
_fp = D + "observation.images.cam_high_res128_crop195-195-256-256.npy"
_nf = os.path.getsize(_fp) // (128 * 128 * 3)
fr = np.memmap(_fp, dtype=np.uint8, mode="r", shape=(_nf, 128, 128, 3))  # raw uint8, not npy
sam = np.load(D + "agent_mask_res128_crop195-195-256-256_sam2raw.npy", mmap_mode="r")
v3 = np.load(D + "agent_mask_res128_crop195-195-256-256.npy", mmap_mode="r")
print("frames", fr.shape, "sam", sam.shape, "v3", v3.shape, flush=True)
n = min(len(fr), len(sam), len(v3))

cov_s = np.asarray(sam[:n]).reshape(n, -1).mean(1)
cov_v = np.asarray(v3[:n]).reshape(n, -1).mean(1)
# frames where SAM2 covers MUCH more than v3 (table-balloon / drift candidates)
gap = cov_s - cov_v
top_balloon = np.argsort(gap)[::-1][:6]
uniform = np.linspace(int(n * 0.05), int(n * 0.95), 6).astype(int)
picks = list(top_balloon) + list(uniform)
tags = ["SAM2>>v3"] * 6 + ["uniform"] * 6

print(f"global cov: SAM2 {cov_s.mean():.3f}  v3 {cov_v.mean():.3f}  "
      f"max SAM2 {cov_s.max():.3f} @frame{cov_s.argmax()}", flush=True)

fig, ax = plt.subplots(3, len(picks), figsize=(2.0 * len(picks), 6.6))
for j, (i, tg) in enumerate(zip(picks, tags)):
    base = np.clip(fr[i].astype(np.float32) * 1.7, 0, 255).astype(np.uint8)
    s = np.asarray(sam[i]) > 0; v = np.asarray(v3[i]) > 0
    os_ = base.copy(); os_[s] = (0.35 * os_[s] + 0.65 * np.array([255, 0, 0])).astype(np.uint8)
    ov_ = base.copy(); ov_[v] = (0.35 * ov_[v] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
    ax[0, j].imshow(base); ax[0, j].axis("off"); ax[0, j].set_title(f"f{i} {tg}", fontsize=7)
    ax[1, j].imshow(os_); ax[1, j].axis("off")
    if j == 0: ax[1, j].set_title("SAM2 raw (red)", fontsize=8, loc="left")
    ax[1, j].text(2, 12, f"cov{cov_s[i]:.2f}", color="yellow", fontsize=7)
    ax[2, j].imshow(ov_); ax[2, j].axis("off")
    if j == 0: ax[2, j].set_title("v3 final (green)", fontsize=8, loc="left")
    ax[2, j].text(2, 12, f"cov{cov_v[i]:.2f}", color="yellow", fontsize=7)
fig.suptitle("ROBOT: SAM2 raw (mask_subtract's raw) vs deterministic v3. Left6 = SAM2 covers >> v3 "
             "(table-balloon?). Does SAM2 keep the gripper / avoid the table?", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/overlay_bright.png", dpi=125)
open(f"{OUT}/summary.txt", "w").write(
    f"ROBOT SAM2 raw vs v3 (robot_3, n={n})\nglobal cov SAM2 {cov_s.mean():.3f} v3 {cov_v.mean():.3f}\n"
    f"max SAM2 cov {cov_s.max():.3f} @f{cov_s.argmax()} (high cov = likely table drift)\n")
print(f"saved {OUT}/overlay.png + summary.txt\n=== DONE ===", flush=True)
