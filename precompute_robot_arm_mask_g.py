"""Replace g's supervision robot_v3 -> robot_arm_mask (SAM2 fusion, fuller arm/claw).
Compute robot_arm_mask on the SAME frames g trains on (video 1/2/3, STRIDE=3), in g's crop
region. robot_arm_mask kernels are tuned for ~224, so run it on the 256 crop
(frame[195:451,195:451]) then resize to 128 (g's resolution). Store {fidx, mask128} per video
+ a QC overlay (robot_arm_mask green vs robot_v3 red, incl. HARD over-plate frames) to
EYEBALL that the fusion mask is fuller / unbroken before retraining g.
Output: outputs/robot_arm_mask_g_cache/  (env: phantom, needs SAM2)
"""
import os, sys, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ".")
sys.path.insert(0, "phantom/submodules/sam2")
sys.path.insert(0, "phantom/submodules/phantom-E2FGVI")
import inpaint_gen_subspace as IG
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from apply_robot_mask_v3 import robot_v3

OUT = "outputs/flow_wm/robot_arm_mask_g_cache"; os.makedirs(OUT, exist_ok=True)
RES = 128
dev = "cuda" if torch.cuda.is_available() else "cpu"
IG._pred = SAM2ImagePredictor(build_sam2("sam2_hiera_l.yaml", IG.SAM2_CKPT, device=dev))


def arm_mask128(c256):
    # robot_arm_mask kernels tuned for ~224 -> run on 224, resize mask back to 128
    c224 = cv2.resize(c256, (224, 224), interpolation=cv2.INTER_AREA)
    m = IG.robot_arm_mask(np.ascontiguousarray(c224))
    return (cv2.resize(m.astype(np.float32), (RES, RES)) > 0.5).astype(np.uint8)


qc = []   # (frame128, arm128, v3_128, hard?)
for v in (1, 2, 3):
    z = np.load(f"{OUT}/frames256_v{v}.npz")            # full-color frames from original video
    fidx_in, frames = z["fidx"], z["frames"]
    fidx, masks = [], []
    for i in range(len(frames)):
        c256 = np.ascontiguousarray(frames[i]); arm = arm_mask128(c256)
        fidx.append(int(fidx_in[i])); masks.append(arm)
        if v == 3 and len(qc) < 14:   # QC on held-out video, mix clean/hard
            c128 = cv2.resize(c256, (RES, RES), interpolation=cv2.INTER_AREA)
            c224 = cv2.resize(c256, (224, 224), interpolation=cv2.INTER_AREA)
            hard = not IG.clear_of_objects(IG.robot_arm_mask(np.ascontiguousarray(c224)), c224)
            qc.append((c128, arm, (robot_v3(np.ascontiguousarray(c128)) > 0).astype(np.uint8), hard))
    np.savez(f"{OUT}/v{v}_armmask.npz", fidx=np.array(fidx, np.int32), masks=np.stack(masks).astype(np.uint8))
    cov = np.stack(masks).reshape(len(masks), -1).mean()
    print(f"video {v}: {len(masks)} frames, robot_arm_mask cov {cov:.3f}", flush=True)

# QC: prefer some hard frames
qc.sort(key=lambda t: not t[3])  # hard first
qc = qc[:8]
def bright(im): return np.clip(im.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
fig, ax = plt.subplots(3, len(qc), figsize=(2.3 * len(qc), 7))
for j, (c, arm, v3, hard) in enumerate(qc):
    b = bright(c); a = arm > 0; r = v3 > 0
    oa = b.copy(); oa[a] = (0.35 * oa[a] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
    ov = b.copy(); ov[r] = (0.35 * ov[r] + 0.65 * np.array([255, 0, 0])).astype(np.uint8)
    ax[0, j].imshow(b); ax[0, j].axis("off"); ax[0, j].set_title("HARD" if hard else "clean", fontsize=8)
    ax[1, j].imshow(oa); ax[1, j].axis("off"); ax[1, j].text(3, 14, f"cov{a.mean():.2f}", color="yellow", fontsize=8)
    if j == 0: ax[1, j].set_title("robot_arm_mask SAM2-fusion (green) -> NEW g supervision", fontsize=8, loc="left")
    ax[2, j].imshow(ov); ax[2, j].axis("off"); ax[2, j].text(3, 14, f"cov{r.mean():.2f}", color="yellow", fontsize=8)
    if j == 0: ax[2, j].set_title("robot_v3 (red) -> OLD g supervision", fontsize=8, loc="left")
fig.suptitle("g's NEW vs OLD supervision on g's own crop (video3). Is SAM2-fusion fuller / claw+rod unbroken?", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/qc.png", dpi=130)
open(f"{OUT}/summary.txt", "w").write("robot_arm_mask (SAM2 fusion) precomputed for g on video 1/2/3, g's 256-crop->128.\n"
                                       "see qc.png: green(new)=robot_arm_mask vs red(old)=robot_v3.\n")
print(f"saved {OUT}/qc.png + v[1,2,3]_armmask.npz\n=== DONE ===", flush=True)
