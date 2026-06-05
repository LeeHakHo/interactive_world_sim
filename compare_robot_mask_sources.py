"""User remembers the robot mask being good (05_robot_mask_e2fgvi.png). That mask is
robot_arm_mask = SAM2 ∪ dark-bridge FUSION (not pure SAM2, not the v3 that supervises g),
AND it was shown only on cherry-picked clean frames (collect_robot skips arm-over-plate).
Honest side-by-side on the SAME frames, BRIGHT, high-dpi: robot_arm_mask (SAM2-fusion, red)
vs robot_v3 (deterministic, what g learns, green), split into CLEAN frames and HARD frames
(arm over plate/cube). Decides whether the fusion mask is a better supervision source for g.
Output: outputs/compare_robot_mask_sources/   (env: phantom, needs SAM2)
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

OUT = "outputs/flow_wm/compare_robot_mask_sources"; os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
IG._pred = SAM2ImagePredictor(build_sam2("sam2_hiera_l.yaml", IG.SAM2_CKPT, device=dev))

wins = np.load(IG.ROBOT_NPZ)["windows"]          # (Nwin, T, 224,224,3)
mid = wins.shape[1] // 2
print(f"windows {wins.shape}", flush=True)

clean, hard = [], []
for w in wins:
    if len(clean) >= 6 and len(hard) >= 6: break
    c = w[mid]
    arm = IG.robot_arm_mask(c)
    ok = IG.clear_of_objects(arm, c)
    if ok and len(clean) < 6: clean.append((c, arm))
    elif (not ok) and len(hard) < 6: hard.append((c, arm))
print(f"collected clean={len(clean)} hard={len(hard)}", flush=True)

picks = [("CLEAN", clean), ("HARD(over plate/cube)", hard)]
allf = clean + hard
labels = ["CLEAN"] * len(clean) + ["HARD"] * len(hard)
nC = len(allf)

def bright(im): return np.clip(im.astype(np.float32) * 1.25, 0, 255).astype(np.uint8)

fig, ax = plt.subplots(3, nC, figsize=(2.4 * nC, 7.4))
for j, ((c, arm), lab) in enumerate(zip(allf, labels)):
    v3 = robot_v3(np.ascontiguousarray(c), bridge=12) > 0
    arm = arm > 0
    base = bright(c)
    oa = base.copy(); oa[arm] = (0.35 * oa[arm] + 0.65 * np.array([255, 0, 0])).astype(np.uint8)
    ov = base.copy(); ov[v3] = (0.35 * ov[v3] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
    ax[0, j].imshow(base); ax[0, j].axis("off"); ax[0, j].set_title(lab, fontsize=8)
    ax[1, j].imshow(oa); ax[1, j].axis("off"); ax[1, j].text(3, 16, f"cov{arm.mean():.2f}", color="yellow", fontsize=8)
    if j == 0: ax[1, j].set_title("robot_arm_mask = SAM2∪dark FUSION (red)", fontsize=8, loc="left")
    ax[2, j].imshow(ov); ax[2, j].axis("off"); ax[2, j].text(3, 16, f"cov{v3.mean():.2f}", color="yellow", fontsize=8)
    if j == 0: ax[2, j].set_title("robot_v3 = what g learns (green)", fontsize=8, loc="left")
fig.suptitle("SAME frames, bright. Fusion mask (red) vs v3 (green). Left=clean, right=HARD (arm over plate/cube). "
             "Is the fusion mask better enough to retrain g on it?", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/compare.png", dpi=135)
open(f"{OUT}/summary.txt", "w").write(
    f"robot_arm_mask (SAM2 fusion) vs robot_v3, same frames, clean={len(clean)} hard={len(hard)}\n"
    "see compare.png: does the fusion mask hold up on HARD (over-plate) frames or only clean ones?\n")
print(f"saved {OUT}/compare.png + summary.txt\n=== DONE ===", flush=True)
