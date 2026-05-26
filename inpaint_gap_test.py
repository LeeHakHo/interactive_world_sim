"""Proper inpaint test: REMOVE the agent (background-filled, not grayed) for both
domains, then measure the DINOv2 human-vs-robot gap.

human: phantom hand-inpainted video + full arm mask (dilated) -> cv2.inpaint the
       remaining arm -> agent-free.
robot: raw cam_high + dark-gripper mask (dilated) -> cv2.inpaint -> agent-free.
Both center-cropped square + resized to 224 for a consistent DINOv2 comparison.
"""
import sys
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts/eval")
sys.path.insert(0, "phantom/submodules/sam2")
from interactive_world_sim.algorithms.common.dinov2_features import DINOv2PatchExtractor
from interactive_world_sim.algorithms.latent_decompose.diagnostics.linear_probe import train_probe_mlp
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CHUNKS = "phantom/data/processed_play_human_aytsai_chunks/play_human_aytsai_chunks"
HUMAN_CHUNKS = ["1001", "2000", "3000"]
ROBOT_VIDS = [f"play_robot_eef/play_robot_{i}_eef/videos/chunk-000/observation.images.cam_high/episode_000000.mp4" for i in (1, 2, 3)]
SAM2_CKPT = "/scr2/yusenluo/interactive_world_sim/phantom/submodules/sam2/checkpoints/sam2_hiera_large.pt"
K = 220  # frames per domain
CX, CY, CW, CH = 195, 195, 256, 256  # WM dataset crop (x,y,w,h)
_predictor = None


def crop_resize(img):  # HWC [0,1], full-res -> WM crop -> 224
    c = img[CY:CY + CH, CX:CX + CW]
    return cv2.resize(c, (224, 224))


def sam_seed_mask(img224, seed_xy):  # seed (x,y) in 224 coords; returns dilated agent mask
    global _predictor
    _predictor.set_image((img224 * 255).astype(np.uint8))
    masks, scores, _ = _predictor.predict(
        point_coords=np.array([seed_xy]), point_labels=np.array([1]), multimask_output=True
    )
    best, ba = None, -1
    for m, sc in zip(masks, scores):
        frac = m.sum() / m.size
        if 0.02 <= frac <= 0.6 and sc > ba:   # agent-sized (arm/gripper can be large)
            best, ba = m, sc
    if best is None:
        best = masks[int(np.argmax(scores))]
    return cv2.dilate(best.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))


def to_crop_coords(x, y):  # full-res (480x640) point -> 224 crop coords
    return int((x - CX) / CW * 224), int((y - CY) / CH * 224)


def inpaint(img_u8, mask_u8):
    return cv2.inpaint(img_u8, (mask_u8 > 0).astype(np.uint8), 3, cv2.INPAINT_TELEA)


def human_frames():
    # original RGB frames (hand+arm visible); masks_arm gives a reliable HAND
    # location to seed SAM2, which then grabs the full connected arm.
    out, ex = [], []
    per = K // len(HUMAN_CHUNKS) + 1
    for ch in HUMAN_CHUNKS:
        masks = np.load(f"{CHUNKS}/{ch}/segmentation_processor/masks_arm.npy", mmap_mode="r")
        cap = cv2.VideoCapture(f"{CHUNKS}/{ch}/video_rgb_imgs.mkv")
        nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, min(nfr, len(masks)) - 1, per).astype(int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
            if not ok: continue
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            crop = crop_resize(fr)
            hm = np.asarray(masks[i])
            ys, xs = np.where(hm)
            mvis = np.zeros(crop.shape[:2], np.uint8)
            if len(ys) == 0:
                painted = crop
            else:
                seed = to_crop_coords(int(xs.mean()), int(ys.mean()))  # hand centroid -> crop
                m = sam_seed_mask(crop, seed)
                mvis = m
                painted = inpaint((crop * 255).astype(np.uint8), m).astype(np.float32) / 255.0
            out.append(painted)
            if len(ex) < 3: ex.append((crop, mvis, painted))
        cap.release()
    return np.stack(out[:K]), ex


def robot_frames():
    import av  # AV1 -> cv2 can't decode; dataset uses PyAV
    out, ex = [], []
    per = K // len(ROBOT_VIDS) + 1
    grip_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    for vp in ROBOT_VIDS:
        container = av.open(vp); stream = container.streams.video[0]
        step = 8; kept = 0
        for fi, frame in enumerate(container.decode(stream)):
            if fi % step != 0:
                continue
            fr = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
            crop = crop_resize(fr)                       # 224 table-center crop
            lum = cv2.GaussianBlur(crop.mean(2).astype(np.float32), (9, 9), 0)
            yx = np.unravel_index(np.argmin(lum), lum.shape)
            m = sam_seed_mask(crop, (int(yx[1]), int(yx[0])))   # seed darkest (gripper)
            painted = inpaint((crop * 255).astype(np.uint8), m).astype(np.float32) / 255.0
            out.append(painted)
            if len(ex) < 3: ex.append((crop, m, painted))
            kept += 1
            if kept >= per:
                break
        container.close()
    return np.stack(out[:K]), ex


@torch.no_grad()
def dino_pool(ext, frames, device):
    feats = []
    for i in range(0, len(frames), 16):
        x = torch.from_numpy(frames[i:i+16]).permute(0, 3, 1, 2).float().to(device)
        feats.append(ext(x).mean(1).cpu())
    return torch.cat(feats)


def main():
    global _predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext = DINOv2PatchExtractor().to(device).eval()
    _predictor = SAM2ImagePredictor(build_sam2("sam2_hiera_l.yaml", SAM2_CKPT, device=device))
    H, hex_ = human_frames(); R, rex = robot_frames()
    print(f"agent-free frames: human={H.shape} robot={R.shape}")
    Hf, Rf = dino_pool(ext, H, device), dino_pool(ext, R, device)
    n = min(len(Hf), len(Rf)); Hf, Rf = Hf[:n], Rf[:n]
    f = torch.cat([Hf, Rf]); l = torch.tensor([0]*len(Hf)+[1]*len(Rf))
    acc = float(np.mean([train_probe_mlp(f, l, device=torch.device("cpu")) for _ in range(3)]))
    allz = torch.cat([Hf, Rf]); mu, sd = allz.mean(0), allz.std(0)+1e-6
    def smmd(a, b): return float(((((a-mu)/sd).mean(0)-((b-mu)/sd).mean(0))**2).sum())
    mmd = smmd(Hf, Rf)
    # within-domain control (random halves) for reference
    def half(X):
        p = torch.randperm(len(X)); h = len(X)//2; return X[p[:h]], X[p[h:]]
    H1,H2 = half(Hf); R1,R2 = half(Rf)
    wh, wr = smmd(H1,H2), smmd(R1,R2)
    accH = float(np.mean([train_probe_mlp(torch.cat([H1,H2]), torch.tensor([0]*len(H1)+[1]*len(H2)), device=torch.device("cpu")) for _ in range(3)]))
    print(f"INPAINTED (agent-free, WM-crop) H-vs-R: probe acc = {acc:.3f}  std-MMD^2 = {mmd:.3f}")
    print(f"  within control: H1H2 MMD={wh:.3f} R1R2 MMD={wr:.3f} | within-H probe={accH:.3f}")
    print(f"  between/within MMD ratio = {mmd/(0.5*(wh+wr)+1e-9):.1f}   (raw baseline was acc1.0 MMD739 ratio~582)")

    # --- color-norm test: match robot per-channel mean/std to human, re-measure ---
    Rm, Rs = R.reshape(-1, 3).mean(0), R.reshape(-1, 3).std(0) + 1e-6
    Hm, Hs = H.reshape(-1, 3).mean(0), H.reshape(-1, 3).std(0) + 1e-6
    R_cm = np.clip((R - Rm) / Rs * Hs + Hm, 0, 1).astype(np.float32)
    Rf2 = dino_pool(ext, R_cm, device)[:len(Hf)]
    f2 = torch.cat([Hf, Rf2]); l2 = torch.tensor([0]*len(Hf)+[1]*len(Rf2))
    acc2 = float(np.mean([train_probe_mlp(f2, l2, device=torch.device("cpu")) for _ in range(3)]))
    mu2, sd2 = f2.mean(0), f2.std(0)+1e-6
    mmd2 = float(((((Hf-mu2)/sd2).mean(0)-((Rf2-mu2)/sd2).mean(0))**2).sum())
    print(f"  + COLOR-MATCHED robot->human: probe acc = {acc2:.3f}  MMD={mmd2:.3f}  ratio={mmd2/(0.5*(wh+wr)+1e-9):.1f}")

    # --- grayscale control: remove ALL colour, keep structure/texture ---
    def to_gray(fr):
        g = 0.299 * fr[..., 0] + 0.587 * fr[..., 1] + 0.114 * fr[..., 2]
        return np.repeat(g[..., None], 3, -1).astype(np.float32)
    Hg = dino_pool(ext, to_gray(H), device); Rg = dino_pool(ext, to_gray(R), device)
    ng = min(len(Hg), len(Rg)); Hg, Rg = Hg[:ng], Rg[:ng]
    fg = torch.cat([Hg, Rg]); lg = torch.tensor([0] * len(Hg) + [1] * len(Rg))
    accg = float(np.mean([train_probe_mlp(fg, lg, device=torch.device("cpu")) for _ in range(3)]))
    mug, sdg = fg.mean(0), fg.std(0) + 1e-6
    mmdg = float(((((Hg - mug) / sdg).mean(0) - ((Rg - mug) / sdg).mean(0)) ** 2).sum())
    print(f"  + GRAYSCALE (all colour removed): probe acc = {accg:.3f}  MMD={mmdg:.3f}  ratio={mmdg/(0.5*(wh+wr)+1e-9):.1f}")

    # QC: per example show raw | mask (whole arm/gripper, red) | inpainted
    fig, axes = plt.subplots(2, 9, figsize=(27, 6))
    for c in range(3):
        for r, (exs, nm) in enumerate([(hex_, "human"), (rex, "robot")]):
            raw, m, painted = exs[c]
            ov = raw.copy()
            mb = np.asarray(m) > 0
            ov[mb] = ov[mb] * 0.3 + np.array([1.0, 0, 0]) * 0.7
            axes[r, 3*c].imshow(raw.clip(0, 1)); axes[r, 3*c].set_title(f"{nm} raw", fontsize=8)
            axes[r, 3*c+1].imshow(ov.clip(0, 1)); axes[r, 3*c+1].set_title(f"{nm} mask", fontsize=8)
            axes[r, 3*c+2].imshow(painted.clip(0, 1)); axes[r, 3*c+2].set_title(f"{nm} inpainted", fontsize=8)
    for a in axes.ravel(): a.axis("off")
    fig.suptitle(f"agent removal (masks_arm-seeded whole arm) — H-vs-R probe={acc:.2f} (raw 1.00)", fontsize=13)
    fig.tight_layout(); fig.savefig("outputs/inpaint_gap_test.png", dpi=120)
    print("saved outputs/inpaint_gap_test.png")


if __name__ == "__main__":
    main()
