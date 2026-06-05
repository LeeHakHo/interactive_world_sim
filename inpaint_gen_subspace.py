"""Generative agent-removal DINO subspace test (phantom env + GPU).

Both domains get the agent removed with the SAME generative video inpainter
(E2FGVI), then frozen DINOv2 measures the human-vs-robot gap.
  human: whole-arm SAM2 mask (hand+forearm+sleeve, plate-negative, cube-subtracted)
  robot: dark-arm SAM2 mask (gripper/arm, plate-negative, cube-subtracted)
Short contiguous temporal windows give E2FGVI the neighbors it needs to fill behind
the agent (incl. plate/cube the agent occludes); we keep the center frames.

Stage A (iws) must have produced outputs/robot_windows.npz (robot is AV1; phantom
env has no `av`). Human chunks are mkv and decoded here directly.

Run: python inpaint_gen_subspace.py [--smoke]
"""
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "phantom/submodules/sam2")
sys.path.insert(0, "phantom/submodules/phantom-E2FGVI")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from E2FGVI.model.e2fgvi_hq import InpaintGenerator  # type: ignore

ROOT = "phantom/data/processed_play_human_aytsai_chunks/play_human_aytsai_chunks"
CHUNKS = ["1001", "2000", "3000"]
SAM2_CKPT = "phantom/submodules/sam2/checkpoints/sam2_hiera_large.pt"
E2FGVI_CKPT = "phantom/submodules/phantom-E2FGVI/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth"
ROBOT_NPZ = "outputs/robot_windows.npz"
CX, CY, CW, CH = 195, 195, 256, 256
T, STRIDE = 10, 2
KEEP_CENTER = 4                       # center frames kept per inpainted window
_IMAGENET_MEAN, _IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SMOKE = "--smoke" in sys.argv
TARGET = 24 if SMOKE else 240          # target agent-free frames per domain
SCAN_PER_CHUNK = 6 if SMOKE else 80    # human candidate windows scanned per chunk
_pred = None


# ---------- DINOv2 + probe ----------
class DINOv2Pooled(nn.Module):
    def __init__(self, device, res=224):
        super().__init__()
        self.res = res
        self.norm = torchvision.transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True, trust_repo=True).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.to(device); self.device = device

    @torch.no_grad()
    def forward(self, x):  # (B,3,H,W)[0,1] -> (B,384)
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=self.res, mode="bilinear", align_corners=False)
        return self.model.forward_features(self.norm(x))["x_norm_patchtokens"].mean(1)


def train_probe_mlp(feats, labels, hidden=128, epochs=10, lr=1e-3, val_ratio=0.2):
    feats, labels = feats.cpu(), labels.cpu().long()
    n = feats.shape[0]; perm = torch.randperm(n); nv = int(round(n * val_ratio))
    vi, ti = perm[:nv], perm[nv:]
    m = nn.Sequential(nn.Linear(feats.shape[1], hidden), nn.GELU(), nn.Linear(hidden, 2))
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); F.cross_entropy(m(feats[ti]), labels[ti]).backward(); opt.step()
    with torch.no_grad():
        return (m(feats[vi]).argmax(-1) == labels[vi]).float().mean().item()


# ---------- color helpers / masks ----------
def crop(img):
    return cv2.resize(img[CY:CY + CH, CX:CX + CW], (224, 224))


def mask_to_crop(m):
    return cv2.resize(m[CY:CY + CH, CX:CX + CW].astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)


def hsv_of(c):
    return cv2.cvtColor(c, cv2.COLOR_RGB2HSV)


def dark_mask(c):
    return cv2.morphologyEx((hsv_of(c)[..., 2] < 75).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def cube_mask(c):
    h, s, v = (hsv_of(c)[..., k] for k in range(3))
    red = ((h < 12) | (h > 168)) & (s > 90) & (v > 35) & (v < 175)
    m = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1 or st[1:, cv2.CC_STAT_AREA].max() < 30:
        return np.zeros_like(m)
    return (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def plate_pt(c):
    h, s, v = (hsv_of(c)[..., k] for k in range(3))
    blue = ((h > 90) & (h < 130) & (s > 70) & (v > 60)).astype(np.uint8)
    n, lab, st, cent = cv2.connectedComponentsWithStats(blue)
    return None if n <= 1 else tuple(cent[1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))].astype(int))


def plate_mask(c):
    h, s, v = (hsv_of(c)[..., k] for k in range(3))
    blue = ((h > 90) & (h < 130) & (s > 70) & (v > 60)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(blue)
    if n <= 1 or st[1:, cv2.CC_STAT_AREA].max() < 200:
        return np.zeros_like(blue)
    return (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def clear_of_objects(arm, c):
    # "good inpaint" frame: arm present but NOT occluding the plate or the cube (arm
    # is out on the uniform table, near its initial/home pose) -> clean fill, objects
    # fully visible. This is the user's frame-selection strategy for robot + human.
    if arm.sum() < 80:
        return False
    if int((arm & (plate_mask(c) > 0)).sum()) > 40:
        return False
    if int((arm & (cube_mask(c) > 0)).sum()) > 8:
        return False
    return True


def big_centroid(m):
    n, lab, st, cent = cv2.connectedComponentsWithStats(m)
    if n <= 1 or st[1:, cv2.CC_STAT_AREA].max() < 20:
        return None
    return tuple(cent[1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))].astype(int))


def keep_touching(mask, seed):
    n, lab = cv2.connectedComponents(mask)
    out = np.zeros_like(mask)
    for c in range(1, n):
        comp = lab == c
        if (comp & (seed > 0)).any():
            out[comp] = 1
    return out


def _finish_arm(c, arm, seed):
    arm = keep_touching(((arm | seed) > 0).astype(np.uint8), seed)
    arm = cv2.morphologyEx(arm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    arm = cv2.dilate(arm, np.ones((7, 7), np.uint8))
    cube = cv2.dilate(cube_mask(c), np.ones((5, 5), np.uint8))
    return (arm & ~(cube > 0)).astype(np.uint8)


def human_arm_mask(c, masks_arm_i):
    hand = mask_to_crop(masks_arm_i > 0)
    if hand.sum() == 0:
        return np.zeros(c.shape[:2], np.uint8)
    dark = dark_mask(c)
    pts, labs = [big_centroid(hand)], [1]
    sp, pp = big_centroid(dark), plate_pt(c)
    if sp:
        pts.append(sp); labs.append(1)
    if pp:
        pts.append(pp); labs.append(0)
    _pred.set_image(c)
    masks, scores, _ = _pred.predict(point_coords=np.array(pts), point_labels=np.array(labs), multimask_output=True)
    valid = [(m, s) for m, s in zip(masks, scores) if 0.03 <= m.mean() <= 0.45]
    arm_sam = (max(valid, key=lambda t: t[0].sum())[0] if valid
               else masks[int(np.argmin([m.mean() for m in masks]))]).astype(np.uint8)
    arm = (arm_sam | hand | keep_touching(dark, (arm_sam | hand))).astype(np.uint8)
    return _finish_arm(c, arm, hand)


def robot_arm_mask(c):
    # Trossen arm+gripper is near-black but THIN (rods + claw). MORPH_OPEN erodes the
    # thin rods and the claw tip is often disconnected from the arm body, so a plain
    # dark-connected mask leaves black bits after inpaint. Fix: dark WITHOUT opening
    # (keep thin rods), DILATE generously to bridge the claw->arm gap, take the
    # connected component, then keep the dark pixels inside it and thicken them. Tight
    # to the arm shape (not a plate-eating blob), so E2FGVI restores the plate.
    hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)
    hh, ss, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    dark = (v < 85).astype(np.uint8)
    plate = ((hh > 90) & (hh < 130) & (ss > 70) & (v > 60)).astype(np.uint8)
    lum = cv2.GaussianBlur(c.mean(2).astype(np.float32), (9, 9), 0)
    yx = np.unravel_index(int(np.argmin(lum)), lum.shape)
    dk = (int(yx[1]), int(yx[0]))
    seed = np.zeros(c.shape[:2], np.uint8); cv2.circle(seed, dk, 4, 1, -1)
    # (1) SAM2 for the complete articulated arm (catches thin rods + claw)
    pts, labs = [dk], [1]
    sp, pp = big_centroid(dark), plate_pt(c)
    if sp:
        pts.append(sp); labs.append(1)
    if pp:
        pts.append(pp); labs.append(0)
    _pred.set_image(c)
    masks, scores, _ = _pred.predict(point_coords=np.array(pts), point_labels=np.array(labs), multimask_output=True)
    valid = [(m, s) for m, s in zip(masks, scores) if 0.02 <= m.mean() <= 0.6]
    sam = (max(valid, key=lambda t: t[0].sum())[0] if valid else masks[int(np.argmax(scores))]).astype(np.uint8)
    # (2) dark-bridged arm for deterministic thin-rod completeness
    bridged = cv2.dilate(dark, np.ones((11, 11), np.uint8))
    arm_dark = (dark & (keep_touching(bridged, seed) > 0)).astype(np.uint8)
    arm = keep_touching(((sam | arm_dark) > 0).astype(np.uint8), (arm_dark | seed).astype(np.uint8))
    arm = cv2.morphologyEx(arm, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    arm = cv2.dilate(arm, np.ones((5, 5), np.uint8))
    # (3) plate protection: drop PURE-plate pixels SAM2 over-grabbed, but KEEP the
    # dark arm lying over the plate (so the arm is removed and E2FGVI refills plate).
    dark_d = cv2.dilate(dark, np.ones((5, 5), np.uint8))
    arm = (arm & ~((plate > 0) & (dark_d == 0))).astype(np.uint8)
    cube = cv2.dilate(cube_mask(c), np.ones((5, 5), np.uint8))
    return (arm & ~(cube > 0)).astype(np.uint8)


# ---------- E2FGVI ----------
def load_e2fgvi(device):
    m = InpaintGenerator().to(device)
    m.load_state_dict(torch.load(E2FGVI_CKPT, map_location=device))
    return m.eval()


def _pad(x, h, w):
    mh, mw = 60, 108
    hp, wp = (mh - h % mh) % mh, (mw - w % mw) % mw
    x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, : h + hp, :]
    return torch.cat([x, torch.flip(x, [4])], 4)[:, :, :, :, : w + wp]


@torch.no_grad()
def inpaint_clip(model, clip, masks):  # clip(T,C,H,W)[0,1], masks(T,H,W)bool
    Tn, C, H, W = clip.shape
    mask_c = masks.float().unsqueeze(1)
    masked = _pad(((clip * 2 - 1).unsqueeze(0)) * (1 - mask_c.unsqueeze(0)), H, W)
    pred, _ = model(masked, Tn)
    pred = ((pred[:, :, :H, :W] + 1) / 2).clamp(0, 1)
    return pred * mask_c + clip * (1 - mask_c)


# ---------- window collection ----------
def _emit(out, fr_mid, mk_mid, mid, half, frames, ex):
    for k in range(mid - half, mid - half + KEEP_CENTER):
        frames.append((out[k].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8))
    if len(ex) < 4:
        ex.append((fr_mid, mk_mid, (out[mid].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)))


def collect_human(e2fgvi, dev):
    frames, ex = [], []
    mid, half = T // 2, KEEP_CENTER // 2
    target = TARGET // len(CHUNKS)
    for ch in CHUNKS:
        masks = np.load(f"{ROOT}/{ch}/segmentation_processor/masks_arm.npy", mmap_mode="r")
        cap = cv2.VideoCapture(f"{ROOT}/{ch}/video_rgb_imgs.mkv")
        nfr = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(masks))
        starts = np.linspace(int(0.05 * nfr), int(0.85 * nfr) - T * STRIDE, SCAN_PER_CHUNK).astype(int)
        got = 0
        for s in starts:
            if got >= target:
                break
            idxs = [s + k * STRIDE for k in range(T)]
            # cheap gate: build only the CENTER mask, keep window only if arm is clear
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idxs[mid])); ok, fr = cap.read()
            if not ok:
                continue
            cmid = crop(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            mmid = human_arm_mask(cmid, np.asarray(masks[idxs[mid]]))
            if not clear_of_objects(mmid, cmid):
                continue
            fr_list, mk_list, ok_all = [], [], True
            for k, i in enumerate(idxs):
                if k == mid:
                    fr_list.append(cmid); mk_list.append(mmid); continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
                if not ok:
                    ok_all = False; break
                c = crop(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                fr_list.append(c); mk_list.append(human_arm_mask(c, np.asarray(masks[i])))
            if not ok_all:
                continue
            clip = torch.from_numpy(np.stack(fr_list)).permute(0, 3, 1, 2).float().to(dev) / 255.0
            mk = torch.from_numpy(np.stack(mk_list) > 0).to(dev)
            out = inpaint_clip(e2fgvi, clip, mk).cpu().numpy()
            _emit(out, cmid, mmid, mid, half, frames, ex)
            got += KEEP_CENTER
        cap.release()
        print(f"  human chunk {ch}: kept {got} clean frames")
    return np.stack(frames), ex


def collect_robot(e2fgvi, dev):
    wins = np.load(ROBOT_NPZ)["windows"]
    frames, ex = [], []
    mid, half = T // 2, KEEP_CENTER // 2
    scanned = 0
    for w in wins:                                   # (T,224,224,3)
        if len(frames) >= TARGET:
            break
        scanned += 1
        cmid = w[mid]; mmid = robot_arm_mask(cmid)
        if not clear_of_objects(mmid, cmid):
            continue
        mk_list = [mmid if k == mid else robot_arm_mask(c) for k, c in enumerate(w)]
        clip = torch.from_numpy(w).permute(0, 3, 1, 2).float().to(dev) / 255.0
        mk = torch.from_numpy(np.stack(mk_list) > 0).to(dev)
        out = inpaint_clip(e2fgvi, clip, mk).cpu().numpy()
        _emit(out, cmid, mmid, mid, half, frames, ex)
    print(f"  robot: kept {len(frames)} clean frames (scanned {scanned}/{len(wins)} windows)")
    return np.stack(frames), ex


# ---------- metrics ----------
@torch.no_grad()
def dino_pool(ext, frames_u8, dev):
    feats = []
    for i in range(0, len(frames_u8), 16):
        x = torch.from_numpy(frames_u8[i:i + 16]).permute(0, 3, 1, 2).float().to(dev) / 255.0
        feats.append(ext(x).cpu())
    return torch.cat(feats)


def probe3(f, l):
    return float(np.mean([train_probe_mlp(f, l) for _ in range(3)]))


def smmd(a, b, mu, sd):
    return float(((((a - mu) / sd).mean(0) - ((b - mu) / sd).mean(0)) ** 2).sum())


def main():
    global _pred
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _pred = SAM2ImagePredictor(build_sam2("sam2_hiera_l.yaml", SAM2_CKPT, device=dev))
    e2fgvi = load_e2fgvi(dev)
    ext = DINOv2Pooled(dev)
    H, hex_ = collect_human(e2fgvi, dev)
    R, rex = collect_robot(e2fgvi, dev)
    print(f"agent-free (E2FGVI) frames: human={H.shape} robot={R.shape}")

    Hf, Rf = dino_pool(ext, H, dev), dino_pool(ext, R, dev)
    n = min(len(Hf), len(Rf)); Hf, Rf = Hf[:n], Rf[:n]
    f = torch.cat([Hf, Rf]); l = torch.tensor([0] * len(Hf) + [1] * len(Rf))
    acc = probe3(f, l)
    allz = torch.cat([Hf, Rf]); mu, sd = allz.mean(0), allz.std(0) + 1e-6
    mmd = smmd(Hf, Rf, mu, sd)
    p = torch.randperm(len(Hf)); h = len(Hf) // 2
    wh = smmd(Hf[p[:h]], Hf[p[h:]], mu, sd); pr = torch.randperm(len(Rf)); hr = len(Rf) // 2
    wr = smmd(Rf[pr[:hr]], Rf[pr[hr:]], mu, sd)
    accH = probe3(torch.cat([Hf[p[:h]], Hf[p[h:]]]), torch.tensor([0] * h + [1] * (len(Hf) - h)))
    print(f"E2FGVI agent-free H-vs-R: probe acc = {acc:.3f}  std-MMD^2 = {mmd:.3f}")
    print(f"  within: H {wh:.3f} R {wr:.3f} | within-H probe {accH:.3f} | ratio = {mmd/(0.5*(wh+wr)+1e-9):.1f}")

    # color-match robot->human + grayscale controls
    Rm, Rs = R.reshape(-1, 3).mean(0), R.reshape(-1, 3).std(0) + 1e-6
    Hm, Hs = H.reshape(-1, 3).mean(0), H.reshape(-1, 3).std(0) + 1e-6
    Rcm = np.clip((R - Rm) / Rs * Hs + Hm, 0, 255).astype(np.uint8)
    Rf2 = dino_pool(ext, Rcm, dev)[:len(Hf)]
    f2 = torch.cat([Hf, Rf2]); mu2 = f2.mean(0); sd2 = f2.std(0) + 1e-6
    print(f"  + color-matched: probe {probe3(f2, l):.3f}  MMD {smmd(Hf, Rf2, mu2, sd2):.3f}")

    def gray(a):
        g = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.uint8)
        return np.repeat(g[..., None], 3, -1)
    Hg, Rg = dino_pool(ext, gray(H), dev)[:n], dino_pool(ext, gray(R), dev)[:n]
    fg = torch.cat([Hg, Rg]); mug = fg.mean(0); sdg = fg.std(0) + 1e-6
    print(f"  + grayscale: probe {probe3(fg, l):.3f}  MMD {smmd(Hg, Rg, mug, sdg):.3f}")

    # QC: examples stacked as ROWS (readable), raw | mask | E2FGVI
    rows = [("human", e) for e in hex_] + [("robot", e) for e in rex]
    nr = len(rows)
    fig, axes = plt.subplots(nr, 3, figsize=(10, 3.2 * nr))
    for r, (nm, (raw, m, painted)) in enumerate(rows):
        ov = raw.copy(); ov[m > 0] = (0.3 * ov[m > 0] + np.array([255, 0, 0]) * 0.7).astype(np.uint8)
        for k, (im, t) in enumerate([(raw, f"{nm} raw"), (ov, "arm mask"), (painted, "E2FGVI inpaint")]):
            axes[r, k].imshow(im); axes[r, k].axis("off")
            axes[r, k].set_title(t, fontsize=10)
    fig.suptitle(f"E2FGVI agent removal, clean (arm clear of plate/cube) frames — H-vs-R probe={acc:.2f}, K={n}/dom", fontsize=12)
    fig.tight_layout(); fig.savefig("outputs/inpaint_gen_subspace.png", dpi=120)
    print("saved outputs/inpaint_gen_subspace.png")

    # save selected agent-free frames + QC examples (per-domain stacks) for re-plotting
    def stack_ex(exs):
        z = np.zeros((0, 224, 224, 3), np.uint8)
        if not exs:
            return z, z[..., 0], z
        return (np.stack([e[0] for e in exs]), np.stack([e[1] for e in exs]), np.stack([e[2] for e in exs]))
    hr, hm, hp = stack_ex(hex_); rr, rm, rp = stack_ex(rex)
    np.savez_compressed("outputs/inpaint_gen_frames.npz", human=H, robot=R,
                        hex_raw=hr, hex_mask=hm, hex_paint=hp, rex_raw=rr, rex_mask=rm, rex_paint=rp)
    print("saved outputs/inpaint_gen_frames.npz")


if __name__ == "__main__":
    main()
