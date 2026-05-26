"""Stage B (run in `phantom` env): inpaint-then-DINO subspace diagnostic.

Consumes val clips dumped by `dump_val_clips.py` (run first in `iws`), removes the
agent (human hand+arm / robot arm) with SAM2 masking + E2FGVI video inpainting
(generates replacement background, NOT a gray hole), then measures whether human
and robot frames land in the same DINOv2 subspace.

This is the "generate the background, don't just mask" upgrade of
`mask_agent_sam2_test.py` (which only grayed the masked region, leaving probe
acc ~1.0). See docs/superpowers/specs/2026-05-26-inpaint-dino-subspace-design.md.

Compares raw (agent present) vs inpainted (agent removed) on the SAME frames:
  - domain probe accuracy (0.5 = aligned, 1.0 = disjoint)
  - RBF-MMD^2, between (H vs R) vs within (H1/H2, R1/R2 = scene-noise floor)
  - PCA scatter + QC montages (raw | mask | inpainted) for visual inpaint check.

DINOv2 + probe are inlined here (no `iws` import) so this runs in `phantom`.

Usage (smoke):  python inpaint_dino_subspace.py --clips <dir> --per-domain 64
Usage (full):   python inpaint_dino_subspace.py --clips <dir> --per-domain 900
"""
import argparse
import os
import sys

import cv2
import numpy as np
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

SAM2_CKPT = "phantom/submodules/sam2/checkpoints/sam2_hiera_large.pt"
E2FGVI_CKPT = "phantom/submodules/phantom-E2FGVI/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------
# DINOv2 ViT-S/14 pooled patch tokens (inlined from
# interactive_world_sim/algorithms/common/dinov2_features.py to avoid iws import)
# ----------------------------------------------------------------------------
class DINOv2Pooled(nn.Module):
    def __init__(self, device, input_res=224):
        super().__init__()
        self.input_res = input_res
        self.normalize = torchvision.transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                    pretrained=True, trust_repo=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, images):  # (B,3,H,W) [0,1] -> (B,384) pooled patch tokens
        if images.shape[-1] != self.input_res or images.shape[-2] != self.input_res:
            images = F.interpolate(images, size=self.input_res, mode="bilinear", align_corners=False)
        images = self.normalize(images)
        return self.model.forward_features(images)["x_norm_patchtokens"].mean(dim=1)


# ----------------------------------------------------------------------------
# Domain probe MLP (inlined from latent_decompose/diagnostics/linear_probe.py)
# ----------------------------------------------------------------------------
def train_probe_mlp(feats, labels, hidden=128, epochs=10, lr=1e-3, val_ratio=0.2,
                    device=torch.device("cpu")):
    feats = feats.to(device)
    labels = labels.to(device).long()
    n = feats.shape[0]
    perm = torch.randperm(n, device=device)
    n_val = int(round(n * val_ratio))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    model = nn.Sequential(nn.Linear(feats.shape[1], hidden), nn.GELU(),
                          nn.Linear(hidden, 2)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        loss = F.cross_entropy(model(feats[train_idx]), labels[train_idx])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(feats[val_idx]).argmax(-1)
        return (pred == labels[val_idx]).float().mean().item()


# ----------------------------------------------------------------------------
# SAM2 agent mask: multi-point seeded on the agent region.
# In THIS data BOTH agents are dark: robot = near-black gripper/arm; human = dark
# long-sleeve arm+hand (this user wears long sleeves, NO visible skin — a skin
# seed wrongly grabbed the red cube while leaving the dark arm). So seed both on
# the DARKEST pixels. Multiple spread-out seeds make SAM2 capture the FULL agent
# (a single darkest pixel often lands on the wrong dark blob and the agent
# survives). The agent = hand (light skin) + forearm (BLACK sleeve), one limb; we
# seed BOTH a skin term and a dark-desaturated term so SAM2 grabs the WHOLE arm,
# not just the sleeve. blue-reject keeps the bowl; the red cube is dark+saturated
# so neither term seeds it (and it's not picked as the largest arm mask).
# ----------------------------------------------------------------------------
_RNG = np.random.default_rng(0)


def skin_dark_scores(img):  # HWC [0,1] -> (skin, dark) per-pixel maps, each ~[0,1]
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    red = ((h < 20) | (h > 160)).astype(np.float32)
    skin = red * np.minimum(s, 0.6) * v          # light reddish hand (dark cube -> low v)
    lum = cv2.GaussianBlur(img.mean(axis=2).astype(np.float32), (9, 9), 0)
    dark = 1.0 - lum                              # black sleeve / gripper = darkest (not table)
    skin = cv2.GaussianBlur(skin, (9, 9), 0)
    return skin / (skin.max() + 1e-6), dark / (dark.max() + 1e-6)


def agent_seed_groups(img, k=6, topn=150):
    """Separate seed groups for hand (skin) and sleeve/gripper (dark). Each group
    is segmented tightly then unioned, so the whole arm+hand is covered without a
    single over-inclusive mask swallowing the table. Robot: skin absent -> dark only."""
    skin, dark = skin_dark_scores(img)
    groups = []
    for sc in (skin, dark):
        if sc.max() > 0.3:                        # this term actually present
            top = np.argsort(sc.ravel())[-topn:]
            sel = _RNG.choice(top, size=min(k, len(top)), replace=False)
            ys, xs = np.unravel_index(sel, img.shape[:2])
            groups.append(np.stack([xs, ys], 1))
    return groups


def _is_blue(img, m):  # masked region predominantly blue -> the bowl/plate, not agent
    r, g, b = img[m].mean(0)
    return b > r + 0.06 and b > g + 0.06


def _is_cube(img, m):  # dark saturated red block -> the cube (a task object), not agent
    r, g, b = img[m].mean(0)
    return r > g + 0.04 and r > b + 0.04 and max(r, g, b) < 0.5


def _pick_tight(predictor, img, pts):
    """Highest-score SAM2 mask in [0.5%, 40%] area, rejecting bowl(blue)/cube(red)."""
    masks, scores, _ = predictor.predict(
        point_coords=pts, point_labels=np.ones(len(pts), dtype=int), multimask_output=True)
    masks = [m.astype(bool) for m in masks]
    cand = [(sc, m) for m, sc in zip(masks, scores)
            if 0.005 <= m.mean() <= 0.40 and not _is_blue(img, m) and not _is_cube(img, m)]
    return max(cand, key=lambda t: t[0])[1] if cand else None


def sam_mask(predictor, img, groups, dilate=15):
    """Union of the tight hand mask and the tight sleeve/gripper mask = the whole
    arm+hand, dilated. Bowl and cube preserved by the rejects."""
    predictor.set_image((img * 255).astype(np.uint8))
    H, W = img.shape[:2]
    agent = np.zeros((H, W), dtype=bool)
    for pts in groups:
        m = _pick_tight(predictor, img, pts)
        if m is not None:
            agent |= m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    return cv2.dilate(agent.astype(np.uint8), k).astype(bool)


# ----------------------------------------------------------------------------
# E2FGVI video inpaint
# ----------------------------------------------------------------------------
def load_e2fgvi(device):
    model = InpaintGenerator().to(device)
    model.load_state_dict(torch.load(E2FGVI_CKPT, map_location=device))
    model.eval()
    return model


def _pad(x, h, w):  # x:(1,T,C,H,W) -> reflection-padded to E2FGVI multiples
    mh, mw = 60, 108
    hp, wp = (mh - h % mh) % mh, (mw - w % mw) % mw
    x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, : h + hp, :]
    x = torch.cat([x, torch.flip(x, [4])], 4)[:, :, :, :, : w + wp]
    return x


@torch.no_grad()
def inpaint_clip(model, clip, masks):
    """clip (T,C,H,W) [0,1], masks (T,H,W) bool -> inpainted (T,C,H,W) [0,1]."""
    T, C, H, W = clip.shape
    mask_c = masks.float().unsqueeze(1)               # (T,1,H,W)
    imgs = (clip * 2 - 1).unsqueeze(0)                # (1,T,C,H,W)
    masked = imgs * (1 - mask_c.unsqueeze(0))
    masked = _pad(masked, H, W)
    pred, _ = model(masked, T)                        # (T,C,Hp,Wp) tanh
    pred = ((pred[:, :, :H, :W] + 1) / 2).clamp(0, 1)
    return pred * mask_c + clip * (1 - mask_c)


# ----------------------------------------------------------------------------
# Color-normalization controls (applied to the inpainted frames before DINO).
#   grayworld: per-image gray-world white balance -> removes global color cast.
#   gray:      luminance only -> removes ALL color, keeps structure/texture.
# If grayworld collapses the gap it was white-balance; if gray still separates,
# the gap is structural (geometry/texture/objects), not color.
# ----------------------------------------------------------------------------
def grayworld(x):  # (T,C,H,W) [0,1] -> per-image channel-mean-equalized
    m = x.mean(dim=(2, 3), keepdim=True)              # (T,C,1,1)
    g = m.mean(dim=1, keepdim=True)                   # (T,1,1,1)
    return (x * g / (m + 1e-6)).clamp(0, 1)


def grayscale(x):  # (T,C,H,W) [0,1] -> luminance replicated to 3 channels
    l = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    return l.repeat(1, 3, 1, 1)


CONFIGS = ["raw", "inpaint", "inpaint+grayworld", "inpaint+gray"]


# ----------------------------------------------------------------------------
# Feature extraction: raw + inpainted (+ color-normalized inpainted) same frames
# ----------------------------------------------------------------------------
@torch.no_grad()
def collect(ext, predictor, e2fgvi, clips_np, device, domain, res, per_domain, n_qc):
    feats = {c: [] for c in CONFIGS}
    qc, seen = [], 0
    for clip_u8 in clips_np:                          # (T,C,h,w) uint8
        clip = torch.from_numpy(clip_u8).float().to(device) / 255.0
        T = clip.shape[0]
        clip = F.interpolate(clip, size=res, mode="bilinear", align_corners=False)
        masks = torch.zeros((T, res, res), dtype=torch.bool, device=device)
        for t in range(T):
            img = clip[t].permute(1, 2, 0).cpu().numpy()
            groups = agent_seed_groups(img)
            masks[t] = torch.from_numpy(sam_mask(predictor, img, groups)).to(device)
        inp = inpaint_clip(e2fgvi, clip, masks)
        inp_gw, inp_g = grayworld(inp), grayscale(inp)
        feats["raw"].append(ext(clip).cpu())
        feats["inpaint"].append(ext(inp).cpu())
        feats["inpaint+grayworld"].append(ext(inp_gw).cpu())
        feats["inpaint+gray"].append(ext(inp_g).cpu())
        if len(qc) < n_qc:
            mid = T // 2
            qc.append((clip[mid].cpu(), masks[mid].cpu(), inp[mid].cpu(),
                       inp_gw[mid].cpu(), inp_g[mid].cpu()))
        seen += T
        if seen >= per_domain:
            break
    return {c: torch.cat(v)[:per_domain] for c, v in feats.items()}, qc


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def probe(X, Y):
    f = torch.cat([X, Y], 0)
    lab = torch.tensor([1] * len(X) + [0] * len(Y))
    return float(np.mean([train_probe_mlp(f, lab) for _ in range(3)]))


def rbf_mmd2(X, Y):
    Z = torch.cat([X, Y], 0)
    d2 = torch.cdist(Z, Z) ** 2
    sigma2 = d2[d2 > 0].median()
    k = lambda A, B: torch.exp(-(torch.cdist(A, B) ** 2) / (2 * sigma2))
    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())


def half(X):
    idx = torch.randperm(len(X)); h = len(X) // 2
    return X[idx[:h]], X[idx[h:]]


def report(tag, H, R):
    n = min(len(H), len(R)); H, R = H[:n], R[:n]
    H1, H2 = half(H); R1, R2 = half(R)
    bt_p, bt_m = probe(H, R), rbf_mmd2(H, R)
    wh_m, wr_m = rbf_mmd2(H1, H2), rbf_mmd2(R1, R2)
    ratio = bt_m / (0.5 * (wh_m + wr_m) + 1e-9)
    print(f"\n=== {tag} (n={n}/domain, dim={H.shape[1]}) ===")
    print(f"{'comparison':18s} {'probe-acc':>10s} {'RBF-MMD^2':>12s}")
    print(f"{'between H vs R':18s} {bt_p:>10.3f} {bt_m:>12.4f}")
    print(f"{'within  H1 vs H2':18s} {probe(H1, H2):>10.3f} {wh_m:>12.4f}")
    print(f"{'within  R1 vs R2':18s} {probe(R1, R2):>10.3f} {wr_m:>12.4f}")
    print(f"MMD ratio between/mean-within = {ratio:.1f}  (1=no gap, >>1=real gap)")
    return bt_p, ratio


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def save_qc(qc, domain, out_dir):
    cols = ["raw", "SAM2 mask", "inpaint", "inpaint+grayworld", "inpaint+gray"]
    n = len(qc)
    fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i, (raw, mask, inp, gw, g) in enumerate(qc):
        r = raw.permute(1, 2, 0).numpy().clip(0, 1)
        ov = r.copy()
        ov[mask.numpy()] = ov[mask.numpy()] * 0.3 + np.array([1.0, 0, 0]) * 0.7
        imgs = [r, ov.clip(0, 1)] + [t.permute(1, 2, 0).numpy().clip(0, 1) for t in (inp, gw, g)]
        for j, im in enumerate(imgs):
            axes[i, j].imshow(im)
            if i == 0:
                axes[i, j].set_title(cols[j], fontsize=9)
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle(f"{domain}: agent removal + color-norm QC", fontsize=12)
    fig.tight_layout()
    p = os.path.join(out_dir, f"qc_{domain}.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"saved {p}")


def pca2(X):
    Xc = X - X.mean(0, keepdim=True)
    _, _, V = torch.linalg.svd(Xc, full_matrices=False)
    return (Xc @ V[:2].T).numpy()


def save_pca(Hf, Rf, out_dir):
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(5 * len(CONFIGS), 5))
    for ax, c in zip(axes, CONFIGS):
        H, R = Hf[c], Rf[c]
        n = min(len(H), len(R))
        Z = pca2(torch.cat([H[:n], R[:n]], 0))
        ax.scatter(Z[:n, 0], Z[:n, 1], s=6, alpha=0.5, label="human", c="tab:orange")
        ax.scatter(Z[n:, 0], Z[n:, 1], s=6, alpha=0.5, label="robot", c="tab:blue")
        ax.set_title(c, fontsize=11); ax.legend(fontsize=8)
    fig.suptitle("DINOv2 pooled features — PCA, colored by domain", fontsize=12)
    fig.tight_layout()
    p = os.path.join(out_dir, "pca.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"saved {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="outputs/inpaint_dino_subspace/clips",
                    help="dir with {robot,human}_clips.npy from dump_val_clips.py")
    ap.add_argument("--per-domain", type=int, default=900)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--n-qc", type=int, default=6)
    ap.add_argument("--out-dir", default="outputs/inpaint_dino_subspace")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ext = DINOv2Pooled(device).eval()
    sam = build_sam2("sam2_hiera_l.yaml", SAM2_CKPT, device=device)
    predictor = SAM2ImagePredictor(sam)
    e2fgvi = load_e2fgvi(device)

    robot_clips = np.load(os.path.join(args.clips, "robot_clips.npy"))
    human_clips = np.load(os.path.join(args.clips, "human_clips.npy"))
    print(f"[run] per_domain={args.per_domain} res={args.res} device={device} "
          f"robot_clips={robot_clips.shape} human_clips={human_clips.shape}")

    Rf, rqc = collect(ext, predictor, e2fgvi, robot_clips, device, "robot",
                      args.res, args.per_domain, args.n_qc)
    Hf, hqc = collect(ext, predictor, e2fgvi, human_clips, device, "human",
                      args.res, args.per_domain, args.n_qc)

    save_qc(hqc, "human", args.out_dir)
    save_qc(rqc, "robot", args.out_dir)
    save_pca(Hf, Rf, args.out_dir)

    summary = {c: report(c, Hf[c], Rf[c]) for c in CONFIGS}
    print("\n========== SUMMARY (between-domain) ==========")
    print(f"{'config':22s} {'probe-acc':>10s} {'MMD ratio':>10s}  (probe 0.5=aligned, ratio 1=no gap)")
    for c in CONFIGS:
        p, r = summary[c]
        print(f"{c:22s} {p:>10.3f} {r:>10.1f}")


if __name__ == "__main__":
    main()
