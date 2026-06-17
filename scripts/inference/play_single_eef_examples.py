"""
Visualize where the motion-weighted score (mscore) comes from, on illustrative frames.

GT action replay -> for selected frames show, side by side:
  GT | Pred | GT motion |GT_t-GT_{t-1}| | Pred motion |Pred_t-Pred_{t-1}| | weighted error (GTmotion * |Pred-GT|)

Picks the WORST (high mscore) and BEST (low mscore) frames among high-motion frames,
so you can see whether the model tracks the moving object, or predicts a static/blurry
scene (which mscore penalizes via the motion-magnitude-match term).

Usage:
    python scripts/inference/play_single_eef_examples.py \
        --ckpt outputs/.../checkpoints/best.ckpt \
        --dataset_dir data/play_robot_v3_hdf5 --split val --episode 0 \
        --max_frames 200 --lam 1.0 --out debug/metric_examples/ep0.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from play_single_eef_metric import (  # noqa: E402
    load_episode_split,
    load_model,
    replay_rollout,
)


def _maps(gt, gt_prev, pred, pred_prev):
    """Per-pixel signed-change maps (channel-mean magnitude) for one frame.

    Returns: GT change |dg|, Pred change |dp|, disagreement |dg - dp| (the mscore integrand).
    """
    g = gt.astype(np.float32) / 255.0
    gp = gt_prev.astype(np.float32) / 255.0
    p = pred.astype(np.float32) / 255.0
    dg = g - gp                                           # signed (H,W,3)
    m_g = np.abs(dg).mean(-1)                             # GT change magnitude
    if pred_prev is not None:
        dp = p - pred_prev.astype(np.float32) / 255.0
        m_p = np.abs(dp).mean(-1)                         # pred change magnitude
        disag = np.abs(dg - dp).mean(-1)                  # signed-change disagreement
    else:
        m_p = np.zeros_like(m_g)
        disag = m_g
    return m_g, m_p, disag


def per_frame_scores(preds0, cam0, actions, model, device, dtype):
    """Per-frame mscore (signed-change agreement), action weight a_t, and mse.

    a_t = ||normalize(act_t) - normalize(act_{t-1})|| — same weight the metric uses, so frame
    selection matches the metric (low-action / noise frames get small a_t and are excluded).
    """
    a_norm = model.normalizer["action"].normalize(
        torch.from_numpy(actions).to(device=device, dtype=dtype)
    ).cpu().numpy()
    scores, a_ts, mses = [], [], []
    prev_g = cam0[0].astype(np.float32) / 255.0
    prev_p = None
    for i, pred in enumerate(preds0):
        t = i + 1
        g = cam0[t].astype(np.float32) / 255.0
        p = pred.astype(np.float32) / 255.0
        mses.append(float(((p - g) ** 2).mean()))
        a_ts.append(float(np.linalg.norm(a_norm[t] - a_norm[t - 1])))
        if prev_p is None:               # first frame has no previous pred -> skip from selection
            scores.append(np.nan)
            prev_g, prev_p = g, p
            continue
        dg = g - prev_g
        mass = float(np.abs(dg).sum()) + 1e-9
        scores.append(float(np.abs(dg - (p - prev_p)).sum()) / mass)
        prev_g, prev_p = g, p
    return np.array(scores), np.array(a_ts), np.array(mses)


def main():
    ap = argparse.ArgumentParser(description="Visualize motion-weighted score on best/worst frames")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset_dir", default="data/play_robot_v3_hdf5")
    ap.add_argument("--split", default="val")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--dec_infer_steps", type=int, default=3)
    ap.add_argument("--max_frames", type=int, default=None,
                    help="Use only the first N frames (e.g. 200 ~= 20s @ 10Hz)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="debug/metric_examples/examples.png")
    args = ap.parse_args()

    h = args.resolution
    cam0, cam1, actions = load_episode_split(args.dataset_dir, args.split, args.episode)
    if args.max_frames:
        cam0, cam1, actions = cam0[:args.max_frames], cam1[:args.max_frames], actions[:args.max_frames]
        print(f"Using first {len(actions)} frames of episode {args.episode}")

    model = load_model(args.ckpt, args.device, args.dec_infer_steps).to(args.device)
    dtype = model.dtype
    preds0, preds1 = replay_rollout(model, cam0, cam1, actions, h, args.device, dtype)

    score, a_t, mse = per_frame_scores(preds0, cam0, actions, model, args.device, dtype)

    # restrict to LARGE-action frames (top quartile of commanded motion), matching the metric's
    # a_t weighting, so low-action / noise-only frames are excluded
    thr = float(np.percentile(a_t[a_t > 0], 75)) if np.any(a_t > 0) else 0.0
    cand = np.where(a_t >= thr)[0]
    cand = cand[~np.isnan(score[cand])]              # drop first frame (nan score)
    if len(cand) < 2:
        cand = np.where(~np.isnan(score))[0]
    worst = int(cand[np.argmax(score[cand])])
    best = int(cand[np.argmin(score[cand])])
    cases = [("WORST (high mscore)", worst), ("BEST (low mscore)", best)]

    col_titles = ["GT", "Pred", "GT change |dg|", "Pred change |dp|", "disagree |dg-dp|"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    for r, (label, idx) in enumerate(cases):
        gt, pred = cam0[idx + 1], preds0[idx]
        gt_prev = cam0[idx]
        pred_prev = preds0[idx - 1] if idx > 0 else None
        m_g, m_p, disag = _maps(gt, gt_prev, pred, pred_prev)
        vmax = max(m_g.max(), m_p.max(), disag.max(), 1e-6)  # shared scale across change maps

        for col in range(5):
            ax = axes[r, col]
            if col == 0:
                ax.imshow(gt)
            elif col == 1:
                ax.imshow(pred)
            elif col == 2:
                ax.imshow(m_g, cmap="magma", vmin=0, vmax=vmax)
            elif col == 3:
                ax.imshow(m_p, cmap="magma", vmin=0, vmax=vmax)
            else:
                ax.imshow(disag, cmap="magma", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[col], fontsize=11)
        axes[r, 0].set_ylabel(label, fontsize=11, rotation=0, ha="right", va="center", labelpad=12)
        axes[r, 1].text(0.5, -0.11,
                        f"frame {idx + 1}    mscore={score[idx]:.4f}    a_t={a_t[idx]:.3f}    mse={mse[idx]:.4f}",
                        transform=axes[r, 1].transAxes, ha="center", fontsize=11, color="#333")

    run = os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))
    plt.suptitle(f"signed-change agreement (mscore)   |   {run}  ep{args.episode}", fontsize=13)
    plt.tight_layout(rect=[0.04, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {args.out}")
    print(f"  WORST frame {worst + 1}: mscore={score[worst]:.4f}  mse={mse[worst]:.4f}")
    print(f"  BEST  frame {best + 1}: mscore={score[best]:.4f}  mse={mse[best]:.4f}")


if __name__ == "__main__":
    main()
