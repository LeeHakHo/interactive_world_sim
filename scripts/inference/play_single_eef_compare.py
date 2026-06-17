"""
Compare TWO checkpoints on the SAME frames to show what mscore captures that MSE does not.

GT action replay for both models on one held-out episode. Among high-action frames, picks
the ones where the two models have similar MSE but very different mscore -> "pixel error says
they're equal, but mscore says one tracks the motion better".

Each selected frame shows:
  GT | GT motion | A:Pred | A:motion | B:Pred | B:motion
with MSE / mscore for each model in the caption.

Usage:
    python scripts/inference/play_single_eef_compare.py \
        --ckpt_a A/best.ckpt --ckpt_b B/best.ckpt \
        --dataset_dir data/play_robot_v3_hdf5 --split val --episode 0 \
        --max_frames 200 --lam 1.0 --n 2 --out debug/metric_examples/compare.png
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


def per_frame(preds0, cam0, actions, model, device, dtype):
    """Return per-frame mscore (signed-change agreement), mse, action weight a_t."""
    a_norm = model.normalizer["action"].normalize(
        torch.from_numpy(actions).to(device=device, dtype=dtype)
    ).cpu().numpy()
    scores, mses, a_ts = [], [], []
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
    return np.array(scores), np.array(mses), np.array(a_ts)


def disagree_map(gt, gt_prev, pred, pred_prev):
    """|dg - dp| channel-mean — the per-pixel mscore integrand (what the metric sees)."""
    dg = gt.astype(np.float32) / 255.0 - gt_prev.astype(np.float32) / 255.0
    if pred_prev is None:
        return np.abs(dg).mean(-1)
    dp = pred.astype(np.float32) / 255.0 - pred_prev.astype(np.float32) / 255.0
    return np.abs(dg - dp).mean(-1)


def gt_change_map(gt, gt_prev):
    return np.abs(gt.astype(np.float32) / 255.0 - gt_prev.astype(np.float32) / 255.0).mean(-1)


def _norm(x):
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo + 1e-9)


def main():
    ap = argparse.ArgumentParser(description="Compare two checkpoints on same frames (mse vs mscore)")
    ap.add_argument("--ckpt_a", required=True)
    ap.add_argument("--ckpt_b", required=True)
    ap.add_argument("--dataset_dir", default="data/play_robot_v3_hdf5")
    ap.add_argument("--split", default="val")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--dec_infer_steps", type=int, default=3)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--n", type=int, default=2, help="Number of frames to show")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="debug/metric_examples/compare.png")
    args = ap.parse_args()

    h = args.resolution
    cam0, cam1, actions = load_episode_split(args.dataset_dir, args.split, args.episode)
    if args.max_frames:
        cam0, cam1, actions = cam0[:args.max_frames], cam1[:args.max_frames], actions[:args.max_frames]

    def rollout(ckpt):
        model = load_model(ckpt, args.device, args.dec_infer_steps).to(args.device)
        dtype = model.dtype
        p0, _ = replay_rollout(model, cam0, cam1, actions, h, args.device, dtype)
        sc, ms, at = per_frame(p0, cam0, actions, model, args.device, dtype)
        del model
        torch.cuda.empty_cache()
        return p0, sc, ms, at

    print(f"Model A: {args.ckpt_a}")
    preds_a, sc_a, ms_a, a_t = rollout(args.ckpt_a)
    print(f"Model B: {args.ckpt_b}")
    preds_b, sc_b, ms_b, _ = rollout(args.ckpt_b)

    # high-action frames, then maximize (mscore disagreement - mse disagreement):
    # frames where mse is similar but mscore differs the most
    thr = float(np.percentile(a_t[a_t > 0], 75)) if np.any(a_t > 0) else 0.0
    valid = ~(np.isnan(sc_a) | np.isnan(sc_b))       # drop first frame (nan score)
    cand = np.where((a_t >= thr) & valid)[0]
    if len(cand) < args.n:
        cand = np.where(valid)[0]
    sc_gap = _norm(np.abs(sc_a - sc_b))
    ms_gap = _norm(np.abs(ms_a - ms_b))
    crit = (sc_gap - ms_gap)[cand]
    picks = cand[np.argsort(crit)[::-1][:args.n]]

    ncol = 6
    col_titles = ["GT", "GT change |dg|", "A: Pred", "A: |dg-dp|", "B: Pred", "B: |dg-dp|"]
    fig, axes = plt.subplots(len(picks), ncol, figsize=(3.0 * ncol, 3.2 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]

    for r, idx in enumerate(picks):
        gt = cam0[idx + 1]
        gt_ch = gt_change_map(gt, cam0[idx])
        a_pred, b_pred = preds_a[idx], preds_b[idx]
        a_dis = disagree_map(gt, cam0[idx], a_pred, preds_a[idx - 1] if idx > 0 else None)
        b_dis = disagree_map(gt, cam0[idx], b_pred, preds_b[idx - 1] if idx > 0 else None)
        vmax = max(gt_ch.max(), a_dis.max(), b_dis.max(), 1e-6)  # shared scale (fair compare)

        imgs = [gt, gt_ch, a_pred, a_dis, b_pred, b_dis]
        is_motion = [False, True, False, True, False, True]
        for col in range(ncol):
            ax = axes[r, col]
            if is_motion[col]:
                ax.imshow(imgs[col], cmap="magma", vmin=0, vmax=vmax)
            else:
                ax.imshow(imgs[col])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[col], fontsize=12)
        axes[r, 0].set_ylabel(f"frame {idx + 1}", fontsize=11, rotation=0, ha="right", va="center", labelpad=10)
        axes[r, 2].text(0.5, -0.10, f"A:  mse={ms_a[idx]:.4f}   mscore={sc_a[idx]:.4f}",
                        transform=axes[r, 2].transAxes, ha="center", fontsize=10, color="#1a5276")
        axes[r, 4].text(0.5, -0.10, f"B:  mse={ms_b[idx]:.4f}   mscore={sc_b[idx]:.4f}",
                        transform=axes[r, 4].transAxes, ha="center", fontsize=10, color="#922b21")

    ra = os.path.basename(os.path.dirname(os.path.dirname(args.ckpt_a)))
    rb = os.path.basename(os.path.dirname(os.path.dirname(args.ckpt_b)))
    plt.suptitle(f"A={ra}   vs   B={rb}   (frames: similar MSE, different mscore)", fontsize=13)
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {args.out}")
    for idx in picks:
        print(f"  frame {idx + 1}:  A mse={ms_a[idx]:.4f} mscore={sc_a[idx]:.4f}  |  "
              f"B mse={ms_b[idx]:.4f} mscore={sc_b[idx]:.4f}")

    # --- scatter: per-frame MSE vs mscore (both models) ---
    scatter_out = os.path.splitext(args.out)[0] + "_scatter.png"
    fig2, ax2 = plt.subplots(figsize=(6.5, 6))
    ax2.scatter(ms_a, sc_a, s=14, alpha=0.5, color="#1a5276", label=f"A ({ra})")
    ax2.scatter(ms_b, sc_b, s=14, alpha=0.5, color="#922b21", label=f"B ({rb})")
    # circle the frames shown in the comparison figure
    ax2.scatter(ms_a[picks], sc_a[picks], s=90, facecolor="none", edgecolor="#1a5276", linewidths=2)
    ax2.scatter(ms_b[picks], sc_b[picks], s=90, facecolor="none", edgecolor="#922b21", linewidths=2)
    both_ms = np.concatenate([ms_a, ms_b])
    both_sc = np.concatenate([sc_a, sc_b])
    r_corr = float(np.corrcoef(both_ms, both_sc)[0, 1])
    ax2.set_xlabel("MSE per frame (lower = better)")
    ax2.set_ylabel("mscore per frame (lower = better)")
    ax2.set_title(f"per-frame MSE vs mscore   (corr = {r_corr:.2f})\n"
                  f"scatter / low corr => mscore measures something MSE doesn't")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(scatter_out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {scatter_out}   (MSE-mscore corr = {r_corr:.2f})")


if __name__ == "__main__":
    main()
