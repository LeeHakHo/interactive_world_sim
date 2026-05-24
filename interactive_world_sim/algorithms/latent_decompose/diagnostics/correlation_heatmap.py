"""Correlation heatmap diagnostic — Phase 0 Step 2.

Pass: off-diagonal (z_task × z_emb) block max < 0.2 (PHASE0_IWS_PLAN_v3.md §2.2).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .linear_probe import _collect_features, _load_lwm


def pca_top_k(x: torch.Tensor, k: int) -> torch.Tensor:
    """Project x (N, D) to its top-k principal components. Centered."""
    x = x - x.mean(dim=0, keepdim=True)
    _, _, v = torch.linalg.svd(x, full_matrices=False)
    return x @ v[:k].t()


def abs_pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|Pearson corr| between every column of a (N, Da) and b (N, Db).
    Returns (Da, Db) tensor in [0, 1]."""
    az = (a - a.mean(0)) / (a.std(0, unbiased=False) + 1e-8)
    bz = (b - b.mean(0)) / (b.std(0, unbiased=False) + 1e-8)
    n = a.shape[0]
    return (az.t() @ bz / n).abs()


def abs_pearson_block_max(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(abs_pearson(a, b).max().item())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--save", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    m, cfg = _load_lwm(args.ckpt, device)
    z_task, _ = _collect_features(m, cfg, "z_task", device)
    z_emb,  _ = _collect_features(m, cfg, "z_emb",  device)
    z_task = z_task[: args.n]
    z_emb  = z_emb[:  args.n]

    p_task = pca_top_k(z_task, 15)
    p_emb  = pca_top_k(z_emb,  10)

    big = torch.cat([p_task, p_emb], dim=1)         # (n, 25)
    corr = abs_pearson(big, big)                     # (25, 25)
    block_max = abs_pearson_block_max(p_task, p_emb)
    print(f"[corr] off-diagonal (task×emb) block max = {block_max:.4f}")

    # Render heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(corr.numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.axhline(14.5, color="white", lw=1)
    ax.axvline(14.5, color="white", lw=1)
    ax.set_title(f"|Pearson|  task×emb block max = {block_max:.3f}")
    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[corr] saved → {args.save}")

    ok = block_max < 0.2
    print(f"[corr] {'PASS' if ok else 'FAIL'} per Phase 0 §2.2")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
