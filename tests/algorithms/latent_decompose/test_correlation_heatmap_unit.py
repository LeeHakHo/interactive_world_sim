"""Pure helper for the correlation heatmap (PCA + abs Pearson)."""
from __future__ import annotations

import torch

from interactive_world_sim.algorithms.latent_decompose.diagnostics.correlation_heatmap import (
    abs_pearson_block_max,
    pca_top_k,
)


def test_pca_returns_correct_shape():
    x = torch.randn(64, 20)
    out = pca_top_k(x, k=5)
    assert out.shape == (64, 5)


def test_block_max_independent_blocks():
    """Two independent random matrices → off-diagonal block max should be
    small (well under 0.2 for n=2000)."""
    torch.manual_seed(0)
    a = torch.randn(2000, 15)
    b = torch.randn(2000, 10)
    m = abs_pearson_block_max(a, b)
    assert m < 0.2, f"independent inputs should have low cross-corr: {m:.3f}"


def test_block_max_correlated_blocks():
    """If we copy columns of A into B, the block max should be ≈ 1."""
    torch.manual_seed(0)
    a = torch.randn(2000, 15)
    b = torch.cat([a[:, :5], torch.randn(2000, 5)], dim=1)
    m = abs_pearson_block_max(a, b)
    assert m > 0.9, f"copied columns should give max corr near 1: {m:.3f}"
