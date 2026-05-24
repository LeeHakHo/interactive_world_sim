"""PooledClassifier: spatial mean-pool → MLP → 2 logits."""
from __future__ import annotations

import torch

from interactive_world_sim.algorithms.latent_decompose.domain_heads import (
    PooledClassifier,
)


def test_shape_contract():
    clf = PooledClassifier(d_in=96, hidden=32)
    z = torch.randn(8, 96, 16, 16)
    out = clf(z)
    assert out.shape == (8, 2)


def test_uses_spatial_mean_pool():
    """Pool over (2, 3): permuting H or W must not change the prediction."""
    clf = PooledClassifier(d_in=96, hidden=32)
    z = torch.randn(2, 96, 4, 4)
    out_a = clf(z)
    z_flip = z.flip(dims=(2, 3))
    out_b = clf(z_flip)
    torch.testing.assert_close(out_a, out_b, rtol=1e-6, atol=1e-6)


def test_gradient_flows():
    clf = PooledClassifier(d_in=96, hidden=32)
    z = torch.randn(4, 96, 8, 8, requires_grad=True)
    loss = clf(z).sum()
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0
