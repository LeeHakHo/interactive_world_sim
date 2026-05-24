"""Pure-MLP probe helper used by linear_probe.py. Real-ckpt loading is
exercised manually off-CI by the human running Step 2."""
from __future__ import annotations

import torch

from interactive_world_sim.algorithms.latent_decompose.diagnostics.linear_probe import (
    train_probe_mlp,
)


def test_probe_separates_linearly_separable_data():
    """Two well-separated Gaussians → probe should easily exceed 90% acc."""
    torch.manual_seed(0)
    n = 200
    feats = torch.cat([
        torch.randn(n, 16) + 3.0,    # class 1
        torch.randn(n, 16) - 3.0,    # class 0
    ], dim=0)
    labels = torch.cat([torch.ones(n, dtype=torch.long),
                        torch.zeros(n, dtype=torch.long)], dim=0)
    acc = train_probe_mlp(feats, labels, hidden=32, epochs=10, lr=1e-2)
    assert acc > 0.9, f"linearly separable case should be easy: got {acc:.3f}"


def test_probe_near_chance_on_random_labels():
    """Random labels → near 50% acc."""
    torch.manual_seed(0)
    feats = torch.randn(400, 16)
    labels = torch.randint(0, 2, (400,))
    acc = train_probe_mlp(feats, labels, hidden=32, epochs=5, lr=1e-2)
    assert 0.35 < acc < 0.65, f"random labels should be near 50%: got {acc:.3f}"
