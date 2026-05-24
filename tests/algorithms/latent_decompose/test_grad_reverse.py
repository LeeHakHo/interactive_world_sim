"""Gradient reversal layer: forward identity, backward flips sign × lambda."""
from __future__ import annotations

import pytest
import torch

from interactive_world_sim.algorithms.latent_decompose.domain_heads import (
    grad_reverse,
)


def test_forward_is_identity():
    x = torch.randn(4, 8)
    y = grad_reverse(x, lambda_=0.7)
    torch.testing.assert_close(y, x)


@pytest.mark.parametrize("lam", [0.0, 0.3, 1.0, 2.5])
def test_backward_negates_and_scales(lam):
    x = torch.randn(3, 5, requires_grad=True)
    y = grad_reverse(x, lambda_=lam)
    y.sum().backward()
    # d/dx of sum(y) is 1 normally; with GRL it becomes -lam.
    expected = -lam * torch.ones_like(x)
    torch.testing.assert_close(x.grad, expected)
