"""Domain classification heads + gradient-reversal layer.

Label convention: human=0, robot=1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambda_)


class PooledClassifier(nn.Module):
    """Spatial mean-pool → 2-layer MLP → 2 logits."""

    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D, H, W) → logits (B, 2). Pool reduces dims (2, 3)."""
        pooled = z.mean(dim=(2, 3))
        return self.net(pooled)
