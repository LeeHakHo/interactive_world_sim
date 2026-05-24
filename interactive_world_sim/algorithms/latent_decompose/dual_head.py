"""Dual-head decompose: two independent projection heads on a shared image
encoder output, producing a task latent and an embodiment latent.

Unlike the channel-split approach (which slices a single feature vector and
forces one slice to forget embodiment via an adversarial loss), the dual-head
approach gives z_task and z_emb their OWN parameters. They start from the same
pooled ViT feature but are projected by separate MLPs, then pushed apart by a
CLUB mutual-information penalty (see club.py) rather than a gradient-reversal
adversary. The decoder is unaffected: it still consumes the original spatial
feature, so reconstruction quality is not sacrificed to win an adversarial game.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ProjHead(nn.Module):
    """Spatial-mean-pool then a 2-layer MLP to a target dim."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, spatial_feat: torch.Tensor) -> torch.Tensor:
        """spatial_feat: (B, D, H, W) -> (B, out_dim). Pools dims (2, 3)."""
        pooled = spatial_feat.mean(dim=(2, 3))
        return self.net(pooled)


class DualHead(nn.Module):
    """Two independent heads on a shared ViT spatial feature.

    forward(spatial_feat) -> (z_task, z_emb), both L2-normalised.

    L2-normalisation replaces the RQ-VAE commitment term as the anti-collapse
    anchor: it keeps each latent on the unit sphere so the CLUB penalty cannot
    trivially shrink the latents toward zero to fake independence.
    """

    def __init__(self, in_dim: int = 384, d_task: int = 288, d_emb: int = 96,
                 hidden: int = 512):
        super().__init__()
        self.task_head = _ProjHead(in_dim, d_task, hidden)
        self.emb_head = _ProjHead(in_dim, d_emb, hidden)
        self.d_task = int(d_task)
        self.d_emb = int(d_emb)

    def forward(self, spatial_feat: torch.Tensor):
        z_task = self.task_head(spatial_feat)
        z_emb = self.emb_head(spatial_feat)
        z_task = nn.functional.normalize(z_task, dim=-1, eps=1e-6)
        z_emb = nn.functional.normalize(z_emb, dim=-1, eps=1e-6)
        return z_task, z_emb
