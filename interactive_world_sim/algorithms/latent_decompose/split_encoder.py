"""SplitEncoder: channel-split wrapper over ViTSpatialEncoder.

Splits the ViT spatial feature map along the embed-dim axis into a
"task" portion and an "embodiment" portion. Decoder consumes
cat([z_task, z_emb], dim=1), which is bit-identical to the original
ViT output (the invariant guarded by test_concat_invariant_bitwise_identity).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SplitEncoder(nn.Module):
    def __init__(self, base_vit: nn.Module, d_task: int = 288, d_emb: int = 96):
        super().__init__()
        embed_dim = int(getattr(base_vit, "embed_dim"))
        assert d_task + d_emb == embed_dim, (
            f"d_task({d_task}) + d_emb({d_emb}) != base_vit.embed_dim({embed_dim})"
        )
        self.base = base_vit
        self.d_task = int(d_task)
        self.d_emb = int(d_emb)

    def forward(self, view_obs: torch.Tensor):
        """view_obs: (B, 3, H, W) → (z_task, z_emb, cls_token).

        z_task: (B, d_task, grid_h, grid_w)
        z_emb:  (B, d_emb,  grid_h, grid_w)
        cls_token: (B, embed_dim)
        """
        spatial_feat, cls_token = self.base(view_obs)
        z_task = spatial_feat[:, : self.d_task]
        z_emb = spatial_feat[:, self.d_task :]
        return z_task, z_emb, cls_token

    @staticmethod
    def concat(z_task: torch.Tensor, z_emb: torch.Tensor) -> torch.Tensor:
        """Reassemble z_task and z_emb on the channel dim (dim=1)."""
        return torch.cat([z_task, z_emb], dim=1)
