"""Tests for SplitEncoder: shape contract and the bit-identical concat invariant.

The "concat invariant" is the load-bearing property of Phase 0: with no new
loss, the network must reproduce the baseline ViT output exactly.
"""
from __future__ import annotations

import pytest
import torch

from interactive_world_sim.algorithms.latent_decompose.split_encoder import (
    SplitEncoder,
)
from interactive_world_sim.algorithms.latent_dynamics.dynamo_ssl_module import (
    ViTSpatialEncoder,
)


@pytest.fixture
def vit():
    return ViTSpatialEncoder(
        img_size=128, patch_size=8, embed_dim=384, depth=2, num_heads=6,
    )


def test_dim_assertion_rejects_bad_split(vit):
    with pytest.raises(AssertionError):
        SplitEncoder(vit, d_task=200, d_emb=100)   # 300 != 384


def test_shapes(vit):
    enc = SplitEncoder(vit, d_task=288, d_emb=96)
    x = torch.randn(2, 3, 128, 128)
    z_task, z_emb, cls = enc(x)
    assert z_task.shape == (2, 288, 16, 16)
    assert z_emb.shape  == (2,  96, 16, 16)
    assert cls.shape    == (2, 384)


def test_concat_invariant_bitwise_identity(vit):
    """SplitEncoder.concat(SplitEncoder(x)) must equal vit(x)[0] up to fp tol."""
    enc = SplitEncoder(vit, d_task=288, d_emb=96)
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        z_task, z_emb, cls = enc(x)
        recombined = SplitEncoder.concat(z_task, z_emb)
        spatial_feat, cls_ref = vit(x)
    torch.testing.assert_close(recombined, spatial_feat, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cls, cls_ref, rtol=1e-6, atol=1e-6)
