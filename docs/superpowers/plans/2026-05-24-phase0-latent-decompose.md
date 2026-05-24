# Phase 0 Latent Decomposition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the SplitEncoder + L_dom + L_adv mechanism inside `LatentWorldModel` Stage 1, plus the Step 2 diagnostics and Step 3 sweep aggregator, following `docs/superpowers/specs/2026-05-24-phase0-latent-decompose-design.md`.

**Architecture:** Channel-split the ViT-S `(B, 384, 16, 16)` spatial feature map into `z_task (B, 288, 16, 16)` and `z_emb (B, 96, 16, 16)`. Reassemble for the existing `spatial_proj` (bit-identical baseline when no new loss). Add domain classifier on `z_emb` (regular CE) and adversarial classifier on `z_task` (gradient reversal + CE). Gate everything behind `cfg.latent_decompose.enabled` so baseline behaviour is preserved.

**Tech Stack:** PyTorch 2.x, PyTorch Lightning, Hydra/OmegaConf, einops, pytest 7.4, ruff.

**Repo conventions to follow:**
- `interactive_world_sim/algorithms/*` for model code
- `configurations/*` for Hydra yaml
- Use type hints; ruff config in `pyproject.toml` is authoritative
- Frequent commits — one logical change per commit, conventional commit prefix (`feat:` / `test:` / `chore:` / `fix:`)
- Always run `pytest <new_test>` after writing it
- Never `git push` (this plan does not push)

---

## Task 0: Pytest scaffold

**Why:** No `tests/` directory exists yet. Create the structure once so subsequent tasks just drop files in.

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/algorithms/__init__.py`
- Create: `tests/algorithms/latent_decompose/__init__.py`
- Create: `tests/integration/__init__.py`
- Modify: `pyproject.toml` — add pytest section

- [ ] **Step 1: Create empty `__init__.py` files**

```bash
mkdir -p tests/algorithms/latent_decompose tests/integration
: > tests/__init__.py
: > tests/algorithms/__init__.py
: > tests/algorithms/latent_decompose/__init__.py
: > tests/integration/__init__.py
```

- [ ] **Step 2: Create `tests/conftest.py`**

```python
"""Shared pytest fixtures for the IWS test suite."""
from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Make every test deterministic. Seeds are reset per-test."""
    seed = int(os.environ.get("IWS_TEST_SEED", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- [ ] **Step 3: Add pytest config to `pyproject.toml`**

Append the following section at the end of `pyproject.toml` (after the existing `[tool.ruff.lint]` block):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
    "integration: tests that instantiate the full LatentWorldModel (slow)",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]
```

- [ ] **Step 4: Smoke-run pytest**

Run: `pytest tests/ -q`
Expected: `no tests ran` exit code 5 (we have no tests yet — but the suite is discoverable).

- [ ] **Step 5: Commit**

```bash
git add tests/ pyproject.toml
git commit -m "chore: pytest scaffold for Phase 0 latent_decompose tests"
```

---

## Task 1: SplitEncoder module

**Files:**
- Create: `interactive_world_sim/algorithms/latent_decompose/__init__.py`
- Create: `interactive_world_sim/algorithms/latent_decompose/split_encoder.py`
- Create: `tests/algorithms/latent_decompose/test_split_encoder_identity.py`

- [ ] **Step 1: Create package `__init__.py`**

```python
# interactive_world_sim/algorithms/latent_decompose/__init__.py
"""Phase 0 latent decomposition: SplitEncoder + domain heads + adv losses.

See docs/superpowers/specs/2026-05-24-phase0-latent-decompose-design.md.
"""
```

- [ ] **Step 2: Write the failing identity test**

```python
# tests/algorithms/latent_decompose/test_split_encoder_identity.py
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
```

- [ ] **Step 3: Run the test, expect failure**

Run: `pytest tests/algorithms/latent_decompose/test_split_encoder_identity.py -v`
Expected: `ImportError: cannot import name 'SplitEncoder'` (file doesn't exist).

- [ ] **Step 4: Implement `split_encoder.py`**

```python
# interactive_world_sim/algorithms/latent_decompose/split_encoder.py
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
```

- [ ] **Step 5: Run the test, expect pass**

Run: `pytest tests/algorithms/latent_decompose/test_split_encoder_identity.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add interactive_world_sim/algorithms/latent_decompose/__init__.py \
        interactive_world_sim/algorithms/latent_decompose/split_encoder.py \
        tests/algorithms/latent_decompose/test_split_encoder_identity.py
git commit -m "feat(latent_decompose): SplitEncoder with concat identity invariant"
```

---

## Task 2: Domain heads (grad_reverse + PooledClassifier)

**Files:**
- Create: `interactive_world_sim/algorithms/latent_decompose/domain_heads.py`
- Create: `tests/algorithms/latent_decompose/test_grad_reverse.py`
- Create: `tests/algorithms/latent_decompose/test_pooled_classifier.py`

- [ ] **Step 1: Write `test_grad_reverse.py`**

```python
# tests/algorithms/latent_decompose/test_grad_reverse.py
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
```

- [ ] **Step 2: Write `test_pooled_classifier.py`**

```python
# tests/algorithms/latent_decompose/test_pooled_classifier.py
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
```

- [ ] **Step 3: Run both, expect import failures**

Run: `pytest tests/algorithms/latent_decompose/test_grad_reverse.py tests/algorithms/latent_decompose/test_pooled_classifier.py -v`
Expected: `ImportError`.

- [ ] **Step 4: Implement `domain_heads.py`**

```python
# interactive_world_sim/algorithms/latent_decompose/domain_heads.py
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
```

- [ ] **Step 5: Run tests, expect pass**

Run: `pytest tests/algorithms/latent_decompose/test_grad_reverse.py tests/algorithms/latent_decompose/test_pooled_classifier.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add interactive_world_sim/algorithms/latent_decompose/domain_heads.py \
        tests/algorithms/latent_decompose/test_grad_reverse.py \
        tests/algorithms/latent_decompose/test_pooled_classifier.py
git commit -m "feat(latent_decompose): grad_reverse + PooledClassifier"
```

---

## Task 3: align_losses + linear_ramp schedule

**Files:**
- Create: `interactive_world_sim/algorithms/latent_decompose/align_losses.py`
- Create: `tests/algorithms/latent_decompose/test_align_losses.py`
- Create: `tests/algorithms/latent_decompose/test_lambda_schedule.py`

- [ ] **Step 1: Write `test_lambda_schedule.py`**

```python
# tests/algorithms/latent_decompose/test_lambda_schedule.py
"""Linear-ramp schedule for lambda_adv."""
from __future__ import annotations

import pytest

from interactive_world_sim.algorithms.latent_decompose.align_losses import (
    linear_ramp,
)


@pytest.mark.parametrize(
    "step, expected",
    [
        (0,     0.0),
        (1999,  0.0),
        (2000,  0.0),
        (6000,  0.15),    # midpoint of [2000, 10000] @ [0.0, 0.3]
        (10000, 0.3),
        (50000, 0.3),
    ],
)
def test_linear_ramp_values(step, expected):
    out = linear_ramp(
        step,
        start_step=2000,
        end_step=10000,
        start_value=0.0,
        end_value=0.3,
    )
    assert abs(out - expected) < 1e-9
```

- [ ] **Step 2: Write `test_align_losses.py`**

```python
# tests/algorithms/latent_decompose/test_align_losses.py
"""compute_L_dom (CE) and compute_L_adv (GRL + CE)."""
from __future__ import annotations

import torch

from interactive_world_sim.algorithms.latent_decompose.align_losses import (
    compute_L_adv,
    compute_L_dom,
)
from interactive_world_sim.algorithms.latent_decompose.domain_heads import (
    PooledClassifier,
)


def test_L_dom_is_plain_cross_entropy():
    clf = PooledClassifier(d_in=96, hidden=32)
    z = torch.randn(8, 96, 16, 16)
    dl = torch.randint(0, 2, (8,))
    loss = compute_L_dom(z, dl, clf)
    # Reference: same MLP applied directly + F.cross_entropy.
    import torch.nn.functional as F
    ref = F.cross_entropy(clf(z), dl)
    torch.testing.assert_close(loss, ref)


def test_L_adv_flips_gradient_on_z_task():
    clf = PooledClassifier(d_in=288, hidden=32)
    z_grl = torch.randn(8, 288, 16, 16, requires_grad=True)
    z_ce  = z_grl.detach().clone().requires_grad_(True)
    dl = torch.randint(0, 2, (8,))

    L_adv = compute_L_adv(z_grl, dl, clf, lambda_adv=0.5)
    L_adv.backward()

    # Plain CE without GRL on the same input — gradient should be opposite sign.
    import torch.nn.functional as F
    plain = F.cross_entropy(clf(z_ce), dl)
    plain.backward()

    torch.testing.assert_close(z_grl.grad, -0.5 * z_ce.grad, rtol=1e-5, atol=1e-6)
```

- [ ] **Step 3: Run both, expect import failures**

Run: `pytest tests/algorithms/latent_decompose/test_lambda_schedule.py tests/algorithms/latent_decompose/test_align_losses.py -v`
Expected: `ImportError`.

- [ ] **Step 4: Implement `align_losses.py`**

```python
# interactive_world_sim/algorithms/latent_decompose/align_losses.py
"""Phase 0 Stage-1 alignment losses + lambda schedule.

L_dom: regular CE on the embodiment classifier — forces z_emb to carry
       embodiment info.
L_adv: gradient-reversal + CE on z_task — pushes the encoder away from
       making z_task domain-discriminable, while letting the classifier
       itself train normally.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .domain_heads import grad_reverse


def compute_L_dom(
    z_emb: torch.Tensor,
    domain_label: torch.Tensor,
    clf_emb: nn.Module,
) -> torch.Tensor:
    return F.cross_entropy(clf_emb(z_emb), domain_label)


def compute_L_adv(
    z_task: torch.Tensor,
    domain_label: torch.Tensor,
    clf_adv: nn.Module,
    lambda_adv: float,
) -> torch.Tensor:
    z_rev = grad_reverse(z_task, lambda_adv)
    return F.cross_entropy(clf_adv(z_rev), domain_label)


def linear_ramp(
    step: int,
    start_step: int,
    end_step: int,
    start_value: float,
    end_value: float,
) -> float:
    """Linear interpolation, clamped at the endpoints. Used by the
    lambda_adv schedule in the LatentWorldModel training loop."""
    step = int(step)
    if step <= start_step:
        return float(start_value)
    if step >= end_step:
        return float(end_value)
    t = (step - start_step) / max(end_step - start_step, 1)
    return float(start_value + t * (end_value - start_value))
```

- [ ] **Step 5: Run tests, expect pass**

Run: `pytest tests/algorithms/latent_decompose/test_lambda_schedule.py tests/algorithms/latent_decompose/test_align_losses.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add interactive_world_sim/algorithms/latent_decompose/align_losses.py \
        tests/algorithms/latent_decompose/test_lambda_schedule.py \
        tests/algorithms/latent_decompose/test_align_losses.py
git commit -m "feat(latent_decompose): align_losses + linear_ramp schedule"
```

---

## Task 4: Dataset `domain_label` field

**Files:**
- Modify: `interactive_world_sim/datasets/latent_dynamics/play_eef_dataset.py:608-628` (the `MixedPlayEEFDataset.__getitem__` block)
- Create: `tests/algorithms/latent_decompose/test_dataset_domain_label.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/algorithms/latent_decompose/test_dataset_domain_label.py
"""MixedPlayEEFDataset must emit a `domain_label` int tensor matching
`embodiment`: human=0, robot=1. The PyTorch default collate stacks
these into (B,) without a custom collate_fn."""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


@pytest.fixture
def mixed_dataset():
    """Tiny in-process MixedPlayEEFDataset. Skipped if data dirs missing."""
    from pathlib import Path

    from interactive_world_sim.datasets.latent_dynamics.play_eef_dataset import (
        MixedPlayEEFDataset,
    )

    robot_root = Path("/scr2/yusenluo/interactive_world_sim/play_robot_eef/play_robot_1_eef")
    human_root = Path("/scr2/yusenluo/interactive_world_sim/human_play_eef_data/play_human_eef_1")
    if not robot_root.exists() or not human_root.exists():
        pytest.skip("real EEF data not present on this host")

    cfg = OmegaConf.create({
        "robot": {
            "domain": "robot",
            "dataset_dirs": [str(robot_root)],
            "video_keys": ["observation.images.cam_high"],
            "obs_keys": ["camera_0_color"],
            "crops": [[195, 195, 256, 256]],
            "resolution": 128, "horizon": 4, "val_horizon": 4,
            "skip_frame": 1, "pad_before": 0, "pad_after": 0,
            "val_ratio": 0.1, "seed": 42, "goal_sample": "intermediate",
            "skip_idx": 1, "cache_root": "/tmp/iws_test_cache_robot",
        },
        "human": {
            "domain": "human",
            "dataset_dirs": [str(human_root)],
            "video_keys": ["observation.images.cam_high"],
            "obs_keys": ["camera_0_color"],
            "crops": [[195, 195, 256, 256]],
            "resolution": 128, "horizon": 4, "val_horizon": 4,
            "skip_frame": 1, "pad_before": 0, "pad_after": 0,
            "val_ratio": 0.333, "seed": 42, "goal_sample": "intermediate",
            "skip_idx": 1, "cache_root": "/tmp/iws_test_cache_human",
        },
        "sample_ratio": 0.5, "seed": 0,
    })
    return MixedPlayEEFDataset(cfg)


def test_per_item_emits_domain_label(mixed_dataset):
    item = mixed_dataset[0]
    assert "domain_label" in item, "MixedPlayEEFDataset must add domain_label"
    dl = item["domain_label"]
    assert isinstance(dl, torch.Tensor) and dl.dtype == torch.long
    assert dl.numel() == 1
    expected = 1 if item["embodiment"] == "robot" else 0
    assert int(dl) == expected


def test_collated_batch_has_domain_label(mixed_dataset):
    loader = DataLoader(mixed_dataset, batch_size=4, num_workers=0)
    batch = next(iter(loader))
    assert "domain_label" in batch
    assert batch["domain_label"].dtype == torch.long
    assert batch["domain_label"].shape == (4,)
    assert set(batch["domain_label"].tolist()).issubset({0, 1})
```

- [ ] **Step 2: Run test, expect fail**

Run: `pytest tests/algorithms/latent_decompose/test_dataset_domain_label.py -v`
Expected: `KeyError: 'domain_label'` (or the skip if data is missing — that's still a pass condition).

- [ ] **Step 3: Modify `MixedPlayEEFDataset.__getitem__`**

In `interactive_world_sim/datasets/latent_dynamics/play_eef_dataset.py`, locate the `__getitem__` method on `MixedPlayEEFDataset` (around line 608). The training branch already does:

```python
        if self._rng.random() < self.sample_ratio:
            item = self.robot[idx % len(self.robot)]
            item["embodiment"] = "robot"
        else:
            item = self.human[idx % len(self.human)]
            item["embodiment"] = "human"
        return item
```

Refactor into a single tail that sets both fields together (covers train + val branches). Replace the whole method body with:

```python
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_val:
            half = len(self) // 2
            if idx < half:
                rob_val = self._robot_val()
                item = rob_val[idx % len(rob_val)]
                emb = "robot"
            else:
                hum_val = self._human_val()
                item = hum_val[(idx - half) % len(hum_val)]
                emb = "human"
        else:
            if self._rng.random() < self.sample_ratio:
                item = self.robot[idx % len(self.robot)]
                emb = "robot"
            else:
                item = self.human[idx % len(self.human)]
                emb = "human"
        item["embodiment"] = emb
        item["domain_label"] = torch.tensor(
            1 if emb == "robot" else 0, dtype=torch.long,
        )
        return item
```

- [ ] **Step 4: Run test, expect pass (or skip if data missing)**

Run: `pytest tests/algorithms/latent_decompose/test_dataset_domain_label.py -v`
Expected: PASS, or `SKIPPED [data not present]`.

- [ ] **Step 5: Commit**

```bash
git add interactive_world_sim/datasets/latent_dynamics/play_eef_dataset.py \
        tests/algorithms/latent_decompose/test_dataset_domain_label.py
git commit -m "feat(dataset): MixedPlayEEFDataset emits domain_label int tensor"
```

---

## Task 5: Dataset `episode_subset` support (Step 3 prep)

**Why:** Step 3 sweep needs per-domain episode subsetting. Lightweight to add now and keeps the per-step changeset focused later.

**Files:**
- Modify: `interactive_world_sim/datasets/latent_dynamics/play_eef_dataset.py` — `PlayEEFDataset.__init__` episode-load loop and a sanity assert

- [ ] **Step 1: Write the test**

```python
# tests/algorithms/latent_decompose/test_dataset_episode_subset.py
"""PlayEEFDataset honours `episode_subset` (list[int] | None) to filter
the episodes that get loaded. None = all (current behaviour)."""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def base_cfg():
    root = Path("/scr2/yusenluo/interactive_world_sim/play_robot_eef/play_robot_1_eef")
    if not root.exists():
        pytest.skip("real EEF data not present on this host")
    return OmegaConf.create({
        "domain": "robot",
        "dataset_dirs": [str(root)],
        "video_keys": ["observation.images.cam_high"],
        "obs_keys": ["camera_0_color"],
        "crops": [[195, 195, 256, 256]],
        "resolution": 128, "horizon": 4, "val_horizon": 4,
        "skip_frame": 1, "pad_before": 0, "pad_after": 0,
        "val_ratio": 0.1, "seed": 42, "goal_sample": "intermediate",
        "skip_idx": 1, "cache_root": "/tmp/iws_test_cache_subset",
    })


def test_default_subset_loads_all(base_cfg):
    from interactive_world_sim.datasets.latent_dynamics.play_eef_dataset import (
        PlayEEFDataset,
    )
    ds = PlayEEFDataset(base_cfg)
    assert len(ds._episodes) > 0


def test_explicit_subset_filters(base_cfg):
    from interactive_world_sim.datasets.latent_dynamics.play_eef_dataset import (
        PlayEEFDataset,
    )
    cfg_full = OmegaConf.create(dict(base_cfg))
    full = PlayEEFDataset(cfg_full)
    full_idxs = [ep["episode_index"] for ep in full._episodes]
    assert len(full_idxs) >= 1
    target = [full_idxs[0]]

    cfg_sub = OmegaConf.create({**dict(base_cfg), "episode_subset": target})
    sub = PlayEEFDataset(cfg_sub)
    assert [ep["episode_index"] for ep in sub._episodes] == target


def test_empty_subset_loads_nothing(base_cfg):
    from interactive_world_sim.datasets.latent_dynamics.play_eef_dataset import (
        PlayEEFDataset,
    )
    cfg = OmegaConf.create({**dict(base_cfg), "episode_subset": []})
    ds = PlayEEFDataset(cfg)
    assert ds._episodes == []
```

- [ ] **Step 2: Run test, expect fail**

Run: `pytest tests/algorithms/latent_decompose/test_dataset_episode_subset.py -v`
Expected: `test_explicit_subset_filters` FAILS (subset is currently ignored).

- [ ] **Step 3: Modify `PlayEEFDataset.__init__`**

In `interactive_world_sim/datasets/latent_dynamics/play_eef_dataset.py`, locate the per-dataset episode-loop in `PlayEEFDataset.__init__` (around line 233-264 — the `for ds_dir in self.dataset_dirs:` loop).

Right BEFORE the loop, add:

```python
        # Optional per-dataset episode filter for Step 3 sweep.
        # None → use all episodes; [] → load nothing; list[int] → keep only those.
        subset = cfg.get("episode_subset", None)
        if subset is not None:
            subset = list(int(i) for i in subset)
            self._episode_subset: Optional[set[int]] = set(subset)
        else:
            self._episode_subset = None
```

INSIDE the loop, right after `ep_idx = rec["episode_index"]`, add a skip:

```python
                    if (
                        self._episode_subset is not None
                        and ep_idx not in self._episode_subset
                    ):
                        continue
```

- [ ] **Step 4: Run test, expect pass (or skip if data missing)**

Run: `pytest tests/algorithms/latent_decompose/test_dataset_episode_subset.py -v`
Expected: 3 PASS, or 3 SKIPPED.

- [ ] **Step 5: Commit**

```bash
git add interactive_world_sim/datasets/latent_dynamics/play_eef_dataset.py \
        tests/algorithms/latent_decompose/test_dataset_episode_subset.py
git commit -m "feat(dataset): PlayEEFDataset honours episode_subset"
```

---

## Task 6: Config additions

**Files:**
- Modify: `configurations/algorithm/latent_world_model.yaml` — add `latent_decompose:` section
- Modify: `configurations/dataset/play_mixed_eef.yaml` — add `episode_subset: null` placeholders

- [ ] **Step 1: Append to `configurations/algorithm/latent_world_model.yaml`**

Add this section at the end of the file (after the existing `metrics:` block):

```yaml

# Phase 0 latent decomposition (HUMAN_ROBOT_ALIGN_PLAN-2 / PHASE0_IWS_PLAN_v3).
# When enabled=false the model is bit-identical to the existing baseline.
# See docs/superpowers/specs/2026-05-24-phase0-latent-decompose-design.md.
latent_decompose:
  enabled: false
  d_task: 288
  d_emb: 96
  lambda_dom: 0.1
  lambda_adv_schedule:
    type: linear_ramp
    start_step: 2000
    end_step: 10000
    start_value: 0.0
    end_value: 0.3
  lr_classifiers: 3.0e-4
```

- [ ] **Step 2: Modify `configurations/dataset/play_mixed_eef.yaml`**

Inside the existing `robot:` block, add a single line `episode_subset: null` (e.g. right after the `cache_root:` line). Do the same inside the `human:` block. Final state:

```yaml
robot:
  ...
  cache_root: /scr2/yusenluo/interactive_world_sim/play_robot_eef/_video_cache
  episode_subset: null
human:
  ...
  cache_root: /scr2/yusenluo/interactive_world_sim/human_play_eef_data/_video_cache
  episode_subset: null
```

- [ ] **Step 3: Verify yaml parse + Hydra composition still works**

Run a Hydra config dry-run if the project has one; otherwise quick-check with python:

```bash
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('configurations/algorithm/latent_world_model.yaml')
assert cfg.latent_decompose.enabled is False
assert cfg.latent_decompose.d_task + cfg.latent_decompose.d_emb == 384
print('OK')
"
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('configurations/dataset/play_mixed_eef.yaml')
assert cfg.robot.episode_subset is None
assert cfg.human.episode_subset is None
print('OK')
"
```

Expected: both print `OK`.

- [ ] **Step 4: Commit**

```bash
git add configurations/algorithm/latent_world_model.yaml \
        configurations/dataset/play_mixed_eef.yaml
git commit -m "config: add latent_decompose section + dataset episode_subset"
```

---

## Task 7: `LatentWorldModel.__init__` + `_build_model` wiring

**Files:**
- Modify: `interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py`

- [ ] **Step 1: Read `cfg.latent_decompose` in `__init__`**

In `LatentWorldModel.__init__`, near where `self.use_dynamo_ssl` is computed (around line 76-92), add the analogous flag — but BEFORE `super().__init__(cfg)` so `_build_model` (called inside `super().__init__`) sees it:

```python
        # Phase 0 latent decomposition flag. Mirrors dynamo_ssl pattern.
        # Active only in Stage 1; Stages 2/3 ignore it.
        ld_cfg = cfg.get("latent_decompose", None)
        self.use_latent_decompose = bool(
            ld_cfg is not None
            and ld_cfg.get("enabled", False)
            and self.training_stage == 1
            and self.use_vit_encoder
        )
        if self.use_latent_decompose:
            assert (
                int(ld_cfg.d_task) + int(ld_cfg.d_emb)
                == int(cfg.dynamo_ssl.vit.embed_dim)
            ), "d_task + d_emb must equal ViT embed_dim"
```

- [ ] **Step 2: Construct SplitEncoder + classifiers in `_build_model`**

Inside `_build_model`, locate the ViT branch (the `if self.use_vit_encoder:` block, around line 153-174). Right AFTER `self.spatial_proj = _proj_cls(...)` is constructed (around line 174), add:

```python
            if self.use_latent_decompose:
                from interactive_world_sim.algorithms.latent_decompose.split_encoder import (
                    SplitEncoder,
                )
                from interactive_world_sim.algorithms.latent_decompose.domain_heads import (
                    PooledClassifier,
                )
                ld = self.cfg.latent_decompose
                self.split_encoder = SplitEncoder(
                    self.vit_encoder, int(ld.d_task), int(ld.d_emb),
                )
                self.clf_emb = PooledClassifier(int(ld.d_emb))
                self.clf_adv = PooledClassifier(int(ld.d_task))
```

- [ ] **Step 3: Sanity unit test (instantiation only)**

```python
# tests/algorithms/latent_decompose/test_lwm_construction.py
"""LatentWorldModel constructs cleanly when latent_decompose.enabled is true
or false. Does not run a training step."""
from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


@pytest.fixture
def base_alg_cfg():
    from pathlib import Path
    cfg_dir = Path(
        "/scr2/yusenluo/interactive_world_sim/configurations/algorithm"
    ).absolute()
    # Minimal compose: load latent_world_model.yaml in isolation; fill the
    # ${dataset.*} interpolations with literals so OmegaConf can resolve.
    with initialize_config_dir(version_base="1.3", config_dir=str(cfg_dir)):
        cfg = compose(config_name="latent_world_model")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.obs_keys = ["camera_0_color"]
    cfg.x_shape = [3, 128, 128]
    cfg.num_latent_channel = 4
    cfg.num_views = 1
    cfg.latent_resolution = 32
    cfg.dynamo_ssl.vit.img_size = 128
    cfg.delta = 0.01
    return cfg


def test_construct_disabled(base_alg_cfg):
    from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
        LatentWorldModel,
    )
    cfg = OmegaConf.create(OmegaConf.to_container(base_alg_cfg, resolve=True))
    cfg.latent_decompose.enabled = False
    m = LatentWorldModel(cfg)
    assert m.use_latent_decompose is False
    assert not hasattr(m, "split_encoder")


def test_construct_enabled(base_alg_cfg):
    from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
        LatentWorldModel,
    )
    cfg = OmegaConf.create(OmegaConf.to_container(base_alg_cfg, resolve=True))
    cfg.latent_decompose.enabled = True
    m = LatentWorldModel(cfg)
    assert m.use_latent_decompose is True
    assert hasattr(m, "split_encoder")
    assert hasattr(m, "clf_emb")
    assert hasattr(m, "clf_adv")
```

- [ ] **Step 4: Run test**

Run: `pytest tests/algorithms/latent_decompose/test_lwm_construction.py -v`
Expected: both PASS.

If the test cannot compose the cfg cleanly because of interpolation references that don't resolve in isolation, the test fixture in Step 3 needs the missing literals filled in. The error message will name the missing key — add it to `base_alg_cfg`.

- [ ] **Step 5: Commit**

```bash
git add interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py \
        tests/algorithms/latent_decompose/test_lwm_construction.py
git commit -m "feat(lwm): construct SplitEncoder + classifiers when latent_decompose.enabled"
```

---

## Task 8: `encoder_forward` — `return_split` kwarg

**Files:**
- Modify: `interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py:372-416` (the `encoder_forward` method)

- [ ] **Step 1: Refactor `encoder_forward` to optionally return per-view split tensors**

Replace the entire `encoder_forward` method with:

```python
    def encoder_forward(
        self,
        obs: torch.Tensor,
        return_split: bool = False,
    ):
        """Forward pass of the encoder.

        Args:
            obs: (B, C, H, W) where C = 3 * num_views.
            return_split: when True AND latent_decompose is active, also
                returns (z_task_list, z_emb_list) — lists of V tensors of
                shape (B, d_task, gh, gw) / (B, d_emb, gh, gw) in
                view-outer order. When False (default) the signature is
                unchanged: returns the normalised spatial latent z only.

        Returns:
            z: (B, C_latent, H_latent, W_latent)
            (z_task_list, z_emb_list): only if `return_split=True` and
                                       `self.use_latent_decompose` is True.
        """
        assert (
            len(obs.shape) == 4
        ), f"Expected obs to have shape (B, C, H, W) but got {obs.shape}"

        z_task_list: list[torch.Tensor] = []
        z_emb_list:  list[torch.Tensor] = []
        emit_split = bool(return_split and self.use_latent_decompose)

        if self.use_resnet_encoder:
            num_views = len(self.obs_keys)
            z_views = []
            for v in range(num_views):
                view_obs = obs[:, v * 3 : (v + 1) * 3]  # (B, 3, H, W)
                if self.use_vit_encoder:
                    if self.use_latent_decompose:
                        z_task, z_emb, cls_token = self.split_encoder(view_obs)
                        spatial_feat = (
                            self.split_encoder.concat(z_task, z_emb)
                        )
                        if emit_split:
                            z_task_list.append(z_task)
                            z_emb_list.append(z_emb)
                    else:
                        spatial_feat, cls_token = self.vit_encoder(view_obs)
                    if self.use_dynamo_ssl and self.detach_rec_from_encoder:
                        spatial = self.spatial_proj(
                            spatial_feat.detach(), cls_token.detach()
                        )
                    else:
                        spatial = self.spatial_proj(spatial_feat, cls_token)
                else:
                    resnet_feat = self.resnet_encoder(view_obs)
                    if self.use_dynamo_ssl and self.detach_rec_from_encoder:
                        spatial = self.spatial_proj(resnet_feat.detach())
                    else:
                        spatial = self.spatial_proj(resnet_feat)
                z_views.append(spatial)
            z = torch.cat(z_views, dim=1)
        else:
            z = self.encoder(obs)

        num_views = len(self.obs_keys)
        c_per_v = z.shape[1] // num_views
        for i in range(num_views):
            z_chunk = z[:, i * c_per_v : (i + 1) * c_per_v].clone()
            z[:, i * c_per_v : (i + 1) * c_per_v] = z_chunk / (
                torch.norm(z_chunk, dim=(1), keepdim=True) + 1e-8
            )

        if emit_split:
            return z, z_task_list, z_emb_list
        return z
```

- [ ] **Step 2: Verify Stage 2 / Stage 3 callers untouched**

Run: `grep -n "encoder_forward" interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py`
Expected: all existing call sites still call `self.encoder_forward(xs)` (no `return_split` kwarg). New code paths added later in Task 9 will pass `return_split=True`.

- [ ] **Step 3: Run construction test again to catch syntactic regressions**

Run: `pytest tests/algorithms/latent_decompose/test_lwm_construction.py -v`
Expected: both PASS.

- [ ] **Step 4: Commit**

```bash
git add interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py
git commit -m "feat(lwm): encoder_forward return_split kwarg (no behaviour change yet)"
```

---

## Task 9: `training_step` Stage 1 loss hook + `lambda_adv_now`

**Files:**
- Modify: `interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py` — Stage 1 branch of `training_step` (around line 792-957) and add a `_lambda_adv_now` method

- [ ] **Step 1: Add `_lambda_adv_now` helper to `LatentWorldModel`**

Place this method near the other helpers (e.g. just before `encoder_forward` at line 372). Body:

```python
    def _lambda_adv_now(self) -> float:
        """Current value of the gradient-reversal scale, per cfg schedule."""
        from interactive_world_sim.algorithms.latent_decompose.align_losses import (
            linear_ramp,
        )
        sched = self.cfg.latent_decompose.lambda_adv_schedule
        return linear_ramp(
            self.global_step,
            int(sched.start_step), int(sched.end_step),
            float(sched.start_value), float(sched.end_value),
        )
```

- [ ] **Step 2: Wire L_dom + L_adv into the Stage 1 non-SSL branch**

In `training_step`, find the Stage 1 branch (`if self.training_stage == 1:`). When `self.use_dynamo_ssl` is False (the `else:` at the very bottom of that branch, around line 952-957), replace:

```python
            else:
                self.log("training/rec_loss", rec_loss)
                output_dict = {
                    "loss": rec_loss,
                }
                return output_dict
```

with:

```python
            else:
                if self.use_latent_decompose:
                    total_loss = self._apply_decompose_losses(batch, rec_loss, xs)
                else:
                    total_loss = rec_loss
                self.log("training/rec_loss", rec_loss)
                self.log("training/loss", total_loss)
                return {"loss": total_loss}
```

Then add the helper method `_apply_decompose_losses` next to `_lambda_adv_now`:

```python
    def _apply_decompose_losses(
        self,
        batch: dict,
        rec_loss: torch.Tensor,
        xs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute and log L_dom + L_adv on a fresh encoder pass that
        also returns the per-view (z_task, z_emb). Adds them to rec_loss
        and returns the total. The encoder pass is identical to the one
        already done by encoder_forward inside the rec_loss path — we
        deliberately do not cache it across the rec_loss/decompose
        boundary to keep this hook decoupled and avoid stale state."""
        from interactive_world_sim.algorithms.latent_decompose.align_losses import (
            compute_L_adv,
            compute_L_dom,
        )

        T = self.cfg.n_frames
        V = len(self.obs_keys)
        dl = batch["domain_label"]                        # (B,) long

        _z, z_task_list, z_emb_list = self.encoder_forward(
            xs, return_split=True,
        )
        z_task = torch.cat(z_task_list, dim=0)            # (V*B*T, d_task, gh, gw)
        z_emb  = torch.cat(z_emb_list,  dim=0)            # (V*B*T, d_emb,  gh, gw)
        # Order: v slowest, b middle, t fastest — matches the cat above
        # because z_task_list[v] has shape (B*T, ...) in (b slow, t fast).
        dl_rep = dl.repeat_interleave(T).repeat(V).to(z_task.device)

        L_dom = compute_L_dom(z_emb, dl_rep, self.clf_emb)
        lam = self._lambda_adv_now()
        L_adv = compute_L_adv(z_task, dl_rep, self.clf_adv, lam)

        lambda_dom = float(self.cfg.latent_decompose.lambda_dom)
        total = rec_loss + lambda_dom * L_dom + L_adv

        self.log("training/L_dom", L_dom)
        self.log("training/L_adv", L_adv)
        self.log("training/lambda_adv", lam)
        return total
```

**Note on the extra encoder pass:** This doubles encoder forward cost on Stage 1, which is acceptable for Phase 0 MVP (one batch per step, ViT-S is small). If profiling later shows it dominates, refactor to thread `z_task_list / z_emb_list` through the existing rec_loss path. Documenting the trade-off here means the next iteration can choose deliberately rather than discovering the cost.

- [ ] **Step 3: Run construction + identity tests**

Run: `pytest tests/algorithms/latent_decompose/test_lwm_construction.py -v`
Expected: PASS (no behaviour change for the construction-only test).

- [ ] **Step 4: Commit**

```bash
git add interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py
git commit -m "feat(lwm): hook L_dom + L_adv into Stage 1 training_step"
```

---

## Task 10: `configure_optimizers` — classifier param group

**Files:**
- Modify: `interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py:280-369`

- [ ] **Step 1: Append classifier param group**

In `configure_optimizers`, inside the `if self.training_stage == 1:` branch, after the existing `param_groups = [...]` assignment and before the `if self.use_dynamo_ssl:` block, insert:

```python
            if self.use_latent_decompose:
                param_groups.append({
                    "params": list(self.clf_emb.parameters())
                              + list(self.clf_adv.parameters()),
                    "lr": float(self.cfg.latent_decompose.lr_classifiers),
                })
```

- [ ] **Step 2: Unit-test optimiser construction**

Add to `tests/algorithms/latent_decompose/test_lwm_construction.py`:

```python
def test_optimizer_has_classifier_group(base_alg_cfg):
    from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
        LatentWorldModel,
    )
    cfg = OmegaConf.create(OmegaConf.to_container(base_alg_cfg, resolve=True))
    cfg.latent_decompose.enabled = True
    m = LatentWorldModel(cfg)
    opt_dict = m.configure_optimizers()
    opt = opt_dict["optimizer"]
    # First two groups are decoder + encoder (existing); third is classifiers.
    assert len(opt.param_groups) >= 3
    clf_lr = float(cfg.latent_decompose.lr_classifiers)
    assert any(abs(g["lr"] - clf_lr) < 1e-9 for g in opt.param_groups), (
        f"No param group with lr={clf_lr} (lrs={[g['lr'] for g in opt.param_groups]})"
    )
```

- [ ] **Step 3: Run test**

Run: `pytest tests/algorithms/latent_decompose/test_lwm_construction.py::test_optimizer_has_classifier_group -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py \
        tests/algorithms/latent_decompose/test_lwm_construction.py
git commit -m "feat(lwm): classifier param group when latent_decompose enabled"
```

---

## Task 11: Integration smoke test (mode=smoke)

**Why:** Catches wiring bugs end-to-end without needing real video data.

**Files:**
- Create: `tests/integration/test_phase0_smoke.py`

- [ ] **Step 1: Write the smoke test using a fake in-memory dataset**

```python
# tests/integration/test_phase0_smoke.py
"""End-to-end smoke for Phase 0 Stage 1 with latent_decompose enabled.

Uses a tiny in-process fake dataset (random tensors) so the test does
not depend on the real video data being present. Verifies that:
  (a) all loss terms are finite,
  (b) rec_loss, L_dom, L_adv are all logged,
  (c) encoder + decoder + classifier param groups all see non-zero grads.

Also includes the G1 "identity under zero loss" sub-test.
"""
from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


@pytest.fixture
def alg_cfg():
    from pathlib import Path
    cfg_dir = Path(
        "/scr2/yusenluo/interactive_world_sim/configurations/algorithm"
    ).absolute()
    with initialize_config_dir(version_base="1.3", config_dir=str(cfg_dir)):
        cfg = compose(config_name="latent_world_model")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    # Fill interpolations with literals (tiny model for speed)
    cfg.obs_keys = ["camera_0_color"]
    cfg.x_shape = [3, 64, 64]
    cfg.num_latent_channel = 4
    cfg.num_views = 1
    cfg.latent_resolution = 16
    cfg.n_frames = 2
    cfg.dynamo_ssl.vit.img_size = 64
    cfg.dynamo_ssl.vit.patch_size = 8
    cfg.dynamo_ssl.vit.depth = 2
    cfg.dynamo_ssl.enabled = False
    cfg.delta = 0.01
    return cfg


def _fake_batch(B=2, T=2, V=1, H=64, W=64, A=8):
    """Build a single batch matching MixedPlayEEFDataset's schema."""
    return {
        "obs": {"camera_0_color": torch.rand(B, T, 3 * V, H, W)},
        "action":         torch.randn(B, T, A),
        "domain_label":   torch.randint(0, 2, (B,), dtype=torch.long),
        "is_early_stop":  torch.zeros(B, 1, dtype=torch.bool),
        "rel_stop_idx":   torch.full((B, 1), T - 1, dtype=torch.long),
    }


def _make_lwm(cfg, enabled: bool):
    from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
        LatentWorldModel,
    )
    from interactive_world_sim.utils.normalizer import LinearNormalizer

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.latent_decompose.enabled = enabled
    m = LatentWorldModel(cfg)
    # Identity normalizer — tests don't care about action scaling.
    norm = LinearNormalizer()
    # The model's set_normalizer signature loads state; build an identity one.
    # Fill .params_dict with identity for "action" and each image key.
    from interactive_world_sim.utils.normalizer import (
        get_identity_normalizer_from_stat, get_image_range_normalizer,
    )
    stat = {
        "min": -torch.ones(8), "max": torch.ones(8),
        "mean": torch.zeros(8), "std": torch.ones(8),
    }
    norm["action"] = get_identity_normalizer_from_stat(stat)
    for k in cfg.obs_keys:
        norm[k] = get_image_range_normalizer()
    m.set_normalizer(norm)
    return m


@pytest.mark.integration
def test_smoke_one_training_step(alg_cfg):
    """Mode=smoke: 1 training_step, assert (a) finite loss, (b) all loss
    terms logged, (c) encoder + decoder + classifier params have grads."""
    m = _make_lwm(alg_cfg, enabled=True)
    m.train()
    # Bypass Lightning by calling training_step directly + manual backward.
    batch = _fake_batch(B=2, T=alg_cfg.n_frames, H=64, W=64)
    out = m.training_step(batch, batch_idx=0)
    loss = out["loss"]
    assert torch.isfinite(loss), f"non-finite loss: {loss}"

    loss.backward()
    # Encoder grad
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.vit_encoder.parameters()
    ), "encoder got no grad"
    # Decoder grad
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.decoder.parameters()
    ), "decoder got no grad"
    # Classifier grad
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.clf_emb.parameters()
    ), "clf_emb got no grad"
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.clf_adv.parameters()
    ), "clf_adv got no grad"
```

- [ ] **Step 2: Run the test, expect any wiring bug to surface**

Run: `pytest tests/integration/test_phase0_smoke.py::test_smoke_one_training_step -v`
Expected: PASS. If it fails, the message will localise the bug (often: a key missing in the cfg fixture, or a normalizer key mismatch). Fix inline.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_phase0_smoke.py
git commit -m "test(integration): Phase 0 Stage 1 smoke test"
```

---

## Task 12: Integration identity test (mode=identity_under_zero_loss, G1)

**Why:** The G1 invariant from the spec — `enabled=true` with zero loss weights must produce a loss trajectory bit-identical to `enabled=false`.

**Files:**
- Modify: `tests/integration/test_phase0_smoke.py`

- [ ] **Step 1: Add the identity test**

Append to `tests/integration/test_phase0_smoke.py`:

```python
@pytest.mark.integration
def test_identity_under_zero_loss(alg_cfg):
    """G1: enabled=true with lambda_dom=0 and lambda_adv.end=0 must produce
    a rec_loss trace bit-identical (rtol=1e-5) to enabled=false."""
    torch.manual_seed(0)
    cfg_off = OmegaConf.create(OmegaConf.to_container(alg_cfg, resolve=True))
    cfg_off.latent_decompose.enabled = False
    m_off = _make_lwm(cfg_off, enabled=False)
    m_off.train()

    torch.manual_seed(0)
    cfg_on = OmegaConf.create(OmegaConf.to_container(alg_cfg, resolve=True))
    cfg_on.latent_decompose.enabled = True
    cfg_on.latent_decompose.lambda_dom = 0.0
    cfg_on.latent_decompose.lambda_adv_schedule.start_value = 0.0
    cfg_on.latent_decompose.lambda_adv_schedule.end_value = 0.0
    m_on = _make_lwm(cfg_on, enabled=True)
    m_on.train()

    # Copy encoder/decoder weights from m_off to m_on so the only delta is
    # the (unused, zero-weighted) classifier heads.
    m_on.vit_encoder.load_state_dict(m_off.vit_encoder.state_dict())
    m_on.spatial_proj.load_state_dict(m_off.spatial_proj.state_dict())
    m_on.decoder.load_state_dict(m_off.decoder.state_dict())

    torch.manual_seed(0)
    batch = _fake_batch(B=2, T=alg_cfg.n_frames, H=64, W=64)
    torch.manual_seed(0)
    out_off = m_off.training_step(batch, batch_idx=0)
    torch.manual_seed(0)
    out_on  = m_on.training_step(batch, batch_idx=0)

    # rec_loss is logged but not returned; the returned "loss" equals rec_loss
    # for m_off and equals rec_loss + 0 * L_dom + 0 * L_adv = rec_loss for m_on.
    torch.testing.assert_close(
        out_off["loss"], out_on["loss"], rtol=1e-5, atol=1e-6,
        msg="G1: zero-weighted decompose loss broke bit-identical baseline",
    )
```

- [ ] **Step 2: Run it**

Run: `pytest tests/integration/test_phase0_smoke.py::test_identity_under_zero_loss -v`
Expected: PASS.

If this fails with non-trivial divergence, the culprit is almost always:
- a non-deterministic op in the ViT (dropout, etc.) — set `cfg.dynamo_ssl.vit.drop_rate = 0.0` in the fixture if not already
- the extra encoder forward pass in `_apply_decompose_losses` perturbing state (e.g. RNG advancement in dropout, batchnorm momentum). Solution: short-circuit when `lambda_dom == 0 and end_value == 0` in `_apply_decompose_losses` (return `rec_loss` immediately without the second pass). Add this short-circuit if needed.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_phase0_smoke.py
git commit -m "test(integration): G1 identity-under-zero-loss invariant"
```

---

## Task 13: `phase0_step1.yaml` experiment config

**Files:**
- Create: `configurations/experiment/phase0_step1.yaml`

- [ ] **Step 1: Inspect the existing experiment config style**

Run: `ls configurations/experiment/ 2>/dev/null && head -30 configurations/experiment/*.yaml 2>/dev/null | head -80`

If no `experiment/` directory exists yet, look at how runs are currently launched (e.g. via `exp_latent_dyn.py`). The minimal viable approach is to write an override yaml that the user passes via `+experiment=phase0_step1`. If the project doesn't use Hydra `experiment` groups, instead provide a CLI override string in the README and skip the file.

- [ ] **Step 2: Write `configurations/experiment/phase0_step1.yaml`** (only if the directory exists or makes sense per Step 1)

```yaml
# @package _global_
#
# Phase 0 v3 Step 1: Stage 1 training with SplitEncoder + L_dom + L_adv.
# See docs/superpowers/specs/2026-05-24-phase0-latent-decompose-design.md.

defaults:
  - override /algorithm: latent_world_model
  - override /dataset: play_mixed_eef

algorithm:
  training_stage: 1
  latent_decompose:
    enabled: true

dataset:
  sample_ratio: 0.5      # balanced 50/50 robot/human per batch
```

- [ ] **Step 3: Verify config composes**

```bash
python -c "
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path

with initialize_config_dir(version_base='1.3',
    config_dir=str(Path('configurations').absolute())):
    cfg = compose(config_name='config', overrides=['+experiment=phase0_step1'])
print('latent_decompose.enabled =', cfg.algorithm.latent_decompose.enabled)
print('dataset.sample_ratio =', cfg.dataset.sample_ratio)
"
```

Expected: `latent_decompose.enabled = True`, `dataset.sample_ratio = 0.5`.

If the project does not have a `config` top-level Hydra entry, replace `config_name='config'` with whatever the actual entry name is (look in `interactive_world_sim/experiments/exp_latent_dyn.py` for `@hydra.main(config_name=...)`).

- [ ] **Step 4: Commit**

```bash
git add configurations/experiment/phase0_step1.yaml
git commit -m "config: Phase 0 step 1 experiment override"
```

---

## Task 14: `linear_probe.py` Step 2 diagnostic

**Files:**
- Create: `interactive_world_sim/algorithms/latent_decompose/diagnostics/__init__.py`
- Create: `interactive_world_sim/algorithms/latent_decompose/diagnostics/linear_probe.py`
- Create: `tests/algorithms/latent_decompose/test_linear_probe_unit.py`

- [ ] **Step 1: Create the package `__init__.py`**

```python
# interactive_world_sim/algorithms/latent_decompose/diagnostics/__init__.py
"""Phase 0 Step 2 diagnostics: linear probe + correlation heatmap + Step 3 agg."""
```

- [ ] **Step 2: Write a unit test for the probe's pure-MLP helper**

```python
# tests/algorithms/latent_decompose/test_linear_probe_unit.py
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
```

- [ ] **Step 3: Run, expect import failure**

Run: `pytest tests/algorithms/latent_decompose/test_linear_probe_unit.py -v`
Expected: `ImportError`.

- [ ] **Step 4: Implement `linear_probe.py`**

```python
# interactive_world_sim/algorithms/latent_decompose/diagnostics/linear_probe.py
"""Linear probe diagnostic — Phase 0 Step 2.

Pass thresholds (PHASE0_IWS_PLAN_v3.md §2.1):
    --target z_task → val accuracy ∈ [50%, 70%]
    --target z_emb  → val accuracy ≥ 95%
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


def train_probe_mlp(
    feats: torch.Tensor,
    labels: torch.Tensor,
    hidden: int = 128,
    epochs: int = 10,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    device: torch.device | None = None,
) -> float:
    """Train an MLP probe on (feats, labels). Return val accuracy."""
    device = device or torch.device("cpu")
    feats = feats.to(device)
    labels = labels.to(device).long()
    n = feats.shape[0]
    perm = torch.randperm(n, device=device)
    n_val = int(round(n * val_ratio))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    d_in = feats.shape[1]
    model = nn.Sequential(
        nn.Linear(d_in, hidden), nn.GELU(),
        nn.Linear(hidden, 2),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        logits = model(feats[train_idx])
        loss = F.cross_entropy(logits, labels[train_idx])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(feats[val_idx]).argmax(-1)
        acc = (pred == labels[val_idx]).float().mean().item()
    return acc


def _load_lwm(ckpt_path: Path, device: torch.device):
    """Load a frozen LatentWorldModel from a Lightning ckpt directory."""
    from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
        LatentWorldModel,
    )
    run_dir = ckpt_path.parent.parent       # ckpt/<...>.ckpt → run_dir
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    m = LatentWorldModel.load_from_checkpoint(
        str(ckpt_path), cfg=cfg.algorithm, map_location=device, weights_only=False,
    )
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, cfg


def _collect_features(
    m, cfg, target: Literal["z_task", "z_emb"], device: torch.device,
):
    """Run val loader, collect (target_pooled, domain_label) pairs."""
    import hydra
    val_dataset = hydra.utils.instantiate(cfg.dataset).get_validation_dataset()
    from torch.utils.data import DataLoader
    loader = DataLoader(val_dataset, batch_size=8, num_workers=0)

    feats, labels = [], []
    obs_key = cfg.dataset.obs_keys[0]
    for batch in loader:
        if "domain_label" not in batch:
            continue
        x = batch["obs"][obs_key].to(device)              # (B, T, 3, H, W)
        B, T = x.shape[:2]
        x = x.reshape(B * T, 3, x.shape[-2], x.shape[-1])
        with torch.no_grad():
            _, z_task_list, z_emb_list = m.encoder_forward(x, return_split=True)
        z_list = z_task_list if target == "z_task" else z_emb_list
        z = torch.cat(z_list, dim=0)                       # (V*B*T, D, gh, gw)
        z_pooled = z.mean(dim=(2, 3))                      # (V*B*T, D)
        feats.append(z_pooled.cpu())
        labels.append(
            batch["domain_label"].repeat_interleave(T).repeat(len(z_list))
        )
    return torch.cat(feats), torch.cat(labels)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--target", choices=("z_task", "z_emb"), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    m, cfg = _load_lwm(args.ckpt, device)
    feats, labels = _collect_features(m, cfg, args.target, device)
    print(f"[probe] collected {feats.shape[0]} samples, dim={feats.shape[1]}")
    accs = []
    for s in range(args.seeds):
        torch.manual_seed(s); np.random.seed(s)
        acc = train_probe_mlp(feats, labels, device=device)
        accs.append(acc)
        print(f"  seed={s}: acc={acc:.4f}")
    mean = float(np.mean(accs)); std = float(np.std(accs))
    print(f"[probe] target={args.target}: {mean:.4f} ± {std:.4f}")

    if args.target == "z_task":
        ok = 0.50 <= mean <= 0.70
    else:
        ok = mean >= 0.95
    print(f"[probe] {'PASS' if ok else 'FAIL'} per Phase 0 §2.1")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run unit tests**

Run: `pytest tests/algorithms/latent_decompose/test_linear_probe_unit.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add interactive_world_sim/algorithms/latent_decompose/diagnostics/__init__.py \
        interactive_world_sim/algorithms/latent_decompose/diagnostics/linear_probe.py \
        tests/algorithms/latent_decompose/test_linear_probe_unit.py
git commit -m "feat(diagnostics): Step 2 linear probe"
```

---

## Task 15: `correlation_heatmap.py` Step 2 diagnostic

**Files:**
- Create: `interactive_world_sim/algorithms/latent_decompose/diagnostics/correlation_heatmap.py`
- Create: `tests/algorithms/latent_decompose/test_correlation_heatmap_unit.py`

- [ ] **Step 1: Write the unit test**

```python
# tests/algorithms/latent_decompose/test_correlation_heatmap_unit.py
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
```

- [ ] **Step 2: Run it, expect import failure**

Run: `pytest tests/algorithms/latent_decompose/test_correlation_heatmap_unit.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `correlation_heatmap.py`**

```python
# interactive_world_sim/algorithms/latent_decompose/diagnostics/correlation_heatmap.py
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
```

- [ ] **Step 4: Run unit tests**

Run: `pytest tests/algorithms/latent_decompose/test_correlation_heatmap_unit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add interactive_world_sim/algorithms/latent_decompose/diagnostics/correlation_heatmap.py \
        tests/algorithms/latent_decompose/test_correlation_heatmap_unit.py
git commit -m "feat(diagnostics): Step 2 correlation heatmap"
```

---

## Task 16: Step 3 sweep aggregator + README

**Files:**
- Create: `interactive_world_sim/algorithms/latent_decompose/diagnostics/aggregate_step3.py`
- Create: `interactive_world_sim/algorithms/latent_decompose/diagnostics/README.md`
- Create: `tests/algorithms/latent_decompose/test_aggregate_step3.py`

- [ ] **Step 1: Write the test (pure-function aggregator)**

```python
# tests/algorithms/latent_decompose/test_aggregate_step3.py
"""Step 3 metric aggregation + Go/No-Go gate."""
from __future__ import annotations

import pytest

from interactive_world_sim.algorithms.latent_decompose.diagnostics.aggregate_step3 import (
    Decision,
    decide,
    summarise,
)


def test_summarise_mean_std():
    rows = [
        {"config": "1.5R_0H", "seed": 0, "fvd": 100.0},
        {"config": "1.5R_0H", "seed": 1, "fvd": 110.0},
        {"config": "1.5R_0H", "seed": 2, "fvd": 120.0},
    ]
    s = summarise(rows, metric="fvd")
    assert s["1.5R_0H"]["mean"] == pytest.approx(110.0)
    assert s["1.5R_0H"]["std"]  == pytest.approx(8.16496580927726)


def test_decide_paper_strong():
    s = {"1.5R_0H": {"mean": 100.0}, "0.5R_1.0H": {"mean": 110.0},
         "0R_1.0H": {"mean": 200.0}}
    d = decide(s, metric_is_lower_better=True)
    assert d.gap == pytest.approx(0.10)
    assert d.outcome == Decision.STRONG


def test_decide_paper_weak():
    s = {"1.5R_0H": {"mean": 100.0}, "0.5R_1.0H": {"mean": 130.0},
         "0R_1.0H": {"mean": 200.0}}
    d = decide(s, metric_is_lower_better=True)
    assert d.outcome == Decision.WEAK


def test_decide_diagnose():
    s = {"1.5R_0H": {"mean": 100.0}, "0.5R_1.0H": {"mean": 200.0},
         "0R_1.0H": {"mean": 200.0}}
    d = decide(s, metric_is_lower_better=True)
    assert d.outcome == Decision.DIAGNOSE


def test_decide_aborts_when_sanity_fails():
    """0R_1.0H must be strictly worse than 1.5R_0H."""
    s = {"1.5R_0H": {"mean": 100.0}, "0.5R_1.0H": {"mean": 105.0},
         "0R_1.0H": {"mean":  90.0}}                  # human-only BETTER → abort
    d = decide(s, metric_is_lower_better=True)
    assert d.outcome == Decision.ABORT
```

- [ ] **Step 2: Run, expect import failure**

Run: `pytest tests/algorithms/latent_decompose/test_aggregate_step3.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `aggregate_step3.py`**

```python
# interactive_world_sim/algorithms/latent_decompose/diagnostics/aggregate_step3.py
"""Step 3 fungibility-test aggregator + Go/No-Go decision.

Per PHASE0_IWS_PLAN_v3.md §3.4:
  gap = (M_{0.5R_1.0H} - M_{1.5R_0H}) / M_{1.5R_0H}  (lower-better metric)
  gap < 0.20   → STRONG, scale up
  gap < 0.50   → WEAK,   consider Phase 4
  gap >= 0.50  → DIAGNOSE
Required sanity: 0R_1.0H must be worse than 1.5R_0H, else ABORT.
"""
from __future__ import annotations

import argparse
import enum
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Decision(str, enum.Enum):
    STRONG    = "STRONG"
    WEAK      = "WEAK"
    DIAGNOSE  = "DIAGNOSE"
    ABORT     = "ABORT"


@dataclass
class DecisionResult:
    outcome: Decision
    gap: float
    primary_metric: str
    reason: str


def summarise(
    rows: list[dict[str, Any]],
    metric: str,
) -> dict[str, dict[str, float]]:
    """Group rows by `config`, compute mean ± std of `metric`."""
    by_cfg: dict[str, list[float]] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(float(r[metric]))
    out = {}
    for c, vs in by_cfg.items():
        mean = statistics.fmean(vs)
        std  = statistics.pstdev(vs) if len(vs) > 1 else 0.0
        out[c] = {"mean": mean, "std": std, "n": len(vs)}
    return out


def decide(
    summary: dict[str, dict[str, float]],
    metric_is_lower_better: bool = True,
) -> DecisionResult:
    need = ("1.5R_0H", "0.5R_1.0H", "0R_1.0H")
    missing = [k for k in need if k not in summary]
    if missing:
        return DecisionResult(
            Decision.ABORT, float("nan"), "n/a",
            f"missing required configs: {missing}",
        )
    M_ref = summary["1.5R_0H"]["mean"]
    M_mix = summary["0.5R_1.0H"]["mean"]
    M_hum = summary["0R_1.0H"]["mean"]

    # Sanity: human-only must be worse than robot-only on the same total budget.
    sanity_ok = (M_hum > M_ref) if metric_is_lower_better else (M_hum < M_ref)
    if not sanity_ok:
        return DecisionResult(
            Decision.ABORT, float("nan"), "primary",
            f"sanity failed: 0R_1.0H not worse than 1.5R_0H "
            f"(M_hum={M_hum:.4f}, M_ref={M_ref:.4f})",
        )

    gap = (M_mix - M_ref) / M_ref if metric_is_lower_better else (M_ref - M_mix) / M_ref
    if gap < 0.20:
        return DecisionResult(Decision.STRONG, gap, "primary",
            "gap < 20% — paper-strong, scale up")
    if gap < 0.50:
        return DecisionResult(Decision.WEAK, gap, "primary",
            "20% ≤ gap < 50% — consider Phase 4 anchor")
    return DecisionResult(Decision.DIAGNOSE, gap, "primary",
        "gap ≥ 50% — diagnose before iterating")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rows-jsonl", type=Path, required=True,
        help="JSONL where each line is {config, seed, fvd, psnr, ...}.",
    )
    ap.add_argument("--metric", default="fvd")
    ap.add_argument(
        "--higher-better", action="store_true",
        help="Set if your primary metric is higher-better (e.g. PSNR).",
    )
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in args.rows_jsonl.read_text().splitlines() if line.strip()]
    summary = summarise(rows, args.metric)
    print(f"[step3] metric={args.metric}, configs:")
    for c, s in sorted(summary.items()):
        print(f"  {c:>12}  {s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")
    d = decide(summary, metric_is_lower_better=not args.higher_better)
    print(f"[step3] gap = {d.gap:.4f}")
    print(f"[step3] {d.outcome.value} — {d.reason}")
    return 0 if d.outcome in (Decision.STRONG, Decision.WEAK) else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write `diagnostics/README.md` with the iteration table**

```markdown
# Phase 0 Step 2 Diagnostics

Pure-function scripts for inspecting a trained `latent_decompose` Stage 1
checkpoint. See `docs/superpowers/specs/2026-05-24-phase0-latent-decompose-design.md`
and `PHASE0_IWS_PLAN_v3.md` §2.

## linear_probe.py

```
python -m interactive_world_sim.algorithms.latent_decompose.diagnostics.linear_probe \
    --ckpt <run>/checkpoints/last.ckpt --target z_task
python -m interactive_world_sim.algorithms.latent_decompose.diagnostics.linear_probe \
    --ckpt <run>/checkpoints/last.ckpt --target z_emb
```

Pass: `--target z_task` acc ∈ [50%, 70%]; `--target z_emb` acc ≥ 95%.

## correlation_heatmap.py

```
python -m interactive_world_sim.algorithms.latent_decompose.diagnostics.correlation_heatmap \
    --ckpt <run>/checkpoints/last.ckpt --save outputs/<run>/corr.png
```

Pass: off-diagonal (task×emb) block max < 0.20.

## Iteration policy (PHASE0_IWS_PLAN_v3.md §2.4 — hard cap 3 retrains)

| Symptom                                   | First action                                 | If still failing                      |
|-------------------------------------------|----------------------------------------------|---------------------------------------|
| `z_task` probe > 75% (disentangle failed) | 2× `lambda_adv_schedule.end_value` (0.6, 1.0)| switch L_adv → CLUB (out of MVP)      |
| `z_task` probe < 45% (over-aligned)       | 0.5× `lambda_adv_schedule.end_value`         | also halve `lambda_dom`               |
| `z_emb` probe < 90%                       | 2× `lambda_dom`                              | sanity-check balanced batch composition|
| corr off-diag > 0.3 but probes pass       | retry with a different seed (likely PCA noise)| skip retraining                       |
| `L_adv` divergent (NaN, exploding)        | drop `lr_classifiers` to 1e-4                | switch to CLUB                        |

## aggregate_step3.py (Step 3 Go/No-Go)

```
python -m interactive_world_sim.algorithms.latent_decompose.diagnostics.aggregate_step3 \
    --rows-jsonl outputs/phase0_step3/rows.jsonl --metric fvd
```

Pass per §3.4: `gap(0.5R_1.0H, 1.5R_0H) < 20%`. Required sanity: `0R_1.0H`
strictly worse than `1.5R_0H`, otherwise the eval is non-discriminative and
the script aborts.
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/algorithms/latent_decompose/test_aggregate_step3.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add interactive_world_sim/algorithms/latent_decompose/diagnostics/aggregate_step3.py \
        interactive_world_sim/algorithms/latent_decompose/diagnostics/README.md \
        tests/algorithms/latent_decompose/test_aggregate_step3.py
git commit -m "feat(diagnostics): Step 3 aggregator + iteration README"
```

---

## Task 17: Full-suite check + summary

**Files:** none

- [ ] **Step 1: Run the whole test suite**

Run: `pytest tests/ -v -m "not integration"`
Expected: all unit tests PASS.

- [ ] **Step 2: Run the integration suite (will need real ckpt or skip on missing data)**

Run: `pytest tests/integration/ -v -m integration`
Expected: smoke + identity tests PASS (they use fake data, no skip needed).

- [ ] **Step 3: Verify git log is clean and linear**

Run: `git log --oneline phantom_dynamo..HEAD` (or `git log --oneline -20`)
Expected: one focused commit per Task 0-16, conventional prefixes throughout.

- [ ] **Step 4 (DO NOT EXECUTE — human-only): Hand off**

This plan stops at code + tests. The actual Stage 1 training run, Step 2
diagnostics on a real ckpt, and Step 3 sweep launch are operator actions
done by the user after Task 17 completes — they involve real GPU hours
and decisions to be made in the moment per the iteration table.

---

## Spec coverage check (for the implementing engineer)

| Spec § | Covered by |
|---|---|
| 2 File tree                              | Tasks 1, 2, 3, 14, 15, 16 (new files); Tasks 4, 5, 6, 7, 8, 9, 10 (modify) |
| 3.1 algorithm.latent_decompose cfg       | Task 6 |
| 3.2 dataset episode_subset cfg           | Tasks 5, 6 |
| 3.3 experiment phase0_step1.yaml         | Task 13 |
| 4.1 SplitEncoder                         | Task 1 |
| 4.2 domain_heads                         | Task 2 |
| 4.3 align_losses                         | Task 3 |
| 4.4 LatentWorldModel hooks               | Tasks 7, 8, 9, 10 |
| 4.5 MixedPlayEEFDataset domain_label     | Task 4 |
| 5 Data flow                              | Tasks 8, 9 (encoder_forward + training_step) |
| 6.1 linear_probe                         | Task 14 |
| 6.2 correlation_heatmap                  | Task 15 |
| 6.3 λ iteration table                    | Task 16 README |
| 7 Step 3 sweep + aggregator              | Task 16 (code); ratios deferred per spec D6 |
| 8 Sanity gates G1-G2                     | Tasks 1, 12 |
| 8 Sanity gates G3-G4                     | Operator-run, see "Step 4" handoff |
| 8 Sanity gates G5-G7                     | Tasks 14, 15 |
| 8 Sanity gates G8-G9                     | Task 16 |
| 9 Testing plan                           | Every task's TDD steps |
| 10 Risks                                 | Tasks 9 ("Note on the extra encoder pass"), 12 (G1 short-circuit fallback) |
| 11 Out-of-scope                          | Honoured — nothing in this plan touches Phase 4 / CLUB / L_pair |
| 12 Delivery sequence                     | Same Task order |
