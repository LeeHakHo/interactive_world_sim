# mask_subtract Residual Decomposition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `mask_subtract` stage-1 latent-decomposition method to the
from-scratch latent world model: an agent-mask-grounded embodiment readout
(`z_emb`, anchored + agent-reconstruction supervised, **detached** from the
encoder-decoder), with `z_scene` formed by projecting the anchored embodiment
direction out of the full latent — then validate that `z_scene` is less
domain-separable than the full latent without collapsing.

**Architecture:** Mirror the existing `emb_film` integration (a `method` value of
`latent_decompose`, additive, baseline-identical when disabled). The embodiment
path reads `stop_grad(F)`; the encoder-decoder is shaped only by reconstruction.
The decoder is the **baseline** decoder (z_scene and z_full are both
`num_latent_channel` channels — no cond-width change, unlike emb_film).

**Tech Stack:** PyTorch, PyTorch Lightning, Hydra/OmegaConf, pytest. Reference
spec: `docs/superpowers/specs/2026-05-28-mask-subtract-residual-decompose-design.md`.

---

## Orientation (read before starting)

Key files and the exact hooks (verified 2026-05-28; line numbers may drift — anchor
on the surrounding method names):

- `interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py` (1735 lines):
  - `__init__` decompose flags: ~L94–L130 (`self.decompose_method`, `self.use_emb_film`,
    `self.use_latent_decompose`, `self.use_dual_head`).
  - `_build_model`: ~L187–L290 (decoder cond width; per-method module construction;
    emb_film builds `EmbHead` + `DomainProbe` at L279–L290).
  - `configure_optimizers`: ~L396–L537 (param groups; emb_film adds emb_head to
    group 1 and probes as aux group ≥2; per-group warmup at L464–L483).
  - `configure_gradient_clipping`: ~L539–L562 (emb_film clips only groups 0,1).
  - `encoder_forward`: ~L868–L962 (per-view ViT → spatial_proj → per-view L2-norm →
    `z`; emb_film appends broadcast code at L953–L958).
  - `_emb_film_probe_loss`: ~L710–L748 (detached domain probes pattern to copy).
  - `training_step` stage-1 branch: ~L1380–L1539 (rec_loss computation L1393–L1459;
    emb_film branch L1520–L1533).
- `interactive_world_sim/algorithms/latent_decompose/emb_film.py` — `EmbHead`,
  `broadcast_emb`, `DomainProbe` (copy `DomainProbe` usage; reuse it directly).
- `interactive_world_sim/datasets/latent_dynamics/play_eef_dataset.py`:
  - per-domain obs assembled at `_sample_window` ~L499–L501:
    `obs_dict[ok] = torch.from_numpy(seq/255.0)` (frames in `[0,1]`).
  - `MixedPlayEEFDataset.__getitem__` ~L621–L643 adds `item["embodiment"]` and
    `item["domain_label"]` (human=0, robot=1). **agent_mask must be added here.**
- `configurations/algorithm/latent_world_model.yaml` ~L156–L198 (`latent_decompose`
  block; add `mask_subtract` params).
- `sbatch/phase0_stage1_mixed_vit.sbatch` — template for the run.

Conventions confirmed: `num_latent_channel` = scene/full latent channels (4);
latent grid = `self.latent_resolution` (square, e.g. 32); 1 view in the run
(`obs_keys=[camera_0_color]`); `domain_label`: human=0, robot=1.

**Test commands:** `pytest tests/test_mask_subtract.py -v` (conda env `iws`).

**Diagnostics folder:** all mask QC visualizations, coverage stats, and the
analysis/record file go in `outputs/mask_subtract_diag/`.

> **CRITICAL — mask quality (user directive):** the mask MUST cover the **whole
> arm** (hand + forearm + sleeve for human; full arm + gripper for robot), NOT just
> the hand. Past mask attempts failed **silently** — they looked right on a casual
> glance but were wrong. A simple per-pixel skin/dark threshold is exactly this
> failure mode (grabs only hand, or only sleeve, or leaks background). Therefore
> the mask is validated with **quantitative guards**, not eyeballing alone:
> (1) coverage fraction in a plausible band (arm ≈ 4–30%, not ~0.5% = hand-only);
> (2) the agent region is a connected component that **touches the image border**
> (a limb enters from off-frame; a hand-only blob does not); (3) blue plate/bowl and
> the dark-red cube are rejected. Verify on MANY frames of BOTH domains before any
> training run.

---

## Phase 0 — Agent mask in the batch

### Task 1: Heuristic agent-mask function

**Files:**
- Create: `interactive_world_sim/algorithms/latent_decompose/agent_mask.py`
- Test: `tests/test_mask_subtract.py`

Source-agnostic by design: the model consumes `batch["agent_mask"]`; this heuristic
is the v1 source (the design names mask completeness as the central lever — a
SAM2/precomputed mask is a later drop-in via the same batch key). Heuristic mirrors
the validated inpaint-diagnostic mask: human = skin(pink) ∪ dark(sleeve), robot =
dark(gripper) only; reject strong-blue (plate/bowl).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mask_subtract.py
import torch
from interactive_world_sim.algorithms.latent_decompose.agent_mask import agent_mask_from_rgb


def _frame(h=16, w=16):
    # background = tan wood (R>G>B, mid luminance), not agent
    f = torch.zeros(3, h, w)
    f[0], f[1], f[2] = 0.55, 0.45, 0.30
    return f


def test_dark_region_is_agent_for_robot():
    f = _frame()
    f[:, 4:8, 4:8] = 0.05  # dark gripper block
    m = agent_mask_from_rgb(f, "robot")          # (1,H,W)
    assert m.shape == (1, 16, 16)
    assert m[0, 5, 5] == 1.0
    assert m[0, 0, 0] == 0.0                       # wood background not masked


def test_skin_region_is_agent_for_human_only():
    f = _frame()
    f[0, 4:8, 4:8] = 0.95; f[1, 4:8, 4:8] = 0.65; f[2, 4:8, 4:8] = 0.65  # pink hand
    m_h = agent_mask_from_rgb(f, "human")
    m_r = agent_mask_from_rgb(f, "robot")
    assert m_h[0, 5, 5] == 1.0                     # human: skin is agent
    assert m_r[0, 5, 5] == 0.0                     # robot: skin NOT agent (dark-only)


def test_blue_plate_rejected():
    f = _frame()
    f[2, 10:14, 10:14] = 0.9; f[0, 10:14, 10:14] = 0.1; f[1, 10:14, 10:14] = 0.1  # blue
    m = agent_mask_from_rgb(f, "human")
    assert m[0, 12, 12] == 0.0


def test_batched_and_temporal_shapes():
    f = _frame()
    batched = f.unsqueeze(0).unsqueeze(0).expand(2, 4, 3, 16, 16).contiguous()
    m = agent_mask_from_rgb(batched, "human")
    assert m.shape == (2, 4, 1, 16, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mask_subtract.py -v -k agent_mask or "dark or skin or blue or batched"`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: agent_mask_from_rgb`.

- [ ] **Step 3: Implement — two functions: color classifier + whole-arm refine**

`agent_mask_from_rgb` = per-pixel color classifier (unit-tested on synthetic blobs).
`whole_arm_mask` = the function the dataset uses: morphological close → keep
connected components that **touch the image border** (the arm/limb) → dilate. This
is what enforces "whole arm, not just hand" and rejects stray hand-only/background
blobs. Verified on real frames in Task 1b (NOT by synthetic unit tests — geometry
needs real data).

```python
# interactive_world_sim/algorithms/latent_decompose/agent_mask.py
"""Agent mask from RGB, by embodiment. WHOLE ARM (hand+forearm+sleeve / full
gripper), not just the hand. v1 source for mask_subtract; the model consumes
batch["agent_mask"], so SAM2/precomputed masks are a drop-in via the same key.
Validated with quantitative guards (coverage + border-touch), not eyeballing —
past masks failed silently.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


def agent_mask_from_rgb(frames: torch.Tensor, embodiment: str) -> torch.Tensor:
    """Per-pixel color classifier. frames (...,3,H,W) [0,1] → (...,1,H,W) {0,1}."""
    assert frames.shape[-3] == 3, f"expected (...,3,H,W), got {tuple(frames.shape)}"
    r, g, b = frames[..., 0, :, :], frames[..., 1, :, :], frames[..., 2, :, :]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    dark = lum < 0.28                                    # black sleeve / gripper
    skin = (r > 0.55) & (r > g + 0.06) & (r > b + 0.06)  # pink hand
    blue = (b > 0.40) & (b > r + 0.06) & (b > g + 0.06)  # plate/bowl — reject
    sat = frames.amax(-3) - frames.amin(-3)
    cube = (r > 0.4) & (r > g + 0.15) & (r > b + 0.15) & (sat > 0.25) & (~skin)  # dark-red cube — reject
    agent = dark if embodiment == "robot" else (dark | skin)
    agent = agent & (~blue) & (~cube)
    return agent.unsqueeze(-3).to(frames.dtype)


def whole_arm_mask(frames: torch.Tensor, embodiment: str,
                   border_only: bool = True, dilate: int = 3) -> torch.Tensor:
    """Refine the color mask into a whole-arm mask: morphological close, keep
    connected components touching the image border (the limb), dilate. Operates
    per-frame. frames (...,3,H,W) → (...,1,H,W) {0,1}."""
    raw = agent_mask_from_rgb(frames, embodiment)[..., 0, :, :]   # (...,H,W)
    lead = raw.shape[:-2]
    flat = raw.reshape(-1, *raw.shape[-2:]).cpu().numpy().astype(bool)
    out = np.zeros_like(flat)
    for i in range(flat.shape[0]):
        m = ndimage.binary_closing(flat[i], iterations=2)
        lab, n = ndimage.label(m)
        if n == 0:
            continue
        if border_only:
            border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
            border.discard(0)
            keep = np.isin(lab, list(border)) if border else np.zeros_like(m)
            # fallback: if nothing touches the border, keep the largest component
            if keep.sum() == 0:
                sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
                keep = lab == (int(np.argmax(sizes)) + 1)
        else:
            keep = m
        if dilate > 0:
            keep = ndimage.binary_dilation(keep, iterations=dilate)
        out[i] = keep
    t = torch.from_numpy(out).to(frames.dtype).to(frames.device)
    return t.reshape(*lead, 1, *raw.shape[-2:])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mask_subtract.py -v`
Expected: 4 `agent_mask_from_rgb` tests PASS (they test the color classifier only).

- [ ] **Step 5: Commit**

```bash
git add interactive_world_sim/algorithms/latent_decompose/agent_mask.py tests/test_mask_subtract.py
git commit -m "feat(decompose): whole-arm agent mask (color + border-touch refine)"
```

### Task 1b: Mask QC on real frames (CRITICAL — verify before training)

**Files:**
- Create: `mask_qc.py` (repo root) → outputs to `outputs/mask_subtract_diag/`

- [ ] **Step 1: Write the QC script**

Pull real frames from `MixedPlayEEFDataset` (both domains), compute `whole_arm_mask`,
save overlay grids (raw | mask-overlay) for ~24 frames/domain, and print/save
coverage-fraction stats + border-touch rate. Save PNGs + a `mask_qc_stats.json` to
`outputs/mask_subtract_diag/`.

- [ ] **Step 2: Run QC and inspect**

Run: `python mask_qc.py` then open the PNGs. **Quantitative gates (must pass):**
median coverage in [0.04, 0.30] per domain; border-touch rate > 0.8; blue/cube not
masked. If a gate fails, tune thresholds in `agent_mask_from_rgb` / `whole_arm_mask`
and re-run — do NOT proceed to training on a failing mask.

- [ ] **Step 3: Commit QC script (PNGs are gitignored outputs)**

```bash
git add mask_qc.py
git commit -m "test(mask_subtract): real-frame mask QC with quantitative gates"
```

### Task 2: Wire agent_mask into the dataset

**Files:**
- Modify: `interactive_world_sim/datasets/latent_dynamics/play_eef_dataset.py` (`MixedPlayEEFDataset.__getitem__`, ~L639–L643)

The mask is computed from the same obs tensor in the item (guarantees crop/resolution
alignment). obs is stored as `item[obs_key]`; in `_sample_window` (L499–L501) frames
are `(T, H, W, 3)` in `[0,1]`. Convert to `(T,3,H,W)` for the mask helper, store
`item["agent_mask"]` as `(T,1,H,W)`.

- [ ] **Step 1: Add mask construction after domain_label**

In `MixedPlayEEFDataset.__getitem__`, immediately after the `item["domain_label"] = ...`
assignment (~L642), before `return item`:

```python
        # Agent mask for mask_subtract decomposition (v1 heuristic source).
        # obs frames are (T,H,W,3) in [0,1]; the mask helper wants (...,3,H,W).
        from interactive_world_sim.algorithms.latent_decompose.agent_mask import (
            whole_arm_mask,
        )
        primary = self.obs_keys[0]
        frames = item[primary]                      # (T,H,W,3) float [0,1]
        frames_chw = frames.permute(0, 3, 1, 2).contiguous()
        item["agent_mask"] = whole_arm_mask(frames_chw, emb)  # (T,1,H,W) whole arm
```

- [ ] **Step 2: Smoke-check the dataset emits the mask**

Run (env `iws`, from repo root):
```bash
python -c "
from omegaconf import OmegaConf
from interactive_world_sim.datasets.latent_dynamics import MixedPlayEEFDataset
cfg = OmegaConf.load('configurations/dataset/play_mixed_eef.yaml')
cfg = OmegaConf.merge(cfg, OmegaConf.create({'horizon':4,'val_horizon':16,'obs_keys':['camera_0_color'],'sample_ratio':0.5}))
ds = MixedPlayEEFDataset(cfg)
it = ds[0]
print('keys:', sorted(it.keys()))
print('agent_mask', it['agent_mask'].shape, it['agent_mask'].dtype, float(it['agent_mask'].mean()))
print('obs', it[ds.obs_keys[0]].shape, 'domain', int(it['domain_label']))
"
```
Expected: `agent_mask` shape `(4,1,H,W)` (matches obs frame count), dtype float,
mean in `(0,1)` (non-trivial coverage). If `play_mixed_eef.yaml` needs more keys,
read it and add required fields; do not change dataset semantics.

- [ ] **Step 3: Commit**

```bash
git add interactive_world_sim/datasets/latent_dynamics/play_eef_dataset.py
git commit -m "feat(dataset): emit agent_mask in MixedPlayEEFDataset items"
```

---

## Phase 1 — `mask_subtract` modules (pure, unit-tested)

All in `interactive_world_sim/algorithms/latent_decompose/mask_subtract.py`.

### Task 3: Removal operators

**Files:**
- Create: `interactive_world_sim/algorithms/latent_decompose/mask_subtract.py`
- Test: `tests/test_mask_subtract.py`

`ortho_proj` removes a unit direction in the **flattened** latent (rank-k out of
C·H·W → bounded, cannot collapse). `gate` multiplicatively suppresses. Direction(s)
are passed in (computed from the anchored embedding by `MaskSubtractHead`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mask_subtract.py (append)
import pytest
from interactive_world_sim.algorithms.latent_decompose.mask_subtract import (
    ortho_proj_remove,
)


def test_ortho_proj_removes_the_direction():
    B, C, H, W = 2, 4, 8, 8
    F = torch.randn(B, C, H, W)
    u = torch.randn(B, C * H * W)
    u = u / u.norm(dim=-1, keepdim=True)
    z = ortho_proj_remove(F, u)                    # (B,C,H,W)
    assert z.shape == F.shape
    # component along u is ~0 after removal
    zf = z.flatten(1)
    comp = (zf * u).sum(-1)
    assert torch.allclose(comp, torch.zeros(B), atol=1e-5)
    # info preserved: removing again is a no-op (idempotent projection)
    z2 = ortho_proj_remove(z, u)
    assert torch.allclose(z, z2, atol=1e-5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_mask_subtract.py::test_ortho_proj_removes_the_direction -v`
Expected: FAIL with `ImportError: ortho_proj_remove`.

- [ ] **Step 3: Implement**

```python
# interactive_world_sim/algorithms/latent_decompose/mask_subtract.py
"""mask_subtract: agent-mask-grounded embodiment readout + residual z_scene.

z_emb = head(stop_grad(F)) ⊙ mask  (agent-region spatial code)
e     = masked_pool(z_emb) → projection → anchored embedding
û     = normalize(map(e))           (embodiment direction, flattened-latent space)
z_scene = F − proj_û(F)             (ortho_proj default; detached û)

Detached from the encoder-decoder (anchor never reshapes the shared latent). No
CLUB, no adversary. See the 2026-05-28 mask_subtract design spec.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ortho_proj_remove(feat: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Remove the unit direction u (B, C*H*W) from feat (B, C, H, W).

    z = feat - <feat, u> u, computed in the flattened latent. u is treated as a
    constant w.r.t. this op's caller (caller passes a detached u).
    """
    B = feat.shape[0]
    flat = feat.reshape(B, -1)                      # (B, D)
    u = F.normalize(u, dim=-1, eps=1e-8)
    comp = (flat * u).sum(-1, keepdim=True)         # (B,1)
    out = flat - comp * u
    return out.reshape_as(feat)


def gate_remove(feat: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Multiplicative suppression: feat * (1 - sigmoid(g)). g broadcasts to feat."""
    return feat * (1.0 - torch.sigmoid(g))
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_mask_subtract.py::test_ortho_proj_removes_the_direction -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add interactive_world_sim/algorithms/latent_decompose/mask_subtract.py tests/test_mask_subtract.py
git commit -m "feat(mask_subtract): ortho_proj / gate removal operators"
```

### Task 4: MaskSubtractHead (readout + direction + z_scene)

**Files:**
- Modify: `interactive_world_sim/algorithms/latent_decompose/mask_subtract.py`
- Test: `tests/test_mask_subtract.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mask_subtract.py (append)
from interactive_world_sim.algorithms.latent_decompose.mask_subtract import MaskSubtractHead


def test_mask_subtract_head_shapes_and_detach():
    B, C, Hl, Wl = 3, 4, 8, 8
    head = MaskSubtractHead(c_scene=C, latent_hw=Hl, d_emb=16, removal="ortho_proj")
    Fl = torch.randn(B, C, Hl, Wl, requires_grad=True)
    mask_lat = torch.zeros(B, 1, Hl, Wl); mask_lat[:, :, 2:5, 2:5] = 1.0
    out = head(Fl, mask_lat)
    assert out["z_emb_spatial"].shape == (B, C, Hl, Wl)
    assert out["e"].shape == (B, 16)
    assert out["z_scene"].shape == (B, C, Hl, Wl)
    # z_emb is zero outside the mask (spatially gated)
    assert out["z_emb_spatial"][:, :, 0, 0].abs().max() == 0.0
    # the embodiment path is detached from F: grad of e w.r.t. F is None
    g = torch.autograd.grad(out["e"].sum(), Fl, retain_graph=True, allow_unused=True)[0]
    assert g is None
    # z_scene DOES carry grad to F (encoder is shaped by it)
    gz = torch.autograd.grad(out["z_scene"].sum(), Fl, allow_unused=True)[0]
    assert gz is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_mask_subtract.py::test_mask_subtract_head_shapes_and_detach -v`
Expected: FAIL with `ImportError: MaskSubtractHead`.

- [ ] **Step 3: Implement**

Append to `mask_subtract.py`:

```python
def _masked_pool(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of z (B,C,H,W) over mask (B,1,H,W) interior → (B,C). Safe if empty."""
    denom = mask.sum(dim=(2, 3)).clamp_min(1.0)      # (B,1)
    return (z * mask).sum(dim=(2, 3)) / denom         # (B,C)


class MaskSubtractHead(nn.Module):
    """Detached embodiment readout + residual scene latent.

    Forward consumes the full latent F and the latent-grid agent mask, returns
    z_emb_spatial (agent-gated code), e (pooled anchored embedding), z_scene
    (F with the anchored embodiment direction projected out). The embodiment path
    reads stop_grad(F); the removal direction is detached so it cannot corrupt e.
    """

    def __init__(self, c_scene: int, latent_hw: int, d_emb: int = 64,
                 removal: str = "ortho_proj", hidden: int = 64):
        super().__init__()
        self.c_scene = int(c_scene)
        self.latent_hw = int(latent_hw)
        self.removal = removal
        self.d_flat = self.c_scene * self.latent_hw * self.latent_hw
        # small conv readout on the (detached) latent
        self.emb_conv = nn.Sequential(
            nn.Conv2d(c_scene, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, c_scene, 3, padding=1),
        )
        self.proj = nn.Sequential(
            nn.Linear(c_scene, hidden), nn.GELU(), nn.Linear(hidden, d_emb),
        )
        # map the anchored embedding to a direction in the flattened latent
        # (ortho_proj) or a gate map (gate).
        self.dir_map = nn.Linear(d_emb, self.d_flat)

    def forward(self, F_latent: torch.Tensor, mask_lat: torch.Tensor) -> dict:
        Fsg = F_latent.detach()
        z_emb_spatial = self.emb_conv(Fsg) * mask_lat           # gated to agent
        e = self.proj(_masked_pool(z_emb_spatial, mask_lat))    # (B, d_emb)
        if self.removal == "gate":
            g = self.dir_map(e).reshape_as(F_latent)
            z_scene = gate_remove(F_latent, g.detach())
        else:  # ortho_proj
            u = self.dir_map(e)                                  # (B, d_flat)
            z_scene = ortho_proj_remove(F_latent, u.detach())
        return {"z_emb_spatial": z_emb_spatial, "e": e, "z_scene": z_scene}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_mask_subtract.py::test_mask_subtract_head_shapes_and_detach -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add interactive_world_sim/algorithms/latent_decompose/mask_subtract.py tests/test_mask_subtract.py
git commit -m "feat(mask_subtract): MaskSubtractHead (detached readout + residual z_scene)"
```

### Task 5: AgentReconHead (grounds z_emb to the agent)

**Files:**
- Modify: `interactive_world_sim/algorithms/latent_decompose/mask_subtract.py`
- Test: `tests/test_mask_subtract.py`

Lightweight conv upsampler from `z_emb_spatial` (C×Hl×Wl) → agent RGB (3×H×W). This
is **not** the diffusion decoder; it is the forcing function pinning z_emb to the
agent. Loss is applied mask-interior only (Task 10).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mask_subtract.py (append)
from interactive_world_sim.algorithms.latent_decompose.mask_subtract import AgentReconHead


def test_agent_recon_head_upsamples_to_image():
    B, C, Hl, Wl = 2, 4, 8, 8
    head = AgentReconHead(c_scene=C, out_hw=32)
    z = torch.randn(B, C, Hl, Wl)
    rgb = head(z)
    assert rgb.shape == (B, 3, 32, 32)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0     # sigmoid output in [0,1]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_mask_subtract.py::test_agent_recon_head_upsamples_to_image -v`
Expected: FAIL with `ImportError: AgentReconHead`.

- [ ] **Step 3: Implement**

Append to `mask_subtract.py`:

```python
class AgentReconHead(nn.Module):
    """z_emb_spatial (B,C,Hl,Wl) → agent RGB (B,3,out_hw,out_hw) in [0,1].

    Upsamples by ×2 conv blocks until out_hw is reached. Used only for the
    mask-interior agent-reconstruction loss (the z_emb grounding signal).
    """

    def __init__(self, c_scene: int, out_hw: int, hidden: int = 64):
        super().__init__()
        self.out_hw = int(out_hw)
        self.stem = nn.Conv2d(c_scene, hidden, 3, padding=1)
        self.block = nn.Sequential(
            nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.to_rgb = nn.Conv2d(hidden, 3, 3, padding=1)

    def forward(self, z_emb_spatial: torch.Tensor) -> torch.Tensor:
        x = self.stem(z_emb_spatial)
        while x.shape[-1] < self.out_hw:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = x + self.block(x)
        if x.shape[-1] != self.out_hw:
            x = F.interpolate(x, size=(self.out_hw, self.out_hw), mode="bilinear",
                              align_corners=False)
        return torch.sigmoid(self.to_rgb(x))
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_mask_subtract.py::test_agent_recon_head_upsamples_to_image -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add interactive_world_sim/algorithms/latent_decompose/mask_subtract.py tests/test_mask_subtract.py
git commit -m "feat(mask_subtract): AgentReconHead (z_emb agent-recon grounding)"
```

### Task 6: SupCon anchor loss (the missing anchor)

**Files:**
- Modify: `interactive_world_sim/algorithms/latent_decompose/mask_subtract.py`
- Test: `tests/test_mask_subtract.py`

Supervised-contrastive on the pooled embedding `e` by `domain_label` (positives =
same domain, negatives = other). With 2 domains this structures z_emb to be
embodiment-discriminative without collapse.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mask_subtract.py (append)
from interactive_world_sim.algorithms.latent_decompose.mask_subtract import supcon_anchor


def test_supcon_anchor_lower_when_clustered_by_domain():
    # two tight clusters by domain → low loss; shuffled → higher loss
    e_good = torch.tensor([[2.0, 0.0], [2.1, 0.1], [-2.0, 0.0], [-2.1, -0.1]])
    dl = torch.tensor([0, 0, 1, 1])
    e_bad = torch.tensor([[2.0, 0.0], [-2.0, 0.0], [2.1, 0.1], [-2.1, -0.1]])
    dl_bad = torch.tensor([0, 1, 0, 1])  # same points, domain-agnostic layout
    assert supcon_anchor(e_good, dl) < supcon_anchor(e_bad, dl_bad)
    # single-domain batch → zero (no positives/negatives to contrast)
    assert float(supcon_anchor(torch.randn(4, 2), torch.zeros(4, dtype=torch.long))) == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_mask_subtract.py::test_supcon_anchor_lower_when_clustered_by_domain -v`
Expected: FAIL with `ImportError: supcon_anchor`.

- [ ] **Step 3: Implement**

Append to `mask_subtract.py`:

```python
def supcon_anchor(e: torch.Tensor, domain_label: torch.Tensor,
                  temperature: float = 0.1) -> torch.Tensor:
    """Supervised-contrastive (InfoNCE) anchor on embeddings e (B, d) by
    domain_label (B,). Returns a scalar; 0 if any sample has no same-domain
    positive (e.g. single-domain batch)."""
    z = F.normalize(e, dim=-1)
    sim = z @ z.t() / temperature                    # (B,B)
    B = z.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    same = domain_label[:, None] == domain_label[None, :]
    pos = same & (~eye)
    if pos.sum() == 0:
        return e.new_zeros(())
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_count = pos.sum(1).clamp_min(1)
    loss = -(log_prob * pos).sum(1) / pos_count
    # only average over anchors that actually have a positive
    has_pos = pos.sum(1) > 0
    return loss[has_pos].mean()
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_mask_subtract.py -v`
Expected: ALL mask_subtract tests PASS.

- [ ] **Step 5: Commit**

```bash
git add interactive_world_sim/algorithms/latent_decompose/mask_subtract.py tests/test_mask_subtract.py
git commit -m "feat(mask_subtract): supervised-contrastive domain anchor"
```

---

## Phase 2 — Integrate into LatentWorldModel

### Task 7: Config + init flags

**Files:**
- Modify: `configurations/algorithm/latent_world_model.yaml` (~L156–L198)
- Modify: `interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py` (`__init__`, ~L107–L130)

- [ ] **Step 1: Add config block**

In `latent_world_model.yaml`, extend the method comment list and append params under
`latent_decompose:` (after the dual_head block ~L197):

```yaml
  # --- mask_subtract (2026-05-28) only ---
  # method: mask_subtract — agent-mask-grounded z_emb (anchor + agent recon),
  #         detached from encoder-decoder; z_scene = F with the anchored embodiment
  #         direction projected out. No CLUB, no adversary. Decoder unchanged
  #         (z_scene and z_full are both num_latent_channel channels).
  d_emb_mask: 64            # pooled embedding width for the anchor
  removal_op: ortho_proj    # ortho_proj | gate
  lambda_agent_rec: 1.0     # mask-interior agent reconstruction weight
  lambda_anchor: 0.5        # domain supcon anchor weight
  lambda_scene_rec: 1.0     # mask-exterior z_scene reconstruction (sufficiency); 0 disables
  anchor_temperature: 0.1
```

- [ ] **Step 2: Add init flag**

In `LatentWorldModel.__init__`, after the `self.use_dual_head = ...` block (~L126),
add:

```python
        self.use_mask_subtract = bool(_decompose_on and self.decompose_method == "mask_subtract")
```

And include `mask_subtract` in the automatic-optimisation set? **No** — mask_subtract
runs under **automatic** optimisation (no two-player game), like emb_film. Leave the
`if self.use_dynamo_ssl or self.use_dual_head:` block (~L151) unchanged.

- [ ] **Step 3: Verify config loads**

Run (env `iws`):
```bash
python -c "
from omegaconf import OmegaConf
c = OmegaConf.load('configurations/algorithm/latent_world_model.yaml')
print(c.latent_decompose.removal_op, c.latent_decompose.lambda_agent_rec, c.latent_decompose.d_emb_mask)
"
```
Expected: prints `ortho_proj 1.0 64`.

- [ ] **Step 4: Commit**

```bash
git add configurations/algorithm/latent_world_model.yaml interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py
git commit -m "feat(lwm): mask_subtract config + use_mask_subtract flag"
```

### Task 8: Build modules in `_build_model`

**Files:**
- Modify: `interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py` (`_build_model`, after the emb_film block ~L290; decoder cond width ~L193)

**Decoder note:** mask_subtract does NOT append channels, so the decoder cond-width
branch (~L193) must remain unchanged for mask_subtract (it already only triggers on
`use_emb_film`). No edit needed there.

- [ ] **Step 1: Construct the mask_subtract modules**

After the emb_film `if self.use_emb_film:` block in `_build_model` (~L290), add:

```python
                if getattr(self, "use_mask_subtract", False):
                    from interactive_world_sim.algorithms.latent_decompose.mask_subtract import (
                        MaskSubtractHead,
                        AgentReconHead,
                    )
                    from interactive_world_sim.algorithms.latent_decompose.emb_film import (
                        DomainProbe,
                    )
                    ld = self.cfg.latent_decompose
                    c_scene = int(self.cfg.num_latent_channel)
                    self.mask_subtract = MaskSubtractHead(
                        c_scene=c_scene,
                        latent_hw=int(self.latent_resolution),
                        d_emb=int(ld.get("d_emb_mask", 64)),
                        removal=str(ld.get("removal_op", "ortho_proj")),
                    )
                    self.agent_recon = AgentReconHead(
                        c_scene=c_scene, out_hw=int(self.cfg.x_shape[1]),
                    )
                    # detached diagnostics: domain-separability of z_scene vs z_full vs z_emb
                    self.probe_scene = DomainProbe(c_scene)
                    self.probe_full = DomainProbe(c_scene)
                    self.probe_emb = DomainProbe(int(ld.get("d_emb_mask", 64)))
```

- [ ] **Step 2: Verify model builds (no run)**

Defer verification to the smoke test (Task 13) — `_build_model` requires a full cfg.
No standalone command here.

- [ ] **Step 3: Commit**

```bash
git add interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py
git commit -m "feat(lwm): build mask_subtract modules in _build_model"
```

### Task 9: Compute z_emb / z_scene in `encoder_forward`

**Files:**
- Modify: `interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py` (`encoder_forward`, ~L868–L962)

mask_subtract needs the agent mask, which lives in the batch, not in `encoder_forward`'s
`obs` arg. Keep `encoder_forward` returning the **full latent F** (so the decoder /
dynamics path is unchanged), and compute z_emb/z_scene in `training_step` where the
batch (mask) is available. So **`encoder_forward` needs NO change for the decoder
path** — F is exactly what the baseline returns. Verify this by reading the method;
the only requirement is that for mask_subtract, `encoder_forward(xs)` returns the
plain normalized latent (it does — mask_subtract is not in the `use_latent_decompose`
or `use_emb_film` branches, so it falls through to `return z`).

- [ ] **Step 1: Confirm fall-through (read-only)**

Read `encoder_forward` ~L897–L962. Confirm that with only `use_mask_subtract` true
(emb_film/latent_decompose false), the per-view loop takes the `else` branch
(`spatial_feat, cls_token = self.vit_encoder(view_obs)`), no split/emb append
happens, and the method returns the plain `z`. **No code change** — add a one-line
comment at the emb_film append guard (~L953) documenting that mask_subtract
intentionally does not append:

```python
        # NOTE: mask_subtract does NOT append channels here — z_scene is derived
        # from F in training_step (needs the batch agent mask), and the decoder
        # consumes the unchanged num_latent_channel latent.
        if self.use_emb_film:
```

- [ ] **Step 2: Commit**

```bash
git add interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py
git commit -m "docs(lwm): note mask_subtract leaves encoder_forward latent unchanged"
```

### Task 10: `_mask_subtract_losses` helper

**Files:**
- Modify: `interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py` (add method after `_emb_film_probe_loss`, ~L748)

Computes: agent-recon (mask interior), supcon anchor, detached probes (z_full /
z_scene / z_emb). Returns `(z_scene, aux_loss)` where `aux_loss` includes the
embodiment-path losses (agent recon + anchor) and the detached-probe CE. The
**scene-reconstruction** term is handled in `training_step` (it needs a decoder
pass), so this helper returns `z_scene` for that.

`z` here is the full latent `F` of shape `(B*T, C_scene, Hl, Wl)`; `batch["agent_mask"]`
is `(B, T, 1, H, W)`.

- [ ] **Step 1: Add the helper**

```python
    def _mask_subtract_losses(self, batch: dict, z: torch.Tensor, seq_len: int):
        """mask_subtract: derive z_scene from F, compute embodiment-path losses
        (detached from the encoder via MaskSubtractHead) + detached probes.

        Returns (z_scene (B*T,C,Hl,Wl), aux_loss scalar). aux_loss = agent recon +
        anchor + detached-probe CE. The mask-exterior scene-reconstruction loss is
        added in training_step (needs a decoder forward).
        """
        import torch.nn.functional as F_
        c_scene = int(self.cfg.num_latent_channel)
        Hl = z.shape[-1]
        # mask → latent grid, flatten time into batch to match z = (B*T,...)
        m = batch["agent_mask"].to(z.device).float()        # (B,T,1,H,W)
        B, T = m.shape[0], m.shape[1]
        m = m.reshape(B * T, 1, m.shape[-2], m.shape[-1])    # (B*T,1,H,W)
        mask_lat = F_.interpolate(m, size=(Hl, Hl), mode="area")
        mask_lat = (mask_lat > 0.5).float()                  # (B*T,1,Hl,Wl)

        out = self.mask_subtract(z, mask_lat)
        z_scene = out["z_scene"]

        # (1) agent reconstruction (mask interior), against the input frames
        xs = batch[self.obs_keys[0]].to(z.device).float()    # (B,T,H,W,3) [0,1]
        xs = xs.permute(0, 1, 4, 2, 3).reshape(B * T, 3, xs.shape[2], xs.shape[3])
        agent_rgb = self.agent_recon(out["z_emb_spatial"])   # (B*T,3,H,W)
        m_img = F_.interpolate(m, size=agent_rgb.shape[-2:], mode="area")
        m_img = (m_img > 0.5).float()
        denom = m_img.sum().clamp_min(1.0)
        L_agent = (((agent_rgb - xs) ** 2) * m_img).sum() / denom

        # (2) supcon domain anchor on the pooled embedding
        from interactive_world_sim.algorithms.latent_decompose.mask_subtract import (
            supcon_anchor,
        )
        dl = batch["domain_label"].to(z.device).long()       # (B,)
        dl_rep = dl.repeat_interleave(T)                      # (B*T,)
        L_anchor = supcon_anchor(
            out["e"], dl_rep,
            temperature=float(self.cfg.latent_decompose.get("anchor_temperature", 0.1)),
        )

        # (3) detached probes (measure only; never reach encoder)
        zf = z.mean(dim=(2, 3)).detach()
        zs = z_scene.mean(dim=(2, 3)).detach()
        ze = out["e"].detach()
        lf = self.probe_full(zf); ls = self.probe_scene(zs); le = self.probe_emb(ze)
        probe_ce = (F_.cross_entropy(lf, dl_rep)
                    + F_.cross_entropy(ls, dl_rep)
                    + F_.cross_entropy(le, dl_rep))
        with torch.no_grad():
            self.log("training/probe_full_acc", (lf.argmax(-1) == dl_rep).float().mean())
            self.log("training/probe_scene_acc", (ls.argmax(-1) == dl_rep).float().mean())
            self.log("training/probe_emb_acc", (le.argmax(-1) == dl_rep).float().mean())
        self.log("training/L_agent_rec", L_agent)
        self.log("training/L_anchor", L_anchor)

        ld = self.cfg.latent_decompose
        aux = (float(ld.get("lambda_agent_rec", 1.0)) * L_agent
               + float(ld.get("lambda_anchor", 0.5)) * L_anchor
               + probe_ce)
        return z_scene, aux
```

**Note for the implementer:** `DomainProbe.forward` expects a pooled `(B, D)` vector
(see emb_film.py L61). Here we pass pooled `(B*T, D)` tensors — consistent.

- [ ] **Step 2: Commit**

```bash
git add interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py
git commit -m "feat(lwm): _mask_subtract_losses (agent recon + anchor + probes)"
```

### Task 11: training_step branch (incl. mask-exterior scene recon)

**Files:**
- Modify: `interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py` (`training_step`, the `elif self.use_emb_film:` chain ~L1514–L1539)

Add a `mask_subtract` branch. The full reconstruction (`rec_loss`, computed at
~L1393–L1459) already trains E/decoder from F. The branch adds: derive z_scene +
embodiment-path losses; optionally a **second decoder forward** on the augmented
z_scene scored **mask-exterior** (sufficiency / anti-collapse).

- [ ] **Step 1: Add the branch**

Replace the `elif self.use_emb_film:` ... `else:` tail (~L1520–L1536) so the chain
reads `... elif self.use_emb_film: ... elif self.use_mask_subtract: <new> else: ...`.
Insert before the final `else:`:

```python
                elif getattr(self, "use_mask_subtract", False):
                    z_scene, aux = self._mask_subtract_losses(batch, z, obs.shape[1])
                    model_loss = rec_loss
                    # (3) mask-exterior scene reconstruction from z_scene
                    #     (sufficiency / anti-collapse). Second decoder forward.
                    lam_sr = float(self.cfg.latent_decompose.get("lambda_scene_rec", 1.0))
                    if lam_sr > 0:
                        pred_scene = self._forward(
                            self.decoder, noisy_xs_t, t, s, external_cond=z_scene,
                        )
                        m = batch["agent_mask"].to(z.device).float()
                        Bb, Tt = m.shape[0], m.shape[1]
                        m = m.reshape(Bb * Tt, 1, m.shape[-2], m.shape[-1])
                        m_img = torch.nn.functional.interpolate(
                            m, size=pred_scene.shape[-2:], mode="area")
                        ext = (m_img <= 0.5).float()             # mask EXTERIOR
                        denom = ext.sum().clamp_min(1.0)
                        L_scene = ((((pred_scene - noisy_xs_s.detach()) ** 2) * ext).sum()
                                   / denom)
                        self.log("training/L_scene_rec", L_scene)
                        model_loss = model_loss + lam_sr * L_scene
                    total_loss = model_loss + aux
                    log_loss = model_loss
```

**Implementer note:** `noisy_xs_t`, `t`, `s`, `noisy_xs_s`, `self._forward` are all in
scope (defined ~L1396–L1414). `external_cond=z_scene` requires z_scene to have
`num_latent_channel` channels — it does (same as F). If `_forward`'s consistency
target differs, mirror exactly how `rec_loss` scores `pred_s` against
`noisy_xs_s.detach()` (~L1443) for the exterior term.

- [ ] **Step 2: Commit**

```bash
git add interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py
git commit -m "feat(lwm): training_step mask_subtract branch + scene-rec sufficiency"
```

### Task 12: Optimizer groups + gradient clipping

**Files:**
- Modify: `interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py` (`configure_optimizers` ~L407–L431, L464; `configure_gradient_clipping` ~L549)

The embodiment modules (`mask_subtract`, `agent_recon`) train but are detached from E
via `stop_grad` inside the head — so their grads only flow through their own params.
Probes are aux (constant LR, excluded from clip). The encoder/decoder are group 0/1.

- [ ] **Step 1: Add mask_subtract param groups**

In `configure_optimizers`, after the emb_film `if self.use_emb_film:` probe-group
block (~L419), add:

```python
            if getattr(self, "use_mask_subtract", False):
                # embodiment path (detached from E): own group, main LR
                param_groups.append({
                    "params": list(self.mask_subtract.parameters())
                              + list(self.agent_recon.parameters()),
                    "lr": self.cfg.lr,
                })
                # detached probes: aux group, constant LR, idx ≥ 2
                param_groups.append({
                    "params": list(self.probe_scene.parameters())
                              + list(self.probe_full.parameters())
                              + list(self.probe_emb.parameters()),
                    "lr": self.cfg.lr,
                })
```

Also extend the per-group-warmup guard (~L464) to include mask_subtract:

```python
            if self.use_dynamo_ssl or self.use_latent_decompose or self.use_emb_film or getattr(self, "use_mask_subtract", False):
```

**Note:** the embodiment group is at idx 2 here, so under `lr_lambda_fn` it gets
constant LR (idx ≥ 2). That is acceptable — the embodiment modules are auxiliary to
the encoder. If warmup is desired for them, that is a tuning follow-up, not required
for the first run.

- [ ] **Step 2: Extend gradient clipping**

In `configure_gradient_clipping` (~L549), change the emb_film guard to also clip
groups 0,1 for mask_subtract (probes/embodiment-path excluded from the global norm):

```python
        if (self.use_emb_film or getattr(self, "use_mask_subtract", False)) and gradient_clip_val:
```

- [ ] **Step 3: Commit**

```bash
git add interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py
git commit -m "feat(lwm): mask_subtract optimizer groups + gradient clipping"
```

---

## Phase 3 — Smoke test and run

### Task 13: Smoke test (few steps, 1 GPU)

**Files:**
- Create: `smoke_mask_subtract.py` (repo root, mirrors `smoke_emb_film.py` if present)

- [ ] **Step 1: Write the smoke script**

```python
# smoke_mask_subtract.py
"""Few-step smoke: build LWM with mask_subtract, run 3 training steps on the mixed
dataset, assert losses are finite and probes log. Run on 1 GPU (env iws)."""
import subprocess, sys

CMD = [
    sys.executable, "main.py",
    "+name=smoke_mask_subtract",
    "algorithm=latent_world_model", "experiment=exp_latent_dyn",
    "dataset=play_mixed_eef",
    "dataset.horizon=4", "dataset.val_horizon=16",
    "dataset.obs_keys=[camera_0_color]", "dataset.sample_ratio=0.5",
    "experiment.training.batch_size=2",
    "experiment.training.optim.accumulate_grad_batches=1",
    "experiment.training.max_steps=3",
    "experiment.training.log_every_n_steps=1",
    "experiment.validation.val_every_n_step=1000000",
    "algorithm.latent_dim=512", "algorithm.action_dim=8",
    "algorithm.training_stage=1",
    "algorithm.dynamo_ssl.enabled=false",
    "algorithm.dynamo_ssl.encoder_backbone=vit",
    "algorithm.latent_decompose.enabled=true",
    "algorithm.latent_decompose.method=mask_subtract",
]
sys.exit(subprocess.call(CMD))
```

- [ ] **Step 2: Run the smoke**

Run (on a GPU node / interactive, env `iws`):
```bash
python smoke_mask_subtract.py 2>&1 | tail -40
```
Expected: 3 training steps complete; log lines include `training/L_agent_rec`,
`training/L_anchor`, `training/L_scene_rec`, `training/probe_scene_acc`,
`training/probe_full_acc`, `training/probe_emb_acc`; no NaN; `training/loss` finite.

- [ ] **Step 3: Fix any integration errors**

Common issues to check if it crashes: (a) `agent_mask` missing in batch → Task 2 not
applied / wrong dataset; (b) channel mismatch in `external_cond=z_scene` → confirm
z_scene has `num_latent_channel` channels; (c) mask interpolation dtype → cast to
float. Iterate until Step 2 passes.

- [ ] **Step 4: Commit**

```bash
git add smoke_mask_subtract.py
git commit -m "test(mask_subtract): few-step training smoke script"
```

### Task 14: sbatch full run

**Files:**
- Create: `sbatch/phase0_stage1_mask_subtract.sbatch` (copy `phase0_stage1_mixed_vit.sbatch`)

- [ ] **Step 1: Write the sbatch**

Copy `sbatch/phase0_stage1_mixed_vit.sbatch` to
`sbatch/phase0_stage1_mask_subtract.sbatch` and change: `--job-name`, `--output`
(`phantom_logs/phase0_stage1_mask_subtract.log`), the `hydra.run.dir` tag and
`+name` to `phase0_stage1_mask_subtract`, and append to the `python main.py` args:

```bash
  algorithm.latent_decompose.enabled=true \
  algorithm.latent_decompose.method=mask_subtract
```

Keep `max_steps=300005`, `every_n_train_steps=20000`, `save_top_k=1`,
`val_every_n_step=10000`, batch_size=2, accumulate_grad_batches=4 (matches the
mixed-vit baseline for fair comparison).

- [ ] **Step 2: Launch the job**

Run:
```bash
sbatch sbatch/phase0_stage1_mask_subtract.sbatch && squeue -u "$USER"
```
Expected: a job id is printed and appears RUNNING/PENDING. Tail the log:
```bash
sleep 90 && tail -30 phantom_logs/phase0_stage1_mask_subtract.log
```
Expected: training started, step counter advancing, the mask_subtract loss/probe
lines present, no crash.

- [ ] **Step 3: Commit**

```bash
git add sbatch/phase0_stage1_mask_subtract.sbatch
git commit -m "run(mask_subtract): phase0 stage1 sbatch"
```

---

## Validation / what to watch (post-launch)

Per the spec success criteria (read the early ckpt at step ~20000):
- `probe_full_acc` high; `probe_scene_acc` **lower than** `probe_full_acc`
  (removal stripped domain signal); `probe_emb_acc` high (z_emb captured domain).
- full reconstruction PSNR not regressed vs emb_film (~40).
- `L_agent_rec` decreasing + agent-recon QC renders the agent (offline, later).
- **ablation:** rerun with `lambda_anchor=0` — expect z_emb to degrade (control that
  reproduces the v4 "slot with no forcing function" failure).

## Out of scope (this plan)

- Cross-domain z_scene **alignment** (OT/whitening on the residual direction).
- **EEF → dynamics** wiring (stage-2).
- Cross-embodiment **recompose** eval.
- SAM2 / precomputed mask source (drop-in via `batch["agent_mask"]`; the heuristic
  is v1). Mask completeness is the central lever — improving it is the first
  follow-up if `probe_scene_acc` does not drop.
