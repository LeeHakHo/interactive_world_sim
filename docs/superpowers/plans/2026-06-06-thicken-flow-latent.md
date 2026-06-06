# Thicken Flow Dynamic Latent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thicken the shared flow dynamic latent with contact + grasp signals and add anti-drift training so the cube stops drifting ("乱动") in autoregressive rollout, without reopening the cross-embodiment 582× separability gap.

**Architecture:** Extend the current single-shot flow WM (`train_flow_wm_scarcity_v3.py`) into `train_flow_wm_scarcity_v4.py`. Two new shared-latent channels — contact distance (object↔EEF tips) and per-domain-normalized grasp openness (`‖eef3[1]-eef3[2]‖`) — feed (a) a grasp-sequence token and (b) a per-point-per-step contact gate `α∈[0,1]` that multiplies predicted displacement (object stays put when `α→0`, enforced by an L1 sparsity prior, no pseudo-labels). Anti-drift = state-noise injection on history + a 2-window multi-step consistency loss. A `--thin` flag exactly reproduces v3 for ablation. Validation is phased: robot-only drift first, then cross-domain scarcity.

**Tech Stack:** Python, PyTorch, NumPy, sklearn (LogReg admission probe), pytest. Data: `outputs/flow_dataset/flow_ds_v3.npz` (tracks `(N,16,48,2)`, vis `(N,16,48)`, eef3 `(N,16,3,2)`, domain, vid — all normalized /224). No CoTracker3 re-run, no dataset regen.

**Shared constants** (defined once at top of `train_flow_wm_scarcity_v4.py`, imported by tests): `K=4, F=12, L=16, Dm=128, P=48, EPOCHS=60, BS=128, LR=3e-4, TEST_ROBOT_VID=12, N_LIST=[1400,400,200,100,50], SEEDS=5`. New: `LAMBDA_GATE=1e-2` (L1 on α), `NOISE_STD=0.01` (normalized coords ≈2.24px), `LAMBDA_CONSIST=0.5`, `STATIC_TAU=0.02` (drift-metric near-static threshold in normalized centroid path length).

---

## File Structure

- Create: `train_flow_wm_scarcity_v4.py` — thick model + helpers + train/eval + Phase-1/Phase-2 runners + `--thin`/`--probe-cg`/`--phase` CLI. Single source of truth (thin vs thick is a flag, not a fork).
- Keep unchanged: `train_flow_wm_scarcity_v3.py` (documented baseline numbers), `gen_flow_dataset_v3.py`, `outputs/flow_dataset/flow_ds_v3.npz`.
- Create: `tests/flow_wm_v4/__init__.py` — empty package marker.
- Create: `tests/flow_wm_v4/test_thick_helpers.py` — fast unit tests on synthetic tensors (no dependence on the 38MB npz): grasp openness, per-domain normalization, contact features, gate invariant (`α=0 → static`), state-noise shape, multi-step chaining shapes, rollout-drift metric.
- Output dirs (created at runtime, per-experiment, per `per_experiment_output_dir` memory): `outputs/flow_wm_v4/probe_cg/`, `outputs/flow_wm_v4/phase1_robot_drift/`, `outputs/flow_wm_v4/phase2_crossdomain/` — each gets a `summary.txt`.

Helper functions live as module-level pure functions in `train_flow_wm_scarcity_v4.py` so tests import them directly:
`grasp_openness`, `fit_grasp_stats`, `normalize_grasp`, `contact_gate_features`, `cg_descriptor`, `inject_state_noise`, `multi_step_consistency`, `rollout_drift_static`, and class `FlowWMThick`.

---

### Task 1: Scaffold v4 from v3 with `--thin` flag (behaves as v3)

**Files:**
- Create: `train_flow_wm_scarcity_v4.py`
- Create: `tests/flow_wm_v4/__init__.py`
- Create: `tests/flow_wm_v4/test_thick_helpers.py`

- [ ] **Step 1: Write the failing test** (`tests/flow_wm_v4/test_thick_helpers.py`)

```python
import torch
import train_flow_wm_scarcity_v4 as v4


def test_constants_present():
    for name in ("K", "F", "L", "Dm", "P", "LAMBDA_GATE", "NOISE_STD",
                 "LAMBDA_CONSIST", "STATIC_TAU"):
        assert hasattr(v4, name), f"missing constant {name}"
    assert v4.K + v4.F == v4.L == 16


def test_thin_forward_shape():
    B = 5
    hist = torch.randn(B, v4.P, v4.K, 2)
    eef3 = torch.randn(B, v4.L, 3, 2)
    g = torch.rand(B, v4.L)
    m = v4.FlowWMThick(v4.P, thin=True)
    pred, alpha = m(hist, eef3, g)
    assert pred.shape == (B, v4.P, v4.F, 2)
    assert alpha is None  # thin path has no gate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'train_flow_wm_scarcity_v4'`.

- [ ] **Step 3: Write minimal implementation** — create `train_flow_wm_scarcity_v4.py` with constants and a thick model whose thin path mirrors v3 exactly. (Gate/grasp-token built but only exercised in later tasks; thin path ignores them.)

```python
"""Step 3 v4: thick flow dynamic latent (contact-gated motion + anti-drift training).

Extends train_flow_wm_scarcity_v3.py. `--thin` reproduces v3 exactly (ablation via
flag, not a fork). Reads outputs/flow_dataset/flow_ds_v3.npz. See
docs/superpowers/specs/2026-06-06-thicken-flow-latent-design.md.
"""
import argparse, os, numpy as np, torch, torch.nn as nn

DS = "outputs/flow_dataset/flow_ds_v3.npz"
K, F, L, Dm, P = 4, 12, 16, 128, 48
EPOCHS, BS, LR = 60, 128, 3e-4
TEST_ROBOT_VID = 12
N_LIST = [1400, 400, 200, 100, 50]
SEEDS = 5
LAMBDA_GATE = 1e-2
NOISE_STD = 0.01
LAMBDA_CONSIST = 0.5
STATIC_TAU = 0.02
device = "cuda" if torch.cuda.is_available() else "cpu"


class FlowWMThick(nn.Module):
    def __init__(self, P, thin=False):
        super().__init__()
        self.thin = thin
        self.inp = nn.Linear(2 * K + 2, Dm)
        self.act = nn.Linear(L * 3 * 2, Dm)            # full-window EEF as action
        self.gctx = nn.Linear(L, Dm)                   # grasp-sequence token (thick only)
        enc = nn.TransformerEncoderLayer(Dm, 4, Dm * 2, batch_first=True, dropout=0.0)
        self.tf = nn.TransformerEncoder(enc, 3)
        self.head = nn.Linear(Dm, F * 2)
        self.gate = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, hist, eef3, g):
        # hist (B,P,K,2), eef3 (B,L,3,2), g (B,L) normalized grasp
        B, P = hist.shape[:2]
        anchor = hist[:, :, -1, :]                                       # (B,P,2)
        obj = self.inp(torch.cat([(hist - anchor[:, :, None]).reshape(B, P, 2 * K), anchor], -1))
        objc = anchor.mean(1, keepdim=True)                              # (B,1,2)
        act = self.act((eef3 - objc[:, :, None]).reshape(B, -1))[:, None, :]
        tokens = [obj, act]
        if not self.thin:
            tokens.append(self.gctx(g)[:, None, :])
        x = self.tf(torch.cat(tokens, 1))[:, :P]                         # (B,P,Dm)
        raw = self.head(x).reshape(B, P, F, 2)                           # displacement from anchor
        if self.thin:
            return anchor[:, :, None, :] + raw, None
        feat = contact_gate_features(anchor, eef3, g)                    # (B,P,F,3)
        alpha = torch.sigmoid(self.gate(feat)).squeeze(-1)              # (B,P,F)
        return anchor[:, :, None, :] + alpha[..., None] * raw, alpha


# --- pure helpers (filled in later tasks) ---
def contact_gate_features(anchor, eef3, g):
    raise NotImplementedError  # Task 3
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q`
Expected: PASS for `test_constants_present` and `test_thin_forward_shape` (thin path never calls `contact_gate_features`). Create empty `tests/flow_wm_v4/__init__.py` so the package imports.

- [ ] **Step 5: Commit**

```bash
git add train_flow_wm_scarcity_v4.py tests/flow_wm_v4/__init__.py tests/flow_wm_v4/test_thick_helpers.py
git commit -m "feat(flow-v4): scaffold thick flow WM with --thin path mirroring v3"
```

---

### Task 2: Grasp openness + per-domain normalization

**Files:**
- Modify: `train_flow_wm_scarcity_v4.py` (add `grasp_openness`, `fit_grasp_stats`, `normalize_grasp`)
- Test: `tests/flow_wm_v4/test_thick_helpers.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_grasp_openness_known_distance():
    eef3 = torch.zeros(2, v4.L, 3, 2)
    eef3[:, :, 1, 0] = 0.3   # tip1 x
    eef3[:, :, 2, 0] = -0.1  # tip2 x  -> distance 0.4
    g = v4.grasp_openness(eef3)
    assert g.shape == (2, v4.L)
    assert torch.allclose(g, torch.full((2, v4.L), 0.4), atol=1e-5)


def test_normalize_grasp_per_domain_range_and_scale():
    # robot grasp ~[0,0.08], human pinch ~[0,0.5]; per-domain norm must map both to [0,1]
    g_raw = torch.cat([torch.linspace(0, 0.08, 50), torch.linspace(0, 0.5, 50)])
    dom = torch.cat([torch.ones(50, dtype=torch.long), torch.zeros(50, dtype=torch.long)])
    stats = v4.fit_grasp_stats(g_raw, dom)
    out = v4.normalize_grasp(g_raw, dom, stats)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # both domains should span most of [0,1] after per-domain normalization
    assert out[dom == 1].max() > 0.9 and out[dom == 0].max() > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q -k grasp`
Expected: FAIL — `AttributeError: module ... has no attribute 'grasp_openness'`.

- [ ] **Step 3: Write minimal implementation** (add to `train_flow_wm_scarcity_v4.py`)

```python
def grasp_openness(eef3):
    # eef3 (...,L,3,2) -> (...,L): distance between the two tips (points 1,2)
    return torch.linalg.norm(eef3[..., 1, :] - eef3[..., 2, :], dim=-1)


def fit_grasp_stats(g_raw, domain):
    # per-domain p5/p95 over flattened grasp values
    g_flat = g_raw.reshape(len(g_raw), -1) if g_raw.dim() > 1 else g_raw[:, None]
    stats = {}
    for d in domain.unique().tolist():
        v = g_flat[domain == d].reshape(-1)
        stats[int(d)] = (torch.quantile(v, 0.05).item(), torch.quantile(v, 0.95).item())
    return stats


def normalize_grasp(g_raw, domain, stats):
    out = torch.zeros_like(g_raw)
    for d, (lo, hi) in stats.items():
        m = domain == d
        out[m] = ((g_raw[m] - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q -k grasp`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_flow_wm_scarcity_v4.py tests/flow_wm_v4/test_thick_helpers.py
git commit -m "feat(flow-v4): grasp openness + per-domain normalization"
```

---

### Task 3: Contact-gate features

**Files:**
- Modify: `train_flow_wm_scarcity_v4.py` (implement `contact_gate_features`)
- Test: `tests/flow_wm_v4/test_thick_helpers.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_contact_gate_features_shape_and_distance():
    B = 3
    anchor = torch.zeros(B, v4.P, 2)            # object at origin
    eef3 = torch.zeros(B, v4.L, 3, 2)
    eef3[:, :, 1, 0] = 0.5                       # future tip1 at distance 0.5
    eef3[:, :, 2, 1] = 0.5                       # future tip2 at distance 0.5
    g = torch.full((B, v4.L), 0.3)
    feat = v4.contact_gate_features(anchor, eef3, g)
    assert feat.shape == (B, v4.P, v4.F, 3)     # [d_tip1, d_tip2, grasp]
    assert torch.allclose(feat[..., 0], torch.full((B, v4.P, v4.F), 0.5), atol=1e-5)
    assert torch.allclose(feat[..., 1], torch.full((B, v4.P, v4.F), 0.5), atol=1e-5)
    assert torch.allclose(feat[..., 2], torch.full((B, v4.P, v4.F), 0.3), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q -k contact`
Expected: FAIL — `NotImplementedError` (placeholder from Task 1).

- [ ] **Step 3: Write minimal implementation** (replace the placeholder)

```python
def contact_gate_features(anchor, eef3, g):
    # anchor (B,P,2), eef3 (B,L,3,2), g (B,L) -> (B,P,F,3): [dist to 2 future tips, grasp]
    B, Pn = anchor.shape[:2]
    tips = eef3[:, K:, 1:3, :]                                   # (B,F,2,2) future 2 tips
    d = torch.linalg.norm(anchor[:, :, None, None, :] - tips[:, None], dim=-1)  # (B,P,F,2)
    gf = g[:, K:][:, None, :].expand(B, Pn, F)                   # (B,P,F)
    return torch.cat([d, gf[..., None]], -1)                     # (B,P,F,3)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q -k "contact or thin"`
Expected: PASS (and thin forward still passes).

- [ ] **Step 5: Commit**

```bash
git add train_flow_wm_scarcity_v4.py tests/flow_wm_v4/test_thick_helpers.py
git commit -m "feat(flow-v4): contact-gate features (object<->future-EEF distance + grasp)"
```

---

### Task 4: Thick forward + gate invariant (α=0 → cube static)

**Files:**
- Modify: `train_flow_wm_scarcity_v4.py` (no code change expected; forward already wired in Task 1 — this task verifies the key inductive-bias invariant)
- Test: `tests/flow_wm_v4/test_thick_helpers.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_thick_forward_shape_and_gate_static_invariant():
    B = 4
    hist = torch.randn(B, v4.P, v4.K, 2)
    eef3 = torch.randn(B, v4.L, 3, 2)
    g = torch.rand(B, v4.L)
    m = v4.FlowWMThick(v4.P, thin=False)
    pred, alpha = m(hist, eef3, g)
    assert pred.shape == (B, v4.P, v4.F, 2)
    assert alpha.shape == (B, v4.P, v4.F)

    # Force the gate fully closed -> predicted future == anchor repeated (cube static)
    with torch.no_grad():
        for layer in m.gate:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
        m.gate[-1].bias.fill_(-50.0)
    pred0, alpha0 = m(hist, eef3, g)
    anchor = hist[:, :, -1, :]
    assert alpha0.max().item() < 1e-3
    assert torch.allclose(pred0, anchor[:, :, None, :].expand(B, v4.P, v4.F, 2), atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q -k "thick or invariant"`
Expected: PASS already if Tasks 1+3 correct (forward wired). If it FAILS, the bug is in the gating algebra in `FlowWMThick.forward` — fix `anchor + alpha[...,None]*raw` so a closed gate yields exactly the anchor. (This task exists to lock the invariant; treat any failure as a real bug to fix here.)

- [ ] **Step 3: Write minimal implementation**

No new code if the invariant test passes. If it failed in Step 2, the only allowed fix is correcting the gating expression in `FlowWMThick.forward` (`return anchor[:, :, None, :] + alpha[..., None] * raw, alpha`) — do not add features.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q`
Expected: PASS (all tests so far).

- [ ] **Step 5: Commit**

```bash
git add train_flow_wm_scarcity_v4.py tests/flow_wm_v4/test_thick_helpers.py
git commit -m "test(flow-v4): lock gate-static invariant (alpha->0 keeps cube put)"
```

---

### Task 5: State-noise injection helper

**Files:**
- Modify: `train_flow_wm_scarcity_v4.py` (add `inject_state_noise`)
- Test: `tests/flow_wm_v4/test_thick_helpers.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_inject_state_noise_shape_and_determinism():
    hist = torch.zeros(6, v4.P, v4.K, 2)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    a = v4.inject_state_noise(hist, v4.NOISE_STD, g1)
    b = v4.inject_state_noise(hist, v4.NOISE_STD, g2)
    assert a.shape == hist.shape
    assert torch.allclose(a, b)                       # same seed -> same noise
    assert a.abs().mean() > 0                          # noise actually added
    # std in the right ballpark (normalized coords)
    assert 0.3 * v4.NOISE_STD < a.std().item() < 3 * v4.NOISE_STD
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q -k noise`
Expected: FAIL — `AttributeError: ... 'inject_state_noise'`.

- [ ] **Step 3: Write minimal implementation**

```python
def inject_state_noise(hist, std, generator):
    # additive Gaussian noise on history object points (normalized coords); cheap DAgger
    noise = torch.randn(hist.shape, generator=generator, device=hist.device) * std
    return hist + noise
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q -k noise`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_flow_wm_scarcity_v4.py tests/flow_wm_v4/test_thick_helpers.py
git commit -m "feat(flow-v4): state-noise injection (cheap DAgger) helper"
```

---

### Task 6: Multi-step consistency helper

**Files:**
- Modify: `train_flow_wm_scarcity_v4.py` (add `multi_step_consistency`)
- Test: `tests/flow_wm_v4/test_thick_helpers.py`

Mechanism: window-1 predicts frames `K..L-1` (12 steps). Take the first `K` predicted frames as a new history, run window-2, and supervise the overlap (window-2 frames that still have GT, i.e. frames `2K..L-1` = 8 steps) against ground truth. This feeds the model its OWN predictions as history — the exact compounding scenario.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_multi_step_consistency_shape():
    B = 4
    tr = torch.randn(B, v4.L, v4.P, 2)            # full clip (B,L,P,2)
    eef3 = torch.randn(B, v4.L, 3, 2)
    g = torch.rand(B, v4.L)
    m = v4.FlowWMThick(v4.P, thin=False)
    cons_pred, cons_tgt = v4.multi_step_consistency(m, tr, eef3, g)
    overlap = v4.L - 2 * v4.K                       # 16 - 8 = 8
    assert cons_pred.shape == (B, v4.P, overlap, 2)
    assert cons_tgt.shape == (B, v4.P, overlap, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q -k consist`
Expected: FAIL — `AttributeError: ... 'multi_step_consistency'`.

- [ ] **Step 3: Write minimal implementation**

```python
def multi_step_consistency(model, tr, eef3, g):
    # tr (B,L,P,2) -> (cons_pred, cons_tgt) each (B,P,overlap,2), overlap = L-2K
    B = tr.shape[0]
    hist1 = tr[:, :K].permute(0, 2, 1, 3)                       # (B,P,K,2)
    pred1, _ = model(hist1, eef3, g)                            # (B,P,F,2) frames K..L-1
    hist2 = pred1[:, :, :K, :]                                  # predicted frames K..2K-1
    pred2, _ = model(hist2, eef3, g)                            # frames 2K..2K+F-1 (own-feedback)
    overlap = L - 2 * K                                         # frames 2K..L-1 still have GT
    cons_pred = pred2[:, :, :overlap, :]
    cons_tgt = tr[:, 2 * K:].permute(0, 2, 1, 3)               # (B,P,overlap,2)
    return cons_pred, cons_tgt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q -k consist`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_flow_wm_scarcity_v4.py tests/flow_wm_v4/test_thick_helpers.py
git commit -m "feat(flow-v4): 2-window multi-step consistency (self-feedback) helper"
```

---

### Task 7: Rollout-drift (static-subset) metric

**Files:**
- Modify: `train_flow_wm_scarcity_v4.py` (add `rollout_drift_static`)
- Test: `tests/flow_wm_v4/test_thick_helpers.py`

Metric definition (primary Phase-1 metric): on held-out robot clips, restrict to the **near-static subset** where the GT cube centroid barely moves (GT centroid path length `< STATIC_TAU`), and report the **predicted** centroid path length there. A drifting model produces large predicted motion on clips where the cube truly didn't move — directly quantifying "cube 乱动". Returns `(drift, n_static)` in pixels (×224).

- [ ] **Step 1: Write the failing test** (append)

```python
def test_rollout_drift_static():
    B, Pn = 10, v4.P
    # GT: all static (centroid never moves); pred: half static, half drifting
    gt = torch.zeros(B, v4.F, Pn, 2)
    pred = torch.zeros(B, v4.F, Pn, 2)
    pred[5:, :, :, 0] = torch.linspace(0, 0.1, v4.F)[None, :, None]  # drift in last 5
    drift, n_static = v4.rollout_drift_static(pred, gt, tau=v4.STATIC_TAU)
    assert n_static == B                              # all GT clips are static
    assert drift > 0                                  # drifting preds counted
    # a fully-static pred -> ~0 drift
    drift0, _ = v4.rollout_drift_static(torch.zeros_like(gt), gt, tau=v4.STATIC_TAU)
    assert drift0 < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q -k drift`
Expected: FAIL — `AttributeError: ... 'rollout_drift_static'`.

- [ ] **Step 3: Write minimal implementation**

```python
def _centroid_path_len(seq):
    # seq (B,T,P,2) -> (B,) total centroid path length in normalized coords
    c = seq.mean(2)                                            # (B,T,2)
    return torch.linalg.norm(torch.diff(c, dim=1), dim=-1).sum(1)


def rollout_drift_static(pred, gt, tau):
    # pred,gt (B,F,P,2); restrict to GT-static clips, report mean predicted centroid
    # path length there, in pixels (x224). Returns (drift_px, n_static).
    gt_len = _centroid_path_len(gt)
    static = gt_len < tau
    n = int(static.sum().item())
    if n == 0:
        return 0.0, 0
    drift = (_centroid_path_len(pred)[static].mean().item()) * 224.0
    return drift, n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q`
Expected: PASS (full suite green).

- [ ] **Step 5: Commit**

```bash
git add train_flow_wm_scarcity_v4.py tests/flow_wm_v4/test_thick_helpers.py
git commit -m "feat(flow-v4): rollout-drift static-subset metric"
```

---

### Task 8: Training loop + ADE/FDE eval (thin & thick), wiring the new losses

**Files:**
- Modify: `train_flow_wm_scarcity_v4.py` (add `load`, `train_eval`, grasp-token normalization plumbing)
- Test: `tests/flow_wm_v4/test_thick_helpers.py` (one fast smoke test on a tiny synthetic loop)

Loss (thick): masked MSE (as v3, visibility-weighted) + `LAMBDA_GATE·mean|α|` (sparsity) + `LAMBDA_CONSIST·` masked-MSE on the multi-step overlap. Thin: masked MSE only (identical to v3). State-noise injected into the history before the forward, every train step.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_train_eval_smoke_tiny(tmp_path):
    # tiny synthetic dataset -> train_eval runs and returns finite ADE/FDE/drift
    torch.manual_seed(0)
    N = 40
    tr = torch.rand(N, v4.L, v4.P, 2) * 0.5 + 0.25
    vis = torch.ones(N, v4.L, v4.P)
    eef3 = torch.rand(N, v4.L, 3, 2)
    dom = torch.ones(N, dtype=torch.long)            # robot-only smoke
    gstats = v4.fit_grasp_stats(v4.grasp_openness(eef3), dom)
    out = v4.train_eval(tr, vis, eef3, dom, gstats,
                        train_idx=torch.arange(0, 30), test_idx=torch.arange(30, 40),
                        thin=False, seed=0, epochs=2)
    for k in ("ade", "fde", "drift"):
        assert k in out and out[k] == out[k]         # finite (not NaN)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q -k smoke`
Expected: FAIL — `AttributeError: ... 'train_eval'`.

- [ ] **Step 3: Write minimal implementation**

```python
def load():
    z = np.load(DS)
    return (torch.from_numpy(z["tracks"]).float(), torch.from_numpy(z["vis"]).float(),
            torch.from_numpy(z["eef3"]).float(), torch.from_numpy(z["domain"]).long(),
            torch.from_numpy(z["vid"]).long())


def _masked_mse(pred, fut, w):
    return ((pred - fut) ** 2 * w).sum() / (w.sum() + 1e-6)


def train_eval(tr, vis, eef3, dom, gstats, train_idx, test_idx, thin, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    g_all = normalize_grasp(grasp_openness(eef3), dom, gstats)          # (N,L)
    m = FlowWMThick(P, thin=thin).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed)
    ngen = torch.Generator(device=device).manual_seed(seed + 777)

    def batches(idx, train):
        idx = idx[torch.randperm(len(idx), generator=gen)] if train else idx
        for i in range(0, len(idx), BS):
            yield idx[i:i + BS]

    for ep in range(epochs):
        m.train()
        for b in batches(train_idx, True):
            h = tr[b, :K].permute(0, 2, 1, 3).to(device)               # (B,P,K,2)
            if not thin:
                h = inject_state_noise(h, NOISE_STD, ngen)
            fut = tr[b, K:].permute(0, 2, 1, 3).to(device)             # (B,P,F,2)
            ef = eef3[b].to(device); gg = g_all[b].to(device)
            hv = vis[b, :K].to(device); fv = vis[b, K:].to(device)
            w = (fv.permute(0, 2, 1) * hv[:, -1:].permute(0, 2, 1))[..., None]
            pred, alpha = m(h, ef, gg)
            loss = _masked_mse(pred, fut, w)
            if not thin:
                loss = loss + LAMBDA_GATE * alpha.abs().mean()
                cp, ct = multi_step_consistency(m, tr[b].to(device), ef, gg)
                # consistency overlap frames 2K..L-1 -> visibility for those frames
                wc = fv[:, K:].permute(0, 2, 1)[..., None]             # (B,P,overlap,1)
                loss = loss + LAMBDA_CONSIST * _masked_mse(cp, ct, wc)
            opt.zero_grad(); loss.backward(); opt.step()

    m.eval(); an = ad = fn = fd = 0.0; preds = []; futs = []
    with torch.no_grad():
        for b in batches(test_idx, False):
            h = tr[b, :K].permute(0, 2, 1, 3).to(device)
            fut = tr[b, K:].permute(0, 2, 1, 3).to(device)
            ef = eef3[b].to(device); gg = g_all[b].to(device)
            hv = vis[b, :K].to(device); fv = vis[b, K:].to(device)
            w = fv.permute(0, 2, 1) * hv[:, -1:].permute(0, 2, 1)
            pred, _ = m(h, ef, gg)
            err = torch.linalg.norm(pred - fut, dim=-1) * 224.0
            an += (err * w).sum().item(); ad += w.sum().item()
            fn += (err[..., -1] * w[..., -1]).sum().item(); fd += w[..., -1].sum().item()
            preds.append(pred.permute(0, 2, 1, 3).cpu()); futs.append(fut.permute(0, 2, 1, 3).cpu())
    pred_all = torch.cat(preds); fut_all = torch.cat(futs)              # (Ntest,F,P,2)
    drift, n_static = rollout_drift_static(pred_all, fut_all, STATIC_TAU)
    return {"ade": an / ad, "fde": fn / fd, "drift": drift, "n_static": n_static}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q -k smoke`
Expected: PASS (runs a 2-epoch tiny loop, returns finite metrics).

- [ ] **Step 5: Commit**

```bash
git add train_flow_wm_scarcity_v4.py tests/flow_wm_v4/test_thick_helpers.py
git commit -m "feat(flow-v4): train/eval loop with gate-L1 + consistency + noise + drift metric"
```

---

### Task 9: `[c,g]` admission probe (converged LogReg, held-out)

**Files:**
- Modify: `train_flow_wm_scarcity_v4.py` (add `cg_descriptor`, `run_probe_cg`, `--probe-cg` CLI)
- Test: `tests/flow_wm_v4/test_thick_helpers.py`

Per `probe_must_converge` memory: use sklearn `LogisticRegression(max_iter=2000)` with a held-out split on a large sample; the descriptor uses **normalized** grasp (what the model actually consumes). Target accuracy ≈ 0.5. The descriptor is the contact distances (object centroid ↔ both tips over the window) plus normalized grasp.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_cg_descriptor_shape():
    N = 7
    tr = torch.rand(N, v4.L, v4.P, 2)
    eef3 = torch.rand(N, v4.L, 3, 2)
    g = torch.rand(N, v4.L)
    desc = v4.cg_descriptor(tr, eef3, g)
    assert desc.shape == (N, v4.L * 2 + v4.L)        # 2 tip-distances per frame + grasp per frame
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q -k descriptor`
Expected: FAIL — `AttributeError: ... 'cg_descriptor'`.

- [ ] **Step 3: Write minimal implementation**

```python
def cg_descriptor(tracks, eef3, g_norm):
    # tracks (N,L,P,2), eef3 (N,L,3,2), g_norm (N,L) -> (N, L*2 + L)
    c = tracks.mean(2)                                          # (N,L,2) object centroid
    tips = eef3[:, :, 1:3, :]                                   # (N,L,2,2)
    d = torch.linalg.norm(c[:, :, None, :] - tips, dim=-1)      # (N,L,2)
    return torch.cat([d.reshape(len(d), -1), g_norm], -1)


def run_probe_cg(out_dir="outputs/flow_wm_v4/probe_cg"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    os.makedirs(out_dir, exist_ok=True)
    tr, vis, eef3, dom, vid = load()
    gstats = fit_grasp_stats(grasp_openness(eef3), dom)
    g_norm = normalize_grasp(grasp_openness(eef3), dom, gstats)
    X = cg_descriptor(tr, eef3, g_norm).numpy(); y = dom.numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    acc = clf.score(Xte, yte); base = max((yte == 0).mean(), (yte == 1).mean())
    msg = (f"[c,g] admission probe (converged LogReg, held-out)\n"
           f"  test acc = {acc:.3f}  (majority baseline = {base:.3f})\n"
           f"  verdict  = {'PASS ~0.5, admissible' if acc < base + 0.10 else 'FAIL leaks domain -> re-normalize/drop'}\n"
           f"  N={len(X)} dim={X.shape[1]}\n")
    print(msg, flush=True)
    open(os.path.join(out_dir, "summary.txt"), "w").write(msg)
    return acc
```

- [ ] **Step 4: Run test to verify it passes; then run the real probe**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q -k descriptor`
Expected: PASS.

Then run the real admission probe (gates the whole approach):
Run: `cd /scr2/yusenluo/interactive_world_sim && python train_flow_wm_scarcity_v4.py --probe-cg`
Expected: prints test acc + verdict and writes `outputs/flow_wm_v4/probe_cg/summary.txt`. **Decision gate:** if acc ≫ baseline (leaks domain), STOP and report — `[c,g]` must be re-normalized before continuing (per spec §3.3). Report the absolute path of `summary.txt`.

- [ ] **Step 5: Commit**

```bash
git add train_flow_wm_scarcity_v4.py tests/flow_wm_v4/test_thick_helpers.py
git commit -m "feat(flow-v4): [c,g] admission probe (converged LogReg, held-out)"
```

---

### Task 10: Phase-1 robot-only drift experiment (thin vs thick)

**Files:**
- Modify: `train_flow_wm_scarcity_v4.py` (add `run_phase1`, wire `--phase 1` CLI + `main`)

Phase-1 isolates the mechanism: train on robot-pool (vid≠12), test on held-out robot vid=12, compare **thin vs thick** on ADE/FDE and the rollout-drift metric. Success = thick drift materially below thin (cube stops moving when GT static), without ADE regression.

- [ ] **Step 1: Add `run_phase1` and CLI** (no unit test — this is an experiment runner; correctness covered by the smoke test in Task 8)

```python
def run_phase1(out_dir="outputs/flow_wm_v4/phase1_robot_drift"):
    os.makedirs(out_dir, exist_ok=True)
    tr, vis, eef3, dom, vid = load()
    gstats = fit_grasp_stats(grasp_openness(eef3), dom)
    test = torch.where((dom == 1) & (vid == TEST_ROBOT_VID))[0]
    rob = torch.where((dom == 1) & (vid != TEST_ROBOT_VID))[0]
    lines = [f"Phase-1 robot-only | train={len(rob)} test(held-out vid={TEST_ROBOT_VID})={len(test)}",
             f"component versions: model=FlowWMThick (v4), data=flow_ds_v3.npz",
             f"{'variant':>6} | {'ADE':>7} | {'FDE':>7} | {'drift_px':>8} | {'n_static':>8}"]
    for thin in (True, False):
        accs = [train_eval(tr, vis, eef3, dom, gstats, rob, test, thin=thin, seed=s)
                for s in range(SEEDS)]
        ade = np.mean([a["ade"] for a in accs]); fde = np.mean([a["fde"] for a in accs])
        drift = np.mean([a["drift"] for a in accs]); ns = accs[0]["n_static"]
        lines.append(f"{'thin' if thin else 'thick':>6} | {ade:7.2f} | {fde:7.2f} | {drift:8.3f} | {ns:8d}")
    msg = "\n".join(lines) + "\n"
    print(msg, flush=True)
    open(os.path.join(out_dir, "summary.txt"), "w").write(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-cg", action="store_true")
    ap.add_argument("--phase", type=int, default=0)
    a = ap.parse_args()
    if a.probe_cg:
        run_probe_cg()
    elif a.phase == 1:
        run_phase1()
    elif a.phase == 2:
        run_phase2()
    else:
        print("specify --probe-cg | --phase 1 | --phase 2", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite (regression guard)**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q`
Expected: PASS (all unit + smoke tests).

- [ ] **Step 3: Run Phase-1 experiment**

Run: `cd /scr2/yusenluo/interactive_world_sim && python train_flow_wm_scarcity_v4.py --phase 1`
Expected: writes `outputs/flow_wm_v4/phase1_robot_drift/summary.txt` with thin vs thick ADE/FDE/drift. **Read the result**: success = thick `drift_px` materially below thin with ADE not worse. Report the absolute path and the verdict to the user (do not just narrate — the summary file is the record).

- [ ] **Step 4: Commit**

```bash
git add train_flow_wm_scarcity_v4.py
git commit -m "feat(flow-v4): Phase-1 robot-only drift experiment (thin vs thick)"
```

- [ ] **Step 5: Decision checkpoint**

Per spec §6.2: only proceed to Task 11 (cross-domain) if Phase-1 shows the gate+anti-drift actually reduces drift. If not, return to systematic-debugging on the gate/consistency before Phase-2. Report Phase-1 outcome and get the user's go/no-go.

---

### Task 11: Phase-2 cross-domain scarcity (human-helps preserved under thick latent)

**Files:**
- Modify: `train_flow_wm_scarcity_v4.py` (add `run_phase2`)

Phase-2 runs the exact v3 scarcity protocol with the **thick** model: robot-only vs robot+human across `N_LIST × SEEDS`, held-out robot vid=12, reporting ADE/FDE Δ. Confirms thickening did not reopen the embodiment gap (human still helps at scarce N).

- [ ] **Step 1: Add `run_phase2`**

```python
def run_phase2(out_dir="outputs/flow_wm_v4/phase2_crossdomain"):
    os.makedirs(out_dir, exist_ok=True)
    tr, vis, eef3, dom, vid = load()
    gstats = fit_grasp_stats(grasp_openness(eef3), dom)
    test = torch.where((dom == 1) & (vid == TEST_ROBOT_VID))[0]
    rob = torch.where((dom == 1) & (vid != TEST_ROBOT_VID))[0]
    hum = torch.where(dom == 0)[0]
    lines = [f"Phase-2 cross-domain (thick) | robot-pool={len(rob)} human={len(hum)} test={len(test)}",
             f"component versions: model=FlowWMThick (v4, thin=False), data=flow_ds_v3.npz",
             f"{'N_rob':>6} | {'robot-only ADE':>15} | {'robot+human ADE':>15} | {'ADE d(+=help)':>13} | {'FDE d':>8}"]
    for N in N_LIST:
        ro, rh = [], []
        for s in range(SEEDS):
            gg = np.random.default_rng(100 + s)
            sub = rob[torch.from_numpy(gg.choice(len(rob), min(N, len(rob)), replace=False))]
            ro.append(train_eval(tr, vis, eef3, dom, gstats, sub, test, thin=False, seed=s))
            rh.append(train_eval(tr, vis, eef3, dom, gstats, torch.cat([sub, hum]), test, thin=False, seed=s))
        a_ro = np.array([x["ade"] for x in ro]); a_rh = np.array([x["ade"] for x in rh])
        f_ro = np.array([x["fde"] for x in ro]); f_rh = np.array([x["fde"] for x in rh])
        lines.append(f"{N:>6} | {a_ro.mean():6.2f}+-{a_ro.std():4.2f}    | "
                     f"{a_rh.mean():6.2f}+-{a_rh.std():4.2f}    | "
                     f"{a_ro.mean()-a_rh.mean():+7.2f}      | {f_ro.mean()-f_rh.mean():+7.2f}")
    msg = "\n".join(lines) + "\n(v3 thin baseline for reference: 50->+6.76, 100->+6.70, 200->+3.50)\n"
    print(msg, flush=True)
    open(os.path.join(out_dir, "summary.txt"), "w").write(msg)
```

- [ ] **Step 2: Run the full test suite (regression guard)**

Run: `cd /scr2/yusenluo/interactive_world_sim && python -m pytest tests/flow_wm_v4/test_thick_helpers.py -q`
Expected: PASS.

- [ ] **Step 3: Run Phase-2 experiment**

Run: `cd /scr2/yusenluo/interactive_world_sim && python train_flow_wm_scarcity_v4.py --phase 2`
Expected: writes `outputs/flow_wm_v4/phase2_crossdomain/summary.txt`. Success = `ADE Δ ≥ 0` (human still helps) at scarce N, comparable to or better than the v3 thin baseline. Report absolute path + verdict.

- [ ] **Step 4: Commit**

```bash
git add train_flow_wm_scarcity_v4.py
git commit -m "feat(flow-v4): Phase-2 cross-domain scarcity under thick latent"
```

- [ ] **Step 5: Update report + memory**

Append a thick-latent section to `FLOW_WM_REPORT.md` (drift fixed? human-helps preserved?), and update memory `reference_flow_wm_three_papers_insight` / add a project memory recording the thick-latent result. Commit only your own files, no `Co-Authored-By`.

---

## Self-Review

**1. Spec coverage:**
- §3.1 contact feature → Task 3. §3.2 grasp openness + per-domain norm → Task 2. §3.3 admission probe → Task 9. §4 contact-gated motion + L1 sparsity → Tasks 4, 8. §5 state-noise injection → Task 5/8; multi-step consistency → Task 6/8. §6.1 one script + `--thin` → Task 1. §6.2 Phase-1 then Phase-2 → Tasks 10, 11. §6.3 metrics (drift/ADE/FDE) → Tasks 7, 8, 10, 11. §6.4 hygiene (per-exp dir, summary.txt, absolute paths, AV1 N/A here, no Co-Authored-By) → Tasks 9–11. §7 success criteria → checked in Tasks 9/10/11 verdict steps. All covered.

**2. Placeholder scan:** Every code step has complete code. The only `raise NotImplementedError` is a deliberate Task-1 placeholder immediately replaced in Task 3 (and its test in Task 3 is the failing test that drives it). No TBD/TODO/"handle edge cases".

**3. Type consistency:** `FlowWMThick.forward` returns `(pred, alpha)` everywhere (thin → `alpha=None`); all callers (`train_eval`, `multi_step_consistency`, tests) unpack two values. `contact_gate_features(anchor, eef3, g) → (B,P,F,3)` consumed by `gate: Linear(3→32→1)`. `grasp_openness → (...,L)`, `normalize_grasp` preserves shape, `g` fed as `(B,L)` to forward and `gctx: Linear(L→Dm)`. `train_eval` returns dict with keys `ade/fde/drift/n_static` used identically in Tasks 10/11. `rollout_drift_static(pred,gt,tau)` takes `(B,F,P,2)`; `train_eval` permutes preds/futs to `(Ntest,F,P,2)` before calling. Consistent.
