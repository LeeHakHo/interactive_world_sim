# Phase 0 — Latent Decomposition (IWS, MVP)

**Date**: 2026-05-24
**Source plan**: `PHASE0_IWS_PLAN_v3.md`
**Codebase entrypoint**: `interactive_world_sim/`
**Branch**: `phantom_dynamo`
**Scope**: Step 1 (training infra) + Step 2 (diagnostics) + Step 3 (mixing-ratio sweep)
**Out of scope (decided)**: Phase 4 SE(3) branch, CLUB swap, L_pair fallback, Stage-2 architecture changes, real-robot eval, new metrics.

---

## 0. Thesis (summarised from v3 plan §"Core thesis")

Stage 1 must decompose the per-image latent into `z = (z_task, z_emb)` such that
`z_task` is embodiment-agnostic (human and robot samples share a subspace) and
`z_emb` is embodiment-specific. If this holds, Stage 2 dynamics — which already
consumes only the spatial latent — should be trainable on mixed
human/robot data, making human data substitute for robot data.

Phase 0 verifies two things in sequence:

1. **Decomposition really happens** — probes + correlation heatmap (Step 2).
2. **Decomposition is useful** — train Stage 2 on R/H mixes, eval on held-out
   robot, compare gap (Step 3).

---

## 1. Architectural decisions (locked)

| # | Decision | Why |
|---|---|---|
| D1 | Phase 0 v3 SplitEncoder approach **replaces** the existing UniT `align_module` direction; UniT module is kept on disk but no longer developed | Brainstorm Q1 — single direction, focused compute |
| D2 | SplitEncoder channel-splits the **ViT-S spatial feature map** `(B, 384, 16, 16)` along the embed-dim axis into `z_task (B, 288, 16, 16)` and `z_emb (B, 96, 16, 16)`. `spatial_proj` consumes `cat([z_task, z_emb], dim=1)` (bit-identical to current pipeline). `cls_token` is untouched. Note: `(B, D, H, W)` ↔ `(B, P=H·W, D)` are equivalent via rearrange — we use the spatial form since that is what `ViTSpatialEncoder.forward` actually returns | Brainstorm Q2 — closest to v3 plan §1.1 and preserves baseline |
| D3 | Stage 1 trained **from scratch** with the new heads — not warm-started from Hyeonhoo's H+R ckpt | Brainstorm Q3 — clean comparison; existing ckpt was the failure-mode baseline |
| D4 | One spec covers Step 1+2+3; implementation will be committed in phases (Step 1 first) | Brainstorm Q4 |
| D5 | "Current baseline" reference for the 5%-tolerance sanity gate = **the baseline run the user is currently re-training** (path to be filled in by the user once that run finishes); we do not re-train another | Brainstorm Q5 |
| D6 | Step 3 R/H mixing ratios are **deliberately left open** in this spec; will be finalised before Step 3 launches based on Step 1+2 results and re-counted in episodes/frames against the actual data budget (12 robot + 3 human episodes) | Brainstorm Q6 |
| D7 | Code organisation pattern: **cfg flag inside `LatentWorldModel`** (mirrors existing `dynamo_ssl` pattern), no subclass | Brainstorm "Approach A" |
| D8 | All Phase 0 new code lives under `interactive_world_sim/algorithms/latent_decompose/` (sibling to `latent_dynamics/`, `align_module/`) | Naming clarification mid-brainstorm |

---

## 2. File tree

```
interactive_world_sim/algorithms/latent_decompose/        (new)
├── __init__.py
├── split_encoder.py
├── domain_heads.py
├── align_losses.py
└── diagnostics/
    ├── __init__.py
    ├── linear_probe.py
    ├── correlation_heatmap.py
    └── aggregate_step3.py

tests/algorithms/latent_decompose/                        (new)
├── test_split_encoder_identity.py
├── test_grad_reverse.py
├── test_pooled_classifier.py
├── test_lambda_schedule.py
└── test_dataset_domain_label.py
tests/integration/
└── test_phase0_smoke.py

interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py   (modify)
configurations/algorithm/latent_world_model.yaml                          (modify)
interactive_world_sim/datasets/latent_dynamics/play_eef_dataset.py        (modify)
configurations/dataset/play_mixed_eef.yaml                                (modify)
configurations/experiment/phase0_step1.yaml                               (new)
configurations/experiment/phase0_step3_<config>.yaml                      (new × 6, after Step 2)
```

---

## 3. Configuration additions

### 3.1 `configurations/algorithm/latent_world_model.yaml` — new section

```yaml
latent_decompose:
  enabled: false               # false → pipeline is bit-identical to baseline
  d_task: 288
  d_emb: 96
  lambda_dom: 0.1
  lambda_adv_schedule:
    type: linear_ramp
    start_step: 2000           # let pure recon stabilise first
    end_step: 10000
    start_value: 0.0
    end_value: 0.3
  lr_classifiers: 3.0e-4       # classifiers need to keep up with encoder
```

### 3.2 `configurations/dataset/play_mixed_eef.yaml` — modify

```yaml
# Per-domain episode subset for Step 3 sweep. null = use all episodes.
robot:
  episode_subset: null
human:
  episode_subset: null
# Phase 0 step 1 default override: balanced batch (50/50)
# sample_ratio: 0.5      # users set this in the experiment cfg, not here
```

### 3.3 `configurations/experiment/phase0_step1.yaml` (new)

Composes `algorithm=latent_world_model dataset=play_mixed_eef` with
`algorithm.latent_decompose.enabled=true` and `dataset.sample_ratio=0.5`.
All other Stage 1 hyperparameters identical to current baseline.

---

## 4. Component contracts

### 4.1 `latent_decompose/split_encoder.py`

```python
class SplitEncoder(nn.Module):
    """
    Channel-split wrapper around the existing ViTSpatialEncoder.

    Args:
        base_vit: ViTSpatialEncoder instance (already constructed by
                  LatentWorldModel._build_model).
        d_task:   first slice width along the channel (embed_dim) axis.
        d_emb:    second slice width. d_task + d_emb MUST == base_vit.embed_dim.

    forward(view_obs) → (z_task, z_emb, cls_token)
        view_obs:  (B, 3, H, W)
        z_task:    (B, d_task, grid_h, grid_w)
        z_emb:     (B, d_emb,  grid_h, grid_w)
        cls_token: (B, embed_dim)               — passed through unchanged

    @staticmethod
    concat(z_task, z_emb) → (B, embed_dim, grid_h, grid_w) torch.cat on dim=1.
    """
```

**Invariant**: when initialised with the same `base_vit`,
`SplitEncoder.concat(*SplitEncoder(view_obs)[:2])` is `allclose(rtol=1e-6)` to
`base_vit(view_obs)[0]` (the spatial_feat).

### 4.2 `latent_decompose/domain_heads.py`

```python
def grad_reverse(x, lambda_: float = 1.0): ...
    # autograd Function: forward identity, backward returns -lambda * grad.

class PooledClassifier(nn.Module):
    """spatial mean-pool → 2-layer MLP (hidden=128, GELU) → 2 logits.
       Label convention: human=0, robot=1.
       forward(z: (B, D, H, W)) → logits (B, 2). Pool reduces dims (2, 3).
    """
```

### 4.3 `latent_decompose/align_losses.py`

```python
def compute_L_dom(z_emb, domain_label, clf_emb):
    """CE on z_emb's domain prediction. Forces z_emb to encode embodiment."""
    return F.cross_entropy(clf_emb(z_emb), domain_label)

def compute_L_adv(z_task, domain_label, clf_adv, lambda_adv):
    """Gradient reversal on z_task → CE. Pushes encoder gradient away from
       domain-discriminability while letting the classifier train normally."""
    z_rev = grad_reverse(z_task, lambda_adv)
    return F.cross_entropy(clf_adv(z_rev), domain_label)
```

### 4.4 `LatentWorldModel` modifications

In `__init__`:
- Read `cfg.latent_decompose` once; cache `self.use_latent_decompose: bool`.

In `_build_model` (after `self.vit_encoder` constructed; gated on Stage 1):
```python
if self.use_latent_decompose and self.training_stage == 1:
    ld = self.cfg.latent_decompose
    self.split_encoder = SplitEncoder(self.vit_encoder, ld.d_task, ld.d_emb)
    self.clf_emb = PooledClassifier(ld.d_emb)
    self.clf_adv = PooledClassifier(ld.d_task)
```

In `encoder_forward` (avoid hidden state — return extras through the call):
- Add a `return_split: bool = False` kwarg. When `False`, signature/behaviour are unchanged (Stage 2/3 callers untouched).
- When `True` (Stage 1 with `use_latent_decompose`), the function additionally returns `(z_task_list, z_emb_list)`, each a list of `V` tensors of shape `(B*T, d_task / d_emb, grid_h, grid_w)` in **view-outer order** (view 0 first, view 1 second, …). Per-view tensors retain the `(b·T + t)` flattening already used by IWS (b slow, t fast).
- The `spatial_feat` fed into `spatial_proj` is always `cat([z_task, z_emb], dim=1)` when split is on, i.e. bit-identical to ViT output.

In `training_step` Stage 1 branch (after `rec_loss` computed):
```python
if self.use_latent_decompose:
    dl = batch["domain_label"]                          # (B,) int 0/1
    B, T = batch_size, n_frames                          # T = horizon
    V = num_views

    # cat order: view-outer, then per-view (b slow, t fast)
    z_task = torch.cat(z_task_list, dim=0)              # (V*B*T, d_task, gh, gw)
    z_emb  = torch.cat(z_emb_list,  dim=0)              # (V*B*T, d_emb,  gh, gw)

    # match the cat order: (v0:b0t0,b0t1,...,bNtN | v1:b0t0,...,bNtN | ...)
    # i.e. v slowest, b middle, t fastest
    dl_rep = dl.repeat_interleave(T).repeat(V)          # (V*B*T,)

    L_dom = compute_L_dom(z_emb, dl_rep, self.clf_emb)
    lam   = self._lambda_adv_now()
    L_adv = compute_L_adv(z_task, dl_rep, self.clf_adv, lam)
    loss = rec_loss + self.cfg.latent_decompose.lambda_dom * L_dom + L_adv
    self.log_dict({"training/L_dom": L_dom, "training/L_adv": L_adv,
                   "training/lambda_adv": lam})
```

**Order convention is load-bearing** — the cat order in `encoder_forward` and
the `repeat_interleave(T).repeat(V)` order in `training_step` must stay in
lockstep. The integration smoke test (G1) is the canary if either drifts.

In `configure_optimizers` Stage 1, when `use_latent_decompose`:
- New param group `{params: list(self.clf_emb.parameters()) +
  list(self.clf_adv.parameters()), lr: self.cfg.latent_decompose.lr_classifiers}`
  appended after the existing encoder/decoder groups.

`_lambda_adv_now(self)`: linear ramp by `global_step` per cfg
`lambda_adv_schedule`.

### 4.5 `MixedPlayEEFDataset` modification

In `__getitem__` after the existing `item["embodiment"] = "robot"|"human"`:
```python
item["domain_label"] = torch.tensor(
    1 if item["embodiment"] == "robot" else 0,
    dtype=torch.long,
)
```
PyTorch default collate will stack these to `(B,)`. No collate fn change needed.

---

## 5. Data flow (Stage 1, `latent_decompose.enabled=true`)

```
batch  ─ obs.camera_0_color: (B, T, 3, H, W)
       ─ action:             (B, T, 8)
       ─ domain_label:       (B,) int

per-view per-frame view_obs (B*T, 3, H, W)
       ↓ split_encoder
       ├── z_task (B*T, 288, 16, 16) ─┐
       ├── z_emb  (B*T,  96, 16, 16) ─┤
       └── cls_token (B*T, 384)       │
                                       │
       cat([z_task,z_emb], dim=1) → spatial_feat (B*T, 384, 16, 16)
                                       │
       spatial_proj(spatial_feat, cls_token) → (B*T, c_per_v, 32, 32)
                                    │
       per-view norm → decoder(diffusion) → rec_loss

       z_task ──┐
       z_emb  ──┤── compute_L_dom (CE on z_emb)
                ├── compute_L_adv (GRL on z_task + CE)
                ↓
       loss = rec_loss + λ_dom · L_dom + L_adv
```

Stage 2 and Stage 3 paths **do not** touch `latent_decompose` — they read
`z = encoder_forward(obs)` exactly as today. Decomposition is purely a Stage 1
encoder-shaping mechanism.

---

## 6. Step 2 — diagnostics

### 6.1 Linear probe — `latent_decompose/diagnostics/linear_probe.py`

CLI:
```
python -m interactive_world_sim.algorithms.latent_decompose.diagnostics.linear_probe \
    --ckpt <path> --target {z_task|z_emb} [--device cuda] [--seeds 3]
```

Flow:
1. Load `LatentWorldModel` from `<ckpt>` with full cfg from the run's
   `.hydra/config.yaml`. Freeze all encoder params.
2. Build mixed val loader; collect `(z_target.mean(dim=1), domain_label)` —
   over multi-view, channel-concat the per-view pooled vectors.
3. For each seed: 8/2 train/val split, fresh 2-layer MLP (hidden=128) for 10
   epochs, AdamW lr=1e-3. Record val accuracy.
4. Print mean ± std and pass/fail vs the threshold for `--target`.

Pass thresholds (v3 plan §2.1):
- `--target z_task` → acc ∈ [50%, 70%]
- `--target z_emb`  → acc ≥ 95%

### 6.2 Correlation heatmap — `latent_decompose/diagnostics/correlation_heatmap.py`

CLI:
```
python -m ...correlation_heatmap --ckpt <p> --save <png> [--n 200]
```

Flow:
1. Load + freeze encoder as above.
2. Collect `(z_task.mean(1), z_emb.mean(1))` for `n` samples.
3. PCA `z_task` → top-15 PCs, `z_emb` → top-10 PCs.
4. Pairwise abs Pearson correlation → 25×25 matrix; render heatmap with
   block boundaries marked. Print off-diagonal (task×emb) block max.

Pass threshold (v3 plan §2.2): off-diagonal block max < 0.2.

### 6.3 λ-tuning iteration policy

v3 plan §2.4 lookup table is reproduced in the README of `diagnostics/`
verbatim. **No automation.** Cap 3 iterations; if still failing, escalate to
Step 2 fallbacks (L_pair / CLUB — both explicitly out of MVP scope and to be
re-spec'd then).

---

## 7. Step 3 — fungibility test

### 7.1 Sweep design (placeholder ratios)

Per D6, the precise R/H mixes are decided once Step 1+2 pass. The structure
is fixed:

- Stage 1: one run on full 12R+3H (Option A in v3 plan §3.2).
- Stage 2 sweep: 5 mixing configs + 1 max-robot reference, 3 seeds each → 18
  Stage 2 runs.
- Eval split: robot held-out only (this is the question the test answers).
- Per-config aggregation: mean ± std over 3 seeds.

Each config is expressed as a Hydra override on `dataset.robot.episode_subset`
and `dataset.human.episode_subset` — no new code path.

### 7.2 `latent_decompose/diagnostics/aggregate_step3.py`

Reads the 18 run dirs (or wandb run IDs), pulls FVD and PSNR per config,
prints a mean±std table, computes:

```
gap = (M_{0.5R_1.0H} − M_{1.5R_0H}) / M_{1.5R_0H}
```

on the chosen primary metric (FVD preferred, PSNR secondary) and emits the
Go/No-Go decision per v3 plan §3.4:
- `gap < 20%` → paper-strong; scale up
- `20% ≤ gap < 50%` → paper-weak; consider Phase 4 anchor
- `gap ≥ 50%` → diagnose

Required sanity (v3 plan §3.4): `0R_1.0H` must be **worse** than `1.5R_0H` on
robot eval. If equal, the eval is not discriminative — flag and abort the Go
decision.

---

## 8. Sanity gates (collected)

| ID | Phase | Gate | Check |
|---|---|---|---|
| G1 | Step 1 unit | `enabled=true, lambda_dom=0, lambda_adv.end=0` → loss series **allclose 1e-6** to `enabled=false` for 100 steps | `tests/integration/test_phase0_smoke.py` (mode: `identity_under_zero_loss`) |
| G2 | Step 1 unit | `SplitEncoder.concat(SplitEncoder(x)[:2])` allclose `vit(x)[0]` rtol=1e-6 | `tests/algorithms/latent_decompose/test_split_encoder_identity.py` |
| G3 | Step 1 train | IWS-native PSNR / FVD on robot val, vs the user's baseline run: PSNR drop ≤ 5%, FVD jump ≤ 20% | reuse `validation_step` |
| G4 | Step 1 train | `L_dom` trends down; `L_adv` allowed to oscillate | wandb visual |
| G5 | Step 2 | `linear_probe --target z_task` acc ∈ [50%, 70%] | script exit code |
| G6 | Step 2 | `linear_probe --target z_emb` acc ≥ 95% | script exit code |
| G7 | Step 2 | corr off-diagonal (task×emb) max < 0.2 | script stdout |
| G8 | Step 3 | `0R_1.0H` worse than `1.5R_0H` on chosen primary metric | `aggregate_step3.py` exits non-zero otherwise |
| G9 | Step 3 Go | `gap(0.5R_1.0H, 1.5R_0H) < 20%` on primary metric | `aggregate_step3.py` |

---

## 9. Testing plan

### 9.1 Unit (`pytest tests/algorithms/latent_decompose/`)

| Test | What it covers |
|---|---|
| `test_split_encoder_identity.py` | G2 + `d_task+d_emb==embed_dim` assert + dtype/shape contract |
| `test_grad_reverse.py` | forward identity; backward grad = -λ · upstream |
| `test_pooled_classifier.py` | input (B,D,H,W) → (B,2); gradient flows; mean-pool over (2,3) |
| `test_lambda_schedule.py` | ramp values at `step ≤ start`, `step ≥ end`, midpoint |
| `test_dataset_domain_label.py` | `MixedPlayEEFDataset[i]` has `domain_label` int tensor matching `embodiment` |

### 9.2 Integration (`pytest tests/integration/test_phase0_smoke.py`)

Two modes (parametrised):

1. `mode=smoke`: `latent_decompose.enabled=true`, 10 steps, asserts (a)
   training loss finite, (b) `rec_loss / L_dom / L_adv` all logged, (c) all
   three param groups (encoder, decoder, classifiers) have non-zero grad on
   step 1.
2. `mode=identity_under_zero_loss` (G1): freeze seed; run 100 steps with
   `enabled=false` vs `enabled=true, lambda_dom=0, lambda_adv.end=0`; assert
   `training/rec_loss` traces are allclose(rtol=1e-6).

Both modes use the smallest available dataset config to stay under 60 s.

### 9.3 Functional

Owned by the user, off-CI: once the baseline rerun completes, compare the
Phase 0 Step 1 run's wandb metrics (PSNR, FVD on robot val) against the
baseline run side-by-side to evaluate G3.

---

## 10. Risks & decisions deferred

| Risk | Mitigation |
|---|---|
| z_emb=96 too narrow for embodiment info | Step 2 probe G6 catches this; v3 §2.4 says double `lambda_dom` |
| L_adv diverges (NaN, explode) | v3 §2.4 says drop `lr_classifiers` to 1e-4 then CLUB. We add NaN guard in `_lambda_adv_now` is unnecessary; existing IWS NaN guard in `optimizer_step` covers gradient NaNs |
| `domain_label` semantics drift in collate | Unit test `test_dataset_domain_label.py` + integration `mode=smoke` both assert |
| Step 3 episode subsets too small (some configs have ≤1 ep) | Decided in D6: defer to Step 3 launch; flag with abort if any subset would produce 0 training windows |
| `cls_token` not split — does it leak embodiment? | Acceptable for MVP. If Step 2 G5 fails persistently, revisit. Not in MVP scope. |

---

## 11. Out of scope (explicit; mirror v3 plan §"OUT OF SCOPE")

- Phase 4 SE(3) task encoder branch
- CLUB replacement for L_adv (Step 2 escalation only)
- L_pair fallback (Step 2 escalation only)
- Stage 2 architectural changes
- Real-robot eval inside Phase 0
- Any new metric beyond IWS-native (PSNR / FVD)

---

## 12. Delivery sequence

1. **Step 1 implementation** (this spec): file tree §2 + components §4 + cfg §3 + tests §9.1, §9.2. Land in one or two PRs.
2. **Step 1 training run** with user's baseline as G3 reference once that completes.
3. **Step 2 diagnostics scripts** (§6). Run; iterate λ ≤ 3 times if needed.
4. **Step 3 sweep design finalisation** — fill in mixing ratios (D6), then `phase0_step3_*.yaml` configs + `aggregate_step3.py`, launch 18 runs.
5. **Decision gate** per G8 + G9.
