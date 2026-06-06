# Thicken Flow Dynamic Latent — Design Spec (Approach B)

> Date: 2026-06-06 · Branch: `phantom_dynamo` · Author: AndreasL9z (+ Claude)
> Status: **Design approved**, ready for implementation plan.
> Context docs: `FLOW_WM_REPORT.md`, `CROSS_EMBODIMENT_WM_ATTEMPTS.md`,
> memory `reference_flow_wm_three_papers_insight` (AMPLIFY / X-Diffusion / Point Policy / HumanEgo).

---

## 0. Problem (one line)

The flow world model (② in Architecture A) hallucinates in autoregressive rollout —
**the cube drifts / moves on its own ("cube 乱动")** even when nothing touches it. The
current dynamic latent (48 object 2D points + 3 EEF points) is **too thin**: the model
gets EEF as a condition but has **no signal or inductive bias that "the cube only moves
when the gripper/hand touches it"**, and it is not robust to feeding its own predictions
back during rollout.

## 1. Goal

Thicken the **shared** dynamic latent with **near-domain-invariant physical signals**
and add **anti-drift training**, so that:

1. The cube stays put when not in contact (kills the dominant "untouched drift").
2. Long autoregressive rollout is stable (compounding controlled).
3. The cross-embodiment property survives: anything added to the shared latent must
   itself be **probe ≈ 0.5** (verified with a converged LogReg, not an undertrained MLP),
   or it is NOT admitted — adding appearance/DINO back would reopen the 582× wall.

This is the route validated by AMPLIFY/X-Diffusion/Point Policy (low-dim motion/point
representation bypasses the embodiment gap) and by HumanEgo (contact/grasp as a
first-class latent citizen + latent-consistency + state-noise injection against drift).

## 2. Non-goals (explicitly deferred to a later iteration = Approach C)

- Point densification toward an AMPLIFY 400-point grid.
- FSQ discrete motion tokens.
- Object 2D orientation estimation (noisy from 2D points).
- Wrist-camera signals — **wrist cam is robot-only**, so it CANNOT enter the shared
  latent (would break cross-domain). It may later serve the robot-domain ③ renderer
  only; out of scope here.
- LPIPS / appearance thickening on ③ (renderer side).

## 3. The two shared-latent physical channels (both derivable on-the-fly)

Crucially, **both signals are computable from the existing `flow_ds_v3.npz` arrays** —
no CoTracker3 re-run, no dataset regeneration. `eef3` is `(N, L=16, 3, 2)` and already
contains the **future 12 frames of EEF** (it is the action conditioning), so per-future-
step contact can be computed against the known future EEF.

### 3.1 Contact / proximity feature `c`
Normalized 224-coord distances between object points (or the object centroid) and the
3 EEF points, at the anchor frame AND at each future step (future EEF is known). Same
coordinate scale as flow. Captures "is the agent near/touching the object".

### 3.2 Grasp openness `g`
`g = ‖eef3[:, 1] - eef3[:, 2]‖` — **structurally identical across domains**:
- human: `eef3 = [wrist(0), thumb_tip(4), index_tip(8)]` → thumb–index pinch distance.
- robot: `eef3 = [base, jaw+, jaw−]` → jaw separation = gripper width.

Normalize **per-domain to [0, 1]** (scale alignment only — this is preprocessing, it does
NOT inject domain-discriminative information; both end as "how open").

### 3.3 Admission gate (hard requirement)
Before wiring `[c, g]` into the model, run a **converged LogReg** (sklearn, large sample,
held-out split — per `probe_must_converge` memory) on `[c, g]` to confirm probe ≈ 0.5.
If it separates the domains, the offending channel is re-normalized or dropped. No channel
enters the shared latent until it passes this body-check.

## 4. Contact-gated motion head (treats root cause #1: untouched drift)

Per-step predicted displacement becomes:

```
Δ = α ⊙ raw          # α ∈ [0,1] per object point, raw = current head output
α = σ(MLP(c, g))     # gate driven only by current contact/grasp (no future leakage)
loss += λ · ‖α‖₁     # sparsity prior
```

**Supervision = pure sparsity (L1), no pseudo-labels** (decided). The model learns
`α ≈ 0` when the contact feature says "far", because that minimizes both prediction error
(the object truly didn't move when far) and the L1 term. No future leakage; the gate only
reads the current contact/grasp.

## 5. Anti-drift training (treats root cause #2: compounding — this is what B adds over A)

- **State-noise injection**: during training, add small noise to the history object points
  then re-anchor — a cheap DAgger approximation so the model is robust to its own errors.
- **Multi-step consistency**: chain two windows in training (predict F → take the last K of
  the prediction as new history → predict again → supervise the second window against GT),
  forcing stability under self-feedback. Complements the existing scheduled sampling
  (HumanEgo latent-consistency idea).

## 6. Implementation & evaluation

### 6.1 One script, flag-toggled ablation (no script sprawl)
- New `train_flow_wm_scarcity_v4.py` with a `--thin` flag that **exactly reproduces v3**
  (single source of truth; thin-vs-thick ablation is a flag, not a fork). `v3` stays as the
  documented baseline numbers. Extends the cleanest existing impl per
  `unify_code_no_script_sprawl`.

### 6.2 Phased validation (decided)
- **Phase 1 — robot-only drift first.** Validate that contact-gate + anti-drift actually
  stops the cube drifting, measured by a new **rollout-drift** metric (cube-centroid drift
  under long chained autoregressive rollout). Establish the mechanism works before touching
  cross-domain — keeps attribution clean.
- **Phase 2 — cross-domain.** Only after Phase 1, run the full v3 scarcity protocol
  (robot-only vs robot+human × `N_LIST=[1400,400,200,100,50]` × `SEEDS=5`, held-out
  robot `vid=12`), reporting ADE/FDE Δ, to confirm the thick latent still preserves
  human-helps.

### 6.3 Metrics
- **rollout-drift** (new, primary for Phase 1): cube-centroid displacement during a long
  chained rollout vs ground truth — directly quantifies "乱动".
- ADE / FDE (per v3) — open-loop accuracy.
- Phase 2: robot-only vs robot+human ADE/FDE Δ (human-helps preserved?).

### 6.4 Hygiene (per memory)
- Per-experiment output directory; write a `summary.txt` (don't narrate results verbally).
- Report all figures/results with **absolute paths**.
- Declare component versions in any closed-loop / rollout test.
- Robot video is AV1 → PyAV, not cv2 (already handled in `gen_flow_dataset_v3.decode`).
- Work in place; commit only my own files; **no `Co-Authored-By`**.

## 7. Success criteria

- Phase 1: rollout-drift on robot held-out drops materially vs the `--thin` baseline
  (cube stays put when not in contact; centroid drift under chained rollout shrinks).
- `[c, g]` admission probe ≈ 0.5 (converged LogReg).
- Phase 2: robot+human ADE Δ stays ≥ 0 (human still helps) at the scarce-N regime,
  i.e. thickening did not reopen the embodiment gap.

## 8. Open risks

- If contact in the data is too noisy, the pure-sparsity α may not latch — fallback is the
  pseudo-label-assisted gate (kept in reserve, not chosen now).
- Per-domain grasp normalization could still leak domain info; the §3.3 probe is the guard.
- Multi-step consistency adds training cost; keep the chain shallow (2 windows) first.
