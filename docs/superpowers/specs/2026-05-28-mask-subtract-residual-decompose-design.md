# Mask-subtract residual decomposition (z_scene = z_full ⊖ z_emb)

Date: 2026-05-28
Branch: phantom_dynamo
Status: approved (design), pending implementation plan

## What this answers / goal

A new stage-1 latent decomposition for the from-scratch latent world model that
factors the observation latent into an **embodiment** part and an
**embodiment-free scene** part, so that human data can eventually help a robot WM.

Win condition (user-stated, two-part — both required):
1. **z_scene less domain-separable** than the full latent (domain probe drops).
2. **z_scene stays information-sufficient** — no collapse. "Even if aligned, the
   information must be enough"; an invariant-but-empty z_scene is a failure
   (this is exactly how the v3 GRL adversary died: PSNR 23.8).

This is the decomposition + z_emb grounding step. Cross-domain numerical
**alignment** of z_scene (residual scene/camera gap) is explicitly **deferred** to
a later step (OT/whitening on the residual linear direction).

## Why this design (what failed before, and the gap it fixes)

Prior cross-embodiment attempts (memory `project_phase0_v5_emb_film_ot`,
`reference_cross_embodiment_wm_attempts`, and `latent_decompose/` code):

- **channel_split (v3, GRL adversary):** destroyed reconstruction (PSNR 23.8) —
  adversarial invariance collapses information.
- **dual_head (v4, CLUB only):** two heads on a shared pooled feature, pushed
  apart by a CLUB MI penalty. **No anchor** — CLUB alone just pushes for
  independence, so z_emb collapsed (`L_dom = ln2`). "A slot with no forcing
  function."
- **emb_film (v5, global appearance code):** gave z_scene a slot to drop
  embodiment but no forcing function → z_scene stayed 0.96 domain-separable; a
  global vector also cannot represent the agent's *spatial* appearance.

**The user's insight:** the missing piece vs the cross-embodiment video-editing
paper (arXiv:2605.03637, "Bridging the Embodiment Gap") is the **anchor** — the
paper structures each latent space with an InfoNCE contrastive loss (positives =
same agent / same task), not just a CLUB independence penalty. Combine that anchor
with: (a) **agent-mask-grounded** supervision of z_emb, and (b) a figurative
**"subtraction"** of z_emb from the full latent to define z_scene.

### Two diagnostics that shape the operator choice

- **mixed-ViT eval (job 45714):** embodiment is encoded in the **channel DC/mean**
  component — robot/human per-channel pooled means have opposite signs. The domain
  signal lives in a **global/low-frequency direction**.
- **inpaint PCA (job 45795):** after removing the agent, the residual domain gap is
  a **low-variance linear direction** (linear probe 1.0 but ~24% top-2 PCA
  variance; the gap is one consistent direction, not the high-variance axes).

Both point to "domain signal ≈ a (few) linear direction(s)" → an **orthogonal
projection** that removes that direction is the matched, low-risk, interpretable
tool. It also **cannot collapse** z_scene (bounded rank-k removal).

## Design

New `latent_decompose.method = mask_subtract`, alongside the existing
`channel_split | dual_head | emb_film` in `configurations/algorithm/latent_world_model.yaml`
and `interactive_world_sim/algorithms/latent_decompose/`. `enabled=false` keeps the
model bit-identical to baseline (same convention as the existing methods).

### Forward pass

```
x ──E──▶ F = z_full ∈ R^{4×32×32}          (existing shared encoder, unchanged)
                │
                ├─ stop_grad ─▶ head_emb ─▶ z_emb_spatial = head_emb(sg(F)) ⊙ M_lat   (agent-region code)
                │                              │
                │                              ├─ D_a(z_emb_spatial) ≈ x · M        (agent reconstruction, mask interior)
                │                              ├─ e = pool_M(z_emb_spatial)         (pooled embedding, d≈64)
                │                              │      └─ InfoNCE(e) by domain        (anchor)
                │                              └─ û_emb = normalize(W · e)          (removal direction)
                │
                ├─ removal ─▶ z_scene = F − ⟨F, sg(û_emb)⟩ · sg(û_emb)              (default: ortho_proj)
                │                              └─ decode(z_scene) ≈ x · (1−M)        (scene reconstruction, mask exterior; sufficiency)
                │
                └─ D(F) ≈ x                                                          (full reconstruction; only loss that shapes E)
```

- **Mask grounding (input side):** `head_emb` reads `stop_grad(F)` and is spatially
  gated by the downsampled agent mask `M_lat` so z_emb only lives on the agent. The
  embodiment encoder is fed agent-region features only (paper's spirit: the
  embodiment encoder sees only the agent).
- **Agent reconstruction (CORE, not optional):** `D_a(z_emb_spatial) ≈ x` on the
  mask interior. **This is the forcing function that pins z_emb to the agent.**
  Without it, the 2-class anchor can be satisfied by *any* domain-correlated
  feature — and per the inpaint diagnostic the domain signal is mostly *scene*, so
  a pure anchor risks making z_emb encode the scene, and then the removed direction
  would be a scene direction, not the agent. z_emb must be a **spatial** code (a
  feature map over the agent region), not a global vector (emb_film lesson: a global
  vector cannot reconstruct spatial agent appearance).
- **Anchor (the missing piece):** InfoNCE on the pooled embedding `e`; positive =
  another frame of the same agent (human/robot), negative = the other domain.
  Structures z_emb to be embodiment-discriminative. With only 2 domains this is
  effectively domain-contrastive — sufficient to define the embodiment direction.
- **Removal operator (figurative "subtraction"), swappable:**
  `removal_op ∈ {ortho_proj (default), affine, gate}`.
  - `ortho_proj` (default): `z_scene = F − ⟨F,û⟩û`. Optionally top-k subspace
    (remove k≥1 orthonormal embodiment directions). **No trainable removal params**
    beyond the anchored direction; bounded rank-k removal → cannot collapse z_scene.
  - `affine`: `z_scene = F − A(z_emb)`, A a small conv/linear (more flexible, handles
    spatially-distributed embodiment; needs the scene-reconstruction signal to train
    A; higher collapse/recon risk).
  - `gate`: `z_scene = F ⊙ (1 − σ(g(z_emb)))` (multiplicative suppression).
  - **û_emb is stop_grad in the removal** so the removal cannot corrupt the
    embodiment code.

### Gradient routing (key design point)

- **embodiment path is fully detached from the encoder-decoder.** `z_emb =
  head_emb(stop_grad(F))`; anchor + agent-reconstruction shape only `head_emb`/`D_a`,
  never `E` or the main decoder. Rationale: the anchor is **domain-discriminative**
  by design; if it backpropped into the shared encoder it would push the shared
  latent to be *more* domain-separable — the opposite of the goal — and risk the
  v3/v4 reconstruction interference. Detaching guarantees the embodiment objective
  can never degrade reconstruction or amplify separability.
- **`E` is shaped only by `D(F) ≈ x` (full reconstruction) and `decode(z_scene) ≈ x`
  (mask-exterior scene reconstruction).** No information-destroying objective acts on
  E this round → F stays full-information.

### Losses (no CLUB, no adversary)

| loss | shapes | purpose |
|---|---|---|
| `L_rec = D(F) ≈ x` | E, D | full reconstruction; keeps F faithful (recon zero-risk) |
| `L_agent_rec = D_a(z_emb) ≈ x` (mask interior) | head_emb, D_a | **pin z_emb to agent appearance** |
| `L_scene_rec = decode(z_scene) ≈ x` (mask exterior) | E, removal | **sufficiency / anti-collapse**; trains removal (affine) |
| `L_anchor = InfoNCE(e)` by domain | head_emb | structure z_emb (the missing anchor) |

Independence between z_emb and z_scene is **structural** (mask localization +
identity anchor + projection), not enforced by an MI penalty or a GRL adversary.

`decode(z_scene)` **reuses the main decoder `D`** (shared weights), not a separate
network. This is consistent because on the mask exterior both `D(F)` and
`decode(z_scene)` target the same `x · (1−M)` (z_scene ≈ F outside the agent region,
where z_emb ≈ 0); the mask interior of `decode(z_scene)` is left unsupervised (no
agent-free target — the user rejected inpaint frames as a target: imperfect and
still domain-separable on DINO). `D_a` is a small **separate** aux head (it must
render the agent from the detached z_emb, so it cannot share gradients with E).

### Information sufficiency / anti-collapse (first-class)

Collapse is structurally blocked this round: (1) E is trained only by
reconstruction → F is full-information; (2) `ortho_proj` removes only a bounded
rank-k subspace → z_scene cannot be zeroed (contrast: GRL can flatten everything).
On top of the structural guarantee:

- **Sufficiency metric, tracked from day 1:** linear probe `z_scene → {EEF position,
  object position, future-frame}` prediction accuracy; mask-exterior reconstruction
  PSNR from `decode(z_scene)`.
- **Hard-floor rule for the later alignment phase:** when OT/whitening is added,
  the dynamics-probe / exterior-PSNR must not drop below a floor, else the alignment
  strength is rejected — same pattern as the OT alpha PSNR floor in
  `project_phase0_v5_emb_film_ot`.

### Agent motion / dynamics

z_scene removes the whole agent, so the agent's motion is supplied to the **dynamics
model** via a separate **EEF-pose stream** (frame-unified: robot world → cam frame,
per the OT memory; reuse `robot_world_to_cam()`). This is a **stage-2** concern;
stage-1 only builds and validates the decomposition. z_scene retains scene/object
dynamics (everything outside the agent region is F unchanged).

## Data flow / dependencies

- **Agent masks** (both domains): reuse the inpaint pipeline's masking — human =
  SAM2 double-positive (hand + dark-sleeve centroids) ∪ `masks_arm` ∪ dark
  connectivity − cube; robot = dark-region connectivity − cube. **Mask quality is a
  known, jittery dependency** (memory `project_inpaint_dino_subspace`,
  `project_arm_mask_sleeve_extension`) and directly bounds z_emb purity.
- **EEF pose** (stage-2): from `play_human_eef` / `play_robot_eef` configs; frame
  unification per the OT design.
- Lives in the existing from-scratch latent WM (4ch × 32×32 latent, diffusion
  decoder, dynamics). `mask_subtract` is additive; baseline unchanged when disabled.

## Success criteria (stage-1)

| signal | target | meaning |
|---|---|---|
| domain probe on z_emb | high (≫0.5) | z_emb captured the embodiment/domain — desired |
| domain probe on z_scene vs z_full | **significantly lower** than z_full | removal stripped agent/domain signal |
| z_scene probe on z_scene | **will NOT hit 0.5** | residual scene/camera gap (deferred) — expected |
| full reconstruction PSNR `D(F)` | ≈ emb_film (~40), no regression | recon not sacrificed |
| mask-exterior PSNR `decode(z_scene)` | healthy | z_scene scene-sufficient (anti-collapse) |
| sufficiency probe (EEF/object/future) | high | z_scene retains dynamics info (anti-collapse) |
| agent-recon QC `D_a(z_emb)` | renders the agent in mask | z_emb grounded to agent (visual gate) |
| **ablation: anchor on/off** | off → z_emb collapses | reproduces v4 failure as control |

## Out of scope (YAGNI, this round)

- Cross-domain **numerical alignment** of z_scene (OT/whitening on the residual
  linear direction). Deferred; sufficiency hard-floor is pre-wired for it.
- **EEF → dynamics** wiring (stage-2).
- Cross-embodiment **recompose** eval (z_scene^human + z_emb^robot → robot frame) —
  the eventual payoff, not this round.
- CLUB / GRL adversary (deliberately excluded — both failed before).

## Known risks

- **(a) Mask quality** jitter → z_emb purity. The whole forcing function rests on
  the agent mask being the agent; QC gate required.
- **(b) Domain signal not a single direction** → `ortho_proj` underfits; mitigation =
  top-k subspace or switch `removal_op=affine` (the swappable operator exists for
  exactly this).
- **(c) Residual scene gap** → z_scene does not auto-align across domains; explicitly
  deferred, not solved here.
- **(d) 4-channel capacity** for the agent code is small; if `L_agent_rec` cannot
  render the agent, increase z_emb capacity (decouple from the 4ch latent width).
