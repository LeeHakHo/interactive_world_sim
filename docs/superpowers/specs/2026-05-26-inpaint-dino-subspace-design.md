# Inpaint-then-DINO subspace diagnostic

Date: 2026-05-26
Branch: phantom_dynamo
Status: approved (approach A), executing autonomously per user request

## Question this answers

If we **remove the agent** (human hand+arm; robot arm) from the workspace video
**and regenerate plausible background** behind it, do human and robot frames land
in the **same DINOv2 subspace**? I.e. is the cross-embodiment domain gap mostly
caused by the *agent pixels*, or also by the *scene/lighting* themselves?

This validates (or kills) the Phase 1 "inpaint as forcing function" direction
recorded in memory `project_phase0_v5_emb_film_ot`: emb_film / OT failed because
they gave the model a slot for embodiment but no *forcing function* to empty it.
Inpainting removes the agent at the *input*, which (hypothesis) makes z_scene
naturally domain-agnostic. Before building the Phase 1 training pipeline, measure
whether the premise even holds in a frozen, training-free feature space (DINOv2).

## Why DINOv2 (not our own encoder)

DINOv2 is a strong, frozen, general vision backbone — if even DINOv2 features
separate the two domains after agent removal, the gap is in the scene, not the
agent, and inpaint-as-forcing-function is weaker than hoped. If they collapse,
the premise holds. This is a clean, model-independent test.

## Key prior result (the gap this fixes)

`mask_agent_sam2_test.py` already does SAM2 point-seeded masking for both domains,
but it **grays out** the masked region (`masked[:, mask] = 0.5`). Result: probe
acc stayed ~1.0 (gray blobs of different shape/position are themselves a domain
cue, and graying destroys background content). The user's point: *not just mask —
also generate the replacement background*. So we swap gray-fill → E2FGVI video
inpainting and re-measure.

## Approach (A — quick diagnostic, throwaway script)

End-to-end, one new script `inpaint_dino_subspace.py`, reusing existing building
blocks. No new dataset, no training.

### Components

1. **Clip source** — `eval_stage1_mixed_vit_alignment.build_domain_val_dataset(CKPT, domain)`
   yields 16-frame clips (`val_horizon=16`) of `obs["camera_0_color"]`, shape
   `(B,T,C,H,W)` in `[0,1]`, 128×128, workspace crop `[195,195,256,256]`
   (1:1-aligned between human and robot). `CKPT` is only used to load the dataset
   config; the model is not loaded.

2. **Agent mask (SAM2 point-seeded)** — reuse `mask_agent_sam2_test.py` logic:
   - robot seed = darkest-pixel centroid (black Trossen gripper)
   - human seed = most skin-like pixel (pink/saturated vs tan wood)
   - SAM2 (`sam2_hiera_large.pt`) → multimask → keep agent-sized (2–45% area),
     best score → dilate. Per-frame (E2FGVI tolerates per-frame masks).

3. **Inpaint (E2FGVI-HQ)** — `phantom/submodules/phantom-E2FGVI/E2FGVI`,
   ckpt `E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth`. Wrapper takes a clip
   `(T,C,H,W)` in `[0,1]` + per-frame masks `(T,H,W)`; normalizes to `[-1,1]`,
   zeros the masked region, reflection-pads to E2FGVI's `(60,108)` multiples,
   runs `model(masked, T)` with all T frames as local frames, denorms, crops,
   composites `pred*mask + orig*(1-mask)`. Returns inpainted clip in `[0,1]`.
   Work at **224×224** (upsample 128→224) so both E2FGVI and DINOv2 see their
   native-ish resolution and raw-vs-inpainted are strictly comparable.

4. **Features (DINOv2 ViT-S/14)** — `DINOv2PatchExtractor`; pooled patch tokens
   `extractor(x).mean(dim=1)` → `(N,384)`. Compute for BOTH raw and inpainted
   versions of the **same** frames (only variable = agent present/absent).

5. **Metrics** — reuse `compare_dino_within_between.py`:
   - `train_probe_mlp` domain-probe accuracy (0.5 = aligned, 1.0 = disjoint),
     mean over 3 seeds.
   - RBF-MMD² (median-heuristic bandwidth).
   - within-domain split (H1/H2, R1/R2) = the **non-agent scene-noise floor**.
   - report `between / mean-within` MMD ratio (1 = no gap, ≫1 = real gap).
   Print a table comparing **raw (agent present)** vs **inpainted (agent removed)**.

6. **Visual QC (hard gate, user-required)** — for the first few human & robot
   clips, save a montage PNG: `raw | mask-overlay | inpainted`. Inspect visually
   that (a) the agent is actually gone, (b) background is plausible (no black hole,
   no obvious smear). Only scale up after this passes.

### Data flow

```
val clip (T,C,H,W)[0,1] 128²
   ├─ upsample→224²
   ├─ SAM2 per-frame mask (seed→multimask→agent-sized→dilate)
   ├─ E2FGVI(masked clip, masks) → inpainted clip 224²
   ├─ DINOv2(raw 224²)       → raw pooled feats   (N,384)
   └─ DINOv2(inpainted 224²) → inpaint pooled feats (N,384)
        → probe-acc + RBF-MMD², within & between, raw vs inpainted
        → PCA scatter (raw vs inpainted, colored by domain)
        → QC montages
```

### Execution

- `sbatch` on `partition-1` (snoopy1), 1 GPU, conda env `iws`. Mirror
  `sbatch/phase0_stage1_mixed_vit.sbatch` env setup.
- **Smoke first**: `--per-domain 64` (≈4 clips/domain) → QC montages → manual
  visual check. Then **full**: `--per-domain ~900`.
- Outputs to `outputs/inpaint_dino_subspace/`: `qc_human_*.png`, `qc_robot_*.png`,
  `pca.png`, and a printed metrics table in the SLURM log.

## Success criteria / interpretation

| Outcome | Reading |
|---|---|
| inpainted between-probe ≈ within-probe ≈ 0.5, MMD ratio → 1 | Premise holds: agent is the gap. Inpaint-as-forcing-function justified. |
| inpainted between-probe still ≫ within, MMD ratio ≫ 1 | Scene/lighting is also a gap. Inpaint alone won't align; need more. |
| QC shows agent not removed / black holes / heavy smear | Mask or inpaint quality bad — fix before trusting any number. |

## Out of scope (YAGNI)

- VACE diffusion inpaint: optional 2nd pass only if E2FGVI result is ambiguous
  (VACE/Wan2.1 not installed; heavy). Not in the first run.
- Reusable preprocessing pipeline / new datasets (user confirmed not needed).
- SAM2 video predictor for temporal mask consistency (per-frame is enough for a
  diagnostic; revisit only if QC shows flicker artifacts driving the result).
- Robot-arm FK-render mask (more accurate but heavier); SAM2 chosen, FK only as
  fallback if SAM2 visibly leaks robot pixels.

## CANONICAL RESULT (inpaint_gap_test.py, job 45741, K=220/domain)

The two-stage `inpaint_dino_subspace.py` pipeline below used a color-heuristic
human mask (inferior) and only 1 val episode/domain (low diversity → probe
saturates at 1.0). The **correct** experiment is `inpaint_gap_test.py`: it seeds
SAM2 from phantom's real `masks_arm` HAND location so SAM2 grabs the WHOLE
connected arm (hand+sleeve) cleanly, keeps the cube, and uses 3 human chunks + 3
robot episodes. Verified visually (`outputs/inpaint_gap_test.png`): whole arm+hand
removed, cube + plate preserved.

| config | probe-acc | MMD ratio |
|---|---|---|
| raw (agent present) | 1.00 | ~582 |
| **inpaint (agent removed)** | **0.939** | **63** |
| inpaint + color-matched (robot→human chan mean/std) | 0.955 | 67 |
| inpaint + grayscale (all color removed) | 0.977 | 86 |
| within-domain control (H1 vs H2) | 0.492 | — |

**Conclusion (canonical):** with a correct, complete agent mask, removing the
agent makes a **large** dent — probe 1.00 → 0.94 and the MMD ratio drops ~9×
(582 → 63). So the agent is a **major** contributor to the cross-embodiment gap.
But a residual remains (probe 0.94, not chance), and it is **structural, not
color**: matching robot color to human (0.955) and removing all color via
grayscale (0.977) do not reduce it — they leave the domains separable on
objects/texture/geometry (bowl vs plate, table grain, layout). So inpainting the
agent helps a lot but does not fully align the observation latents; the residual
scene/object gap needs matched collection or an explicit cross-scene treatment.

---

## Pipeline results (inpaint_dino_subspace.py, job 45740 — superseded by above)

Note: human mask here is a color-heuristic (skin+dark union) and val data is 1
episode/domain, so probe saturates at 1.0; numbers are kept for the record but the
canonical result above (correct masks_arm seeding, diverse data) supersedes them.

### Details (2026-05-26, job 45740, n=900/domain — color-heuristic mask)

**Mask evolution (two user corrections):** (1) human user wears long sleeves, no
visible skin — the first skin-seed masked the red **cube** (task object) and left
the arm; (2) pure dark-seed then masked only the **black sleeve**, leaving the pink
**hand**. Final mask seeds BOTH a skin term (light pink hand) and a luminance-dark
term (black sleeve / robot gripper), segments each tightly and **unions** them =
the whole arm+hand. blue-reject keeps the bowl/plate; cube-reject (dark saturated
red) keeps the cube. Robot has ~no skin → dark seed only → gripper.

Visual QC **passed**: the whole arm+hand (human) and the gripper (robot) are
removed; the blue bowl/plate, red cube, and the robot's manipulated object are all
preserved; E2FGVI fills plausible background, no black holes.

Probe acc = **1.000 in every config** (n=900). Between-domain RBF-MMD²:

| config | MMD² | vs raw |
|---|---|---|
| raw (agent present) | 0.7741 | — |
| inpaint (agent removed) | 0.5943 | −23% |
| inpaint + grayworld (white-balance removed) | 0.5946 | ≈ inpaint (no effect) |
| inpaint + grayscale (all color removed) | 0.7080 | still fully separable |

**Conclusion — premise does NOT hold, and the residual gap is STRUCTURAL, not
color.** (1) Removing the agent shrinks the gap a real amount (MMD −23%) so the
agent *is* part of it, but the domains stay **perfectly separable** (probe 1.000).
(2) Per-image white-balance normalization (grayworld) changes the gap by ~0 (0.594
→ 0.595) — so it is **not** a global color cast. (3) In **grayscale** the domains
are *still* perfectly separable (probe 1.000) — so the residual gap is **structural**:
different objects (bowl vs plate), table texture/grain, layout, geometry. PCA
(`outputs/inpaint_dino_subspace/pca.png`) shows two disjoint clusters along PC1 in
all four configs.

**Implication for Phase 1 inpaint-as-forcing-function:** inpainting the agent (and
even normalizing color) will not align the two domains' observation latents — the
scenes/objects themselves differ structurally. Alignment needs matched collection
(same workspace/objects/camera) or an explicit cross-scene treatment, not just
agent removal. Consistent with `project_stage1_hr_alignment_eval` (trained encoder
also fully disjoint, MMD ratio 627).

## Known caveats

- Long-sleeve human (memory `project_arm_mask_sleeve_extension`): SAM2 skin seed
  may grab only the hand, leaving the sleeved forearm → residual embodiment cue.
  QC will reveal this; note it in the result rather than silently trusting it.
- Inpaint artifacts are a *shared* confound (same backbone both domains), not a
  domain cue, so they don't bias the between-vs-within comparison.
- Background objects/lighting genuinely differ per episode; within-domain split
  calibrates that floor.
