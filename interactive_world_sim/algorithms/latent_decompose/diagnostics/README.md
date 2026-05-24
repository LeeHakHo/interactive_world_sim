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
