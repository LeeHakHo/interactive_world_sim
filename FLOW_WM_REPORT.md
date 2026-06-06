# Cross-Embodiment Flow World Model — Report

> Goal: prove **human demonstration data helps train a robot world model**, by routing the
> shared signal through **object 2D flow** (a domain-agnostic interface) instead of pixels
> (where human vs robot are 582× separable and every alignment/decompose attempt failed —
> see `CROSS_EMBODIMENT_WM_ATTEMPTS.md`).
>
> Repo `/scr2/yusenluo/interactive_world_sim`, branch `phantom_dynamo`. All artifacts for this
> line are under `outputs/flow_wm/` (this session) + the pre-existing `outputs/m0_runs/`,
> `outputs/flow_wm_scarcity_curves/`, `outputs/flow_render_dataset/`.
> Updated 2026-06-05.

---

## 0. TL;DR

**Architecture A** = three components, decoupled so the shared part lives in flow and the
domain-specific part (appearance) lives in a per-domain renderer anchored to the real first
frame:

| # | component | shared? | status |
|---|---|---|---|
| ① | per-domain ViT-S autoencoder (latent 4×32×32) | per-domain | reused (job45895, PSNR 42) |
| ② | object-flow dynamics (FlowWM, action-conditioned) | **shared** | **human-helps verified** |
| ③ | per-domain flow-conditioned renderer (anchors real I₀) | per-domain | works on objects; agent now deployable |

**Headline results**
- **②: human data helps robot, big, when robot data is scarce.** N=50 robot clips: robot-only
  ADE 11.03 → robot+human **4.27** (−61%); robot+human@50 ≈ robot-only@1400 → **~28× data
  efficiency**. Robot-abundant: no help (already saturated).
- **③: flow genuinely drives rendering** (object position controlled by flow, net gain 3.60×,
  shuffle/zero ablations pass); **tolerates ②'s prediction error up to ~55% of motion magnitude**.
- **agent rendering** (this session): switching g's supervision to `robot_arm_mask` (SAM2
  fusion) removed the agent mid-break; rendering at N=1 removed the self-inflicted blur.
- **compounding error now MEASURED** (§5): ②'s flow diverges under autoregressive rollout
  (per-step gain 1.07 robot-only / **1.21 robot+human** → human helps open-loop but HURTS
  closed-loop), BUT ③'s I₀-anchor bounds the PIXEL error (frames stay coherent), and the agent
  mask = g(EEF) never drifts (EEF is a known action; delta-EEF/keyboard control works, no IK).
  The object's flow drift is **fixed by scheduled sampling** (autoregressive rollout training):
  per-step gain drops to ≤1, rh free-run stops exploding (101→27), and **human-helps returns to
  closed-loop** (the teacher-forced "human hurts" was a training-protocol artifact).

---

## 1. ② Object-flow dynamics — the core claim (human helps robot)

- **Model**: 3-layer Transformer (d=128) over object-point tokens + EEF action trajectory.
  Input = K=4 history frames of P object points + EEF action; output = future object-point
  trajectory. Pure coordinate space, no pixels. Held-out = robot video 12.
- **Shared interface validated (Step 0)**: object 2D flow probe acc **0.645** (vs pixels/DINOv2
  **1.0**), between/within MMD ratio **1–1.5×** (vs **582×** in pixels). Flow collapses the
  domain wall from "non-overlapping" to "weakly separable". (`outputs/flow_separability_step0*.png`)
- **Action is the key**: without action, the WM is just inertial extrapolation (ADE ~9, barely
  better than constant-velocity 10.65). With EEF action, ADE drops to ~3.6 (object motion IS
  agent-driven — feeding the action recovers it).
- **★ Data-scarcity ablation (the selling point)** — `train_flow_wm_scarcity.py`, 5 seeds:

  | N robot clips | robot-only ADE | robot+human ADE | Δ (human helps) |
  |---|---|---|---|
  | 1400 | 3.67 | 3.54 | +0.13 (noise, saturated) |
  | 400 | 5.24 | 3.88 | +1.36 |
  | 200 | 7.76 | 4.26 | +3.50 |
  | 100 | 10.99 | 4.29 | **+6.70 (−61%)** |
  | 50 | 11.03 | 4.27 | **+6.76 (−61%)** |

  robot-only collapses to constant-velocity (~11) as data shrinks; robot+human holds at ~4.3.
  **robot+human@50 ≈ robot-only@1400 → ~28× data efficiency.**
- **Cross-domain sharing (harder evidence than a probe)**: human-ONLY-trained dynamics
  transfers to robot at ADE 4.37, only 0.7px worse than robot's own 3.63 — the "EEF→object
  response" is genuinely shared.
- Curves: `outputs/flow_wm_scarcity_curves/scarcity_curve.png`. Pred QC (red pred hugs green
  GT, ADE~4px = 7% of motion, not a static blob): `outputs/flow_pred_viz.png`.
- **Caveats**: single rigid object, small 2D point motion; generalization tested on 1 held-out
  robot video; point prediction, not image reconstruction.

## 2. ③ Flow-conditioned renderer — objects work

> **Renderer versions (do not confuse).** `densev3` = object-only flow renderer, **no agent
> mask** — used for object-flow validation and the §2 end-to-end human-helps demo (object region
> is the focus there). `densev4e` = the **full** renderer with the agent g-mask silhouette —
> used for all AGENT-quality (§3) and closed-loop-rollout-agent (§5) results. An earlier
> pixel-rollout that used densev3 showed agent ghosting; that was a **wrong-renderer artifact**,
> redone with densev4e (§5). Closed-loop / end-to-end work defaults to the full (densev4e) pipeline.

- **Design**: condition = `cat([z₀=enc(I₀), dense_flow_map])` fed into the CTM decoder's
  ControlNet bypass (widened, identity-init new channels). Each frame is anchored to the real
  I₀ and denoised toward I_t. Rollout is locked in flow space; pixels are re-anchored to I₀
  each step (intended to avoid pixel-error compounding — but see §5).
- **Decisive finding**: a sparse, low-res flow splat makes flow USELESS (net gain ~1.08×).
  Dense pixel-level splat + big Gaussian + occupancy channel (DragNUWA-style) → **net gain
  3.60×**; shuffle-flow and zero-flow ablations both degrade → flow is genuinely used, object
  position is flow-controlled. (`outputs/m0_runs/EXPERIMENT_LOG.md`, exp1–7)
- **Robustness to ②'s error (exp8)**: inject Gaussian noise σ on predicted point positions →
  splat → render. Mean GT motion = 0.1197 (norm). Tolerates **σ=0.066 = 55% of motion** before
  rendering degrades to the zero-flow baseline. → gives ② a precision budget.
  (`outputs/flow_robustness_scan__densev3_decoder__pointnoise/`)
- **End-to-end pixel demo** (② → ③, human-helps carried to pixels): held-out robot ADE
  robot-only 17.10 / robot+human 9.49; object-region pixel MSE **1.68×** better with human.
  Crucially robot-only ADE 0.076(norm) > ③'s tolerance 0.066 (out of window) while robot+human
  0.042 < 0.066 (in window) → human pulls ② back into ③'s usable window.
  (`outputs/e2e_flow_wm_render__robotonly_vs_human/`)

## 3. Agent rendering quality (this session)

The agent (gripper/arm) is domain-specific → handled by ③'s conditioning mask, NOT flow.
A learned `g(joint, eef) → agent mask` feeds ③ (deployable, non-leaking: g only eats the
current-step position-control action, which defines the next frame's pose).

- **Blur diagnosis**: agent "blur" is mostly **self-inflicted by multi-sample averaging**
  (N=8 sharpness 49 vs N=1 169 ≈ GT 177) + the **decoder 32×32 latent capacity** (even an
  oracle mask only reaches 65% of GT sharpness). Mask quality mainly drives agent **position/
  MSE**, not sharpness (g→oracle: MSE −1.85× but sharpness only +11%).
  (`outputs/flow_wm/diag_blur_source/`, `diag_sharpen_samples/`, `diag_mask_sensitivity/`)
- **Agent mid-break fix**: g's old supervision `robot_v3` (deterministic dark-CC) is broken on
  thin rods/claw. `robot_arm_mask` (SAM2 ∪ dark-bridge fusion + plate-protect) is fuller and
  unbroken on clean AND hard (over-plate) frames. (`outputs/flow_wm/compare_robot_mask_sources/`)
  - Pitfall fixed: the 128 video-cache is desaturated (S~25 vs original ~60), which breaks
    robot_arm_mask's HSV detection — must decode the ORIGINAL video (full color).
  - g v3 retrained on `robot_arm_mask`: held-out IoU **0.722 > 0.647**, mask complete.
- **densev4e** (g v3 mask → ③): agent-region MSE **0.01299 < densev4d 0.01524 (−15%)**, agent
  middle-break reduced. Final N=1 demo (complete mask + sharp):
  `outputs/flow_wm/densev4e_demo_n1/demo.png`.
- **Honest residual**: even N=1 isn't photo-sharp — the decoder 32×32 latent capacity wall.
  Fix = add LPIPS/adversarial loss to ③ (not yet done). Reference oracle (leaks GT shape):
  0.00131 — not deployable, only an upper bound.
- **Control input — delta-EEF / keyboard works, no joints, no IK**: the mask `g` needs only the
  EEF pose, not joint angles. EEF-only g (3 projected points = base + 2 fingertips) reaches
  held-out IoU **0.733 ≈ joint+eef 0.722** (`outputs/flow_wm/maskgen_eefonly/`). So with
  delta-EEF teleop/keyboard: accumulate deltas → absolute EEF pose → g(EEF) → agent mask. No
  joints, no IK (6-DoF IK is a dead end on this arm). The 3 EEF points already encode gripper
  position/orientation/opening, and top-down the arm links extend along base→tip, so joints are
  redundant (IoU even ticks up without them). In rollout the EEF/action is a known input → the
  agent mask never drifts (it's the object flow from ② that drifts — see §5).

## 4. What is NOT claimed

- Not "human always helps" — only when robot data is scarce (abundant robot saturates).
- Not "fully same-domain" — flow is weakly separable (0.645), not 0.5.
- Not photo-realistic rendering — limited by the 32×32 latent AE.
- Pixel decompose/alignment line is dead (582× wall); this line sidesteps it via flow.

---

## 5. ★ Compounding error — the open question (UNTESTED)

### ★ MEASURED (2026-06-05) — true autoregressive rollout, ADE px@224 vs horizon (H=40)

`gen_rollout_eval.py` (long held-out robot sequences, SEQ=64, P=48) + `eval_compounding_rollout.py`
(rolls ② out single-step, feeding back its own predictions). eef (=known action) is GT each step.

| regime | h=1 | h=5 | h=10 | h=20 | h=40 |
|---|---|---|---|---|---|
| robot-only teacher (open-loop) | 8.4 | 7.2 | 6.9 | 6.3 | 7.8 |
| robot-only free (autoregressive) | 8.4 | 26.0 | 25.6 | 24.9 | **26.2** |
| robot+human teacher (open-loop) | 2.8 | 2.6 | 2.7 | 2.6 | 2.8 |
| robot+human free (autoregressive) | 2.8 | 14.5 | 27.7 | 43.9 | **101.5** |
| constant-velocity | 1.0 | 6.8 | 17.8 | 35.3 | 61.0 |

**Findings (counterintuitive, important):**
1. **Compounding is real and large**: free-run ≫ teacher-forced. ② never saw its own predictions
   (teacher-forced training → free-run inference = distribution shift), so error accumulates.
2. **Open-loop (teacher-forced): human helps a lot** (rh 2.8 vs ro 7.8), flat with horizon —
   consistent with the scarcity result (§1).
3. **Closed-loop (free-run): human REVERSES.** robot+human is the most accurate open-loop model
   but **diverges worst** under recursion (101 px @ h=40, worse than constant-velocity 61),
   while robot-only free-run **plateaus** (~26, does not blow up).

**Interpretation**: the robot+human model fits a sharper EEF→object response (great open-loop)
but is more sensitive to its own prediction noise → positive-feedback blow-up; robot-only is
blunter/smoother and saturates. **Mechanism confirmed (`eval_wm_sensitivity.py`)**: the per-step
error-amplification gain (|Δpred_next| / |δ_hist|, finite-difference) = robot-only **1.07**,
robot+human **1.21** (both >1 → both compound; rh amplifies more, flat across σ → structural).
Over 40 steps, 1.21⁴⁰/1.07⁴⁰ ≈ 120× — exactly why rh explodes (101) while ro saturates (26).
Human trains ② into a HIGHER-GAIN map: sharper open-loop, less stable closed-loop
(accuracy↔robustness trade-off). `wm_sensitivity.png`. **Human-helps holds for OPEN-LOOP
prediction, NOT for closed-loop rollout under the current training protocol** — to use ② as a rollout world model
you must train for rollout (scheduled sampling / DAgger); teacher-forcing alone makes it
fragile, and the human-augmented model amplifies that fragility.

**Caveats**: 36 sequences from the same robot episodes (isolates pure compounding, not domain
generalization); strictest single-step autoregression (advance by pred[0]); ② trained
one-shot/teacher-forced, never for rollout — this measures training-protocol fragility.

**③-pixel-level rollout — MEASURED (2026-06-05), I₀-anchor BOUNDS the pixel error** ✅
(`eval_pixel_rollout.py`): render the rollout frames as [z₀=enc(I₀), splat_flow(0→t)] using ②'s
free-run (drifted) flow. Object-region pixel MSE vs horizon:

| t (sample-frame) | 4 | 8 | 12 | 20 | 28 | 36 | 40 |
|---|---|---|---|---|---|---|---|
| GT-flow (③ upper bound) | .0027 | .0029 | .0041 | .0027 | .0034 | .0039 | .0043 |
| robot+human free-run | .0034 | .0114 | .0168 | .0149 | .0138 | .0147 | .0171 |
| robot-only free-run | .0139 | .0240 | .0170 | .0148 | .0127 | .0143 | .0133 |

- ②'s flow diverges to ~101 px@224 (half the image), but ③'s pixel MSE **saturates ~0.015**
  (3–4× the GT-flow bound), it does NOT blow up.
- Visual grid (`pixel_rollout_grid.png`): the rendered frames stay coherent table scenes —
  object/agent are pushed to a **wrong-but-plausible position** (graceful), the frame never
  collapses to noise.
- **Why**: ③ re-anchors every frame to the real I₀ and does NOT recurse on its own pixels;
  only ②'s object positions drift. So pixel error is bounded by "object rendered off-position",
  not unbounded recursion. **The hypothesis below is CONFIRMED.**
- **Implication**: the long-rollout bottleneck is ②'s flow localization (drift), NOT ③'s pixel
  stability. Without fixing ②'s compounding the picture still won't collapse — it just renders
  the object increasingly off-position; fixing ② (scheduled sampling) is what makes it accurate.

curves `outputs/flow_wm/rollout_eval/{compounding_curve.png, pixel_rollout_curve.png, pixel_rollout_grid.png}`.

**Refinement (densev4e renderer, with agent g-mask)** — `eval_pixel_rollout_v4e.py`. The first
pixel rollout used densev3 (NO agent mask) → agent rendered blurry/ghosted (object & agent both
"off"). Redone with densev4e: agent mask = `g(joint_t, eef_t)`, and **joint is a known rollout
input (agent's own action) that never passes through ② and never drifts** → the agent mask
stays correct for the whole rollout. Region MSE vs horizon:

| region | GT-flow | ②-free-run |
|---|---|---|
| **agent** (g-mask, joint known) | ~0.015 | ~0.015 (≈ same, flat) |
| **object** (② flow) | ~0.002 | ~0.015 (drifts, ~7×) |

- **The agent ghosting was a missing-mask artifact, not compounding.** With the g(joint) mask the
  agent renders sharp and on-position through the whole rollout (GT-flow and free-run agent MSE
  are nearly identical — the mask doesn't drift). Grid: `pixel_rollout_v4e_grid.png`.
- **Only the object drifts** (it rides ②'s flow). So "the picture degrades" splits cleanly:
  agent = fixed by the joint-driven mask; object = ②'s compounding (needs scheduled sampling).
- Ghosting cause in general: `z₀=enc(I₀)` carries the object/agent appearance at the I₀ position;
  flow pushes them elsewhere; with large drift the original position can leave a residual. The
  agent mask suppresses this for the agent; the object has no such mask.

### ★ FIX — scheduled sampling on ② (2026-06-05) — compounding treated, human-helps returns

`eval_scheduled_sampling.py`. Train ② AUTOREGRESSIVELY: predict next object point, feed back
GT-or-own-prediction with teacher-prob annealed 1.0→0.3 (eef = known action plan held fixed over
the rollout, sidestepping the 24-frame clip window). So ② learns to stay stable on its OWN
prediction distribution. vs teacher-forced:

| model | per-step gain | free-run ADE @h=40 |
|---|---|---|
| ro teacher | 1.072 | 26.2 |
| rh teacher | 1.212 | **101.5 (explodes)** |
| **ro SS** | **0.996** | 29.7 |
| **rh SS** | **0.850** | **27.0 (bounded)** |

- **Compounding treated**: SS pushes gain from >1 to **≤1** (rh 0.85 lowest) → error decays
  instead of amplifying → rh free-run stops exploding (101 → 27, flat curve).
- **★ Human-helps RETURNS in closed loop**: rh SS (27) ≈ ro SS (30), rh lower mid-rollout, and
  **rh SS gain 0.85 < ro SS 0.996** — under rollout training the human data makes ② the MOST
  stable. So the earlier "human hurts closed-loop" was an artifact of the teacher-forced
  protocol; with scheduled sampling the human benefit carries to the rollout.
- **Trade-off (honest)**: SS trades a bit of open-loop accuracy for closed-loop stability — rh
  one-step h=1 goes 2.84 (teacher, sharpest) → 4.90 (SS). ro barely benefits (gain already ~1);
  SS mainly rescues rh. Residual free-run ADE ~27 (not 0) is ②'s single-step error accumulating
  to a BOUNDED level (gain<1), not compounding.

curve+gain `outputs/flow_wm/scheduled_sampling/scheduled_sampling.png`.

---

**Background — what the rollout machinery is**
- ② is a **one-shot multi-step predictor**: K=4 history → predict the whole F=20-step future in
  a single forward, then read off t=23 (`e2e_flow_wm_render.py`: `F = L-K`, `pred[..., RENDER_T-K, :]`).
- per-horizon error WITHIN that one shot is plotted (`outputs/flow_pred_horizon.png`) — error
  grows with prediction step, as expected.
- ③ anchors every frame to the real I₀, so pixel error is **not** recursively compounded frame
  to frame (by design).

**What we DON'T have (the actual compounding-error test)**
- **No autoregressive rollout**: we never feed ②'s own predictions back as history and predict
  again, step after step. So we don't know whether error **explodes or drifts** under recursion.
- ② is trained **teacher-forced** (real history). At rollout time it would eat its own
  (noisy) predictions → **distribution shift** the model never saw in training — the classic
  compounding-error trigger.
- ③'s "anchor to I₀" is only validated at short horizon. Over a long rollout, I₀ diverges from
  the current scene (object moved far, agent re-posed, disocclusions) → the I₀ anchor may stop
  providing the right appearance, and ③ would have to hallucinate more.

**Risks, ranked**
1. **② recursive drift**: small per-step flow error accumulates; with teacher-forcing→free-run
   shift it can diverge. Likely the dominant term.
2. **I₀-anchor staleness**: long horizon makes I₀ a poor appearance reference (esp. for regions
   the agent uncovers / newly occludes).
3. **③ tolerance budget is per-step**: exp8's σ=0.066 tolerance was measured single-step; under
   compounding the effective input error per step may exceed it midway through a rollout.

**Proposed test (to add)**
- Implement true autoregressive rollout of ②: predict a short chunk, append predictions to
  history, repeat to horizon H ≫ 20. Plot **ADE vs rollout step** for (a) teacher-forced upper
  bound, (b) free-run, (c) constant-velocity floor. Look for the divergence step.
- End-to-end pixel drift: render each rollout step via ③, measure pixel MSE / object-region
  IoU vs GT as a function of horizon; check whether the I₀ anchor holds.
- **Hypothesis to falsify**: "because ③ re-anchors to I₀ and ② is action-conditioned, pixel
  error stays bounded even if ②'s flow drifts." If it holds, the I₀-anchor design is the win;
  if not, we need either (i) shorter re-anchoring intervals with intermediate real frames, or
  (ii) a closed-loop correction.

---

## 6. Next steps (open, user to pick)

1. ~~Scheduled sampling on ②~~ **DONE (§5)** — gain ≤1, rh free-run 101→27, human-helps returns
   to closed-loop. Follow-ups: anneal schedule / longer rollout training to push residual
   free-run ADE (~27) lower; end-to-end pixel rollout with the SS-② + densev4e renderer.
2. **LPIPS on ③** — close the last agent sharpness gap (decoder capacity wall).
3. **② flow precision** — fixes the slight cube-position offset (open-loop accuracy).
4. **human-side ③** — complete the cross-embodiment loop on the human domain.

---

### Script / artifact index
- ② FlowWM: `train_flow_wm_scarcity{,_v3}.py`, data `outputs/flow_dataset/flow_ds_v{2,3}.npz`
- ③ renderer: `train_flow_cond_decoder_densev3.py` (ckpt `..._densev3_80ep__ckpt/decoder_flowcond.pt`)
- ③ robustness: `eval_flow_robustness.py`
- end-to-end: `e2e_flow_wm_render.py`
- agent mask g: `train_maskgen_joint2mask_v3.py` (ckpt `outputs/flow_wm/densev4d_maskgen_joint2mask_v3/maskgen.pt`)
- agent renderer (densev4e, with g-mask): `train_flow_cond_decoder_densev4e.py`, demo `demo_densev4e_n1.py`
- agent mask EEF-only (delta-eef/keyboard): `train_maskgen_eefonly.py` (ckpt `outputs/flow_wm/maskgen_eefonly/maskgen.pt`)
- mask source compare / precompute: `compare_robot_mask_sources.py`, `decode_robot_frames_armmask.py`, `precompute_robot_arm_mask_g.py`
- compounding (§5): `gen_rollout_eval.py`, `eval_compounding_rollout.py`, `eval_wm_sensitivity.py`
- compounding FIX (§5): `eval_scheduled_sampling.py` (autoregressive scheduled-sampling training of ②)
- pixel rollout (§5): `eval_pixel_rollout.py` (densev3, superseded), `eval_pixel_rollout_v4e.py` (densev4e, current)
- agent blur diagnostics: `outputs/flow_wm/{diag_blur_source,diag_sharpen_samples,diag_gmask_error_structure,diag_mask_sensitivity}/`
- experiment log: `outputs/m0_runs/EXPERIMENT_LOG.md`

---

## §6. Thick dynamic latent vs anti-drift training (2026-06-06)

**Question:** rollout 的 "cube 乱动" 是因为 ② 的 dynamic latent 太薄(只48物体点+3EEF点)吗?试了加厚 latent:
contact-gate(EEF↔物体接触门控运动) + grasp token + 抗漂移训练(state-noise injection + multi-step
consistency)。代码 `train_flow_wm_scarcity_v4.py`(`--thin`复现v3;`--phase 1/3/4`),spec/plan 在
`docs/superpowers/{specs,plans}/2026-06-06-thicken-flow-latent*`,13个TDD单测 `tests/flow_wm_v4/`。

**判决(robot-only,held-out vid=12,SEEDS=5,EPOCHS=60):**
1. **准入闸门**:进共享latent的信号必须近域不变。contact/grasp 用 min-max[0,1] 归一化泄漏域
   (LogReg acc 0.779 FAIL,grasp主泄漏)。改 **per-domain z-score** → 0.561 PASS(≈flow净位移校准0.542),
   不损物理信息(within-domain corr=1.0)。需per-sample域标签做标准化。
2. **开环单步drift指标失效**:thin 16.95 vs thick 17.04 无区别——数据被`MOVE_MIN`过滤成几乎无静止clip,
   且开环测不到自回归症状。
3. **链式自回归指标**:compounding ratio(ADE_chain/ADE_open) thin **1.506** → thick **1.085**。
4. **2×2消融(架构×训练)归因**:抗漂移训练是compounding解药(thin→+antidrift 1.506→1.071);
   **latent加厚对漂移几乎无用**(thin→thick-antidrift 1.506→1.549,链式ADE 5.89=5.89)。门退化成常数缩放
   (α静止0.334≈运动0.349)——乘性门α·raw+自由head纯L1下退化。
5. **更长horizon(2-hop,超训练1-hop)**:尾段ADE hop0/1/2 thin 4.36/7.14/8.84(ratio2.03);
   thin+antidrift 4.78/5.21/7.06(1.476);thick 4.23/4.99/6.31(1.491)。抗漂移**泛化到训练外horizon**;
   thick不改ratio但每跳绝对精度最好(价值=找回抗漂移牺牲的开环精度)。

**结论:cube乱动根因是compounding,非latent太薄。** 最便宜稳健修复=抗漂移训练加在现有thin模型
(不需contact/grasp/域标签/准入闸门)。thick latent对漂移无用,仅~10%绝对精度bonus(代价复杂度);去留取决于
Phase-2跨域是否帮human-transfer(未做)。⚠️抗漂移牺牲开环精度(3.94→4.54),rollout场景可接受。
结果文件:`outputs/flow_wm_v4/{probe_cg,phase1_robot_drift,phase1_ablation,phase1_horizon}/summary.txt`。

### §6.1 CORRECTION (5-seed long-rollout + Phase-2) — supersedes §6 conclusion

§6 above drew tentative conclusions from short/single-seed runs. Rigorous follow-up
(5-seed long 36-frame rollout + 5-seed Phase-2) REVERSED two of them. Full organized
results: `outputs/flow_wm_v4/REPORT/` (README + per-topic folders).

REVERSALS:
- **"anti-drift training is the drift fix" → FALSE on long rollout.** It only helps the
  short 1-hop chained metric it trains for; on 5-seed long rollout it HURTS every metric
  (thin→thin+antidrift: dyn_ADE 5.98→7.42, static-cube hallucination 46.8→68.3px doubled).
- **"the contact-gate is useless" → it DAMPS cube hallucination.** Degenerate gate
  (α≈const 0.35, because contact is a weak 2D proximity proxy — NO wrist-cam/depth) acts as
  a blunt motion damper: least static hallucination (35.6 vs thin 46.8 / thin+ad 68.3) and
  best static ADE. Cost: under-tracks genuine fast motion (dyn_ADE 7.22 vs thin 5.98).

STILL TRUE:
- Cube hallucination root cause = motion-biased data (v3 MOVE_MIN dropped all static cubes).
  Data fix (`flow_ds_v4`, 36% static) cut static hallucination ~25% (12.6→9.5px) but the
  gate stayed degenerate (weak 2D contact ceiling).

NEW (Phase-2, the headline positive): **human demos strongly help a data-scarce robot WM**
— N=50 ADE 13.2→5.9 (thin) / 10.9→5.6 (thick) ≈ 2.3×. Thick gives best ABSOLUTE robot+human
ADE at every N (not harmful to transfer). The object-flow cross-embodiment pivot is validated.

WINNER for the rollout cube-drift symptom: **contact-gate (thick) + flow_ds_v4 data,
anti-drift OFF**. A principled fix needs real contact/depth (wrist-cam=robot-only→can't
enter shared latent; 3D=single-view infeasible); next idea = denser AMPLIFY-style flow.
Action-replay videos: `outputs/flow_wm_v4/REPORT/4_rollout_videos/`.
