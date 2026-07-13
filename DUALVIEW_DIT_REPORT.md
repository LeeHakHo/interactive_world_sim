# Dual-View DiT ③ 正式化 REPORT — flow 接口 vs naive action 条件化

> 自包含报告。spec: `docs/superpowers/specs/2026-07-12-dualview-dit-formalize-design.md`;
> plan: `docs/superpowers/plans/2026-07-12-dualview-dit-formalize.md`;
> 执行日志: `.superpowers/sdd/progress.md`。分支 phantom_dynamo,2026-07-12/13 完成。

## 0. 一句话结论

**在同一 DiT backbone 上,object-flow 空间条件显著优于 IWS-stage2 式 eef-FiLM 条件(双视角均 >2σ),
且端到端(② 预测 flow 驱动 ③)几乎无损贴住 GT-flow 天花板、仍显著赢 naive;真 IWS stage2 整管线(CMLatentDynamics
DF)在同数据同口径下垫底。** 两个诚实 nuance:(a) 对 eeffilm 的优势有一大块来自"空间注入 vs 向量 FiLM"机制差——
OSCAR-skeleton 式的 eefsp 臂在易视角(cam_high)甚至反超 flow(cam_low 上 flow 均值优但不显著),物体 flow 内容的
可靠净增益是**端到端可预测性/可控性**(eefsp 无 ② 可预测通道);(b) ③ 层 human co-train 三臂全部不显著(复刻先验"human-helps 主战场在 ②")。

## 1. Setup(组件版本)

- **数据**:`outputs/flow_render_dataset_can_dual/`(robot 2700×L48 + human 1800×L24,cam_high+cam_low 双视角,
  48 点物体 track + 3 点 eef,`low_valid` 过滤)。heldout split `rng(0)` 固定(150 条),与训练 seed 无关;
  eval = heldout 中 motion top-24 条,持久化 `eval_seqs_n24.json`,全部 run 同一集合。
- **③ backbone**:add-DiT D384×8×6(project_detmem_dit 赢家),frozen 16ch VAE(ostris/vae-kl-f8-d16),
  双视角 token 联合注意力([z0_v0,prev_v0,z0_v1,prev_v1] 各 256 token)。
- **训练配方(三臂完全一致)**:BS8, LR2e-4, 60ep, PREV_DF 0.3;robot latent-MSE + decode-LPIPS(LAM=1),
  human obj-region pixel MSE;co-train robot+human(MIX=rh)或 robot-only(MIX=r)。
- **②(e2e 用)**:`outputs/cross_embodiment_wm/dualview_wm/wm_dual.pt`(DualLWC 双视角联合 flow WM,另一 session 训)。
- **eval(单一权威)**:`obj_lpips_audit`(物体 footprint 质心 64×64 crop LPIPS;footprint<5 点记无效并报 det-rate)
  + PSNR;所有臂含 IWS baseline 同一函数同一 seq 集。全表 det-rate = 1.00,无"物体消失"污染。
- **主脚本**:`exp_scel_dualview_dit_formal.py`(train/eval/gif/e2e 四模式)、`exp_dualview_iws_stage2.py`、
  `run_dualview_formal_sweep.sh` + `sbatch/dualview_formal_train.sbatch`、`agg_dualview_formal.py`。

**三条件臂**(controlled:同 backbone、同训练预算、唯一变量=条件注入;baseline 经 IWS/OSCAR 文献核对升级):

| 臂 | 内容 | 注入 | 文献锚点 |
|---|---|---|---|
| flow(ours) | 物体 flow splat + footprint (3ch/view) | spatial-add | object-motion 接口 |
| eefsp | 只 splat 3 个 eef 点(无物体 flow,3ch 同构) | spatial-add(同 flow) | ≈OSCAR 2D-skeleton 空间条件轻量类比 |
| eeffilm | 全量 eef 向量 3点×[x,y,dx,dy]×2view=24-dim(指尖隐含 grip) | MLP→adaLN-FiLM | =IWS stage2 `action_emd`→`cond_emb_layers` 同构 |

## 2. M1 Controlled 消融(replay GT 条件,3 seed,mean±std)

| cond | crossview | mix | v0 obj-LPIPS | v1 obj-LPIPS | v0 PSNR | v1 PSNR |
|---|---|---|---|---|---|---|
| flow | 1 | rh | 0.2556±0.0066 | **0.2790±0.0033** | 22.69 | 22.08 |
| flow | 0 | rh | 0.2533±0.0036 | 0.2729±0.0029 | 22.61 | 21.99 |
| eefsp | 1 | rh | **0.2284±0.0123** | 0.2913±0.0140 | 23.81 | 22.55 |
| eefsp | 0 | rh | 0.2313±0.0127 | 0.3089±0.0262 | 23.81 | 22.21 |
| eeffilm | 1 | rh | 0.2784±0.0114 | 0.3218±0.0212 | 22.93 | 22.18 |
| eeffilm | 0 | rh | 0.2752±0.0195 | 0.3230±0.0176 | 22.92 | 22.12 |
| IWS-stage2(DF, external) | - | - | 0.3238 (n=1) | 0.3705 (n=1) | 19.54 | 18.41 |

(完整表含 robot-only 行与 det-rate:`outputs/cross_embodiment_wm/dualview_dit_formal/ablation_table.md`)

**主判据(variance 门控,flow vs eeffilm cv1)**:
- v0: d=+0.0228 > 2×pooled_std(0.0186) → **PASS**
- v1: d=+0.0428 > 2×pooled_std(0.0304) → **PASS**
- **flow 接口显著优于 IWS 式 eef-FiLM,非 seed 噪声。** 且 flow 臂 seed 方差全场最小(v1 std 0.003),训练最稳。

**eefsp 臂判读(内容 vs 注入机制,诚实)**:
- cam_high:eefsp 0.2284 **反超** flow 0.2556(差 >2σ)。易视角下,agent 构型的空间渲染条件已足够让 ③ 内隐推断
  物体运动(can 任务物体大多随夹爪走)——**复刻 OSCAR"skeleton 空间条件最强"的发现**。
- cam_low:flow 0.2790±0.003 vs eefsp 0.2913±0.014,flow 均值优但 d=0.0123 = 1.2σ,**<2σ 不显著(n=3,如实报)**。
  "flow 内容的净价值"由显著证据承担:e2e 可预测性(eefsp 没有可由 ② 预测的物体运动通道,§3)+ flow vs eeffilm 门控。
- 结论改写:**"flow 赢 naive action"成立,但拆开看 = "空间注入 ≫ 向量 FiLM"(机制,两空间臂共享)+
  "物体运动内容"(flow 独有,在难视角/端到端/可控性上兑现)。** paper 叙事必须按此写,不可把全部差距归给接口内容。

**cross-view 联合注意力消融(cv1 vs cv0=attention-mask 阻断,参数量不变)**:
三臂 cv0/cv1 差异全部在噪声内(flow 甚至 cv0 略好)。**负结果:本设置下两视角可独立渲染,联合注意力不是增益来源**
(单测保证 mask 无泄漏,`tests/test_dualview_formal.py`)。

## 2.5 M1c Human-helps(robot-only vs robot+human,cv1,3 seed)

| 臂 | v0 Δ(r−rh) | v1 Δ(r−rh) | 判定 |
|---|---|---|---|
| flow | −0.0036 (σ0.005) | −0.0051 (σ0.003) | 不显著 |
| eefsp | +0.0004 (σ0.009) | +0.0032 (σ0.010) | 不显著 |
| eeffilm | +0.0058 (σ0.011) | +0.0003 (σ0.016) | 不显著 |

**③ 渲染层 human co-train 对任何接口都不帮**(Δ>0=帮,全部 <2σ)。复刻先验([[project_dualview_dit_flow_vs_eef]]
的 +0.03 天花板判死)。**human-helps 的主战场在 ②**(object-flow WM human-help 1.7-2×,
project_human_helps_end2end),③ 只负责把 ② 的增益无损渲染出来(见 §3)。

## 3. M2 端到端 ②→③(pred-flow 驱动,n=24;2026-07-13 勘误后的干净数字)

> **勘误**:初版 e2e 用的 ②(`dualview_wm/wm_dual.pt`)与 ③ 的 split 代码不等价,24 条 eval seq 有 22 条
> 在该 ② 的训练集内(体检报告 CAN_DATA_AUDIT_2026-07-13.md,CRITICAL 项)。已用 `SPLIT=okfirst` 重训
> **对 eval seq 可证明零接触**的 ②(`dualview_wm_cleansplit/wm_dual.pt`,commit 23d13de)并重测。
> 泄漏版数字(ADE 2.07/2.33px,e2e 0.270/0.291)作废,以下为干净数字。

| 列 | v0 obj-LPIPS | v1 obj-LPIPS |
|---|---|---|
| GT-flow → ③(③ 天花板) | 0.263 | 0.282 |
| **② pred-flow → ③(端到端,干净 ②)** | **0.270** | **0.298** |
| eef-FiLM → ③(naive) | 0.285 | 0.345 |

- **② ADE:cam_high 2.71px / cam_low 3.67px**(H=20;验收线 ≲4px,过)。
- **端到端比天花板差 0.007/0.016,仍明显赢 naive** → ②→③ 全链在无泄漏条件下成立:核心结论经勘误存活,
  cam_low 的天花板差距比泄漏版略大(0.016 vs 0.009),如实报。这是 eefsp(agent 条件)给不了的:
  它没有可由 ② 预测、可交互操控的物体运动通道。
- 注意:summary.txt 里的 VERSIONS 行仍打印旧 ckpt 路径(静态字符串),实际用的 ② 由 `WM_PT` env 指定
  为 cleansplit 版,以 e2e 日志(dvf_e2e_clean_53263)为准。
- ② 输入 view1 遮挡点按训练约定填 0.5(修过一处 train/inference 填充不一致,commit 523b7ae)。

## 4. M2.5 External baseline:真 IWS stage2(CMLatentDynamics,DF)

- **是什么**:IWS stage2 原生 dynamics 模块(Conv3d 时空 backbone + `action_emd`→per-block FiLM 注入,模块零修改)
  + diffusion-forcing 噪声训练 + AR 滑窗多步去噪采样(与 `latent_world_model.dynamics_forward` 同构),
  在同 frozen VAE latent(双视角 stack 32ch)、同数据、同监督配方、同 eval 上独立训练(40ep,34.1M params,
  action=同 eeffilm 的全量 24-dim)。
- **不是什么**:未复刻 IWS 的 CTM 蒸馏细节与 conv encoder → backbone+接口两变量混杂,**非 controlled 证据**
  (controlled 由同 backbone 的 eeffilm 臂承担);回答"我们的 flow-DiT 相对现成 IWS stage2 整体如何"。
- **结果:0.324 / 0.371(PSNR 19.5/18.4),全场最差**;比同信息量的 eeffilm(同注入机制、更强 backbone?否——
  DiT vs Conv3d 不可比,故只作 external 行)。n=1 seed,标注。

## 5. 风险与诚实声明

1. eefsp 在 cam_high 反超 flow——"接口价值"叙事必须拆成机制(空间注入)+内容(物体运动)两层,见 §2。
2. ③ 层 human-helps 三臂全不显著(§2.5)——human-helps 论据全部由 ② 层证据承担,paper 不得引用 ③ 层。
3. IWS external 行 n=1、且 DF 训练是简化复刻(x0-pred,无 CTM 蒸馏),数字只作量级参考。
4. cross-view 负结果基于本任务/数据;不同任务(需跨视角一致性约束的)可能不同。
5. cam_low 的 tracks 有 4.07% NaN(遮挡),eval 用 low_valid+det-rate 防护;det-rate 全场 1.00。
6. 途中两次磁盘满事故导致 eeffilm s1/s2 重跑(结果无影响,输出树已迁 /scr symlink)。
7. **e2e 泄漏勘误(2026-07-13)**:初版 e2e 的 ② 训练集含 22/24 eval seq,已重训干净 ② 重测(§3);
   全管线体检报告见仓库根 `CAN_DATA_AUDIT_2026-07-13.md`(标注/clips 层 CLEAN,消费层此一项 CRITICAL,
   另有 NaN 约定/vis 死参数两个无实测影响的脆弱点)。

## 6. 复现与产物

```bash
# 单格训练(env: COND=flow|eefsp|eeffilm CROSSVIEW=1|0 SEED MIX=rh|r EPOCHS=60)
sbatch --export=ALL,SCRIPT=exp_scel_dualview_dit_formal.py,COND=flow,CROSSVIEW=1,SEED=0,MIX=rh sbatch/dualview_formal_train.sbatch
# 全 sweep(A=主门控6, B=eefsp+crossview12, C=human-helps6)
STAGE=A bash run_dualview_formal_sweep.sh   # 然后 B, C
# 聚合+门控+human-helps 判定
python agg_dualview_formal.py               # -> outputs/cross_embodiment_wm/dualview_dit_formal/ablation_table.md
# 端到端 / 对比 gif
MODE=e2e python exp_scel_dualview_dit_formal.py
MODE=gif python exp_scel_dualview_dit_formal.py
# IWS external baseline
sbatch --export=ALL,SCRIPT=exp_dualview_iws_stage2.py,SEED=0 sbatch/dualview_formal_train.sbatch
```

- 消融表:`outputs/cross_embodiment_wm/dualview_dit_formal/ablation_table.md`
- 每格 run:`outputs/cross_embodiment_wm/dualview_dit_formal/{cond}_cv{0|1}_s{seed}[_ronly]/{metrics.json,summary.txt,dvdit.pt}`(实体在 `/scr/yusenluo/iws_overflow/formal_runs/`,symlink)
- e2e:`outputs/cross_embodiment_wm/dualview_dit_formal/e2e/{metrics.json,summary.txt,gifs/seq*_cam{high,low}.gif}`
- 三臂+IWS 对比 gif:`outputs/cross_embodiment_wm/dualview_dit_formal/compare/gifs/seq*_cam{high,low}.gif`
- IWS baseline:`outputs/cross_embodiment_wm/dualview_iws_stage2/s0/{metrics.json,summary.txt,iws_dyn.pt(state_dict)}`
- 单测:`tests/test_dualview_formal.py`(13 项:mask 无泄漏/三臂 shape/eval det-rate/ADE/DF schedule 等)
