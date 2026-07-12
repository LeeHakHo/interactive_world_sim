# dual-view DiT ③ 正式化:flow 域不变接口 > naive action(paper-ready)

> 目标:把"flow 接口 > naive action 接口"这一核心结果,在 dual-view DiT ③ 上做成 paper-ready——
> **controlled**(同 backbone 唯一变量=接口)、**端到端**(② pred-flow 非 replay 理想 flow)、**多 seed 严谨**、可复现。
> 分支 phantom_dynamo。数据 flow_render_dataset_can_dual。现有脚本 `exp_scel_dualview_dit.py`。
> 前置:[[project_dualview_dit_flow_vs_eef]](已出 flow>eef)、[[project_dualview_dit_formalize]](转向记录)。

## 0. 现状与本正式化要补的四个不够 solid 处

现状:`exp_scel_dualview_dit.py`(275 行)= DualViewDiT ③(两视角联合注意力 add-DiT + frozen 16ch VAE,
co-train robot+human),flow-cond vs eef-cond 已出判决(cam_high obj-LPIPS 0.326 vs 0.393,cam_low 0.342 vs 0.427)。
**但四处不够 solid,本正式化逐一补**:
1. **replay GT flow**:render() 喂 GT tracks/eef。→ M2 端到端接 ② pred-flow。
2. **baseline 严谨性**:eef-cond 是"IWS-stage2-style"近似,未论证等价、未和真 IWS 对照。→ M1 controlled 论证 + external ref。
3. **单 seed 30ep**:LPIPS 差距 0.067 可能被 seed 噪声吞。→ M1 多 seed variance-first。
4. **消融不全**:cross-view 联合注意力的价值没单独量化。→ M1 attention-mask 消融。

## 1. 成功判据(paper claim 成立)

1. **Controlled 三方(replay GT flow,同 DiT backbone,唯一变量=条件注入)**:flow(ours)在 cam_high+cam_low
   两视角 obj-LPIPS 都优于 eef-FiLM(=IWS stage2 注入机制),且 **≥3 seed 差距 > seed std**(统计显著,非噪声)。
2. **端到端**:用 ② pred-flow(非 GT)驱动 ③,flow 仍优于 eef-FiLM → 整条 ②→③ pipeline 成立,而非仅理想 flow 下。
3. **Cross-view 消融**:量化联合注意力 vs 阻断跨视角(参数不变)的 obj-LPIPS 差,给出其价值。
4. **External baseline(重要 baseline)**:真 IWS stage2 整 pipeline(CMLatentDynamics,Conv3d backbone,原生
   action-conditioned)同数据同口径训练+eval,进对照表;注明非 controlled(backbone 不同),但 flow-DiT 应整体不输它。

## 2. 对比哲学(本正式化的 solid 核心)

**controlled ablation 是主证据**:固定 DiT backbone + frozen VAE + 同训练预算/seed,**唯一变量 = 条件注入方式**:
- **flow**:per-view 物体 flow splat(dx,dy)+ footprint,spatial-add 注入(`ce` 卷积 → emb_cond,加到 token)。域不变物体运动接口(ours)。
- **eef-FiLM**:4-dim 单 eef 点 [x,y,dx,dy] → `act_emd` Linear → per-block FiLM scale/shift。**代码级等价于 IWS stage2**:
  `cm_latent_dynamics.ResnetBlock` 的 `cond_emb_layers: Linear(cond_dim, dim_out*2)` → `h=h*(1+cond_scale)+cond_shift`,
  与本脚本 eef-cond 的 adaLN FiLM 同构。故 eef-FiLM = "IWS stage2 的 action 注入机制,移植到同一 DiT backbone"。
  这样 flow vs eef 就隔离了"接口"单一变量。

**external baseline(重要 baseline,正式项非 stretch)**:真 IWS stage2 整 pipeline(CMLatentDynamics + Conv3d 时空
backbone,原生 action-conditioned 注入)独立训一个,同数据同 eval 口径报 obj-LPIPS/PSNR。它 backbone 与我们不同 →
**backbone+接口两变量混杂,不能作为"接口价值"的 controlled 证据**(controlled 证据仍由同 backbone 的 eef-FiLM 承担),
但它回答"我们的 flow-DiT 相对现成 IWS stage2 整体如何"——paper 的对照表必须有这一行(用户 2026-07-12 定为重要 baseline)。

## 3. 组件分解(3 里程碑,C/D 贯穿)

### M1 — Controlled 硬消融(replay GT flow)
- **三方注入 × cross-view {ON, OFF} × seed {0,1,2}**,同 DiT backbone。
- **cross-view OFF = attention-mask 阻断**(不是拆成两个网络):block 内 attention 加 mask,禁止 view-0 token 看 view-1 token
  (反之亦然),**参数量/容量不变**,只断跨视角信息流 → 干净隔离"跨视角联合"的价值。
- **多 seed variance-first**:先 seed{0,1,2} 各跑一遍,量 obj-LPIPS 的 seed std;若 std 接近 flow-eef 差距(0.067),
  加 seed 到 5 或延长 epoch,直到差距 > 2×std 才下结论。别用单点差声称赢。
- **更长训练**:EPOCHS 30 → 60(或到 val 收敛),记 loss 曲线。
- **eval 口径审计(单一权威)**:per-view obj-region LPIPS(object footprint 附近 64×64 crop)+ PSNR;
  审计现有 `obj_lpips` 的 crop clamp 逻辑(line 192 的 min/max)是否稳健;**报 object 检测率**,footprint 点 <5 的帧
  记为无效而非计入(避免"物体消失"被当好,[[feedback_cube_metric_nan_trap]])。
- **M1c human-helps 数据混合消融**:MIX ∈ {rh(co-train 现状), r(robot-only)} × {flow, eeffilm} × seed{0,1,2},
  cv 固定 1。对比各臂的 human 增益 Δ,回答"flow 接口是否更能利用 human 数据"(诚实报,不作硬门)。
- 交付:`(3 cond × 2 crossview + 2 cond × 2 mix) × 2 view` obj-LPIPS/PSNR 表(带 seed mean±std)+ 每格 gif。

### M2 — 端到端接 ② pred-flow
- `rollout_dualview(wm_dual.pt, tracks, eef, H)` 产双视角 pred-flow(现成 ② ckpt),替换 render() 的 GT flow。
- **三列隔离 ②/③ 误差**:`GT-flow→③(③天花板) | ②pred-flow→③(端到端) | eef-FiLM→③(naive)`,4 列含 GT。
  额外报 **② flow ADE px(pred vs GT)** 作中间诊断——让读者看到 ② 误差吃掉多少,而非把 ③ 的糊错怪到接口。
- 判据:端到端列 flow 仍优于 eef-FiLM(哪怕被 ② 误差拉低),且 GT-flow 列是上界。
- 交付:端到端 4 列 gif(两视角)+ metric（含 ② ADE）。

### M3 — 可复现 pipeline + 交付
- 探索脚本 → 正式:env/config 清晰(COND/CROSSVIEW/SEED/EPOCHS)、训练日志、`metrics.json`、
  `DUALVIEW_DIT_REPORT.md`(自包含:设置/消融表/端到端/结论/图路径)、全套 gif、drive 上传。
- 单一权威 eval 函数(M1 审计后的口径),别多份口径。

## 4. 固定项 / 非目标(避免 scope 蔓延)

- **human-helps 消融纳入正式化(用户 2026-07-12 推翻旧固定项)**:project 的核心命题是 human 数据帮 robot,
  必须比较 robot-only vs robot+human → **M1c**:MIX ∈ {rh, r} × {flow, eeffilm} × seed{0,1,2}(cv1)。
  claim 方向:flow 接口是否比 naive eef 从 human 数据里榨出更多帮助(Δ = LPIPS(r-only) − LPIPS(r+h) 按臂对比)。
  注意先验:③ 层 human-helps 曾判天花板 +0.03([[project_dualview_dit_flow_vs_eef]]),结果可能仍是"③ 层不帮、
  帮在 ②"——**诚实报两种结局都进 paper**,不作硬门;e2e 的 ② human-helps(1.7-2×)已另有判决支撑整条线。
- **数据固定** can_dual;逐视角 2D flow(不做 3D 三角化,[[project_dualview_point_correspondence_qc]] 证 48 点独立采样
  3D 是假象)。注意 cam_low 的 tracks_low/vis_low 质量,eval 用 low_valid mask。
- **backbone 固定** add-DiT([[project_detmem_dit]] 赢家),不重开 backbone 搜索。

## 5. 风险与预案

| 风险 | 预案 |
|---|---|
| seed std 吞掉 0.067 差距 | variance-first:先测 std,不足则加 seed/epoch 到 差距>2×std;老实报"不显著"也是结果 |
| ② pred-flow 太差,端到端 flow 反输 eef | 三列隔离 + 报 ② ADE;若 ② 是瓶颈,明确"接口在 GT-flow 上界成立、端到端受 ② 限",诚实入 report |
| external baseline 真 IWS stage2 接线大/配置重 | 正式项但独立里程碑(M2.5),不阻断 M1/M2;controlled(eef-FiLM)仍是接口价值主证据;接线优先复用 IWS 现成 config/trainer,别重写 |
| cross-view attention-mask 实现错(信息泄漏) | 单测:构造 view-1 全零输入,验 view-0 输出在 OFF 下与含/不含 view-1 无关 |
| 本 session shell 输出损坏 | 一切 ground truth 用 Read/Write/python os/git 直查;实现放 shell 正常的 fresh session,不在坏环境勉强跑训练 |
| iws env / GPU | iws conda；跑前 nvidia-smi 选空卡；不用 subagent(本 session 委托反复幻觉,controller 亲自) |

## 6. 里程碑顺序与产物

1. **M1a**:cross-view attention-mask + 单测(信息不泄漏)。
2. **M1b**:三方 × crossview × 多 seed sweep（replay GT）→ 消融表 + variance 判定。
2.5. **M1c**:{flow, eeffilm} × {rh, r-only} × 3 seed(cv1)→ human-helps Δ 对比行。
3. **M2**:接 rollout_dual(exp_scel_dualview_wm)→ 端到端 3 列 + ② ADE。
4. **M2.5**:external baseline 真 IWS stage2(CMLatentDynamics 原生 action-conditioned)同数据同口径 → 对照表补行。
5. **M3**:pipeline 整理 + REPORT + gif + drive。
产物根 `outputs/cross_embodiment_wm/dualview_dit_formal/`;报告 `DUALVIEW_DIT_REPORT.md`。
