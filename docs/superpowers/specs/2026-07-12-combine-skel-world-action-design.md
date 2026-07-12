# 结合 skel + world action featurization:同时拿 human-helps 与最终精度

> 目标:突破 velocity_action 线得出的"单一 featurization 不可两全 Pareto 前沿"。
> 分支 phantom_dynamo。数据 flow_render_dataset_v3。协议严格复用 `exp_scel_agentframe`。
> 前置结论见 memory `project_velocity_action_featurization` / `VELOCITY_ACTION_LOG.md`。

## 1. 问题与成功判据

**Pareto 前沿(已证,N=100 rh↓ / human-helps Δ↑ 反序):**
`world` rh 8.93 / Δ +0.68(精度上界,迁移下界) … `skel` rh 11.3 / Δ +11(迁移上界,精度下界)。
无一单一 featurization 在最终精度上支配 world。根因:让 action 对预测有用的接触几何 = embodiment-specific = 破坏迁移那份。

**成功判据(用户锚定 1,路径从"先验机制"起步):**
- **判据 1(最终目标)**:某结合机制在 held-out robot ADE 上 **rh ≤ 8.93(不输 world)** 且 **human-helps Δ 明显 > +0.68**。即落到 Pareto 图上 world 点的**左上方**。
- **起步(先看机制成不成立)**:该机制的 rh 落在 world(8.93)与 skel(11.3)之间、同时 Δ 显著为正且 > world;并在渲染层(4 列 gif)看得见 cube 跟得比 skel 紧。达到即"机制 work",再调优冲判据 1。

**核心设计原则(所有机制共用):精度来源纯 robot(world 通道),human 只从共享通道(skel)帮进来。** human 样本永不更新 world 精度通道。

## 2. 启发来源

- **LaST-HD(arxiv 2606.23685)**:不在 action 输入端对齐,而在 forward-dynamics 的 shared latent 端对齐——auxiliary world model 产 unified latent target,监督跨具身表示对齐,mixed co-train + human-correction。→ 印证"共享性在 dynamics 侧"。`align` 机制直接落地它。
- 我们的否定线 `mask_subtract` / `contact_gate`:代理指标↑ ≠ 干预有用 → 判据必看最终 rh,不单看 Δ。

## 3. 机制定义(sweep 成员)

记 `skel_feat = VelLWC._featurize(eef3,objc; "skel")`(同域,ACT_DIM Lw·10),`world_feat = ..."world"`(绝对,Lw·6)。`is_h`=样本是否 human(idx ≥ Nr)。

**锚(复用现有 VelLWC,作上下界):**
- `world` — 纯 world 通道。精度上界 / 迁移下界。
- `skel` — 纯 skel 通道。迁移上界 / 精度下界。

**结合机制(新 `CombLWC`,forward 收 `is_h` mask):**

- **A1 加性 residual(主攻)**:
  ```
  act = act_shared(skel_feat)  +  α · masked_world(world_feat, is_h)
  masked_world 对 human 样本输出置 0(human 只走 shared 主体)
  ```
  shared 是主体(human+robot 共训,human 帮);world 是 robot-only 精度残差,看得到主体(残差头输入可拼接 shared token,LaST-HD 式"只补差")。
  **防残差独吞(成败手)**:对 world 残差输出加 L2 幅度惩罚 `λ_res·‖masked_world‖²`(默认 λ_res=1e-3),迫使"能靠 shared 解释的别用残差";α 默认 0.5,小初始化残差头。消融 λ_res∈{0, 1e-3, 1e-2}。

- **align(LaST-HD)**:
  ```
  robot 主预测走 world path(保精度);human 主预测走 shared path(human world disjoint 无意义)
  consistency: L_align = ‖ z_world(robot) − z_shared(robot).detach() ‖²   (逼 world-path 继承 shared 结构)
  z_shared 由 robot+human 共训(human 帮它)
  ```
  λ_align 默认 0.3,消融 {0.1,0.3,1.0}。

- **warm(两阶段课程)**:
  ```
  Stage1: train skel path, idx = robot+human           # backbone 学 shared dynamics(human 帮)
  Stage2: 加 world residual 头, load Stage1 backbone(低 lr/部分 freeze), idx = robot  # 拉精度
  ```
  human-helps 以 warm-start 存进 backbone。消融 Stage2 是否 freeze backbone。

**备选(先不纳入,Other 可后补)**:`aux` 多任务辅助头;`adversarial`(gradient-reversal,高风险对照);`target-side Δ`(与 featurization 正交,可叠加)。

## 4. 实现

**位置**:新文件 `exp_scel_combine_action.py`,`import exp_scel_velocity_action as V` 复用 `V._featurize`(经 VelLWC 实例)/ `V.ACT_DIM` / 协议常量;`import exp_scel_agentframe as X` 复用 DS/HELDOUT/ade_world。**不改动已验证的 velocity 线代码**(避免回归)。遵守 unify:一个文件承载 A1/align/warm 机制族,`COMBINE` env 切换,不每机制一脚本。

**`CombLWC(A.FlowWM_LWC)`**:
- `__init__`:`act_shared = Linear(ACT_DIM["skel"], Dm)`;`act_world = Linear(ACT_DIM["world"], Dm)`;`combine`∈{a1,align,warm};α/λ 超参。
- `_feat_pair(eef3,objc)`:复用 velocity 的 skel/world featurize(拷两段或实例化一个 VelLWC 取其 `_featurize`)。
- `trunk(hist, eef3, is_h)`:object tokens 不变;按 `combine` 组装 act token(A1 加性 + mask;align 双 path 返回 z_world/z_shared;warm 单 path 由 stage 决定)。返回 (x, anchor[, aux_latents])。
- forward 需把 `is_h` 透传;`ade_world`/`rollout_lwc` 评估时全 robot → `is_h=False`(world 残差全开),无需改评价代码,只需 forward 对缺省 `is_h` 取全 False。

**训练 `train_comb(tracks,vis,eef,idx,combine,seed)`**:以 `V.train_feat` 的 SS loop 为骨架,增量:
- batch 内按 `b >= Nr` 得 `is_h`,传入 forward。
- A1:主 CE loss + `λ_res` 残差正则。
- align:主 CE(robot 走 world、human 走 shared)+ `λ_align·L_align`。
- warm:两次调用(stage1 skel all / stage2 world residual robot,载入 backbone)。
- 其余(scheduled sampling p 调度、vel_to_class、加权 CE、opt)逐字复用。

**main**:沿用 velocity 的 N_ROB_LIST/SEEDS/ro-vs-rh 双训练 + `X.ade_world` 评价 + `_save`/`_plot`。锚 world/skel 走 `V.train_feat`,结合机制走 `train_comb`。输出目录 `outputs/cross_embodiment_wm/combine_action/`。

## 5. 评价与交付

- **协议**:held-out robot ADE px@224(H=16),HELDOUT=150,ro(N_rob robot)vs rh(+1800 human),Δ=ro−rh。N∈{20,50,100,400}×3 seeds。**逐字复用 `X.ade_world`,不新造口径。**
- **主交付**:Pareto 图(横 rh 精度、纵 human-helps Δ),标出 world/skel 锚 + 各结合机制,画出"world 左上方"目标区;`summary.txt` 表。
- **渲染补齐**:对达到起步门槛/突破判据 1 的机制,复用 `exp_scel_velocity_render.py` 出 4 列 gif `GT | GT-flow→③ | world-② | <机制>-②`,眼检 cube 跟随,遵 gif layout 硬要求 + 眼检 + 组件版本声明。上传 drive。
- **落盘**:结果写 `COMBINE_ACTION_LOG.md`(option 表:机制/rh/Δ/是否突破/为什么),更新 memory。

## 6. 风险与预案

| 风险 | 预案 |
|---|---|
| A1 残差独吞 → 退化回 world、Δ 归零 | λ_res 正则 + α 小 + 消融;监控 world 残差范数占比 |
| align λ 难调、L_align 与主 loss 抢梯度 | λ_align 消融;先 SMOKE 验梯度不爆 |
| warm stage2 遗忘 shared | freeze/低 lr 消融;对比 stage1-only 的 Δ |
| 都达不到判据 1(前沿可能真硬) | 起步门槛即有价值:量化"能逼近多少",作为"action gap 靠机制消不掉"的加固证据(诚实负结果同样入 paper,同 mask_subtract/contact_gate 族) |
| iws 环境 / GPU | iws conda env;跑前 nvidia-smi 选空 GPU |

## 7. 里程碑

1. `CombLWC` + `train_comb`(A1)+ SMOKE(2 epoch,验 forward/domain-mask/正则不爆)。
2. A1 full sweep(N×seed)→ Pareto 表,判定是否达起步门槛。
3. align + warm 补齐 sweep,合表。
4. 突破点(或最优点)渲染补齐 4 列 gif + 上传 drive + 落盘 log/memory。
