# COMBINE action featurization 研究日志（skel+world 结合机制，LaST-HD 启发）

> 目标：突破 velocity_action 线的 Pareto（单一 featurization 不能同时拿 human-helps 和最终精度）。
> 2026-07-12 深夜由用户定向：结合 paper 启发改方法（human-helps + 精度双赢）、解决"只是把
> action-cond 换成 pred-flow-cond"的太简单 concern、基础架构用双视角联合预测 + DiT。
> spec: `docs/superpowers/specs/2026-07-12-combine-skel-world-action-design.md`
> 另一 session 并行做 ③ formalize（`exp_scel_dualview_dit_formal.py`，三臂+crossview mask），
> 本线=②机制（`exp_scel_combine_action.py` v3 / `exp_scel_dualview_comb.py` can_dual 双视角）+
> 端到端接线（`exp_scel_dualview_e2e.py`）。

## 0. LaST-HD (arxiv 2606.23685) 精读要点（2026-07-12，PDF 在仓库根）

- 机制：**独立的 action-conditioned world model**（unpaired human+robot 混训，冻结）产
  forward-dynamics latent target，cosine 监督主模型 latent reasoning；`L = L_act + λ·L_latent`。
- 核心 insight（与我们 object-flow 线互证）："推苹果产生的物体运动与具身无关"——
  **action 条件下的预测性特征**编码物理后果所以跨域共享；视觉未来帧特征只编码外观、不共享。
- 关键消融（Fig 3b）：action-conditioned WM 目标 **73%** > SigLIP 视觉 66% > 无 action 条件 WM
  63% > 无 latent 监督 60%。→ 目标必须"action-conditioned + 预测性 + 被真实未来 grounding"。
- 训练配方：Stage1 混合 co-train；Stage2 human 在线纠错（真机 DAgger，暂不适用于我们离线 eval）。

## 1. Option 表（机制族）

| option | 定义 | 状态 | 为什么 |
|---|---|---|---|
| `world`(v3)/`dummy5`(dual) | 绝对星座 action（接触几何，disjoint） | 锚 | 精度上界/迁移下界（velocity 线判决） |
| `skel` | OSCAR 骨架分解（同域） | 锚 | 迁移上界/精度下界 |
| `a1` 加性 residual | shared(skel) 主体 + robot-only world 残差（human 置 0），λ_res 防独吞 | sweep 中 | 主攻机制（spec §3） |
| `align` 自蒸馏 | 同网络双 path，L2 拉 world→shared.detach()（只 robot） | sweep 中（仅 v3） | LaST-HD 的简化近似；目标是移动靶——预期弱于 align_wm |
| `align_wm` ★ | **LaST-HD faithful**：冻结混训 skel-WM 教师（被 flow CE grounding）+ cosine 全样本对齐，主路径纯 world | sweep 中 | 修正 align 三处偏离（独立冻结教师/grounding/cosine+双域监督）；教师复用 sweep 的 skel 锚（同 idx 同 seed→Δ 诚实，成本≈0） |
| `warm` 两阶段 | stage1 skel 混训 → freeze backbone + stage2 world 残差 robot-only | sweep 中（仅 v3） | 课程式；风险=stage2 遗忘 |
| `aux`/`adversarial`/`target-side Δ` | — | 未纳入 | spec 备选，Other 可后补 |

## 2. 实现与协议

- v3 线：`exp_scel_combine_action.py`（commit dd5bdda 加 align_wm）。协议逐字复用
  exp_scel_agentframe（HELDOUT=150，held-out robot ADE px@224 H=16，ro vs rh，N×3seeds）。
- 双视角线：`exp_scel_dualview_comb.py`（commit f8e9023）。`DualCombLWC(DualLWC)`：
  2P token+view emb 基础上加 act_shared（双视角 skel）/act_world（双视角 dummy5=基类 act）；
  can_dual **L24**（robot 2700+human 1800 统一批），heldout 150 robot（low_valid 过滤），
  每视角 drift px@128（H=20），R_SS=16。**human-helps 首次进 dual-view ②**（原 wm_dual 只 robot）。
- 端到端：`exp_scel_dualview_e2e.py`。四列 `GT | GT-flow→③ | ②pred-flow→③ | eef-cond(naive)`，
  ③=dualview_dit 2026-07-07 探索版 ckpt（正式版待另一 session M1 出品后换），报 ② ADE 隔离误差。
  L24 heldout=②正当 heldout；窗口级重叠是项目既有协议（18 条源视频），报告注明。

## 3. 运行记录

- 2026-07-12 深夜：v3 sweep sbatch **53157**[0-2]（seed 切分，6 方法×4N），dual sweep sbatch
  **53158**[0-2]（4 方法×3N）。合并用各 OUT 的 `res.json`。
- e2e 管线先用 dummy5 `wm_dual.pt` 验通（`dualview_e2e/plumb_dummy5/`）；align_wm ckpt 由
  sweep 保存（`wm_alignwm_rh_N*.pt`），出来后换 `WM2` 重跑；dummy5 同 split 配对 ckpt 需补训。

## 4. 结果

### 4.1 e2e 管线验通（2026-07-12 深夜，②=dummy5 wm_dual.pt 探索版，⚠️其训练为 L48 全量，
对 L24 heldout 有窗口泄漏——最终数字待 sweep 配对 ckpt 重跑；组件版本已声明）

| cam | GT-flow→③(天花板) | ②pred-flow→③(端到端) | eef-cond(naive) |
|---|---|---|---|
| high objLPIPS | 0.299 | **0.307** | 0.382 |
| low objLPIPS | 0.301 | **0.308** | 0.372 |
| high/low PSNR | 21.95/21.81 | 21.89/21.68 | 20.91/21.16 |

② flow ADE 2.12/2.26 px@128。**端到端列几乎贴住天花板（gap 0.007-0.008 LPIPS），naive eef
被甩开 −0.064~−0.075**→"② pred-flow 误差不吃掉 flow 接口优势"成立（M2 判据,探索版③上）。
**眼检坐实**（`plumb_dummy5/eyeball_lastframes.png`）：②pred 列罐子成形与天花板同质,eef 列
物体涂抹/消失,三 seq 两视角一致。gifs `outputs/cross_embodiment_wm/dualview_e2e/plumb_dummy5/gifs/`。

### 4.2 sweep 结果（待 53157/53158 完成填写）

（占位：v3 Pareto 表 / dual-view Pareto 表 / align_wm e2e 重跑 / BREAKTHROUGH 判定）
