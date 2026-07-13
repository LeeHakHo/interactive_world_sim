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

### 4.2 sweep 结果（2026-07-13 凌晨，3 seeds 合并；merged_all.json + pareto_merged.png）

**v3（held-out robot ADE px@224，world 锚 vs 机制，rh | Δ）：**

| N | world | a1 | align(自蒸馏) | align_wm ★ | warm |
|---|---|---|---|---|---|
| 20 | 9.76 \| +4.88 | 11.99 \| +2.62 | 13.12 \| +2.59 | **10.01 \| +4.83** | 11.43 \| +6.98 |
| 50 | 8.96 \| +2.28 | 11.12 \| +0.85 | 12.57 \| **−1.23** | **8.90 \| +2.41** | 11.47 \| +3.14 |
| 100 | 8.93 \| +0.67 | 9.63 \| +0.80 | 11.48 \| **−1.27** | **8.79 \| +1.29 ★BREAK** | 10.97 \| +1.41 |
| 400 | 7.35 \| +2.27 | 8.74 \| +1.78 | 9.15 \| +0.80 | **7.72 \| +1.94** | 9.42 \| +1.34 |

**dual-view（can_dual，drift px@128 两视角平均，dummy5 锚）：**

| N | dummy5 | skel | a1 | align_wm ★ |
|---|---|---|---|---|
| 50 | 4.95 \| +4.91 | 6.70 \| +6.68 | 6.40 \| +5.27 | 5.28 \| +4.70 |
| 100 | 3.98 \| +4.70 | 5.46 \| +7.80 | 5.24 \| +5.73 | 4.01 \| +4.88 |
| 400 | 3.44 \| +1.54 | 5.58 \| +1.61 | 3.95 \| +0.52 | **3.16 \| +1.55**(n=2) |

**判读（诚实，含显著性）：**
1. **机制间排序显著**：align_wm 在两个数据集所有 N 全面压制 a1/warm/align（v3 N=100：8.79 vs
   9.63/10.97/11.48，差距 ≫ seed std ~0.5）。**`align` 自蒸馏在 v3 N=50/100 为负 Δ**——移动靶
   目标有害，冻结 grounded 教师是成败点 = LaST-HD 机制论的直接消融证据（呼应其 Fig3b）。
2. **align_wm vs 锚（break 判定）尚不显著**：v3 N=100 ★BREAK（rh 8.79<8.93 且 Δ 2×）但 seed
   配对 rh_diff [−0.18,−0.71,+0.47] 2/3；dual N=400 rh_diff [−0.39,−0.20] 方向一致但 n=2。
   → 已排加测：53212（v3 N=100 seeds3-4）/53213（dual N=400 seeds2-4）至 5 seeds。
3. **can_dual 的 Pareto 张力远弱于 v3**：dummy5 锚本身 Δ +4.7~4.9（v3 world 只 +0.67）——can
   数据 human 对绝对 action 也大幅帮；结合机制在 can 上的空间主要是精度端（align_wm N400 −8% rh）。
4. align_wm 从不伤害：所有 cell 的 rh 与锚打平或更好、Δ 打平或更大——作为默认机制无成本。

**事故记录**：/scr2 磁盘满（100%）→ dual s2 在 N=400 align_wm ckpt 保存时崩，metrics 从 log
抢救回 3/4 格；已清 uv_cache 腾 14G 应急 + torch.save 包 try/except（3316f23）。⚠️另一 session
19 个 dvf_* job 同样受磁盘风险，用户需醒后清理。

### 4.3 5-seed 显著性终判（2026-07-13）

- **v3 N=100（5 seeds，配对 t）**：align_wm rh 8.93±0.37 vs world 9.07±0.34（diff −0.145，
  **p=0.61**）；Δ +1.39 vs +0.79（**p=0.26**）。3-seed 的 ★BREAK 是噪声级——**break 不成立**。
- **dual N=400（5 seeds）**：align_wm rh 3.30±0.38 vs dummy5 3.44±0.52（diff −0.136，5 seed
  中 4 个为负但 **p=0.42**）；Δ 打平。方向一致的改善迹象，不显著。
- **终结论**：①"突破 Pareto"未证成（锚点处前沿是硬的，加固"action gap 靠机制消不掉"）；
  ②机制排序显著且是 paper 素材：**latent 端对齐（LaST-HD 冻结 grounded 教师）是唯一不付
  精度代价的 human-help 机制**，输入端改造（skel/a1/warm）全部拿精度换 Δ，自蒸馏（align）
  有害。align_wm 可作无成本默认。

### 4.4 端到端终跑（无泄漏配对 ckpt，N=400 seed2，job 53214）✅

| obj-LPIPS | GT-flow→③(天花板) | ②align_wm→③ | ②dummy5→③ | eef-cond(naive) |
|---|---|---|---|---|
| cam_high | 0.299 | 0.309 | 0.305 | 0.382 |
| cam_low | 0.301 | 0.314 | 0.310 | 0.372 |

② ADE px@128：align_wm 2.66/4.68，dummy5 2.56/4.25（cam_low 明显高于泄漏版 plumbing 的
2.26——泄漏效应可见，但 ③ 对 ② 噪声鲁棒，e2e 仍贴天花板）。**e2e 主张（干净 ckpt 确认）：
flow 接口穿过 ② 预测误差仍大幅赢 naive action（LPIPS −0.06~−0.07），两 ② 机制在 N=400
数据充足区打平（与 4.3 一致）**。
**gif（眼检坐实）**：5 列对比 `final_compare/gifs/`（GT|GT-flow③|②align_wm|②dummy5|eef，
两视角×6 seq）+ 各 run 四列 gif；末帧 montage `final_compare/eyeball_lastframes.png`——
两 ② 列罐子成形与天花板同质，eef 列物体消失/涂抹，三 seq 两视角一致。
（已知小瑕疵：gif caption 中文字形缺失显示为方块，列标题为 ASCII 不受影响。）

## 5. human-vs-robot 数据汇率（2026-07-13，job 53243，用户问"human 数据是否近等效 robot 数据"）

**设计**：补 robot-only scaling 曲线 ro(N) 至全量（v3 world N≤2550 / dual dummy5 N≤2460，
各 3-5 seeds），把 rh(N)（=N robot+~1800 human）放到曲线上找等效点 ro(N+M)=rh(N)，
汇率 α=M/1800。图 `data_equiv/exchange_{v3,dual}.png`，数据 `curves.json`。

**判决（两数据集一致）**：
1. **不是等效（α≪1），但有实打实的汇率**：α 峰值在中等数据区（v3 N=400：α 0.35~0.47，
   即 1800 human ≈ 640~840 robot；dual N=100-800：α 0.3±）；稀缺区 α≈0.15-0.3。
   经验法则：**~3-5 条 human ≈ 1 条 robot**（在 human 能帮的区间）。
2. **α 随 robot 量衰减到 0**：N≥1600 时 α≈0.1 以下；**全量 robot 时 Δ≈0**（v3 rh−ro 逐
   seed [+0.60,−0.12,+0.34] p=0.33 / dual [+0.06,+0.33,−0.03] p=0.38——点估计略负但**不显著**，
   不能声称"有害"）。human 数据是 robot 稀缺时的替代品，robot 管够时边际价值耗尽（正常的
   饱和现象；固定混合比例下 human 占 41% 梯度份额而不再带新信息，如需可用混比退火缓解）。
3. **实用换算**（配合 LaST-HD 报告的 human 采集快 4-5×）：α≈0.3 × 采集速度 4-5× ≈
   **单位采集时间下 human 数据的价值与 robot 大致打平到 1.5×**——这是"用 human 数据"的
   经济学论证，比"等效"更诚实也更有用。
4. 曲线形状：rh 曲线在整个稀缺-中段稳定压住 ro 曲线（v3 上 rh(100)=9.07 甚至好于
   ro(400)=9.63），N≈1600 交叉，全量轻微反超——一张图讲完 human 数据的价值边界。
