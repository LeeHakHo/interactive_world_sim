# SCEL — Structured Cross-Embodiment Latent (结构化跨具身共享表示)

Date: 2026-06-16
Branch: phantom_dynamo
Status: design (pending user review → implementation plan)

## 目标 / What this answers

把现有 object-flow 跨具身管线（48 物体点 + 3 eef 点 + footprint 渲染）升级成一个**结构化**的
human/robot 共享表示，让 human 演示数据更强、更稳地 help robot world model，**终极定位 = 可看的
simulator（生成 future video）**。不是 AMPLIFY 那种无差别密集网格撒点，而是有语义角色的节点-边图 +
受物理约束的预测目标。

Win condition（用户认可的判据）：
1. **下游 human-helps**（robot-only vs robot+human，端到端**渲染 cube 位置误差** px，统一口径
   `outputs/cross_embodiment_wm/`）—— 这是**唯一拍板判据**。
2. ② 的 flow 预测误差/长程漂移**下降**（ADE + 长程 rollout，对比现状 thin LWC 的 ~6.9px@40）。
3. ③ 渲染的外观/状态表达力**提升**（治当前 footprint 只能贴轮廓的糊）。
4. 框架对 **scale 友好**——后续会有更多**任务/数据**（task diversity；**是否多物体交互未定**，
   单 cube 单任务只是初步实验）。

probe（域可分性）**仅作诊断，不作门**——见下方"方法论铁律"。

## 方法论铁律（用户校准，写死）

- **probe 是充分不必要条件**：通道 probe≈0.5 → 几乎必然能迁移（充分）；但 probe RED（可分）**不代表
  不能迁移**（AMPLIFY grid 0.82 仍 +47% help 是反例）。所以 probe 只当**便宜的绿灯加速器/诊断**，
  RED 不砍方案，一切靠下游 human-helps 拍板。
- **张力（有条件，非铁律）：*无选择性地*加厚共享 latent 倾向重新引入域可分性**（差异藏在细节里）。
  但这是**方法缺陷不是普遍规律**——两次已知失败各有具体病根，**不能据此否定一切加厚**：
  - **AMPLIFY grid 400 点**：整图**均匀撒点**，必然撒到 agent/背景 → 把 agent 运动卷进共享 flow
    （agent 指纹 probe 0.987）。病在"撒点无选择性"，**不在"厚"**。
  - **thick-latent（contact/grasp gate）**：2D 弱 proxy（无 force/depth/wrist-cam）下 gate 退化成常数。
    病在"信号质量差"，**不在"厚"**。
  - → **结构化、有选择性、受约束的加厚/改造**（只在物体上、刚体约束、agent-frame）是**开放问题、未验证**，
    正是 SCEL 要探索的。盲目均匀加厚有风险，但 ≠ 加厚必死。
- **降 ② flow 误差：优先"加约束"（已有正面证据），结构化加厚是待验候选（不预先否定）**：grid *无选择性*
  加厚回归 ADE 11.1 **输** thin 9.0；LWC 赢（4.8 vs 9.2）靠**有界窗口分类**（约束自由度）+ scheduled
  sampling。→ 约束是确定有效的杠杆；选择性加厚须过下游判据，但不预先排除。

## 核心架构原则：厚度路由 per-domain，共享层加约束保持薄而 invariant

| 用户的"弱" | 放哪解决 | 怎么做 |
|---|---|---|
| #1 跨具身共享弱 | **共享层** | 不加厚，换**更受约束/更 invariant 的坐标系与预测目标**（agent-frame 相对坐标 + 速度归一化 + feasibility 加权）|
| #2 agent 表示太粗 | **拆两层** | 共享层 = grasp frame（薄、invariant）；per-domain ③ = agent shape **做厚**（human 用 `MANO_LEFT/RIGHT.pkl` 渲结构化手 mask，robot 用夹爪 mask），不进共享 |
| #3 渲染表达力弱 | **per-domain ③** | 免费加厚区：footprint+GAN 或 flow-conditioned latent decode；mask-then-color 两段渲染 |
| #4 物体状态表达弱 | **共享层** | 2D 单视角受限（已知）；加约束（刚体 pose）而非加维度。注：富物体-关系红利需多物体场景，而新数据**是否多物体未定** → **不作赌注** |

per-domain（③ 渲染、agent shape）是"免费加厚区"——robot 的厚度帮不了 human、反之亦然，但无所谓，
共享的只是"grasp→object 动力学"。

**注（非禁令）**：共享层**并不禁止**加厚——默认先用 per-domain 免费加厚区 + 共享层加约束（风险最低、
有正面证据）；但共享层的**结构化、选择性**加厚（区别于 AMPLIFY 均匀撒点）是**开放候选**，只要过下游
human-helps 判据就可采纳。"厚度路由 per-domain"是默认优先级，不是铁律。

## ① 数据（复用 `gen_flow_render_dataset_v3eef.py`，加派生字段，零重标注）

现有张量（`outputs/flow_render_dataset_v3/clips_{robot,human}.npz`）：
`frames(N,L,128,128,3)`、`tracks(N,L,48,2)`、`eef(N,L,3,2)`、`vis(N,L,48)`、`joint`(robot)、`vid`。

离线派生（一次，新脚本 `build_scel_graph.py` 或扩展 gen 脚本）：
- **object 节点** = `[cx, cy, scale]`：质心 = 48 点 mean；scale = 48 点散布（std 或 bbox 边长）。
  **无朝向**——48 点是 CoTracker 随机种、其 PCA 主轴反映撒布方式非物理朝向，且近方形刚体 + 2D 单视角
  下朝向退化、是噪声，砍掉。
- **agent 节点** = `[gx, gy, cosφ, sinφ, w]`：来自现有 eef 3点（base+2 指尖）的真实 3D pose 投影
  重参数化 → 位置/朝向/开合。朝向**可靠**（真实几何投影，非 PCA），保留。
- **per-domain 速度 z-score 统计量**（agent 节点速度归一化用，thick-latent 验证过的口径）。
- **relation 边**（**仅多物体场景，未定**）= 物体质心相对位移；**contact 边接口预留，第一期关闭**
  （label 取不到：无力/depth/wrist-cam，只有弱 2D proxy，thick-latent 证明会退化成常数）。

label 可得性：90% 免费（现有数据重参数化）；唯一硬骨头 contact 已推到第二期。

## ② 共享动力学（扩展 `amplify_wm.py` 的 LWC，不新开文件）

- **骨干**：图 transformer over 节点（object 节点 + agent 节点 + 现有 act 注入），LWC **有界窗口
  速度分类头** per 节点，**scheduled sampling**（R_SS）。每个节点通道带**自己的预测损失**
  （MaskWAM App C 护栏：side-input 会被忽略）。agent 节点速度**归一化**后再分类。
- **结构化改造候选（加约束降 flow 误差，可消融）**：
  1. **刚体平移约束 + 小残差**（廉价补充，**非主推**）：② 预测 object 的**整体平移**（1 个质心速度）+
     每点小残差（残差正则到小），而非 48 点各自乱跑。自由度 96 → 2+ε。
     ⚠️**不预测旋转**：2D 近方形 cube 旋转退化、48 点 PCA 主轴非物理朝向（与砍 object 朝向**同因**，
     含旋转会自相矛盾），预测 SE(2) 旋转 = 预测噪声。完整 SE(2)（含旋转）标**高风险可选，默认关**。
     难度 ≈ 现有 LWC（主预测只 1 个质心速度，更简单）。去掉旋转后边际不大，仅作廉价叠加。
  2. **agent-frame 相对坐标**（★★ 主推，一箭双雕）：object 运动表达在 grasp frame 坐标系（相对夹爪/手的
     运动），而非世界系。object 主要跟 agent 走 → 相对运动更小更可预测（降误差）+ 天然抵消 agent
     速度指纹（增 invariant，治 #1）。**这是约束主力**，候选 1 只是补充。
  3. **action 编码加厚**：eef 3点 → +速度/加速度/未来 action plan。action 是**已知输入**，加它不增
     预测自由度，只增条件信息，不犯共享性。
  4. （正交训练杠杆）**SS R_SS 16→更长**，直接治 compounding 漂移。
  5. **agent 多关键点（跨域对应 schema）**：agent 节点从 eef 3点 → K 个**功能对应**关键点（human 手与
     robot 夹爪映射到**相同语义角色**：腕/基座、拇指尖↔一指、食指尖↔另一指…）。⚠️**必须跨域语义对应**，
     否则就是 AMPLIFY 均匀撒点重演（各加各的分域几何）。agent 是分域核心 → 高风险，过 probe 诊断 + 下游
     判据；sweet spot 可能是少量功能点（4–5）。human 端从 MANO/检测的 21 关节里选对应点；robot 端夹爪
     DoF 有限，对应点取 base + 2 指尖（+ 可选指根）。
- **跨域 transition-delta 一致性正则（新，把"动力学共享"显式写成损失）**：核心 = human 和 robot 中
  **相似的 transition，其共享 latent 的 delta（Δz / 预测 flow delta）也应相似**。这比"static latent
  invariance"（要求观测对齐）更对路——我们要的是**动力学共享**，不是观测对齐；也可视为"共享 latent"的
  一部分（用户提出）。
  - **配对**：按 (state, action) 近邻配 human↔robot transition（相似 eef 位移 + 相似 object 起始状态），
    batch 内软配对（相似度加权）。
  - **损失**：`L_consist = Σ w_ij · ‖ Δz_i^human − Δz_j^robot ‖`，`w_ij` = transition 相似度；可选
    对比式（相似 delta 拉近、不相似推远）。
  - 与候选 2（agent-frame 相对坐标）**互补**：一个在表示层让运动 invariant、一个在损失层强制 Δz 一致，
    可叠加。配对噪声用下面的 feasibility 加权缓解。probe 仅诊断，判据走下游 human-helps。
- **feasibility 加权**（X-Diffusion 原理，治闭环反噬 gain 1.21）：闭环反噬可能只来自 robot-不可行的
  human 子集；per-trajectory feasibility 加权或替代一刀切，待评估。也用于上面 transition 配对的降噪
  （human 不可行子集降权）。

## agent shape（per-domain dead-end head，复用现有 g-mask）

grasp frame → g-mask silhouette（`train_maskgen_eefonly` 路线，已验 held-out IoU 0.73）。**做厚**：
human 用 MANO 渲结构化手 mask、robot 用夹爪 mask。dead-end，**不进共享图**，只喂 ③。

## ③ 渲染（per-domain 加厚区，复用 `exp_v3_human_helps_pixels.py`）

基线：I0 + object footprint（从 pose 重建凸包）+ g-mask（现有 footprint+novis 管线，H=40 GTflow 0.9px）。

加厚候选（对比实验，治 #3 渲染糊）：
- **A. footprint + GAN**（稳、忠实、低风险）：加 PatchGAN adversarial loss（`LAMBDA_GAN` 已留接口）治
  L2/LPIPS 回归糊。
- **B. flow-conditioned latent decode**（厚、高回报、高风险）：latent decode **严格条件在 ② 的 flow +
  I0 anchor 上、禁止自主推运动**（每帧重锚 I0，flow 当硬条件）。
  - ⚠️ **铁律：别让 ③ 当第二个 WM**。已弃的 latent-predict（`exp_v3_iws_latent_predict.py`，预测 z_t）
    长程幻觉 11.5px vs 像素 1.9px——病根是它自主推运动。B 与它的唯一区别 = "flow 锁得够不够死"，
    **必须实测长程是否还幻觉**。
- **C. mask-then-color 两段渲染**（备选 3）：先渲结构层（object occupancy + agent mask）再上色，分离
  位置/形状正确性与外观。两篇 paper 共识 mask-only ≥ RGB-only（RoboTwin 88.8 vs 87.3）。

## 备选主表示 plan B：Object-occupancy 共享表示（MaskWAM 移植）

与坐标图主线**并行、head-to-head**（同下游判据，谁 human-helps 强用谁）：
- 共享层 = **object occupancy mask 的未来序列**（不是坐标点）。② 在 mask latent 空间预测物体未来形状/
  位置（带自己预测损失 + App C 护栏）。
- 卖点：连续形状比点更结构化；object mask 天然 embodiment-invariant（manipulator 不在里面）；contact
  几何化（mask 边界重叠）；mask 软误差可能比 48 点硬 ADE 更宽容。
- 风险：回 2D 栅格、需可靠 object mask 监督（`cube_mask_v3` robot 有、human 帧待验）；latent-predict
  判弃教训（但 object-mask 是窄得多的目标，非全外观，可能逃过幻觉，需重测）。

## （条件扩展）promptable 多物体（MaskWAM mask-prompt）

**仅当**新数据含多物体交互时启用（**未定**）：首帧 mask prompt 指定 active object，一个 WM 处理多物体
场景。relation 边同理——多物体才有意义。这两项是**投机性接口**，不是第一期赌注，不阻塞主线。新数据若是
"更多任务/仍单物体"，则 scale 红利改走**跨任务泛化 + human 覆盖更多任务模式帮 robot**（见 M5）。

## 判决矩阵 / 评估

- **主判据**：下游 human-helps（端到端渲染 cube 位置 px，robot-only vs robot+human，统一口径
  `outputs/cross_embodiment_wm/`，per-experiment 独立目录 + `summary.txt` 存盘）。
- **辅助**：② ADE + 长程 rollout 漂移（vs thin LWC 6.9px@40）；③ 渲染质量（含 GT-flow 列作渲染天花板）。
- **诊断**：probe（grasp 抽象/agent-frame 后 agent 可分性是否降），不作门。
- **铁律**：cube_pos_err 的 nan 陷阱——必报检测率 + 消失记罚 + 逐帧眼检；跨方法比较同序列同设置；
  声明用的组件版本；mask 质量眼检 overlay。

## 里程碑

- **M0**：派生字段 + 图数据构建；pose/grasp/scale 重建 overlay **眼检**对（robot+human 各几帧）。
- **M1**：图 ② LWC+SS 训通，单 cube human-helps **不输**现状散点 flow（持平即过）。第一期实打实的红利
  **不赌未来数据**，落在 M2/M3（降 ② 误差 + 治渲染糊 + 增 invariant），这些**单任务单 cube 就能验**。
- **M2**：共享层改造消融（刚体变换 / agent-frame 相对 / action 加厚 / SS 加长 / **agent 多关键点** /
  **transition-delta 一致性正则**）→ ② flow 误差是否下降 + human-helps 是否变强（ADE + 长程 + 下游）。
- **M3**：③ 渲染加厚对比（footprint+GAN vs flow-conditioned latent decode vs mask-then-color）治糊 #3。
- **M4**：备选 plan B（object-occupancy）head-to-head vs 坐标图主线（同下游判据）。
- **M5**（条件，依新数据形态）：若多物体 → relation 边 + promptable 多物体接口 smoke；若"更多任务/仍
  单物体" → 跨任务泛化 + human 覆盖更多任务模式的 human-helps 验证。

## 范围 / 非目标

- 第一期定位 = **enabling refactor + agent 侧验证 + 降 ② 误差 + 治渲染糊**，**不是**单 cube 上 human-helps 暴涨。
- **scale 维度 = 更多任务/数据多样性**（task diversity），**不是确定的多物体交互**；第一期红利**不寄托未来数据**。
- **contact 边**第一期不做（label 取不到，第二期 + 更好传感）。
- **3D / 6D pose** 出局（Point Policy 证单视角 depth 致命）。
- **World-Action joint / policy** 出局（用户定位 = WM 不是 policy）。
- object 朝向出局（2D 不可靠）。

## 风险

- 单 cube object 状态红利小（已知）；**不靠"多物体兑现"兜底**（新数据未定多物体）——第一期红利改落在
  降 ② 误差 + 治糊 + 增 invariant（单任务可验）。
- agent-frame 相对坐标实现需小心：grasp frame 自身在动，相对坐标的积分/anchor 要对齐（M2 验）。
- flow-conditioned latent decode 可能仍幻觉（B 的成败赌点，M3 实测）。
- MANO 手 mask 渲染需对齐到 v3 crop 坐标系（per-domain，不影响共享层）。

## 复用 / 不重开脚本（feedback: 别脚本 sprawl）

- ② 改造：扩展 `amplify_wm.py`（FlowWM_LWC / train_lwc_ss / rollout_lwc）。
- ③ 渲染：扩展 `exp_v3_human_helps_pixels.py`（Renderer / make_flow / FOOTPRINT / LAMBDA_GAN 接口已留）。
- 数据：扩展 `gen_flow_render_dataset_v3eef.py` 或新 `build_scel_graph.py`（仅派生，复用现有 npz）。
- 评估：走 `exp_v3_full_vs_scarce_human.py` / `eval_pixel_H40.py` 现有口径。
- 备选 B（occupancy）：可复用 `cube_mask_v3` + latent decode 脚手架。
