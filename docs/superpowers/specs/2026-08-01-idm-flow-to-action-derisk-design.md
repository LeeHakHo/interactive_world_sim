# IDM: object-flow + agent-trace → robot action (Step 0 de-risk) — Design

**日期**: 2026-08-01
**状态**: 设计已与用户逐段确认(§1 数据流 / §2 网络 / §3 闭环判据),待用户审 spec → writing-plans。

---

## 0. 一句话

学一个 inverse dynamics model(IDM),把 `(object-flow + agent接触trace + grip)` 映射到 robot action `(Δjoint + grip)`,让 human demo 未来能被解成可执行 robot 动作(Do-As-I-Do / Behavior-Prompting-Policy 式)。**本 spec 只做 Step 0:robot-only、offline、闭环判据的 de-risk** —— 证"flow+agent-trace 到底够不够解出能驱动物体的 action"。

## 1. 动机与前置结论(为什么这么设计)

- object-flow 是我们已证的**域不变跨具身接口**(② flow-cond 明显赢 eef-cond)。但 **object-flow 单独对 IDM 是病态的**:物体 flow 只在**接触时**带 action 信息,机械臂自由空间伸手时 flow≡0 → action 无解。故必须加 **agent 自身运动**。
- agent 侧用 **mp 接触 trace**(接触点 c + 指轴 + 开合),**不用重建全臂 skel**:实测重建全臂关节轻度 OOD(去 j5 后 2.4×)、且 skelik 当动作有 recovery-from-rest 自回归塌缩;mp 是我们最好的动作 rep(丢了 OOD 的腕,保真实接触/朝向)。
- 输出 **Δjoint 不用 Δeef**:6DoF IK 是死路(memory `feedback_trossen_overlay_ik`);直接出关节增量避开 IK,且 robot 有 joint GT 做监督。

## 2. 分期(本 spec 只做 Step 0)

| Step | 内容 | 本 spec? |
|---|---|---|
| **Step 0** | robot-only、offline:robot 数据训 IDM,held-out 闭环判据 | ✅ **是** |
| Step 1 | human demo:human flow+trace → IDM → action → 前向 WM 里验证复现 human 物体运动 | ❌ 后续 |
| Step 2 | 真机执行 | ❌ 很后面 |

## 3. 表示与数据流

> ★**输入集是起点假设,不是定论**:哪些组件真帮 IDM 是经验问题 → Step 0 **含输入消融**(见 §4.5),用闭环判据挑最小够用集。下面是起点。

### 输入(每个时刻 t 的双向窗口 `[t-K, t+F]`)
- **object-flow**: 物体 48-track 2D 轨迹(双视角),**含未来段** F —— 动作的"效果"可见,是逆模型 well-posed 的关键。
- **agent 接触 trace**: mp 式(接触点 c + 指轴单位向量 + 开合标量),逐帧。
- **grip**: 归一化开合(0-1)。
- **eef-proprio**: 末端位姿 —— **由 agent-trace 本身携带**(eef-pose 就是 trace 的几何部分),**不单独设关节 proprio 输入**。

### ★关键设计选择
1. **proprio = eef-pose(在 agent-trace 里),不喂 joint-angle**:human 推理时没有 robot 关节,但 eef-pose 从 human trace 拿得到 → 跨具身可用。代价:IDM 从 eef-pose 学 `eef→Δjoint`(同一 eef-pose 可能多关节解,IDM 学典型解)= **position-only-IK 式映射**,正好和 memory "IK 必须 position-only" 一致,不碰 6DoF 死路。
2. **action = Δjoint(6-dof)+ grip**:关节增量,避开 IK。

### 数据源(现成)
- `outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz`: `tracks`/`tracks_low`(48-track object-flow,双视角)、`eef`/`eef3d`(agent 3 点)、`joint`(2700,48,6+)、`grip`、`vid`、`low_valid`。
- 监督标签:`Δjoint[t] = joint[t+1]-joint[t]`(6-dof);grip 取 `grip[t]`。
- Split:episode-split `HELDOUT_VIDS=100,102`(无帧泄漏,和 ②/③ 一致)。

## 4. IDM 网络

- **架构**: VPT 式双向窗口 → 中心帧动作。窗口 `[t-K, t+F]` 的 (object-flow, agent-trace, grip) flatten → 小 temporal net(MLP 或小 transformer,和现有组件同量级)→ 中心帧 `Δjoint(6)+grip(1)`。对 t 滚动得整条动作轨迹。
- **contact-aware**: grip/接触做显式输入通道(grip-aware 一贯有用)。
- **监督**: robot 数据 supervised regression(MSE / Huber on Δjoint;grip 用 MSE 或 BCE)。
- **为什么双向窗口**: 单看 t 之前推不出动作(自由空间 flow=0);把物体**未来怎么动**放进输入,IDM 反推"什么动作造成了这个效果"。
- 参数:`K,F` 复用 ② 的量级(K=4,F=20)作起点,可 sweep。

## 4.5 输入消融(Step 0 的一部分 —— 挑最小够用输入集)

输入组件用**闭环判据(§5)判**,不靠先验。精简网格(每臂训一个 IDM,同 held-out 评):

| 臂 | 输入 | 测什么 |
|---|---|---|
| **A0** | 只 object-flow | 薄基线,预期**自由段崩**(坐实病态) |
| **A1** | flow + agent-trace(mp) | 加 agent 运动,自由段是否被救 |
| **A2** | A1 + 显式 grip 通道 | grip timing 是否需要单列 |
| **A3** | A1 但 **causal 窗口(只过去)** vs 双向 | 未来 object-flow(VPT 假设)到底帮不帮逆推 |
| A4(可选) | agent-trace 变体:mp vs 原始 eef 3 点 vs 只接触点 c | 接触 trace 的最简形式 |

- **判据**: 每臂报 eef-recon(自由/接触分开)+ object-flow-recon;**挑通过闭环的最小集**当 IDM 定稿。
- **YAGNI**: A0/A1/A3 是核心必跑(证病态 + 救没救 + 窗口方向);A2/A4 视 A1 结果决定要不要。
- 这样"输入端组件调整"是**实验驱动**的,不在 plan 里拍死。

## 5. 闭环判据(Step 0 de-risk 的成功判据)

held-out robot demo 上:
1. IDM 滚动预测整条 `Δjoint` 轨迹。
2. 从 GT `joint_0` **积分**得关节轨迹 → **FK(pinocchio,phantom env)得 eef 轨迹**(ee_gripper_link + 两指 carriage → 3 点 eef)。
3. **(a) eef 重构误差**: FK-eef 轨迹 vs GT eef 轨迹(mm/px),**自由段 / 接触段分开报**(病态在自由段;接触段 = grip 闭合或物体在动的帧)。
4. **(b) object-flow 复现**: FK-eef → mp 接触 trace → 喂**前向 ②(mp robot WM,`mp_r_all/wm_dual.pt`)** → 预测 object-flow vs GT object-flow(px)。**判"解出的 action 真能驱动物体"。**
5. **报**: eef-recon(mm,自由/接触分开)、object-flow-recon(px)、grip 准确率。眼检并排 overlay(GT eef vs FK-eef vs GT object vs 复现 object)。

### 判据方向(green/red gate)
- **PASS**: 接触段 eef 重构 + object-flow 复现都好。阈值**相对标定**(避免拍脑袋绝对值):① 接触段 object-flow-recon ≈ ② 自身 GT-flow 天花板(即用 GT eef 喂 ② 的 object-flow 误差,~2-3px 量级)的同量级;② 接触段 eef-recon 在物体 track 噪声/tracking 尺度内。具体数值在 plan 阶段用 held-out 上"GT-action 经同一 FK+② 管线"的误差当天花板来标定。→ IDM 站得住,进 Step 1(human)。
- **FAIL**: 自由段 eef 崩 → eef-proprio+trace 不够约束自由空间 → 回头加厚(物体 3D/旋转)或转 forward-WM planning。

### 环境
- IDM 训练/推理:`iws`(torch)。
- FK:`phantom`(pinocchio),和 skel 建管线一致。
- 前向 ② 评估:`iws`,加载 `mp_r_all/wm_dual.pt`。

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| 自由空间 Δjoint 多解(病态) | 输入含 eef-pose(agent-trace)+ 双向窗口;闭环判据**自由/接触分开报**直接暴露 |
| IDM 自回归积分漂移(skelik 教训) | Step 0 先测**从 GT joint_0 积分**;若漂移大,报 teacher-forced(每步 GT proprio)vs 自积分,定位是否 exposure bias |
| grip timing(何时闭合/松手) | 显式 grip 通道 + 判据里单列 grip 准确率 |
| position-only 映射丢关节解信息 | 接受(跨具身必要);判据看 eef 重构是否够好,而非关节 MSE |

## 7. 交付物(Step 0)

- `idm_train.py`(训练,输入集经 env/config 可切 → 支持 §4.5 消融)+ `idm_eval.py`(闭环判据,含 FK + 前向 ② + 并排 overlay gif)。
- **输入消融表**(§4.5 各臂 × eef-recon/object-flow-recon,自由/接触分开)→ 挑最小够用输入集。
- held-out 闭环 summary + Drive 上的 overlay gif。
- PASS/FAIL 判决(基于选定输入集),决定是否进 Step 1。

## 8. 明确不做(scope 边界)

- 不做 human demo 推理(Step 1)。
- 不做真机(Step 2)。
- 不做 forward-WM planning(仅作为 FAIL 后的备选,不在本 spec)。
- 不重建全臂 skel 当输入(实测 OOD + 脆)。
