# 交互草图(Interaction Sketch)跨具身框架 — 设计文档

> 2026-07-31。设计者与用户(claudezhaoyang11)brainstorm 定稿。
> 目标读者:接手实现的工程师 / 下个 session。前置背景见 [[project_human2robot_translate_ro3]]、[[project_ik_eef2skel_human_agent]]、[[project_multihead_aux_wm]]。

## 0. 一句话

把 human/robot 视频都编码成**同一种域无关的抽象"交互草图"**(发生了什么:物体怎么动、agent 在哪、有没有抓着),再从草图 **decode 回像素**。草图是**看得见的简笔画**(每通道一张可解释的图)。框架 = **encode → 草图 → decode**;human→robot 翻译、数据生成等都是**下游应用**。

**差异化(护城河)**:领域内抽象运动/交互接口的工作(μ₀ interaction-trace、UMA、AMPLIFY、OSCAR)都是**草图→policy**;我们**decode 回像素**(+ object-centric + agent-skel 统一)。

## 1. 目标与非目标

**目标**:一个可复用的 **encode→草图→decode** 系统。
- 定义完整的草图表示(全通道)。
- 两个编码器(human 视频→草图、robot 视频→草图),结构统一,只 agent 分支不同。
- 一个 decoder(草图 + z0 → 像素),robot 重建为主训练目标,human 通过辅助头利用。

**非目标(本 spec 不含,属下游应用)**:
- human→robot 翻译的**锚来源/匹配**(是翻译这个下游应用的事,已用状态感知检索 de-risk,见 [[project_human2robot_translate_ro3]])。
- 草图→policy 下游。
- ② 预测草图演化(本 spec 只做 encode/decode,不做 dynamics 预测)。

## 2. 全局约束(Global Constraints)

- **数据版本对齐**:一律 retrack flow + human L48 + episode-split(heldout vids 100,102)。栽过 3 次数据版本错配 [[feedback_eval_data_version_match]]。
- **环境**:tracker/编码器几何在 numpy;IK/FK 在 conda `phantom`(真 pinocchio);decoder+WanVAE 在 `.venv_wan`。跨环境按现有两步式(sidecar 落盘)。
- **GPU 走 sbatch**(partition-1),别 nohup/登录节点 [[feedback_use_sbatch_not_nohup]]。
- **只 git add 自己的文件**,不加 Co-Authored-By [[iws untracked working tree]]。
- **每中间产物出可视化**(草图各通道 = 原帧|通道|叠加三联)[[feedback_visualize_intermediates]];eval 默认传 Drive [[feedback_upload_gifs_gdrive]]。
- **坐标系**:草图 canonical 形态 = **3D 世界系**(我们有 depth + 双视角 + 真标定 + tracks3d + eef3d),投影到 cam_high/cam_low 2D 喂 decoder。

## 3. 组件与接口

三个单元,清晰边界:

```
[原始视频 + 本体感觉/手姿]
        │  encode (§4)
        ▼
   [交互草图]  ← 8 通道 / 每视角 2D(3D canonical 投影而来)· 可视化
        │  + z0(decode 目标的首帧)
        │  decode (§6)
        ▼
     [像素]
```

- **SketchEncoder**(§4):video + agent 状态 → 草图。robot/human 共用除 agent-skel 分支。
- **ContactDetector**(§5):3D 几何量 → (attachment 标量, contact-point)。几何式 + 学习式,可插拔同接口。
- **SketchDecoder**(§6):(草图, z0) → 像素。MultiHeadVideoWM;主头 robot-only,aux 头 robot+human。

## 4. §1+§2 — 草图表示 & 编码器

### 4.1 草图通道(3D canonical → 投影 2D per view)

canonical 形态在 **3D 世界系**算,给 decoder 时投影到 cam_high/cam_low 各一张 2D 图。8 通道:

| # | 通道 | 3D 内容 | 投影后(decoder 输入,每视角) | robot 来源 | human 来源 |
|---|---|---|---|---|---|
| 1–3 | object-flow | tracks3d 的 3D 位移 | 2D flow(dx,dy,footprint) | tracker | tracker |
| 4 | agent-skel | 3D joint(grip 烘入手指开合) | 2D 9 点线画 | joint→**FK** | 手(HaMeR)→eef3→**IK**→joint→FK |
| 5 | grip 标量 | 标量(视角无关) | 广播成一层 | 本体感觉 grip | 手开合→自身range映射 |
| 6 | agent-trace | agent 3D 历史轨迹(eef3d) | 2D 拖尾 | eef3d 轨迹 | eef3d 轨迹 |
| 7 | attachment 标量 | 标量(物体是否被夹爪绑定) | 广播成一层 | ContactDetector | ContactDetector |
| 8 | contact-point | 3D 接触点 | 2D splat | ContactDetector | ContactDetector |

- **grip 两处都在**:烘在 skel 手指开合里 + 单独标量通道(用户定)。
- **warp 不在草图里**:warp 是外观派生,归 decoder(§6 从 z0+flow 算)。草图保持纯抽象动态。
- **全可视化**:每通道都是可解释图 → paper 可展示 `encode → 看得见的草图 → decode`。

### 4.2 编码器结构

统一原则:**共用管线在 3D 世界系算,只 agent-skel 分支按具身不同,最后统一投影双视角 2D**。
- object-flow:tracks3d → 3D 位移 → 投影(**改动点**:现在 flow_cond 用 2D tracks,升到 tracks3d)。
- agent-skel:已是 3D→投影(fk_skel2d_dual);human 走 IK([[project_ik_eef2skel_human_agent]] 的 ik_net)。
- agent-trace:eef3d 历史 → 投影拖尾(**新**)。
- grip:robot 本体感觉 / human 自身range映射(GRIP_MAX=0.04,human 用自身百分位映射,避免饱和)。

## 5. §2 — ContactDetector(attachment + contact-point)

新组件,两具身同一套(在 3D 算)。**两种实现,可插拔同接口,当 ablation 比下游**:

- **几何启发式(默认/bootstrap,不需标注)**:
  `attachment = σ(夹爪闭合程度) · σ(3D 夹爪-物体距离小) · σ(物体3D速度 与 夹爪3D速度 相关性)`(三条件软合成 0..1)。
  `contact-point(3D) = 夹爪指尖与物体最近点`。
- **学习式(升级)**:用几何规则**自举标签** → 训一个接触检测头 → 下游比几何式。
- 接口:`ContactDetector(tracks3d, eef3d, grip) → (attachment: (T,), contact_pt3d: (T,3))`。

## 6. §3 — SketchDecoder

- **架构**:MultiHeadVideoWM(= VideoDiT + 可选 aux 头),Wan-latent(C=48),z0 pin,rf 采样。承接 grip_ro([[project_mh3_warp_retrack_renderer]] 的 `epsplit_L48_mh/grip_ro`)。
- **输入通道**:8ch 草图 + warp(3,decode 时从 z0+flow 算) = **11ch**(每视角)。**重训**(grip_ro 是 flow+skel+warp 7ch)。
- **训练目标 = robot 重建**:encode robot 视频 → 草图 → decode 回 robot 像素。z0 = 该 clip **自己的首帧**(无匹配问题)。
- **主头 robot-only + aux 头 robot+human**(用户关键补充,利用 human 不重蹈"输入robot输出human"矛盾):
  - 主 latent 头(rf recon):loss **mask 到 batch 里 robot 那半**(human 不进主 recon MSE)。
  - aux 头(抽象目标,**DINO** 默认已验证 [[project_multihead_aux_wm]],可插拔):robot+human 全 batch 训,塑共享 trunk。
  - 推理只用主头(纯 robot),但 trunk 已吃过 human → scarce 区受益 [[project_human_helps_renderer_exp]]。
  - **代码改动**:`train_multihead_wm.py::losses` 现在 main 全 batch 算,须改成 main 只算 robot 子集、aux 算全 batch。
- **z0 = 普通输入**:训练时 = 自身首帧;下游应用自己负责 z0 来源(翻译=状态感知检索,已 de-risk)。

## 7. 数据流(端到端)

1. **建草图 sidecar**(phantom + numpy):robot/human clips → tracks3d flow + skel2d(FK/IK)+ grip + eef3d trace + ContactDetector → 投影双视角 → 8ch 草图 npz(每视角 2D,pool16,retrack/L48 对齐)。中间产物出三联可视化。
2. **训 decoder**(.venv_wan sbatch):MultiHeadVideoWM,11ch(8草图+3warp),robot 重建主头 robot-only + aux(DINO)robot+human。episode-split。
3. **验证**(见 §8)。

## 8. 成功判据 & 测试

**框架层(本 spec)**:
- **robot 重建保真**:heldout robot clip,encode→草图→decode vs GT,报 **render LPIPS(含 agent)** + cube_px + 眼检并排 [[feedback_measure_real_deliverable_metric]]。目标 ≈ grip_ro 水平(LPIPS ~0.055–0.062)或更好。
- **草图可视化核验**:每通道 原帧|通道|叠加 三联,眼检 8 通道都对(flow/skel/grip/trace/attachment/contact)[[feedback_visualize_intermediates]]。
- **aux 头帮扶(scarce 消融)**:scarce robot 下,主头 robot-only + aux(robot+human) vs 无 aux,比 render LPIPS/cube。判 human 经 aux 有没有帮到。
- **ContactDetector 消融**:几何 vs 学习,比下游(重建保真 + 下游翻译穿模率)。

**下游(非本 spec,记录)**:翻译穿模率(状态感知锚)、数据生成保真。

## 9. 被否决的选项(留档,可复活 [[feedback_log_all_options]])

- **纯 2D 每视角草图**:曾选,后因"我们有 depth+双视角+真标定"改 3D canonical+投影。
- **纯 3D decoder**(不投影):decoder 要重造投影层,工程重,否。
- **warp 留在草图**:破坏"草图=纯抽象动态"解耦,移出归 decoder。
- **learned latent 草图**(非显式通道):域不变 latent 是老大难(DINO 582× 可分),且非"简笔画";用显式通道。
- **主头含 human recon**:重蹈"输入robot输出human"矛盾;human 只走 aux 头。
- **框架层解决锚匹配**:锚匹配是翻译下游的事,框架 z0 只是普通输入。
