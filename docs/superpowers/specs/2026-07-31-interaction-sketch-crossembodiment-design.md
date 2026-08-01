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
- **② 的重新实现/重训**:rollout 机制在 §6.5 规定,但 ② 预测器 **引用现有 `rollout_dual`**(只预测 object-flow);attachment/contact 由 ContactDetector 导出。本 spec 不重造/重训 ②。

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

### 4.1 草图通道(★已实现 9 通道,每视角 2D)

每视角一张 2D 图(cam_high/cam_low),128 建 pool 到 16×16。**实现为 9 通道**(原设计 8ch + 补的 world_dz):

| # | 通道 | 内容(decoder 输入,每视角 2D) | robot 来源 | human 来源 |
|---|---|---|---|---|
| 0–2 | object-flow | 图像2D flow(dx,dy,footprint) | 2D tracker | 2D tracker |
| 3 | **world_dz** | 物体世界z位移 splat(物理竖直, 把lift从平移分离) | tracks3d(depth反投影) | tracks3d(★2026-08-01 augment补) |
| 4 | agent-skel | 2D 9点线画(grip烘入手指) | joint→FK sidecar | 手→**IK**→FK sidecar |
| 5 | grip 标量 | 广播成一层 | 本体感觉 grip | 手开合→自身range(sidecar grip饱和弃用) |
| 6 | agent-trace | eef3d 历史投影拖尾 | eef3d | eef3d |
| 7 | attachment 标量 | 广播(物体是否被夹爪绑定) | ContactDetector 2D | ContactDetector 2D |
| 8 | contact-splat | 接触点2D splat | ContactDetector 2D | ContactDetector 2D |

- ★**为何 2D + world_dz 而非纯3D-canonical**: object-flow 用两域都有的图像2D tracks(human物体轨迹原只2D); 物理竖直靠**单独 world_dz 通道**(从两域parquet都有的depth反投影的tracks3d, `augment_clips_tracks3d.py`)补回——图像flow看不出lift, dz给出来。can任务有lift/place, 这个通道有实价值(12-22cm抬升→通道0.6-0.86)。
- **grip 两处**:烘在skel手指 + 单独标量。**warp 不在草图**(归decoder §6, 从z0+flow算)。**全可视化**(每通道可解释图)。

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
- **输入通道**:9ch 草图 + warp(3,decode 时从 z0+flow 算) = **12ch**(每视角)。**重训**(grip_ro 是 flow+skel+warp 7ch)。
- **训练目标 = robot 重建**:encode robot 视频 → 草图 → decode 回 robot 像素。z0 = 该 clip **自己的首帧**(无匹配问题)。
- **主头 robot-only + aux 头 robot+human**(用户关键补充,利用 human 不重蹈"输入robot输出human"矛盾):
  - 主 latent 头(rf recon):loss **mask 到 batch 里 robot 那半**(human 不进主 recon MSE)。
  - aux 头(**域共享 object-centric 目标**,robot+human 全 batch 训,塑共享 trunk)。
  - ★**aux 目标必须域共享**(用户抓):**DINO 全帧 disjoint(582× 可分),两域两个不相干目标 → trunk 按域分流、不被逼共享 → 否决**。改用**物体**(同一罐子,object-flow probe 0.645 基本同域):
    - **主形式 = 物体位置 heatmap**(每视角在物体质心放 2D 高斯,trunk 从带噪 latent 预测)——空间、易监督、逼 trunk 定位共享物体,有 related-work 先例。
    - 备选/补充:物体 3D 位置回归(tracks3d 质心)、contact heatmap。可 ablate(heatmap vs 回归)。
    - 非循环:虽 object-flow 在 cond,但从**带噪 latent** 预测**干净物体位置**是非平凡去噪任务;担心循环可换未来物体运动。
  - 推理只用主头(纯 robot),但 trunk 已吃过 human → scarce 区受益 [[project_human_helps_renderer_exp]]。
  - **代码改动**:`train_multihead_wm.py::losses` 现在 main 全 batch 算,须改成 main 只算 robot 子集、aux 算全 batch;aux 目标 builder 出物体 heatmap(替 DINO)。
- **z0 = 普通输入**:训练时 = 自身首帧;下游应用自己负责 z0 来源(翻译=状态感知检索,已 de-risk)。

## 6.5 Dynamics & Rollout(world-model 闭环)

框架不止 encode→decode(那只是渲染器/autoencoder)。作为 **world model 必须能自回归预测未来**。关键:**草图 = world-model 的状态空间**,8 通道分两半——

| 半 | 通道 | 角色 |
|---|---|---|
| **控制/动作** | agent-skel、grip、agent-trace | **输入**:policy/teleop/human-demo 命令的 agent 轨迹 |
| **状态/响应** | object-flow、attachment、contact-point | 物体对 agent 的**响应** |

**预测器现状(重要):我们只有 object-flow 预测器 ②,attachment/contact 没有独立预测器。**
- **②** = 现成 `rollout_dual`(object-flow WM,action=agent eef 星座 dummy5/mp),从数据学 grasp carry-follow(grip 闭合→物体跟随),非硬编码 [[project_grasp_dynamics_hardcoded_blocker]]。**只预测 object-flow**。
- ★**attachment/contact 不单独预测,由 ContactDetector(§5)从 rollout 状态确定性导出**:ContactDetector 在 **encode 和 rollout 用同一套** `(tracks3d, eef3d, grip) → (attachment, contact_pt3d)`。所以 ② 只需预测 object-flow(它已隐含 grasp 效应),contact 两通道随预测出的物体位置 + 命令的 agent **自洽导出**。这就是"只有 flow 预测器"够用的原因。

**自回归 rollout 循环**:
1. 状态 = 物体 3D track 点(+ 当前 agent 状态);给 agent 动作序列(skel/grip/trace = 控制)。
2. 每步:agent 动作 → **②** 预测物体下一步 3D 运动(含 grasp 跟随)。
3. 用(命令 agent + ②预测 object + **ContactDetector 导出** attachment/contact)**建下一帧草图**。
4. **③** decode 草图 → 像素。回到 2。

→ 完整 WM = **② (草图/点空间动力学) → ③ (草图→像素 decode)**,草图是状态接口。e2e 就是 `eval_e2e_combined`:② rollout → 建草图 cond → ③ sample。③ 在 **GT 草图**训、用在 **②预测草图**(warp+grip 让 ③ 对 ② 误差鲁棒 [[project_gripwarp_e2e_robustness]])。

**边界**:② 是**引用现有**(不在本框架 spec 重实现/重训);本框架贡献 = 统一草图作状态接口 + ③ decode + contact 导出闭环。**未来选项(YAGNI,现由导出满足)**:学习式 contact 预测头 / 让 ② 直接预测 attachment。

## 7. 数据流(端到端)

1. **建草图 sidecar**(phantom + numpy):robot/human clips → tracks3d flow + skel2d(FK/IK)+ grip + eef3d trace + ContactDetector → 投影双视角 → 8ch 草图 npz(每视角 2D,pool16,retrack/L48 对齐)。中间产物出三联可视化。
2. **训 decoder**(.venv_wan sbatch):MultiHeadVideoWM,11ch(8草图+3warp),robot 重建主头 robot-only + aux(DINO)robot+human。episode-split。
3. **验证**(见 §8)。

## 8. 成功判据 & 测试

**框架层(本 spec)**:
- **robot 重建保真**:heldout robot clip,encode→草图→decode vs GT,报 **render LPIPS(含 agent)** + cube_px + 眼检并排 [[feedback_measure_real_deliverable_metric]]。目标 ≈ grip_ro 水平(LPIPS ~0.055–0.062)或更好。
- **草图可视化核验**:每通道 原帧|通道|叠加 三联,眼检 8 通道都对(flow/skel/grip/trace/attachment/contact)[[feedback_visualize_intermediates]]。
- **aux 头帮扶(scarce 消融)**:scarce robot 下,主头 robot-only + aux(robot+human) vs 无 aux,比 render LPIPS/cube。判 human 经 aux 有没有帮到。★aux 目标须**域共享**(物体 heatmap),别用 disjoint 的 DINO;可再 ablate heatmap vs 3D 回归。
- **ContactDetector 消融**:几何 vs 学习,比下游(重建保真 + 下游翻译穿模率)。

**下游(非本 spec,记录)**:翻译穿模率(状态感知锚)、数据生成保真。

## 9. 被否决的选项(留档,可复活 [[feedback_log_all_options]])

- **纯 2D 每视角草图**:曾选,后因"我们有 depth+双视角+真标定"改 3D canonical+投影。
- **纯 3D decoder**(不投影):decoder 要重造投影层,工程重,否。
- **warp 留在草图**:破坏"草图=纯抽象动态"解耦,移出归 decoder。
- **learned latent 草图**(非显式通道):域不变 latent 是老大难(DINO 582× 可分),且非"简笔画";用显式通道。
- **主头含 human recon**:重蹈"输入robot输出human"矛盾;human 只走 aux 头。
- **DINO 当 aux 目标**:全帧 DINO human/robot 完全 disjoint(582× 可分),两域两个不相干目标,trunk 不被逼共享 → 换域共享 object-centric(物体 heatmap)。
- **框架层解决锚匹配**:锚匹配是翻译下游的事,框架 z0 只是普通输入。

## 10. Related Work — 交互如何建模 & 我们的差异化(2026-08-01 调研)

领域共识:**"抽象交互表示"是跨具身正确接口**;分歧在**表示→policy vs decode→像素**。

| 交互表示 | 代表 | 去向 | 与我们关系 |
|---|---|---|---|
| **物-物相对位姿 SE(3)** | Human2Any(2606.28813) | 组合规划/policy | agent-free, 缺关系任务时可借(加物-物通道) |
| **3D 交互点 trace**(物体/手/工具/接触区) | **μ₀**(Furong Huang UMD, 2606.13769) | trace→policy(★不渲染) | ★最强对标+最强质疑: 批评pixel模型"浪费容量在外观"→只policy |
| **物体 flow**(2D/3D场) | Im2Flow2Act "Flow as cross-domain interface"(2407.15208), AMPLIFY(FSQ motion token) | policy | =我们object-flow通道 |
| **统一骨架/retarget** | OSCAR, DexWM(ACTION-Δ vs STATE), MT-π | policy | =我们agent-skel(IK统一, 像Human2Any retarget) |
| **学习式latent action** | LAPA, Genie, LAC-WM(ICML26) | WM/policy | 学习式对照(见下), 我们手设计绕开582×墙 |
| **★我们:交互草图** | object-flow+world_dz+agent-skel+grip+contact+warp, decode→像素 | **decode→像素** | 差异化=渲染 |

### 我们的差异化(护城河) & 核心质疑
- **几乎全场 表示→policy;只有我们 decode 回像素。** 这既是差异化也是 μ₀ 正面质疑点。
- **decode值不值**: 支持=①human→robot数据生成(翻译已验证)②可视化/可解释WM③想象/验证; 质疑(μ₀)=渲染烧容量在外观。护城河成立⇔"渲染下游价值"撑得住这句质疑。

### 学习式 latent action 怎么做(LAPA/Genie/LAC-WM)
- **无action标签, 从视频反推latent动作**: (1)**IDM** 看相邻两帧(x_t,x_{t+k})→推离散latent z(VQ码本); (2)**FDM** (x_t,z)→预测x_{t+k}; (3)**瓶颈**逼z只编码"变化/动作"(能从x_t看到的静态丢掉)。
- **跨具身**: z从像素转移学, 不绑robot action space; human"抬"和robot"抬"视觉转移像→映到相近z→同套latent action共享。
- **用法**: LAPA=Stage1学码本→Stage2视频预训policy预测z→Stage3少量真标签z→真action; LAC-WM=WM条件在统一latent action→迁移未见具身更好; Genie=同机制生成可交互世界。
- **vs 我们**: 他们**学**z(不透明VQ, 靠瓶颈逼共享粗动作); 我们**手设计**通道(靠已知共享object-flow probe0.645)。两条路通往同一目标(共享粗动作接口): 学习式(可扩展/不透明)vs手设计(可解释/可控/能decode)。★学习式能work正因瓶颈让z只抓粗动作(共享)不抓细外观(disjoint)——同我们object-flow道理, 只是自动vs显式。

### 可选升级(从related work借, YAGNI待需)
1. **物-物相对位姿通道**(Human2Any)→关系任务(A放进B), 现单物体flow缺关系。
2. **点trace表示**(μ₀)替/补图像通道(更3D紧凑), 已部分做(tracks3d→dz)。
3. **学习式latent通道**(LAPA)——但域不变latent难(582×), 手设计正为绕它。
