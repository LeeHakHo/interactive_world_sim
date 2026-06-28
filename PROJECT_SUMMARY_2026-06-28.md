# Cross-Embodiment World Model — Project Summary (2026-06-28)

> 自包含总结,给协作者 / 网页版 Claude 当背景。涵盖 setup、尝试过的所有方向、结果、结论。
> 仓库 `/scr2/yusenluo/interactive_world_sim`(IWS),分支 `phantom_dynamo`。

---

## 0. 一句话

**目标:训一个 cross-embodiment world model,让 human 演示数据帮助 robot world model(human helps robot)。**
**当前主线结论:像素级跨具身不可行 → 转 object-flow(物体 2D 流)作跨具身接口 → human 确实帮 robot,且帮助走 object-flow 这条通道(不走渲染层)。**

---

## 1. Setup

### 1.1 问题设定
- human 用手、robot 用 Trossen 夹爪,做相似桌面任务(play 数据:桌上蓝框工作区 + 红 cube)。
- 假设(曾经):human/robot 共享场景/物理,只差 agent(手 vs 夹爪)。**此假设被证伪**(见 §3.1)。

### 1.2 数据
- robot: `play_robot_v3_*_eef`(18 eps,2 相机 cam_high + cam_right_wrist),crop **(190,225,210,205)→128**。
- human: `play_human_v3_eef_*`(12 eps,只有 cam_high)。**human 有 eef(腕+2指尖)但无 gripper width、无 joint。**
- 处理后 clip 数据集:`flow_render_dataset_v3`(robot 2700 + human 1800 clips,L=24)、`flow_render_dataset_v3_grip`(robot,L=48,带 grip 字段)。
- npz 字段:frames / tracks(48-pt object flow)/ eef(3点投影到图像 2D)/ vis / joint / grip / vid。
- **关键:eef 是 world eef 投影到图像的 2D 坐标(不是 world 3D)** —— object-flow 整个建模在图像 2D 平面,因为 2D 图像是 human/robot 共享空间(world 3D 不共享)。

### 1.3 架构 A(当前主架构)= 三组件
- **① object-flow**:CoTracker3 抽 cube 的 48-pt 2D 轨迹(+ eef 3点)。
- **② FlowWM**(`FlowWM_LWC` / grasp-aware `GripLWC`):action(eef 相对 cube + grip)→ 预测 cube 2D flow,15×15 速度网格分类,scheduled-sampling 训练。
- **③ 渲染器**(`detmem` = DetMemRenderer):flow + gmask 条件 → 像素。16ch VAE(ostris,frozen)+ latent transition(z_prev→z_next 确定性)+ decode-LPIPS + prev-memory 自回归。**不用 flow-matching**。
- 端到端:eef → ② → cube flow → ③ → pixel。

### 1.4 环境
- conda env `iws`(`/scr/yusenluo/anaconda3/envs/iws`);跑需 `HF_HUB_OFFLINE=1 VAE_NAME=ostris/vae-kl-f8-d16 PYTHONPATH=<repo>`;GPU 1/5/7。robot 视频 AV1(用 PyAV `av` 解,cv2 不行)。

---

## 2. 尝试过的所有方向(含舍弃)

| 方向 | 方法 | 结果 | 采用? |
|---|---|---|---|
| **像素/特征对齐** | OT(EgoBridge 风格,Sinkhorn+DTW) | 跟重建打架,PSNR 39→25 | ❌ 死路 |
| | channel_split(GRL 对抗) | min-max 不稳 | ❌ |
| | dual_head(CLUB 互信息) | z_emb 塌缩 | ❌ |
| | emb_film(架构分解 global embedding) | 重建健康 PSNR 39,但 z_scene 仍 probe 0.96 可分 | ⚠️ 不对齐 |
| | mask_subtract(rank-1 残差正交投影减"具身轴") | 只移除 1.78% 能量,z_scene 仍 1.0 可分;agent 区(0.94)不比整帧(0.97)更分域 → 无集中"具身方向"可减 | ❌ 设计性死结 |
| | inpaint 去 agent | gap 582×→30× 但 probe 仍 ≈1.0;残差=场景纹理+伪影非agent | ⚠️ 不充分 |
| **object-flow** | cube 48-pt 2D flow ② + 渲染 | 物体 flow 跨具身同域(probe 0.645 vs 像素1.0);human helps cube ADE 降25% | ✅ **主线** |
| | AMPLIFY 密集网格 flow(400点+FSQ) | probe 门 RED(0.82)但下游 human-helps 仍+39~47% → probe 太严是误判 | ⚠️ 备选 |
| **③ 渲染器**(7 arch) | pixel renderer / temporal-SPADE / **detmem-16ch** / WEAVER(flow-matching) / lpips-lat / DF / pixel-footprint | **detmem-16ch 最锐**(LPIPS 0.138<pixel 0.141);temporal-SPADE 最稳 | ✅ detmem |
| **grasp/contact** | contact gate(接触门控干预) | AUC 高(0.95)但 gate 干预无用+有害(baseline ②本就接触敏感) | ❌ 否定 |
| | grip②(夹爪开合入 ②) | drift 改善 35%,但 grasp 只 5/30 学到,weld 未破 | ⚠️ 部分 |
| | IK adapter(eef→joint,position-only+temporal) | in-dist joint RMSE<<STD,gmask IoU 0.80 | ✅ 用于 agent 渲染 |
| **结构化表示** | agent-frame 相对坐标 | 强降 ②误差(-38%)但"用结构替代 human"→**杀 human-helps**,与命题冲突 | ❌ 战略冲突 |
| **latent 跨具身** | residual(Δz)vs direct(绝对z)latent WM 混训 | **residual 帮 robot**(起点不跨域但 dynamics 变化跨域);与 object-flow 帮幅相当 | ✅ 转折 |
| **keyboard 交互 demo** | 组合动作驱动 ②/③(平移/开合/box) | cube 可控已证;开合可控但 grasp 不稳;box 绕圈回原点 sanity | ✅ demo |

---

## 3. 关键结果(数字)

### 3.1 跨具身可分性(为什么放弃像素)
- 原始观测 DINOv2:linear probe **1.0**(完全可分),between/within 距离比 **582×**。
- 物体 2D flow:probe **0.645**,MMD ratio 1-1.5×(基本同域)→ object-flow 绿灯。

### 3.2 human helps(走 object-flow ②)
- `13_v3_human_helps_flow`:② robot-only vs robot+human,held-out robot cube flow ADE:
  N_rob 50→ 25.2 vs **19.3**(Δ+5.9);100→ +6.2;200→ +5.8(scarce regime human 显著帮);400→ -6.7(数据足不需 human)。
- 穿到像素:渲染 cube 位置准 **2.4×**(28→12px)。

### 3.3 ★ keyboard 四列对比 + object-flow ablation(2026-06-28,本 session)
四列 `GT | IWS-naive | ours-noflow | ours`,v3_grip 同批 seq,cam_high,frame K..L-1 对齐:

| metric | IWS-naive | ours-noflow | ours |
|---|---|---|---|
| PSNR↑ | 14.8 | 23.4 | **24.0** |
| LPIPS↓ | 0.214 | 0.080 | **0.062** |
| cube 位置↓ | 17.9px | 6.0px | **2.2px** |

- **IWS-naive** = hyeonhoo 的 2-view VAE+latent transition + eef naive co-train(EgoBridge 框架,无 OT)。
- **ours-noflow** = 我们**同一个 VAE+latent transition(detmem)**,但 cond 换 eef-splat **去掉 object-flow**,robot+human co-train。
- **ours** = object-flow ②③(detmem + flow)。

**三结论(控制变量,诚实)**:
1. **ours-noflow ≫ IWS**(23.4 vs 14.8,cube 6.0 vs 17.9)→ 同样"无flow+eef+co-train",**我们的 VAE+latent 实现碾压 hyeonhoo;赢 IWS 主因是架构实现,不是 flow**。
2. **ours > ours-noflow**(cube 2.2 vs 6.0,~2.7×)→ object-flow 增益**集中在 cube 位置精度**,整帧 PSNR/LPIPS 差距小;seq-dependent(简单跟随 noflow 够,复杂/长 rollout flow 关键)。
3. **ours-noflow rh(robot+human cube 3.1)≈ ro(robot-only 3.0)**→ **human 帮不上渲染层**(③外观任务,human/robot 外观 582× 可分)。

**★★ 核心论点(由 #2+#3)**:**object-flow 是 human→robot 迁移的载体/接口。** human-helps 走 ② object-flow 通道;去掉 flow(ours-noflow),human 帮助接不进来(rh≈ro)。

### 3.4 渲染器
- detmem-16ch:LPIPS 0.138 < pixel 0.141(latent route 锐度翻盘赢 pixel)。四要素(16ch VAE+确定性+decode-LPIPS+prev-memory)缺一不可。

### 3.5 grasp(未解)
- ② 把 cube 无条件 weld 在 eef;grip 数据存在(被 MOVE_MIN 滤掉)。重切 v3_grip + grip 入②:drift 改善 35%,但 grasp 只 5/30 学到、weld 未破。开环设 grip 无接触物理(已知 open issue)。

---

## 4. 当前结论

1. **像素/特征级跨具身不可行**:human/robot 观测 582× 可分,所有对齐/分解/inpaint 失败;gap 是场景级、分布式,不是可减的"具身轴"。
2. **object-flow 是跨具身接口**:物体 2D flow 在两域同域;human helps 走这条通道。
3. **human 帮 ② dynamics,不帮 ③ 渲染**:渲染是外观任务,human/robot 外观不同帮不上(rh≈ro);human 的知识通过 object-flow dynamics 迁移。
4. **赢 baseline(IWS naive co-train)= 架构实现 + object-flow 精度**:我们的 VAE+latent transition 实现本身远好于 hyeonhoo;object-flow 再加 cube 精度 2.7×。
5. **渲染器 detmem-16ch 最锐**(latent route 赢 pixel)。
6. **latent residual 也跨具身**(Δz>>绝对z)→ object-flow 非唯一接口。
7. **grasp/weld 未解**:cube 无条件焊在 eef,grip 入②改善但未破;开环无接触物理。

---

## 5. 未决 / 下一步

- **z(深度)task**:object-flow 是 2D 投影,平面 task 够(eef z 钉平面);未来有 z 运动的 task 需 2.5D(2D flow 同域骨干 + depth 通道)或 3D point tracking。这是 2D 路线的边界。
- **grasp 接触约束**:grip 离散两档缺接触物理(close 下限应=cube宽度 if 抓握 else 0)。
- **② 静止 OOD 自漂移**:数据 MOVE_MIN 滤静止 → ② 没学"夹爪不动 cube 不动";纯原地命令(pump)cube 漂,移动命令好。
- **paper story**:object-flow 作显式主接口(可控+跨具身)vs related work 把 object motion 当辅助监督。

---

## 6. Related work 对标

- **OSCAR(2606.04463,可引)**:别人版 cross-embody WM = VAE+latent transition + 2D skeleton 统一条件 + human(MANO)+robot naive co-train。证明 human warm-start 帮 robot、structured condition 绕外观 gap、latent-action 不精确。**它就是我们的 IWS-naive baseline 的同类**;它没做机制分析(residual)、没 object-centric、没可控 demo = 我们空地。
- **TRI Human-WM slide**(confidential):human WM,structured condition(skeleton+warping)+ explicit supervision(MANO/mask/object dynamics)。warping 思路可借鉴治 detmem 外观锚死。
- **UMA(Cornell)**:object motion 跨具身接口,印证 object-flow 方向(但它给 action 不渲染像素)。
- 我们的差异化:**object-flow 作显式主接口 + 机制分析(human-helps 走 flow 通道,residual)+ 可控/轻量**。

---

## 7. 关键文件 / 文档索引
- 这条 keyboard 四列对比 + ablation:`exp_keyboard_3way.py` + `exp_detmem_eef_cotrain.py`,输出 `outputs/cross_embodiment_wm/keyboard_3way_iws/`(INDEX.md + gifs + summary)。
- 早期像素对齐全记录:`CROSS_EMBODIMENT_WM_ATTEMPTS.md`(§4.1 含 OSCAR/TRI 对标)。
- flow WM 线总 report:`FLOW_WM_REPORT.md`。
- 渲染器/grasp/keyboard 历程:`RENDERER_GRASP_RESEARCH_LOG.md`、`RENDERER_IMPL_DETAILS.md`。
- IWS-naive ckpt load 配方:`checkpoints/epoch=3-step=250000.ckpt`(hyeonhoo 2view H+R,build_cfg+stage2+dynamo_ssl.encoder_backbone='conv2d')。
