# ③ Cond 注入升级:VAE-encode flow/skel + token fusion(照 FlowWAM/OSCAR)设计 (2026-08-02)

**Goal:** 把 ③ 渲染器的 spatial cond(object-flow / agent-skel / warp)注入方式,从"栅格图 avg-pool 到 16×16 + channel concat"(丢亚格细节)升级为"全分辨率栅格过同一个 Wan VAE → 专属 patch-embed → token 融合",保住亚 16px 细节,提升渲染画质 + agent 落位精度。

**Architecture:** 复用现有 Wan-VAE latent video-DiT(MultiHeadVideoWM)。改的只有 cond 进 DiT 的那一段:去掉 `pool16` avg-pool,cond 栅格走 VAE-encode + 独立 patch-embed + token-add 融合。DiT 主干 warm-start 自 grip_ro_tL12。

**Tech Stack:** .venv_wan(Wan2.2 VAE + video DiT),train_multihead_wm.py,eval_e2e_combined.py。

## Global Constraints
- Wan VAE:16× 空间压缩(256²→16×16),4× 时序(48 帧→tL=12)。cond 必须对齐 latent 的 (tL,16,16)。
- 只改 cond 注入,不动 VAE、不动 ② object-flow WM。
- go-forward 数据口径:retrack + L48 + episode-split(HELDOUT_VIDS=100,102),tL12。
- 冻结 Wan VAE(不微调);cond patch-embed 从 RGB patch-embed 权重初始化(FlowWAM:收敛更快)。

---

## 1. 问题(为什么改)

现状(eval_e2e_combined / build_can_cond):
- cond = 栅格化的 [flow 3ch + skel 1ch] 或 [flow 3ch + grip 1ch + warp 3ch],在 128² 渲出;
- **`pool16 = avg_pool2d(128→16, 8×)`** → 一个 8×8 像素区平均成一格 → **夹爪骨架线、flow 向量的亚格细节被抹平**;
- 作为 channel concat 进 DiT(16×16×Ccond ‖ 16×16×Cz)。

**16×16 grid 本身不粗**(VAE 能从 16×16×C 重建清晰 256 图,roundtrip LPIPS 0.029,细节在通道里);**粗的是 avg-pool 在进 DiT 前就把 cond 细节丢了**。诊断确认(project_decode_error_budget):agent 区 78% 误差在预测层,但 cond 注入粗化会限制 ③ 能表达的 agent 细节(夹爪朝向/尖端,session 2026-08-01 眼检到的糊)。

## 2. Related work 地基(2026-08-02 调研)

- **全领域无人用裸 avg-pool**;正经方法都在池化前放 learnable encoder。
- **OSCAR(2606.04463)**:骨架栅格**过同一个 Wan2.1 VAE** → 独立 patch-embed → **token 相加**融合。骨架几何存 VAE 通道码,不被空间抹掉。
- **FlowWAM(2607.13017)**:flow RGB 图**过同一个 Wan2.2 VAE** → 独立 patch-embed → **flow+RGB token concat 做 joint self-attention**(各自 RoPE + 流身份 embed),patch-embed 从 RGB init。**和我们架构最像(Wan VAE + dual-stream DiT + flow cond)。**
- **VideoComposer(2306.02018)**:STC-encoder = 2 conv + avg-pool(池化前先 conv 编码,非裸池化)。
- **ControlNet / Cosmos-Transfer**:并行 encoder + 多尺度/多深度注入(每 7 block 插一个 control block)。
- **Track2Act(2405.01527)/ AMPLIFY(2506.14198)**:稀疏点 → cross-attention token(无空间池化,位置精确)/ 局部窗口分类解码找回亚格精度。

**结论:latent-DiT flow/skel WM 的最佳实践 = (b-VAE)VAE-encode cond + 专属 patch-embed + token 融合;稀疏点用(c)cross-attn。OSCAR/FlowWAM(都 Wan-VAE,和我们同栈)独立收敛到同一配方 = 强背书。**

## 3. 设计(架构)

### 3.1 cond 分两类处理
- **稠密/栅格 cond → VAE-encode(b-VAE)**:object-flow field(3ch)、warp(3ch,本就是 warp 的 RGB)、agent-skel-raster(1ch 线画,可复制成 3ch 灰度)。
  - 全分辨率(256²)渲出 cond 栅格视频 (tL_rgb=48, 256,256, C) → **过同一个冻结 Wan VAE** → cond latent (tL=12, 16,16, Cz)。
  - 每个 cond 流一个**独立 patch-embed**(init 自 RGB patch-embed)→ DiT hidden。
- **稀疏 agent 点 → cross-attn token(c)(可选变体)**:skel 9 关节 / 夹爪点当 token,DiT latent query cross-attend(位置精确,不池化)。作为 skel-raster 的对照臂。
- **非空间标量(grip)**:保持 FiLM / 单 token(不 VAE-encode,grip 是开合标量)。

### 3.2 融合:OSCAR 式 token-add(主)
- 各 cond 流 patch-embed 后**逐 token 相加**进 video token(OSCAR 式,cheap,token 数不变)。
- ★为什么 add 不 concat:我们只**条件**于 flow(输入自 ②),不像 FlowWAM 要**预测** flow;纯条件用 token-add 够且省(FlowWAM concat 是因它双向预测两流)。
- 变体 B(如需更强):flow token concat 做 joint self-attn(FlowWAM 式)。作为 A/B。

### 3.3 warm-start
- DiT 主干 + RGB patch-embed:warm-start 自 grip_ro_tL12(mh_ema.pt)。
- 新 cond patch-embed:init 自 RGB patch-embed 权重。
- 冻结 Wan VAE。

## 4. De-risk 先行(别直接全量重训)

**Step 0 判决实验**:小设定(scarce robot 或 N 中等)下,同 DiT、同数据,只换 cond 注入:
- **臂 P0(baseline)**:现状 avg-pool + channel concat。
- **臂 P1(VAE-encode + token-add)**:本设计主方案。
- **臂 P2(可选)**:skel 走 cross-attn token(稀疏臂)。

**判据(每指标一套权威实现 + summary)**:
1. **render LPIPS(含 agent 区)**:P1 < P0(尤其 agent region-weighted);眼检并排(agent 落位/夹爪朝向/尖端细节)。
2. **agent 落位精度**:渲染帧上夹爪/骨架位置 vs GT(px),P1 更准 = cond 细节保住了。
3. **收敛**:P1(patch-embed RGB-init)是否比 P0 同 step 更快/更低。

PASS = P1 明显赢 P0(LPIPS + 眼检 agent 细节)→ 进全量重训 go-forward ③;否则记录负结果(avg-pool 够用,瓶颈在别处)。

## 5. 成功/失败与风险

- **成功**:VAE-encode cond 让 ③ 渲染更锐、agent 细节(夹爪朝向)更准 → 直接改善 human→robot translate 的糊 + latent 分布的 off-manifold 边缘。
- **风险**:
  - Wan VAE 在**非自然图像(flow/skel 栅格)**上编码可能次优 → 缓解:flow 用 RGB 格式编码(FlowWAM 做法),skel 线画复制成灰度 3ch;patch-embed 可学。
  - token 数:VAE-encode 每 cond 流 = 一份 16×16 token;多流 add 不涨 token,concat(变体 B)涨。
  - 计算:cond 也过 VAE encode(推理时多几次 VAE forward)→ 可预编码 cond latent 缓存(像 wan_latents 那样离线存 cond latent)。
- **不做**:不动 ② object-flow WM;不换 VAE;不追 legacy tL6。

## 6. 与我们其它线的关系
- 这是 ③ 渲染器的**独立改进点**,和"latent 分解喂 policy"([[project_multihead_aux_wm]] 相关的 dynamic-map 分解)正交:一个管"③ 怎么把 cond 渲得更细",一个管"哪些 cond 分量喂 policy"。
- 若 PASS,grip_ro_tL12 之后的 canonical ③ 应带此 cond 注入。
- 承接 [[project_flow_warp_renderer]] / [[project_renderer_latent_route]] / [[reference_flow_repr_survey]]。

## 7. 开放决策(待用户 review)
1. 主方案 token-add(OSCAR)vs concat+joint-attn(FlowWAM):建议先 add(省、够条件用),必要再 B。
2. skel 走 VAE-encode(和 flow 一起)还是 cross-attn token(稀疏臂):建议 de-risk 都上做 P1/P2 对照。
3. cond latent 是否离线预编码缓存(省推理 VAE forward):建议是(照 wan_latents 模式)。
