# ③ 视频 backbone 选型 —— 强 related work 精读提取(2026-07-22)

4 篇并行精读的原始要点,支撑 migration spec §10 的 backbone 决策。目的:**他们在我们独特方法(②object-flow/dual-view/human-helps)之外,靠什么拿到渲染质量**。

## OSCAR (2606.04463) — Omni-Embodiment Action-Conditioned WM
- Backbone: **DiT + rectified-flow**, 从 **Cosmos-Predict2.5-2B** finetune(2B)。首帧 I₀ 干净编码覆盖第一帧,只 denoise 未来(image-to-video)。
- VAE: **WAN 2.1 视频 VAE**(时序压缩 latent)。
- 条件: **2D skeleton 光栅成 RGB 视频** → 过**同一 VAE** → skeleton latent 与 target 同 shape → 两个 patch embedder(PE_v/PE_s)**token 相加**注入(非 cross-attn/adaLN/concat)。human MANO 同格式统一。
- 拿到质量的关键: **强预训练视频先验填细节**(texture-free skeleton 只管粗运动)+ **首帧 I₀ 锚外观** + texture-free 条件避免过拟合具身外观(→ 跨具身/human 数据能帮)+ **warm-start human 混训**(robot-only 15k 步再混,PSNR 24.24 vs from-scratch 23.87 vs robot-only 23.48)+ 激进数据过滤防 freeze-frame。
- Loss: rectified-flow 速度回归。单 GH200。18万 episode。metrics 前 49 帧、15fps。
- ★对我们: **借** 加性 latent-token 条件、首帧 I₀ 锚、texture-free 条件、warm-start、过滤;**别盲抄** 它质量主要来自 2B 先验(我们没有→需更强首帧/prev-memory/warp 或加密条件);单视角(dual-view 是我们的)。

## WEAVER (2606.13672) — ★最贴我们的蓝图
- Backbone: **latent 空间 DiT 从头训 928M**(32 层/16 头/dim1536/RMSNorm/RoPE/QK-Norm/SwiGLU);**flow-matching**;预测未来 latent 非像素(frozen decoder 管外观)。**从头训打赢微调的 1.5B Ctrl-World**。
- VAE: 冻结 **SD3 图像 VAE 逐帧**(非视频 VAE;时序建模在 DiT 里)。也 token 化并预测 proprio 状态。
- 长 rollout("Longer")四机制: **(a)Diffusion Forcing**(chunk 内逐帧独立噪声→抗漂移,FID 40+步不涨,最高杠杆) **(b)稀疏长记忆(每5帧)+ 短历史(最近2帧)双上下文** **(c)causal 时序 attn**(每 block) **(d)多视角预测当 consistency 机制**(处理遮挡)。
- 条件: **action/flow 当 token 拼接** in-context(非 cross-attn/adaLN)+ flow τ timestep emb。★动作用**绝对 joint position(经 adapter 从 velocity 转)**,直接喂 velocity 质量差 —— 印证"条件 featurization 是一等设计"。
- "Faster": **KV-cache memory/history(固定噪声 k=1,省30%)** + SPRINT 丢 token + cosine schedule + **ReFlow 少步蒸馏(NFE=4)**。
- Loss: flow-matching 速度 MSE + proprio(0.1)+ 解耦 reward/critic 头。4×H100 10 天预训练。
- ★对我们: **直接搬** Diffusion Forcing(首先)、稀疏memory+短history、token 拼接条件、KV-cache、cosine+ReFlow、联合预测状态、latent+frozen-decoder split。**分歧**: 它图像VAE逐帧+DiT里做时序;我们 Wan 视频VAE(时序压缩进 codec)→ memory/history 定义在 **latent chunk 粒度**。

## DexWM (2512.13644) — 手-物 WM
- Backbone: **CDiT(Navigation WM 式)但当回归器用**(直接回归未来 latent,不去噪,为速度)。从头训,4 尺寸,默认 **XL 456M**(32 块/dim1024)。更大单调更好。
- 表示: **冻结 DINOv2-L patch 特征**(非像素/非 object-centric,448 token)。像素靠**单独训的 RAE ViT 解码器**(L1+LPIPS+adversarial,仅可视化)。**不是像素/视频 VAE WM**(对我们像素渲染目标只是弱证据)。
- 时序: 自回归多步,history 最多 9 帧/4s 窗口,horizon 4s=20帧@5Hz。确定性。**非均匀时序采样提升泛化**。
- 条件: **单一动作向量 AdaLN 注入每个 block**(非分置通道)。action=3D 手关键点差 + 相机 pose delta(132维),**canonical 帧解耦手/相机运动**。跨具身用 **dummy 关键点**(夹爪→同心圆5指,radius=开合)统一。
- ★拿质量关键: **Hand Consistency 辅助 loss(λ=100)** —— 全帧 latent loss 低估小手区,加一个预测 12 指尖/腕 heatmap 的头重罚,+34% PCK。直接回归(非去噪)提速。human 预训练+跨具身混训都帮。
- ★对我们: **借** AdaLN 注入稀疏几何(可选)、**★agent/物体辅助定位 loss(对应我们渲染质量在 agent+物体区)**、**相机 pose delta 条件 + canonical 帧(dual-view 解耦相机)**、非均匀时序采样。**别抄** 它 feature-space WM(非真像素渲染证据)、无 object-centric/分置通道结构。

## IWS 自建 ③ + stage2(我们现状)
- **IWS stage2 CMLatentDynamics**: 3D-conv U-Net(空间下采样被 Identity 掉)+ **causal 时序 attn + rotary** + **action FiLM(action_emd→scale/shift)** + diffusion(v-pred)。绑 in-house stage1 latent。
- **自建 ③ DualViewDiTG**: 逐帧 **DiT**(D384/depth8/heads6)+ **cross-view joint attention**(两视角同序列)+ **spatial-add flow cond**(conv 栈→加到 z0 token)+ prev-frame 自回归 + 确定性 **MSE+LPIPS**(无 flow-matching)。冻结 ostris-16ch 图像 VAE。gmask/warp 作额外条件通道。
- **两者无代码复用**(概念类比 detmem≈stage2 换条件)。detmem/temporal-SPADE 是两个赢家(纯 conv+prev-memory / 像素域 SPADE)。
- ★对 Wan 迁移: **保留** DualViewDiTG 的 **cross-view joint attention + flow/gmask 空间条件通道**(差异化);prev-memory hack 被视频 VAE 时间压缩 + 视频 DiT 时序 attn 天然取代;stage2 的 causal-temporal-attn + action 注入接口可借鉴。

## 综合裁决(→ spec §10.4)
自训小视频 DiT(D512/depth12/heads8,Wan latent chunk)+ 空间&causal时序 attn + dual-view cross-view attn;条件 token 级(object-flow spatial-add + agent 通道 + warp + 相机 extrinsic);**首帧 I₀ 锚 + rectified-flow + Diffusion Forcing + agent/物体定位辅助 loss + agent 区 LPIPS**;chunk 自回归 + 稀疏memory/短history;cosine 采样 + 后续 ReFlow。detmem 确定性作备选 ablation。
