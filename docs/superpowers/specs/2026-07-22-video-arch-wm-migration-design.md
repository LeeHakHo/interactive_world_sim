# 视频架构 WM 迁移设计(FlowWAM 式,自训小视频 DiT)

日期:2026-07-22 · 分支:phantom_dynamo · 数据集:play_robot_can(dual-view)

## 0. 一句话

把渲染 ③ 从「逐帧 latent(ostris f8-d16)」全面迁移到「**时序 latent 视频架构**」:
**Wan2.2 VAE(冻结)+ 自训小视频 DiT + chunk 自回归**,分辨率升到 **256×256 原生**。
② object-flow 世界模型**保留不变**,继续当跨具身接口(human-helps / dual-view / object-centric 差异化全保住)。

## 1. 目标与北极星

- **直接目标**:验证「视频 latent 架构」下,我们的 object-flow ② + 跨具身机制仍然成立且渲染更好(时序一致、256 清晰)。
- **北极星**:用 WM 生成数据 BC 一个 policy;coke-can pick&place 任务;pipeline = 采 demo → policy 在 WM 里 rollout → 测成功率(先用 VLM 判成功)。**核心用途 = 替代真机验证,要求 WM eval 与真机 eval 一致。**
- 这要求 WM 是 **action-conditioned + 任意 horizon 可交互 rollout** → 决定了必须 chunk 自回归(见 §3)。

## 2. 范式与组件

| 组件 | 现在(逐帧) | 迁移后(视频) |
|---|---|---|
| VAE | ostris f8-d16(16ch,逐帧) | **Wan2.2 VAE 冻结**(z=48,空间 16×,时序 4× 因果) |
| ③ 渲染 | `DualViewDiTG` 逐帧 latent | **自训小视频 DiT**:T=17 帧/chunk → 5 latent 帧,时序 attention,chunk 自回归 |
| ② WM | dummy5 object-flow,dual-view | **不变**(track 空间,VAE 无关) |
| 数据 | 128 crop | **256 原生重 crop**(从 640×480:cam_high 390→256 / cam_low 480→256,均降采样,非放大) |
| 条件 | flow_cond+agent+warp 逐帧 128² | 同构:256² splat → 降到 latent 尺寸(16×16),沿时间拼 chunk |

**Wan2.2 VAE 确认事实**(读 safetensors):z_dim=48;encoder 输入 12ch(3×2×2 patchify)+ 3 段空间 8× = 共 16× 空间;含 time_conv 的时序 4× 因果压缩。→ 256px 图 = **16×16×48** latent(vs 128px 的 8×8,足够装罐子细节)。T 帧 → `1+(T-1)/4` 个 latent 帧(首帧因果单独)。

## 3. Chunk 自回归契约(北极星地基)

- **单位 = T=17 帧 chunk**(Wan 家族标准 4k+1,4× 因果 → 5 latent 帧;显存友好)。
- **每 chunk**:输入 = 上一 chunk 末帧 latent(warm-start)+ 该段 ② flow 条件序列(逐帧 splat→latent 尺寸,时间维拼成 chunk)→ 输出 = 该段 5 latent 帧 → VAE decode 出 17 帧。
- **任意 horizon** 靠 chunk 累积。policy 每 chunk 喂一段动作 → ② rollout → ③ 渲染 → VLM 判成功。契约与现逐帧自回归同构,只是单位从 1 帧变成 T 帧。
- dual-view 保留:两 view stream(与现 `DualViewDiTG` 一致),加时序轴。

## 4. 两个 Ablation(本次一等公民实验)

### A. flow warp on/off
- 重测 `warp_preview`(rigid Umeyama 搬 footprint)在**视频架构**下是否仍帮。
- 旧逐帧结论:cam_low +19% / cam_high 微弱。**新架构时序信息更多,warp 可能变冗余** → 重判有价值。
- 变量唯一 = 条件通道是否含 warp 3ch;其余固定。

### B. agent mask vs agent skeleton(OSCAR 式)
agent 条件通道两种表示,对比渲染质量(尤其 agent 区):
- **mask**:现有 maskgen 剪影(joint7→silhouette,`maskgen_caneef` IoU0.91/low 0.84)。
- **skeleton**:URDF 关键点投 2D 连成线(OSCAR 2D skeleton;v3 上 flowskel 线画曾胜 flow,见 `project_flowskel_agent_repr`)。
- **骨架来源(标定已解决)**:真标定 `calib/*_REAL.json` 自洽可用(2026-07-22 重投影 median 0.01px),含 `color_to_base`。→ **URDF FK 出关节 3D(基座系)→ color_to_base 投影 → 2D 连线**,干净可行,不再需要"训 joint→2D 网络凑合"。
  - 前置确认(plan 第一步):把 FK 投影的臂叠到真图眼检对齐。
- 变量唯一 = agent 通道用 mask 还是 skeleton;其余固定。

## 5. 标定与 depth 利用(纠正旧"标定坏了")

- **标定不坏**:aytsai 真标定(`can_rig_real_calib.json` + `rgb_cam_calib_can{,_low}_REAL.json`)早到位,已接进 `exp_scel_dualview_wm.py`(`_CAL`),2026-07-22 重投影验证自洽(median 0.01px / p90 0.03px)。memory `project_can_rig_calibration_broken` 已纠正为「已解决」。
- **depth**:本地 parquet 被 strip(`DEPTH_STRIPPED.txt`),全量在 HF `aytsaiusc/play_robot_can_1_eef`。depth_scale=0.001 m/unit,depth↔color 14.9mm 基线,depth 有独立内参(不与 RGB 逐像素对齐)。
- **用途**:(a)2D track/eef 直接 depth-lift 成真 3D(不靠三角化);(b)骨架深度前后遮挡排序;(c)未来 3D 条件。→ plan 加一步「重下全量 parquet 取 depth」。

## 6. de-risk 里程碑(先证再重训,失败早停)

1. **Wan VAE 加载通** — native safetensors loader(不升级 diffusers 0.29.2);encode/decode smoke(单张随机图往返)。
2. **256 数据 regen** — 从 640×480 raw 按现成 crop 参数重 crop 到 256 原生;**单帧 + 短 clip roundtrip LPIPS** 对标 ostris-16ch(证 16× VAE 在 256 不掉质量;若掉再议分辨率/VAE)。
3. **FK 骨架对齐眼检** — URDF FK→color_to_base 投 2D,叠真图确认(解锁 ablation B)。
4. **单 chunk 视频 ③ 训通** — replay(GT flow 条件),LPIPS + 时序一致性 + 眼检。
5. **接 chunk 自回归 + ② pred-flow + 两个 ablation(A warp / B mask-vs-skel)**。
6. (北极星,后续独立 spec)policy-in-WM sandbox + VLM 成功率。

## 7. 指标(测真交付物,别用代理)

- **③ 渲染 LPIPS(含 agent)** + 时序一致性(相邻帧 warp 一致/闪烁) + **眼检渲染 vs GT 并排全要素**(罐+agent 都看)。
- ② ADE 仅当代理,不作判据。
- gif = 主交付,统一 `save_combined_gif`(Rendered 行 + Flow overlay 行 + GT 锚),布局不自造(遵 `feedback_gif_eval_layout`)。
- 北极星阶段追加 **VLM 成功率**。

## 8. 待解风险 / 开放项

- **R1 分辨率×VAE**:16× 空间在 256 = 16×16 latent,罐子细节是否够 → 里程碑 2 实测定;不够则考虑 512 或更低压缩 VAE。
- **R2 小视频 DiT 容量**:自训小模型在我们数据量上时序质量上限未知 → 里程碑 4 判;不足再考虑加深/借先验(但不走 5B 微调,已排除)。
- **R3 FK 骨架对齐**:标定自洽 ≠ FK 基座系与真臂像素对齐,需眼检(里程碑 3);若偏,回退用 eef 3 点 + depth 近似骨架。
- **R4 lift_high 水平漂移**(已知 bug,见 `project_can_zlift_keyboard`):纯垂直指令 x 也动,迁移后一并复查是 ② 编码还是 ③ 渲染串扰。
- **R5 depth 重下**:HF 全量 parquet 体积大,取 depth 列即可,别全拉。

## 9. 不做(YAGNI)

- 不微调 Wan2.2-TI2V-5B(排除,基建重、非我们卖点)。
- 不改 ② object-flow 表示(dummy5 相对已定论最优,见 `project_can_zlift_keyboard` abs/rel)。
- 不做无关重构;v3 数据集本次不迁移(集中 can)。

## 10. ③ 视频 DiT backbone —— ground 到强 related work(2026-07-22 补,4篇并行精读)

用户准则:迁移目的=**更好渲染质量**,backbone 必须参考强 work,不能随手搭小 DiT(见 `feedback_video_backbone_ground_in_refs`)。精读 OSCAR(2606.04463)/WEAVER(2606.13672)/DexWM(2512.13644)/IWS 自建 ③,提取"他们在我们独特方法之外靠什么拿到渲染质量",综合如下。

### 10.1 共识(定我们 backbone 的地基)
- **自训中小 DiT 预测 latent + 冻结 codec 管外观** 是三家共同架构。★WEAVER 928M **从头训**明确打赢微调的 1.5B(Ctrl-World)→ **坐实我们选项 A(自训,不微调 5B)是对的**,且可比 WEAVER 更小(object-flow 条件比自由动作生成强约束得多)。
- **条件注入 = token 级(加/拼),非 cross-attn/adaLN**:OSCAR=条件当"第二路视频"过同一 VAE 后 latent token **相加**(parallel patch embedder);WEAVER=flow/action **token 拼接** in-context;DexWM=AdaLN(唯一用 adaLN 的,但它条件是 132 维向量非空间图)。→ 我们现 DualViewDiTG 的 **spatial-add flow cond 正确**,保留。

### 10.2 他们"额外"拿到渲染质量/长 rollout 的机制(我们要借的)
1. **★Diffusion Forcing(WEAVER 首要)**:chunk 内**逐帧独立噪声 level** 训练 → rollout 容忍自己不完美的过去预测,FID 40+ 自回归步不涨。**最高杠杆抗漂移,backbone 无关,首先采纳。**
2. **★首帧 I₀ 锚点(OSCAR)**:首个时序 latent 用真首帧的干净编码覆盖,只预测未来帧 → 外观/场景接地、具身无关。**我们无大先验,这个锚点尤其关键**(补偿"预训练先验填细节"我们没有)。
3. **★agent/物体辅助定位 loss(DexWM HC,λ=100)**:全局 latent/像素 loss 会低估小 agent+物体区;加一个预测 object-flow 点/agent 骨架 heatmap 的辅助头重罚。**直接服务"渲染质量在 agent+物体区"**,与我们 measure-real-deliverable 准则一拍即合。
4. **稀疏长记忆 + 短历史双上下文(WEAVER)**:每 k 帧留一个 memory + 最近若干帧 history → 任意 horizon 有界开销。视频 VAE 已时间压缩,故在 **latent chunk 粒度**上定义 memory/history。
5. **flow-matching + cosine schedule + ReFlow 少步蒸馏(WEAVER/OSCAR)**:长 rollout 质量 + 交互速度(keyboard/policy 用)。
6. **warm-start human 混训(OSCAR:robot-only 先训再混 human,warm-start > from-scratch)**:接我们 human-helps 线。
7. **数据过滤防 freeze-frame 塌缩(OSCAR:最短长度/有意义动作/静态相机)**:小模型无强先验更易塌缩,便宜采纳。

### 10.3 关键张力 + 裁决
- **flow-matching(WEAVER/OSCAR 视频 WM 都用)vs 确定性回归(DexWM direct-regress + 我们 detmem 发现确定性更锐)**。裁决:**主路 = flow-matching + Diffusion Forcing**(真·视频 WM、长 rollout 抗漂移的证据在 WEAVER;detmem 那个"确定性更锐"是逐帧 ③ 结论,不含长 rollout),**辅以 agent/物体定位 loss + decode-LPIPS(agent 区)保锐**。detmem 确定性作为 M4 备选 ablation(若 flow-matching 太糊)。
- **无大先验补偿**:OSCAR 质量主要来自 2B Cosmos 先验,我们没有 → 靠(a)首帧 I₀ 锚点强接地 (b)prev-memory/warp 搬运真像素(我们 detmem/flow-warp 线)(c)必要时条件加密(object-flow 网格化)。

### 10.4 我们最终 ③ backbone(grounded)
- **小视频 DiT 从头训**:D≈512 depth≈12 heads≈8(介于自建 384/8 与 WEAVER 1536/32);操作 Wan latent chunk(48ch, tL 帧, 16×16)。
- **每 block:空间 attn + causal 时序 attn**(WEAVER/IWS-stage2);**dual-view cross-view joint attention**(我们 DualViewDiTG 差异化,保留)。
- **条件**:object-flow(splat 256→降到 latent 16×16 spatial-add)+ agent 通道(mask/skel = ablation B)+ warp(ablation A)+ per-view 相机 extrinsic(DexWM dual-view)。token 级拼/加,非 adaLN。
- **首帧 I₀ 锚点**(OSCAR):首 latent = 真首帧编码,只预测未来。
- **目标 = 3rectified-flow + Diffusion Forcing**(逐帧独立噪声)+ 辅助 agent/物体定位 loss(DexWM HC 式)+ agent 区 decode-LPIPS。
- **chunk 自回归 + 稀疏 memory/短 history**(WEAVER)→ 任意 horizon;cosine 采样;后续 ReFlow 蒸馏提速。
- 差异化不变:② object-flow 跨具身接口、dual-view cross-view attn、human-helps、object-centric。
- 参考出处:OSCAR §3.1-3.2、WEAVER §3.1-3.2/Table2、DexWM §3.2-3.3、IWS `exp_scel_dualview_gmaskcond.py`/`cm_latent_dynamics.py`。综合细节存 `docs/.../2026-07-22-video-backbone-grounding.md`(reader 原始提取)。
