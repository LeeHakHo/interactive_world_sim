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
