# Cross-Embodiment Video Editing baseline (human→robot 翻译 + co-train WM)

Date: 2026-05-28
Branch: phantom_dynamo
Status: design approved (方案 B + LoRA 注入)，待用户过目 spec
Paper: `2605.03637v1.pdf` — "Bridging the Embodiment Gap: Disentangled Cross-Embodiment Video Editing"

## 这个 baseline 回答什么

复用论文方法把 **human play 数据生成式翻译成 robot 视频**，再 co-train 你们的 latent
WM，给出 `CROSS_EMBODIMENT_WM_ATTEMPTS.md` §6 方向1 的硬数字：**"human 数据（经一个
比 inpaint 更强的翻译器变成 robot 后）到底帮不帮 robot WM？"**

为什么是这篇而不是继续 decompose/align/inpaint：你们已验证 human↔robot 观测在 DINOv2
里完全可分（probe 1.0 / 582×），像素/特征级**对齐**这条路撞墙。这篇换路 —— **不对齐，
改用一个 internet-scale 视频扩散先验 (VACE) 直接生成 robot 视频**，把"agent 不同"这件事
交给生成模型去 hallucinate，而不是去强行对齐两个不重叠的分布。

## 关键约束（决定了所有设计取舍）

- **不能用 overlay**（Phantom 式渲染叠加）——那正是论文打败的 baseline，质量差。必须真生成。
- **算力：只有 2× RTX A6000（48GB/卡）**，是论文 8×H200 的 ~9% 显存。
  - 显存能 fit（VACE-1.3B 冻结 ≈2.6GB；batch1/卡 + ~25 帧 + 256 分辨率 + grad checkpointing）。
  - 真正的瓶颈是 **吞吐/收敛质量**：global batch ~2 vs 论文 40。
- **play 数据没有 task 文本标签**，也没有现成物体轨迹。

## 与论文的差异（明确写出，便于日后 cite "我们的复现"）

| 论文 | 本复现 | 原因 |
|---|---|---|
| Wan2.1-VACE-1.3B 冻结 | 同款，冻结 | 用户指定同款 backbone |
| 15-block mirrored adapter A_ψ（可训练 ~半个 backbone） | **LoRA(rank 16–64) on DiT + 条件 token 注入** | 2 卡塞不下/训不动几百 M adapter；LoRA 收敛快、显存小，仍是"冻结 backbone + 注入条件"同思路 |
| z_task = 文本 T + EEF 运动 M_s + 物体轨迹 O_s | z_task = **M_s + O_s**（无 T） | play 数据没干净 task 文本 |
| 双对比 = CLUB + task-contrast + emb-contrast | **CLUB + emb-contrast**（丢 task-contrast） | task-contrast 需 task 分组正样本；play 无 task 标签。z_task 由 M_s+O_s 几何信号构造 → invariance 主要由**输入构造**保证 |
| O_s 6DoF（Grounding DINO+SAM2+VDA+ICP） | O_s **position 为主**（SAM2+VGGT 深度 lift，先不做 ICP 6DoF） | 沿用你们 IK/OT 结论：6DoF 是死路，position-only 鲁棒 |
| 8×H200, batch40, 81 帧/5s, 2 epoch | 2×A6000, batch1/卡, ~25 帧, 256 分辨率, 1 epoch + 有限 step | 算力 |
| 下游 = ACT BC policy | 下游 = **你们的 latent WM (emb_film 那条) co-train** | 项目是 WM，且直接回答 §6 方向1 |

## 架构

```
                 ┌─────────── 可训练 ───────────┐
 M_s (EEF运动) ─► E_task ─┐
 O_s (物体轨迹)─►        │ z_task ─┐
                          │        ├─► 作为额外条件 token 注入 VACE VCU 条件流
 C (末端静态图)─► E_emb ──┘ z_emb ─┘
                          │
 masked source video ────►│ (VACE 原生 inpainting 条件，复用现有 agent mask)
                          ▼
              ┌───────────────────────────┐
              │  Wan2.1-VACE-1.3B (冻结)   │  + LoRA(rank16–64) on DiT
              │  3D-VAE latent + DiT       │
              └───────────────────────────┘
                          ▼
                 重建/生成视频 (Flow Matching)

 CLUB q_φ(z_emb|z_task)  ← 仅用于 disentangle loss，不在生成路径上
```

- **冻结 backbone**：Wan2.1-VACE-1.3B（3D-VAE：时间 4×、空间 8× 压缩；DiT + VCU 多模态条件）。bf16 + gradient checkpointing 必开。
- **E_task**：M_s 与 O_s 各自 projector → 小 Transformer → MLP fuse → `z_task`（论文隐层 64 量级）。
- **E_emb**：静态末端图 C 过 **frozen VACE-CLIP** patch features → 小 Transformer → `z_emb`。
- **注入**：`z_emb`（本就来自 reference 图，天然契合 VACE 的 reference 条件）与 `z_task` 作为额外条件 token 进入 VACE 现有 VCU/cross-attn 条件流；LoRA 让 DiT 学会用这些条件。
- **masked source video**：复用你们现成 agent mask（`masks_arm` / SAM2）把 agent 抠掉，背景/物体由 masked video 携带。这把你们成熟的 inpaint 资产直接接进来。

## 数据准备

- 域：human = `play_human_aytsai`（phantom chunks），robot = `play_robot_eef`（AV1，PyAV `av` 解）。
- **M_s（EEF 运动）**：robot EEF 在 Trossen world frame，经静态外参 `T_cam_world`（URDF FK `cam_high_color_optical_frame→tabletop_link` 求逆，`reproject_test.py` 已验证）**position-only** 转到相机系；human EEF 本就在相机系。两域统一帧 —— 这是你们 OT 时踩过的坑，必须先做。
- **O_s（物体轨迹）**：SAM2 追踪共享物体（蓝盘/红块）+ VGGT 深度（`gen_phantom_depth_vggt.py`）lift 到 per-frame 3D 位置。position 为主。
- **C（末端静态图）**：robot 取夹爪 crop 帧；human 取手 crop 帧。
- **masked video**：human 用现成 arm mask 抠手臂+手；robot 用 SAM2 抠夹爪（沿用 `inpaint_gap_test.py` 的 crop+SAM2 路子）。
- **训练 clip**：~25 帧（论文 81）、256 分辨率（论文 480p）。

## 训练（自重建 + 方案 B loss）

- **自重建**（无需配对跨具身数据）：每个视频（human 或 robot）从噪声 + 自身 `z_task`/`z_emb` + 自身 masked video 重建自己。
- `L = L_FM + λ_dis · L_CLUB + λ_emb · L_emb_contrast`
  - `L_FM`：rectified-flow / flow-matching velocity MSE（论文主损失）。
  - `L_CLUB`：变分上界 minimize MI(`z_task`, `z_emb`)，**每 10 步更新一次 q_φ**（论文做法），防 `z_emb`/`z_task` 互相泄漏。
  - `L_emb_contrast`：InfoNCE，正样本 = 同 agent 不同帧的 `z_emb`，负样本 = 不同 agent。
  - **丢掉 `L_task_contrast`**（无 task 标签；invariance 靠 M_s+O_s 输入构造）。
  - λ 起点沿用论文 `λ_dis=1.0, λ_emb=0.5`；AdamW lr 1e-5。
- **优化器分组**：LoRA + E_task + E_emb 为主组；CLUB 的 q_φ 单独优化器（像 GAN 的判别器，detach 输入只训自己），沿用你们 dual_head/CLUB 的手动双优化器经验。
- **分阶段落地（实现顺序）**：
  - **v0**：只 `L_FM`，先把"VACE + LoRA + 条件注入 + masked video"自重建跑通，确认能学出 robot 外观重建（≈论文 w/o DC ablation）。
  - **v1**：加 CLUB + emb-contrast（完整方案 B）。

## 推理（cross-generation：human → robot）

对每条 human play 片段：
- `z_task` ← human 视频的 M_s + O_s；
- `z_emb` ← robot 夹爪静态图；
- masked video ← human 视频抠掉手臂+手；
- → VACE 去噪 50 步、scale 1.0 → 输出 **human-derived robot 视频**。

产出一批"由 human 演示翻译来的 robot 视频"，存成下游 WM 可读的格式。

## 下游 co-training

- 用 **真实 robot 视频 + 生成 robot 视频** co-train 你们的 latent WM（emb_film 那条）。
- 对照实验：**robot-only WM** vs **robot + 生成robot WM**。
- 指标：你们现有的 rollout PSNR / 多步预测误差。
- 输出：human 数据经强翻译后**是否真的拉高 robot WM** 的硬数字。

## 评测指标

1. **翻译质量**：FVD / PSNR / LPIPS（论文用）+ 视觉 QC（robot 形态正确？human 的 task 运动保留？背景/物体一致？）。
2. **disentangle 诊断**：复用你们的 linear probe / MMD（`diagnostics/linear_probe.py`、`compare_dino_within_between.py`）—— `z_task` 域应趋向不可分（invariant），`z_emb` 应可分。
3. **下游**：上面的 robot-only vs robot+生成 WM 对比。

## 显存预算 / 缩减档位（A6000 48GB × 2）

- 必开：bf16 + gradient checkpointing on 冻结 backbone。
- batch 1/卡（global 2），~25 帧，256 分辨率。
- 1 epoch + 有限 step，先看翻译质量再决定是否拉长。
- Fallback（若仍紧或太慢）：再降帧/分辨率；LoRA rank 调小；只对 DiT 末几层挂 LoRA。

## 风险与未决

1. **VACE 是否"认识" Trossen 夹爪**（小众机器人）：自重建会教它重建 robot 外观，但 zero-shot 形态可能不完美 → 视觉 QC 把关；必要时增大 robot 自重建比重。
2. **2 卡收敛质量**：global batch ~2，生成质量可能上不去 → 下游对比不可信（garbage in）。先 v0 小规模验证翻译质量是否达标，再投入下游。
3. **O_s 抽取稳定性**：SAM2 在 play 数据上的长程追踪可能丢，VGGT 深度尺度。
4. **生成伪影**：和 inpaint 同款担忧 —— 翻译视频本身可能是下游噪声源；这恰恰是这个 baseline 要测量的（伪影 vs 信息量谁占上风）。

## 实现里程碑（供 writing-plans 展开）

- **M1 数据管线**：M_s 统一帧 + O_s 抽取 + C 提取 + masked video（大量复用现有脚本）。
- **M2 VACE 集成**：下载/加载 VACE-1.3B、冻结、挂 LoRA、接通 VCU 条件 + z_task/z_emb token 注入；v0 自重建 smoke（极小规模）。
- **M3 训练 v0→v1**：L_FM 跑通 → 加 CLUB+emb-contrast；翻译质量 QC + disentangle 诊断。
- **M4 cross-generation**：批量把 human play 翻成 robot 视频。
- **M5 下游 co-train**：robot-only vs robot+生成，出对比数字。
