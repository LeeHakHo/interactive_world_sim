# 视频架构 WM 迁移 实现计划

> **For agentic workers:** 本计划为探索性 GPU 研究工作,按里程碑推进;每个里程碑 = 可视化产物 + 验证 gate + commit。失败早停、结论存盘(summary.txt)、中间产物出三联图给用户。

**Goal:** 把渲染 ③ 从逐帧 ostris latent 迁到 Wan2.2 VAE(冻结)+ 自训小视频 DiT + chunk 自回归,256 原生;② object-flow 不变;含 warp / mask-vs-skel 两 ablation。

**Architecture:** 冻结 Wan2.2 VAE 编码时序 latent(z=48,空间16×,时序4×因果)→ 自训小视频 DiT 在 latent 上做 chunk 自回归渲染,② flow 逐帧 splat 降到 latent 尺寸当空间条件,dual-view 双流。

**Tech Stack:** PyTorch 2.7 / iws conda env(`/scr/yusenluo/anaconda3/envs/iws/bin/python`)/ Wan VAE native safetensors / 8×A6000。

## Global Constraints

- Python 一律用 `/scr/yusenluo/anaconda3/envs/iws/bin/python`(base numpy2 崩 cv2)。
- 只 git add 自己的文件;commit 不加 Co-Authored-By。
- 每里程碑独立输出文件夹 `outputs/video_arch_wm/<milestone>/`,结论写 `summary.txt`,不口述。
- 可视化用标准布局(gif 用 `save_combined_gif`;三联图=原帧|产物|叠加);中间产物(skel/mask/VAE roundtrip/条件通道)全出图给用户,附绝对路径。
- 数据集 = play_robot_can dual-view;真标定 `calib/*_REAL.json`(已验证自洽)。
- 指标测真交付物:③ 渲染 LPIPS(含 agent)+ 眼检 vs GT;② ADE 仅代理。
- 失败早停:任一里程碑 gate 不过,先出诊断图 + summary,不硬推下一步。

---

## Milestone 1: Wan VAE 加载 + roundtrip smoke

**Files:**
- Create: `wan_vae.py`(native safetensors loader + encode/decode 封装)
- Create: `tests/test_wan_vae.py`
- Out: `outputs/video_arch_wm/m1_vae_smoke/`

**Interfaces:**
- Produces: `WanVAE(ckpt).encode(x)->z`(x:(B,3,T,H,W)[0,1] → z:(B,48,t,H/16,W/16));`.decode(z)->x`;`.roundtrip(x)->x_hat`。时序因果:T=4k+1 → t=k+1。

- [ ] Step 1: 探 native 权重结构 → 确定 encoder/decoder 前向(参照 Wan2.1/2.2 官方 WanVAE 实现;3D conv + time_conv + causal padding)。若能定位官方 `wan` 包/`diffusers` 分支实现则复用,否则移植最小前向。
- [ ] Step 2: 写 `WanVAE`,load safetensors 到模块;encode/decode 单张图(T=1)smoke,断言 shape。
- [ ] Step 3: 单帧 roundtrip(随机真 can 帧,256²)→ 出 原图|重建|差图 三联 PNG;打印 LPIPS/PSNR。
- [ ] Step 4: 短 clip roundtrip(T=17)→ 出 gif(原 vs 重建);验时序 latent 帧数 = 5。
- [ ] Gate: 重建 LPIPS 合理(<0.15 量级)+ 眼检不糊;写 summary.txt。commit。

## Milestone 2: 256 原生数据 regen + VAE 对标 ostris

**Files:**
- Create: `build_can256_dataset.py`(从 640×480 raw 按 CROPS 重 crop→256,dual-view,存 clip)
- Modify: `exp_scel_dualview_dit.py`(latent cache 支持 Wan VAE + 256;cache 路径含 vae 名+res 防串味,已有机制)
- Out: `outputs/video_arch_wm/m2_data256/`

**Interfaces:**
- Produces: 256 dual-view can clip 数据集(帧 256²,tracks/eef crop-norm 不变,复用现有 tr/ef/fr 结构但 fr=256)。

- [ ] Step 1: 写 crop→256 脚本,cam_high (60,60,390,390)→256 / cam_low (0,0,640,480) 取中心方块→256;出 3 clip 抽帧对照图(128 旧 vs 256 新)。
- [ ] Step 2: 全量 regen(或先子集),存盘;打印数量/尺寸。
- [ ] Step 3: **VAE 对标** — 同一批 can 帧,Wan VAE(256) roundtrip LPIPS vs ostris-16ch(128 及 256上采样)。出并排 gif + 数字表。
- [ ] Gate: Wan@256 roundtrip 不输 ostris(理想更清晰,尤其罐子);写 summary。commit。若明显输 → 触发 R1(议 512/换 VAE),先停给用户看。

## Milestone 3: FK 骨架投影 + agent 条件三联图(解锁 ablation B)

**Files:**
- Create: `can_skeleton.py`(URDF FK → 基座系关节 3D → color_to_base 投 2D → 连线光栅到 (H,W) 通道)
- Create: `tests/test_can_skeleton.py`
- Out: `outputs/video_arch_wm/m3_skel/`

**Interfaces:**
- Produces: `skel_channel(joint7, view)->(H,W)`float[0,1] 线画;与 `gmask_imgs`(mask)同签名可互换当 agent 通道。

- [ ] Step 1: 定位 URDF + FK(`Trossen_Analysis/` 有 urdf;pinocchio FK,memory 记 FK 与数据集 eef 差 1e-9)。取关节链 3D 点。
- [ ] Step 2: `color_to_base` 投 2D(用 `can_rig_real_calib.json` 的 color_to_base;crop-norm)。
- [ ] Step 3: **眼检对齐** — FK 投影骨架叠真图(两视角,多帧),出三联 原帧|骨架|叠加。**这是 R3 gate**:骨架必须压在真臂上。
- [ ] Step 4: 连线光栅成通道;出 mask vs skeleton 并排图(同帧)。
- [ ] Gate: 骨架叠加对齐(眼检);写 summary。commit。若不对齐 → 回退 eef 3 点+depth 近似,记录。

## Milestone 4: 单 chunk 视频 DiT ③ 训通(replay)

**Files:**
- Create: `video_dit.py`(小视频 DiT:latent chunk (B,2,48,5,16,16) + 逐帧 flow 条件降到 16×16,时序 attention,dual-view 双流)
- Create: `train_video_dit.py`(replay 训练:GT flow 条件 + Wan latent 目标,flow-matching 或 L2+LPIPS,先 L2 起)
- Out: `outputs/video_arch_wm/m4_chunk_replay/`

**Interfaces:**
- Consumes: WanVAE(M1)、256数据(M2)、agent 通道(M3 的 mask 或 skel)。
- Produces: `VideoDiT.render_chunk(z_prev_last, flow_cond_seq)->latent_chunk`;`rollout_chunks(...)`任意 horizon。

- [ ] Step 1: 定义 `VideoDiT`(参照现 `DualViewDiTG` 加时序轴;token=latent patch + 时间/视角 emb)。前向 shape smoke。
- [ ] Step 2: 数据管线:一个 clip → Wan 编码成 latent chunk + 逐帧 flow 条件(splat 256→降 16×16)。
- [ ] Step 3: 训单 chunk replay(GT flow),小步先过拟合 1 clip 验证能重建 → 出 gif(GT|render|flow overlay)。
- [ ] Step 4: 扩到训练集训到收敛(sbatch,多 GPU);记 LPIPS/时序一致性曲线。
- [ ] Gate: replay 渲染 LPIPS 贴天花板 + 眼检罐+agent 到位 + 时序不闪;summary。commit。

## Milestone 5: chunk 自回归 + ② pred-flow + 两 ablation

**Files:**
- Create: `eval_video_rollout.py`(chunk 自回归:② rollout flow → ③ 逐 chunk 渲染,任意 horizon)
- Create: `run_video_ablations.py`(A: warp on/off;B: mask vs skel)
- Out: `outputs/video_arch_wm/m5_rollout_ablation/`

**Interfaces:**
- Consumes: 现有 ② `wm_dummy5`(rollout_dual)、M4 VideoDiT。
- Produces: 自回归多 chunk gif;ablation 指标表。

- [ ] Step 1: chunk 自回归 rollout:上一 chunk 末 latent warm-start,② pred-flow 喂条件,累积 H≥40。出 gif(GT|render|flow overlay,双视角)。
- [ ] Step 2: **Ablation A**(warp on/off):同 seq 同 ckpt,唯一变量 warp 通道;LPIPS 表(cam_high/low 分开)+ 并排 gif。
- [ ] Step 3: **Ablation B**(mask vs skel):训两个 ③ 变体(仅 agent 通道不同),LPIPS + agent 区 LPIPS + 并排 gif。
- [ ] Step 4: 汇总报告 `REPORT.md`;gif 上传 drive `iws_evals/2026-07-2x_video_arch_wm/`。
- [ ] Gate: 自回归不漂移 + 两 ablation 出决定性结论;summary + REPORT。commit。

## Milestone 6(北极星,后续独立 spec,今晚不做)

policy-in-WM sandbox + VLM 成功率 → 另开 spec。

---

## 今晚自主执行顺序

M1 → M2 → M3(可与 M2 并行) → M4 → M5。每里程碑跑完出可视化+summary,给用户留绝对路径;遇 gate 不过先出诊断图停下等看。长训练用 sbatch 后台,期间推进可并行的下一里程碑准备工作。
