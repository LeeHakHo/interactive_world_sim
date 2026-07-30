# 跨具身 Object-Flow 世界模型 — 架构 / 数据流 / Shape / 实验全史

> 事无巨细技术文档,给作者本人 + 接手 Claude。目标:照此能复现 / 写 paper。
> 生成 2026-07-30。**架构断言带 file:line(读真实代码);shape 用 conda `iws` 实测 npz;实验结论带 memory/report 出处。**
> 区分标注:**[代码]** = 读源码/实测 npz 得到;**[记忆]** = memory/report 转述(可能是当时观测,已尽量交叉核对)。

分支 `phantom_dynamo`,仓库 `/scr2/yusenluo/interactive_world_sim`。

---

## 0. 目录

1. 总览(三模块 I/O)
2. 数据(数据集 / clips 构建 / npz 全字段 shape / episode-split + 泄漏史)
3. ② object-flow WM(DualLWC)
4. ② dual-view DiT(flow-cond 逐帧渲染,旧线)
5. ③ Wan 视频 WM(video DiT + 多头)
6. cond 结构(7ch 逐通道)
7. 全实验史 + 结论表
8. 当前状态 + 开放问题 + 下一步

---

## 1. 总览

命题:**human 演示数据帮训 robot 世界模型**,把共享信号走 **object 2D flow**(域不变接口)而非像素(human/robot 像素 582× 可分,所有对齐/分解尝试全败,见 `CROSS_EMBODIMENT_WM_ATTEMPTS.md`)。**[记忆]** `FLOW_WM_REPORT.md` §0。

管线三模块(每个独立训练):

```
① CoTracker object-flow 提取 (数据侧, 非学习)
      物体红罐 → HSV/GDINO+SAM2 种子 → 48 点网格 → CoTracker3 逐帧 track
      → 归一化 [0,1] 2D 轨迹 (per view)

② action → object-flow WM  (DualLWC, exp_scel_dualview_wm.py)
      输入: K=4 帧历史物体点 (双视角 2P=96) + 动作(eef 派生星座, K+F=24 帧窗口)
      输出: 未来每点每步 W×W=225 类速度分类 → 积分成 2D 轨迹
      → 预测 object-flow(域不变接口), scheduled-sampling 自回归

③ flow + cond → Wan-latent → render  (video DiT, video_dit.py / train_video_dit.py)
      输入: Wan2.2 causal 视频 VAE latent (C=48, tL, 16×16, 双视角)
            + cond(flow splat + agent + warp, 逐 latent 帧, 16×16)
      输出: rectified-flow 速度 → 采样整段 latent → 冻结 Wan VAE decode → 像素
      → "渲染"= 冻结 VAE decode;③ 本质是 latent 空间未来预测(动力学)
```

三模块 I/O 速查表:

| 模块 | 训练脚本 | 输入 | 输出 | 关键 shape |
|---|---|---|---|---|
| ① flow 提取 | `gen_flow_render_dataset_caneef.py` + `retrack_sam2_full.py` | 视频帧 224² | 48 点 track (归一化) | tracks (N,L,48,2) |
| ② object-flow WM | `exp_scel_dualview_wm.py` (`DualLWC`) | hist (B,2P,K,2) + eef (B,K+F,n_raw,2)×2view | logits (B,2P,F,W²) | K=4,F=20,W=15,Dm=384 |
| ③ Wan 视频 WM | `train_video_dit.py` (`VideoDiT`) / `train_multihead_wm.py` (`MultiHeadVideoWM`) | z_τ (B,2,48,tL,16,16) + cond (B,2,Ccond,tL,16,16) | v_pred (B,2,48,tL,16,16) | D=512,depth=12,tL=12/6 |

★术语校正(用户,`project_multihead_aux_wm`):③ 不叫"渲染器",③ = **未来 Wan-latent 预测**(latent 动力学);**渲染 = 冻结 VAE decode**。外观由冻结 VAE + I₀ 锚固定不学。**[记忆]**

---

## 2. 数据

### 2.1 数据集(单场景 play data)

**[记忆]** `SESSION_REPORT_2026-07-29` §7 + `project_multihead_aux_wm`(数据真相节):

| 域 | lerobot 目录 | episode | clip N | L(帧) | 说明 |
|---|---|---|---|---|---|
| robot | `human_play_data/play_robot_can_{1..18}_eef` | **18** (vid 100–117, 各 150) | 2700 (low_valid 2590) | **48** | ~3h play,每 ep ~10min 切 150 chunk |
| human | `human_play_eef_data/play_human_can_eef_{1..12}` | **12** (vid 0–11) | 1800 (valid 1795) | **24**(L48 重建中) | ~2h 连续 play,30fps |

**[代码]** 目录列表 `build_can256_latents.py:21-22`;`gen_flow_render_dataset_caneef.py:43-44`。vid 映射:robot vid = 100+idx,human vid = idx(`build_can256_latents.py:23-25`)。

★关键:**play data = 连续同场景**(同 rig / 同红罐 / 同桌 / 同两机位)。不同 episode ≠ 不同分布,是同场景不同时间段 → **单场景数据天花板**。position/traj 多样但无场景/物体多样性。**[记忆]** `SESSION_REPORT_2026-07-29` §7。这是后面所有"满量 human 不帮"结论的根因。

**双视角**:`cam_high`(俯视,view0)+ `cam_low`(斜视,近相机,view1)。**[代码]** crop `gen_flow_render_dataset_caneef.py:34` — high=(60,60,390,390)、low=(0,0,640,480,全帧压方,因近相机工作区溢 FOV)。标定 `calib/rgb_cam_calib_can_REAL.json` / `..._low_REAL.json`(REAL rig 标定,名义 URDF 位姿陈旧 273mm/9.5°,见 `project_can_rig_calibration_broken`)。**[代码]** `exp_scel_dualview_wm.py:73-77`。

物体点:红罐,`can_mask` HSV `((h>168)|(h<6))&(s>110)&(v>50)` → 连通域挑 → 凸包填充,肤色被 sat 门排除。**[代码]** `gen_flow_render_dataset_caneef.py:77-106`。robot z 不锁平面(can 任务是 3D)。robot eef 三点 = base + 两指尖(`LINK6_TO_EE=[0.156,0,0]`,`FINGER_X=0.0865`,指开口 = obs_right_gripper width),`gen_flow_render_dataset_caneef.py:118-129` + `gen_flow_render_dataset.py:23`。human eef 三点 = wrist + 两指尖(parquet `observation.eef.position_right/rotation_matrix_right/width_right`),`gen_flow_render_dataset_caneef.py:133-149`。

### 2.2 clips 构建(重叠滑窗)

**[代码]** `gen_flow_render_dataset_caneef.py`:
- 常量 `gen_flow_render_dataset.py:21`:`L, S, P = 24, 4, 48`(L=clip 帧数,S=子采样步,P=物体点数)。can 用 `CLIP_L` 覆盖 → robot L=48。
- `gen_flow_render_dataset_caneef.py:37`:`CLIP_STRIDE, TARGET_PER_VID, MAXF = 12, 150, 18100`;`TRACK_RES, IMG_RES = 224, 128`(`:36`)。
- 滑窗:`:179` `starts = range(0, len-L*S, CLIP_STRIDE)`;`:184` `idxs = [s0 + k*S for k in range(L)]`。→ 每 clip 跨源帧 **L×S = 48×4 = 192 帧(robot)/24×4 = 96 帧(human)**,但每 **12 帧**起一窗口。
- keep 规则(`:193-199`):物体质心位移 ≥ MOVE_MIN(10.0)或(robot)grip 开合 ≥ GRIP_MOVE(0.012)。→ 过滤静止 clip(有偏置:滤掉静止 → 抗漂移长 rollout 有害,`project_thick_latent_antidrift_ablation`)。
- CoTracker3(`cotracker3_offline`,`:237`)在 224² 上 track P=48 点,归一化 /224 存 float16。

**[记忆]** ★**重叠滑窗 = 泄漏根源**:`CLIP_STRIDE=12` 是子采样 `S=4` 整数倍 → 相邻 clip 落同一 4 帧网格 → 共享的是**逐帧完全相同**源帧。`project_clip_heldout_leakage`。

### 2.3 GDINO+SAM2 re-track(干净轨迹)

原始 CoTracker + 红 HSV init 有漏检(漏银顶/漂手),启发式种子打地鼠失败。改 **Grounding-DINO("a can")→ 紧 bbox → SAM2 box prompt + 中心正点 → clean mask → 网格 48 点 → CoTracker3**。**[代码]** `retrack_sam2_full.py`:GDINO `gdino_bbox`(`:26-35`,threshold 0.2)、`sam2_from_bbox`(`:38-43`)、`grid48`(`:85-94`)、`per_frame_vis`(`:97-102`,离质心 >2.5×median+4 判不可见)。seek 解码只取目标帧(`:58-82`)。输出 `{domain}_{view}.npz` = tracks(N,L,48,2)+vis+retrack_ok。**全量 0% 失败**(`project_multihead_aux_wm` §干净tracks判决)。**[记忆]**

`assemble_retrack_clips.py`:把 re-track 的 tracks/vis 合进 clips,`retrack_ok` = 两视角都成功(`:18`)→ `clips_*_retrack.npz`。env `BASE`/`HCLIP`/`DOMAINS`(L48 重建参数化,`L48_REBUILD_LOG.md`)。**[代码]**

### 2.4 human 真腕回填(realwrist)

clips 的 human eef[:,:,0] 是退化"指尖中点"(非真腕,差真腕 ~26px)。`build_human_wrist_low.py`:从 parquet `observation.eef.kpts21_3d_world`(HaMeR 21 点,腕=kpt0)取 world 3D 腕 → REAL 标定投影 high+low → `wrist_sidecar_human_low.npz`(wrist2d_high_proj / wrist2d_low / wrist_valid_low)。**[代码]** `:38-46,54-74`。`build_human_realwrist_clips.py`:用真腕替换 eef 腕槽,无效帧回退中点 → `clips_human_L24_retrack_realwrist.npz`。**[代码]** `:16-27`。真腕有效率 ~74%(`project_can_data_audit`;须 wrist_valid 过滤)。**[记忆]** 真腕判决 = 迁移侧帮 human_only 不救满量(`project_bidirectional_transfer_3x2` / `project_wrist_axis_negative`)。

### 2.5 clips npz 全字段 shape(iws 实测)

**[代码]** `outputs/flow_render_dataset_can_dual/clips_robot.npz`(robot,N=2700,L=48):

| 字段 | shape | dtype | 语义 |
|---|---|---|---|
| `frames` | (2700, 48, 128, 128, 3) | uint8 | cam_high 帧 128² |
| `frames_low` | (2700, 48, 128, 128, 3) | uint8 | cam_low 帧 128² |
| `tracks` | (2700, 48, 48, 2) | float16 | high 视角 48 点物体 flow,归一化 [0,1] |
| `tracks_low` | (2700, 48, 48, 2) | float16 | low 视角(遮挡有 NaN,4.07%) |
| `vis` / `vis_low` | (2700, 48, 48) | float16 | 逐点可见性权重 |
| `eef` / `eef_low` | (2700, 48, 3, 2) | float16 | 3 点 eef [base/wrist, 指1, 指2] 归一化 |
| `eef3d` | (2700, 48, 3, 3) | float32 | eef 世界 3D(m) |
| `tracks3d` | (2700, 48, 48, 3) | float32 | 物体点 depth-lift 世界 3D |
| `tracks3d_valid` | (2700, 48, 48) | bool | |
| `joint` | (2700, 48, 7) | float32 | robot 7 关节(human 无) |
| `grip` | (2700, 48) | float32 | obs_right_gripper 物理开口(human 无) |
| `vid` | (2700,) | int64 | episode id(robot 100–117) |
| `fidx` | (2700, 48) | int32 | 源视频帧下标(re-track/latent 编码用) |
| `low_valid` | (2700,) | bool | cam_low 该 clip 是否有效 |

human `clips_human_L24.npz`(N=1800,L=24):同结构但**无 joint/grip**,`tracks3d/eef3d` 有。`_retrack` 版:tracks/vis 变 float32 + 加 `retrack_ok`。`_retrack_realwrist` 版:eef/eef_low 也变 float32(真腕替换)。**[代码,实测]**

L48 human 重建:`outputs/flow_render_dataset_can_dual_L48/clips_human_L48*.npz`,L=48,其余同(`L48_REBUILD_LOG.md`)。

sidecar(`skel_sidecar_*`,agent 通道用):**[代码,实测]**
- `skel_sidecar_robot.npz`:skel2d_high/low (2700,48,**9**,2) + segments (8,2) + grip。
- `skel_sidecar_robot_v2.npz`:(2700,48,**8**,2)(v2 拓扑,② skel action 用 idx [5,6,7,4],`exp_scel_dualview_wm.py:154`)。
- `skel_sidecar_human.npz`:(1800,24,**4**,2)(wrist/fin1/fin2/forearm_stub)。

### 2.6 ★episode-split + 泄漏史(必读)

**[记忆]** `project_clip_heldout_leakage` + `SESSION_REPORT_2026-07-29` §1:

**泄漏根因(实锤)**:重叠滑窗(§2.2)+ `split_okfirst` 按 clip 下标切、**无 vid 分组** → 严重 heldout 泄漏。robot 154 heldout / 2436 pool 实测:
- 18/18 episode 全部横跨 train/heldout;
- 147/154 (95%) heldout clip 与训练共享 ≥1 帧;
- **131/154 (85%) 共享 ≥50% 完全相同源帧**;26/154 (17%) 近重复(≥90%)。

→ **所有绝对 drift / LPIPS 偏乐观**;相对 delta(同 N 同 heldout 匹配臂)大概率仍成立(泄漏对称)。

**三个 split 函数(代码演进)**,`exp_scel_dualview_wm.py`:
- `split_okfirst`(`:555`,旧,泄漏):`perm = rng(0).permutation(np.where(ok)[0]); ho,pool = perm[:150],perm[150:]`。③-style filter-then-permute(修 audit item5 的 ②/③ split 不等价,但仍按 clip-idx,泄漏)。
- `split_by_episode`(`:567`,★修复,commit 7868fa8):按 vid 分组,`HELDOUT_VIDS` 里整段 episode 作 heldout,其余训练池,都 ok-过滤 → 无跨-episode 帧泄漏。照 IWS `play_eef_dataset.py:378`。**默认空 = 旧行为逐字节不变**。
- legacy(`:606`):permute 全部再 filter(最旧)。

留出集选择:干净重训用 `HELDOUT_VIDS=100,102`(robot 279 clip)/ `HELDOUT_VIDS_H=0,1`(human)。③ 侧 `train_multihead_wm.py:49-52` 同口径(vids 是 18 个 episode id,`np.repeat` 展成 per-clip,`~np.isin` 排除)。**[代码]**

★**IWS 自己不漏**:`play_eef_dataset.py:378` = episode 级 val。只有 bolt-on 的 ②/③ 偏离了 house standard。**[记忆]**

---

## 3. ② object-flow WM(DualLWC)

主脚本 `exp_scel_dualview_wm.py`;基类 `amplify_wm.py`(`FlowWM_LWC`)。

### 3.1 基类 FlowWM_LWC(AMPLIFY 式局部窗口分类)

**[代码]** `amplify_wm.py:33-71`。设计动机:AMPLIFY(2506.14198 Table 11)证明"每点每步 W×W 局部速度分类" > 连续回归(回归糊向 0-motion,不能建多峰运动)。

- 构造 `FlowWM_LWC(P, Dm=384, layers=3, heads=Dm//64=6, W=15, vel_half=0.12)`(`:35`)。
- `inp = Linear(2*K+2, Dm)`(`:39`):历史 K 帧相对 anchor 位移 (2K) + anchor 绝对 (2) → token。
- `act = Linear((K+F)*3*2, Dm)`(`:40`):动作(基类是 3 点 eef × K+F 帧 × 2 坐标)。
- `tf = TransformerEncoder(EncoderLayer(Dm, heads, Dm*2, dropout=0), layers=3)`(`:41`)。
- `head = Linear(Dm, F*W*W)`(`:43`):每 token 出 F 步 × W²=225 类。
- 速度网格 `vel_grid(W, vel_half)`:W×W 覆盖 ±vel_half 的 (dx,dy) 单元格中心(`:19-23`)。
- `vel_to_class`(`:26-30`):速度 → 最近类 idx;`expected_vel`(`:62-65`):softmax × 网格 → 软期望速度。

K/F/常量:**[代码]** `e2e_flow_wm_render.py:30-31` `K, L = 4, 24; F = L-K` → **K=4 历史帧,F=20 预测步,窗口 Lw=K+F=24**。注:即使 robot clip L=48,WM 仍滑 24 帧窗口。速度类 W=15(W²=225)。

### 3.2 DualLWC(双视角联合)

**[代码]** `exp_scel_dualview_wm.py:307-412`。继承 `FlowWM_LWC`,`Dm=384, layers=3, W=15, vel_half=VEL_HALF=0.06`(`:456`、`:68`)。

- `P1 = P`(每视角点数 48),点 token 总数 `2P=96`(两视角拼)。
- `view_emb = Parameter(2, Dm)`(`:314`):学习视角嵌入。
- `n_tok`(动作星座点数,`:313`):cpt=1 / dummy5·dummy5g·dummy5rt·mp·dhc·dummy5rs=5 / skel=4。
- `act = Linear((K+F)*n_tok*2*2, Dm)`(`:315`):动作 tokens × 2 视角 → Dm。
- `act_grip = Linear(K+F, Dm)`(`:317`,仅 dummy5g/mp/dhc/dummy5rs):归一化 grip 序列 → Dm 加到动作 embedding。
- `head_mode`(`:325-329`):`two` → `head_h = deepcopy(head)`(human 头暖启);`film` → `dom_film = Embedding(2, 2*Dm)`(域 → γ,β,zero-init = 恒等)。
- `dhc` → `hc = MLP(Dm→Dm/2→4)`(DexWM HC 头,从 trunk 预测绝对接触 c,`:319`)。
- `raster/rasterg` → `act_ras = RasterAct(...)`(CNN 光栅图)+ `act_pos = Linear(...)`(`:321-324`)。

**forward `fwd_dual(hist, eef_a, eef_b, dom)`**(`:359-412`):
1. anchor = hist 最后帧(`:361`);obj token = `inp([相对位移(2K); anchor])` + view_emb(`:363-365`)。
2. 物体质心 oc(per view,`:366-367`)→ 动作点减质心变对象相对坐标(`:404-405`)。
3. 动作星座 `_act_pts`(`:343-357`):见 §3.3。
4. `act = Linear([d5a; d5b])`,dummy5g/mp/dummy5rs 加 grip 嵌入(`:406-409`)。
5. `x = tf([obj; act])[:, :P2]`(`:410`);`logits = _readout(x,dom).reshape(B, 2P, F, W²)`(`:411`)。

**`_readout(x, dom)`**(`:331-341`):single → `head(x)`;two → 按 dom 选 `head_h/head`;film → `head(x*(1+γ)+β)`(域 embedding 出 γ/β 调制 trunk 特征,EgoWAM 式)。getattr 默认 single 兼容旧 ckpt。

**输入输出 shape**:hist (B,2P,K,2) [view0 点 | view1 点];eef_a/b (B,K+F,n_raw,2) per view;logits (B,2P,F,W²);anchor (B,2P,2)。

### 3.3 动作表示逐个(★核心实验轴)

**[代码]** 构造在 `_act_pts`(`:343-357`)+ `load_action_tokens`(`:253-304`)。全部对象质心相对(去绝对位置泄漏动机见 `project_action_dflow_separability`:action 端输入支撑 disjoint probe 1.0,是根因;velocity/差分同域 0.53)。

| ACTION | n_tok | 构造 | grip | 出处/结论 |
|---|---|---|---|---|
| `dummy5`(默认) | 5 | DexWM 虚拟星座:[wrist, c, c+ax, c−ax, c+perp],c=两指中点,ax=半指轴(`:80-85`) | 无 | 基线 |
| `dummy5g` | 5 | dummy5 三点建星座 + 第 4 槽 = 归一化物理 grip(robot obs / human 指距代理,`:278-291`) | 显式标量 | grasp-aware,`project_grip_grasp_aware_wm` |
| `dummy5rt` | 5 | 几何 retarget:腕/开口冻成 canonical 常数 W0/R0(per view,`:98-109`) | 无 | ★**判负**(probe 0.9994,冻死抹抓取,ro 2.928 最差) |
| `mp`(★赢) | 5 | Point-Policy 中点星座**丢腕**:c + 半尺度归一 R0 的四点十字(`:120-128`) | act_grip 标量 | ★**赢家**(ro 2.083 最好),开口留域中性标量 |
| `dummy5rs` | 5 | breathing 星座:R=R0·(真实开口/域中位数),开合留几何(`:138-147`) | act_grip 标量 | ≈mp 无净增益,mp 仍赢 |
| `cpt` | 1 | 纯接触点 c(单点,`:356`) | 无 | 更差(3.199),证 mp 不只是单点 |
| `dhc` | 5 | DexWM Δ+HC:喂 Δ 星座(同域)+ grip,HC 头预测绝对 c(λ=100,`:386-397,432-434`) | act_grip | DexWM 式,未成头条 |
| `skel` / `skelv3` | 4 | v2 同构骨架(robot idx[5,6,7,4] / human 4 点)/ 尺度不变结构 token(`:177-190`) | — | skelv3 丢定位精度,输 world 基线 |
| `raster`/`rasterg` | — | OSCAR 式局部光栅图(64² 线画 CNN)+ 末端位置向量(`:205-250`) | rasterg 加标量 | 见 `raster_sidecar_*` |

### 3.4 训练(scheduled sampling)

**[代码]** `train_dual`(`:448-498`):`m = DualLWC(P2//2, action, head_mode, Dm=384, layers=3, W=15, vel_half=0.06)`;AdamW lr=`SSm.WM_LR`;epochs=`SSm.WM_EPOCHS`(smoke=2)。

- SS loss `_ss_loss`(`:415-438`):teacher prob `pteach` 从 1.0 退火到 0.3(`:470`);R_SS 步自回归(`R_SS_ENV=32`,`:69`);每步预测下一帧速度类 CE,vis 加权 `Vv[K+h]*Vv[K-1]`;buf 用 teacher-forcing / 自预测混合(`:436-437`)。dhc 加 `100*hc_aux`(`:433`)。
- MIX=rh 混训(`:479-496`):★修复(2026-07-20)——**每 optimizer step 同含 1 robot batch + 1 human batch**(一次 backward),robot batch 少则 cycle 复用。修前"先训完 robot 再训 human"致稀缺时 human batch ~6× 堆结尾 → 梯度被 human 主导 → 假"human 有害"。**[代码+记忆]**
- R_SS_h = min(R_SS, L_human−K)(human L24 → R_SS_h=20)。

**主流程 main**(`:578-690`):加载 clips → split(`HELDOUT_VIDS` → episode-split;否则 okfirst/legacy)→ `subsample_robot_pool(NROB)`(robot-scarce,rng(SEED) 确定性子集,`:540-552`)→ MIX=rh 加 human cotrain(`HUMAN_N` 下采样,`:641-644`)→ train → rollout eval。

### 3.5 rollout + drift 口径

**[代码]** `rollout_dual`(`:501-513`):从 K 帧历史自回归 H 步,`nxt = buf[-1] + expected_vel(logits[:,:,0])`,滑动动作窗口 `efA[:, h:h+Lw]`(+末帧 pad)。→ (B,H,2P,2)。自回归 **horizon 任意**(`feedback_rollout_horizon_autoregressive`:44 步 ≈ 20 步不漂移)。

- 度量(`:659-669`):`drift_px = norm(pred−gt)*IMG`,IMG=128。H=40(env,`:68`)。cam_high(view0)/cam_low(view1)分开报。lift 子集 = z 跨度 >0.08。triangulated 3D err(双视角射线交,`:525-537`)对 depth-lift GT。
- 输出 `metrics.json` / `summary.txt`:drift_px_cam_high/low、tri3d_err_mm。

---

## 4. ② dual-view DiT(flow-cond 逐帧渲染,旧线)

主脚本 `exp_scel_dualview_dit.py`(探索版)+ `exp_scel_dualview_dit_formal.py`(正式版,44k 行)。这是 **Wan 迁移前**的逐帧 ③ 渲染器,用 **frozen ostris/vae-kl-f8-d16(16ch)**。

**[代码]** `DualViewDiT`(`exp_scel_dualview_dit.py:49-93`):
- 双视角联合渲染,token = [z0_v0(256), z0_v1(256), prev_v0, prev_v1] + view/type emb,cross-view joint attention(`:74-93`)。backbone = `exp_scel_latent_dit.Block`(add-DiT,project_detmem_dit 赢家),GRID=16,zdim=`latent_ch()`=16。
- COND=flow:per-view object-flow splat + footprint(3ch)→ CNN `ce` → spatial-add(`:61-64,79-81`)。
- COND=eef:单 eef 点向量 [x,y,dx,dy](4-dim)→ `act_emd` MLP → FiLM(IWS stage2 `action_emd` 同构,`:66-67,84-85`,ada=True)。

`flow_cond`(`:32-39`):`splat128([tracks;eef], ...)` → (dx,dy) 两通道 + footprint(物体凸包填充)→ (3,128,128)。`eef_point`(`:42-46`):naive baseline。

训练(`:150-189`):co-train robot(latent-MSE + decode-LPIPS,LAM=1)+ human(obj-region pixel MSE),PREV_DF 0.3,BS8,LR2e-4,60ep。

**结论(DUALVIEW_DIT_REPORT.md,3 seed 门控)**:**[记忆]**
- **flow-cond 显著优于 IWS 式 eef-FiLM**(双视角 >2σ 门控 PASS):flow v0 0.256/v1 0.279 vs eeffilm 0.278/0.322。
- 诚实拆解:优势一大块来自"空间注入 vs 向量 FiLM"机制(eefsp = 空间注入 eef 点,cam_high 甚至反超 flow 0.228)。flow 独有净价值 = **端到端可预测性**(eefsp 无 ② 可预测通道)。
- **e2e ②→③ 几乎贴 GT-flow 天花板**:GT-flow 0.263/0.282,② pred-flow 0.270/0.298,eef-FiLM 0.285/0.345。② ADE 2.71/3.67px(H20)。→ **③ 对 ② flow 误差宽容**(改准 ② 对渲染增益极小 → rh≈ro@③)。
- **③ 层 human co-train 三臂全不显著**(human-helps 主战场在 ②)。
- cross-view joint attention 本设置无增益(负结果)。
- IWS stage2(CMLatentDynamics DF)external baseline 垫底 0.324/0.371。

★用户后续纠正(`STRATEGY_2026-07-30` §3):以后 flow-cond vs naive 要照 **FlowWAM Fig.4 方法学**(固定 backbone/VAE/数据只换 cond,别拿"我们③ vs IWS③" 混实现差),补 cube_px 控制 metric。

---

## 5. ③ Wan 视频 WM(video DiT + 多头)

★用户 2026-07-22 定的迁移(`project_wan_vae_migration`):从 ostris-16ch 图像 VAE + 逐帧 ③ **全面迁移到 Wan2.2 causal 视频 VAE + 视频 DiT**(选项2 = FlowWAM 式)。背书:OSCAR 用 Cosmos VAE / FlowWAM 用 Wan2.2 VAE 都是视频 VAE。**[记忆]**

### 5.1 Wan2.2 VAE(冻结)

**[代码]** `wan_vae.py`(用 `.venv_wan/bin/python`,官方 `diffusers.AutoencoderKLWan` + `Wan-AI/Wan2.2-TI2V-5B-Diffusers` 的 vae/)。官方规格(`wan_vae.py:7`):**z_dim=48, base_dim=160, 空间 16× 下采样,时序 4× 因果**(temperal_downsample=[F,T,T]),含官方 latents_mean/std(48 维,`:31`)。

- 编码:(1,3,L,H,W) → (48, tL, 16, 16),**tL = 1 + (L−1)//4**(`build_can256_latents.py:46`)。L48 → tL12,L24 → tL6。256² 帧 → 16×16 latent。
- **[代码,实测]** `latents_all.npz`(robot):lat/lat_low (2700, **48**, **12**, 16, 16) f16;`latents_human_all.npz`:(1800, 48, **6**, 16, 16)。L48 human 重建:(1800, 48, 12, 16, 16)。
- de-risk(`project_wan_vae_migration` M1/M2):Wan@256 roundtrip LPIPS cam_high 0.037/low 0.030,PSNR35 → **VAE 非瓶颈**(<③ 渲染误差 ~0.12)。迁移理由 = 时序 latent + 视频 DiT。

### 5.2 时序对齐(latent 帧 ↔ 原始帧)

**[代码]** `build_can_cond.py:64`:latent 帧 k ↔ 原始帧 `rf = 0 if k==0 else min(4k, L-1)`(Wan 因果 VAE 映射)。cond 在 128² 建再 avg-pool 到 16²(`:21,51-53`)。build_can256 latent 编码在 256²(`build_can256_latents.py:18,36`)。

### 5.3 video DiT 架构

**[代码]** `video_dit.py`,`VideoDiT(C=48, Ccond=7, V=2, gh=16, gw=16, D=512, depth=12, heads=8, patch=2)`(`:78`),grounded 到 WEAVER/OSCAR/DexWM/IWS。参数 ~80M。

- patchify(`:92-95`):每 latent 帧 (C,16,16) → patch=2 → P=(16/2)²=64 token/帧,`x_embed = Linear(C*4, D)`(`:82`)。
- 条件注入 = **OSCAR 加性**:cond 同 patchify → `c_embed = Linear(Ccond*4, D)` 加到 latent token(`:83,106`)。
- pos(空间)+ view_emb(V=2)(`:84-85,108`)。
- Block ×12(`:47-74`):(a) 空间 + cross-view joint attention(同一 latent 帧内 V×P token 互 attend,非因果);(b) causal 时序 attention(同空间位置跨帧,因果);(c) MLP;每段 **adaLN-zero(逐帧 τ 调制)**(`:54,60-61`)。
- timestep = 逐帧独立 τ(Diffusion Forcing),`TimestepEmb`(`:23-33`)。
- 目标 = rectified-flow 速度(x1−x0),final adaLN + zero-init head → v_pred (B,V,C,T,16,16)(`:113-117`)。

**输入输出 shape**:z_τ (B,2,48,tL,16,16) 已加噪;τ (B,tL) 逐帧;cond (B,2,Ccond,tL,16,16);v_pred 同 z 形。

### 5.4 训练(train_video_dit.py)

**[代码]** `train_video_dit.py`:
- flow-matching:x0=noise,x1=data,xt=(1−t)x0+t·x1,目标 v=x1−x0(`:58-71`)。首帧 τ=1(clean,I₀ 锚,不算 loss,`:66-70`)。
- **PRED**(`:25`):`abs`(绝对 z)| `delta`(consecutive-Δz,`to_delta`/`from_delta` `:50-55`,human 共享 per-step 动力学,project_human_helps_renderer_exp 主线)。
- **MIX**(`:26`):`""` robot-only | `human`(HFRAC 比例混,`:110-119`)。LAT_H/COND_H = human latent/cond。
- **INIT**(`:30`):warm-start ckpt(OSCAR 式续训)。**TLCAP**(`:24`):统一 tL(robot12→6 匹配 human24)。**CCOND**(`:23`):截 cond 前 CCOND 通道(warp-off=4)。
- 默认 STEPS=40000,BS=16,LR=1e-4,DEPTH=12,D=512(`:20-21`)。EMA(0.999)。采样 `sample`(`:74-86`,20 步 Euler,首帧钳 z0)。
- batch:z = stack([lat[sel], lat_low[sel]], 1) → (B,2,48,tL,16,16)(`:114,118`)。
- eval:latent recon MSE(绝对 z 域,delta 已 cumsum 回,`:137-139`)。

### 5.5 多头 MultiHeadVideoWM

**[代码]** `video_multihead_wm.py`,`MultiHeadVideoWM(VideoDiT)` subclass(`:47-79`)。★硬约束:不改 video_dit.py/train_video_dit.py(在飞实验在用),subclass 复用全部 trunk + 主头(parity Δ=0,可 load 现有 ckpt)。

- `aux = ModuleDict`,每 aux 头 `AuxHead(D, out_ch, patch, V, ph, pw, gh, gw)`(`:29-44`,与主头同结构 final adaLN + Linear + unpatchify,**zero-init** 初始预测=0)。
- `forward(z_τ, τ, cond, want_aux=False)`:主 v_pred(与 VideoDiT 逐字一致,parity)+ `{name: aux_pred (B,V,out_ch,tL,16,16)}`(`:73-79`)。
- 设计 = EgoWAM 多头 world model,但**保留可解码 Wan-latent 头**(→VAE decode=渲染),抽象辅助头(DINO/depth/mask/HC)**只训练期塑造共享 trunk**。差异化 EgoWAM:他们扔 world head 不解码。**[记忆]** `project_multihead_aux_wm`。

**训练** `train_multihead_wm.py`:镜像 train_video_dit + aux loss。L = LAM_MAIN·main_rf_loss + Σ LAM_k·MSE(aux_k(trunk), 干净目标_k)(`:162-163`)。env `AUX`(逗号/+列)、`LAM_DINO/LAM_MASK/...`、`DINO_R/DINO_H`。aux 从同一带噪 forward 的 trunk 预测干净目标,不算首帧(`:76-91`)。`HELDOUT_VIDS` episode-split(`:49-52`)。

**[代码,实测]** aux 目标 npz `outputs/video_arch_wm/aux_targets/`:
- `dino_robot.npz`:feat (200,2,**32**,12,16,16) f16(DINOv2 ViT-S/14 → PCA KDIM=32,保留 ~84% 方差)+ pca_mean/comp。
- `mask_robot.npz`:feat (2000,2,**2**,12,16,16)(2ch:物体+agent 占据,MaskWAM 式)。
- `hc_robot.npz`:feat (2000,2,**3**,12,16,16)(3ch eef heatmap,DexWM HC 头)。
- human 版 tL=6。

### 5.6 latents/cond shape(iws 实测)

**[代码,实测]**

| npz | shape | 说明 |
|---|---|---|
| `latents_all.npz` lat/lat_low | (2700, 48, 12, 16, 16) f16 | robot,C=48,tL=12 |
| `latents_human_all.npz` | (1800, 48, 6, 16, 16) | human L24,tL=6 |
| `wan_latents_can_dual_L48/latents_human_all.npz` | (1800, 48, 12, 16, 16) | human L48,tL=12 |
| `cond_all.npz` cond | (2700, 2, **7**, 12, 16, 16) f16 | robot 7ch gmask,V=2 |
| `cond_skel_all.npz` | (2700, 2, 7, 12, 16, 16) | robot 7ch skel agent |
| `cond_human_eef3_all.npz` | (1800, 2, **4**, 6, 16, 16) | human,NOWARP 4ch(flow3+eef3 hint) |
| `cond_human_eef3_all_retrack_realwrist.npz` | (1800, 2, 7, 6, 16, 16) | human 7ch realwrist |

训练用 **CCOND=4**(丢 warp,§6)对齐 robot(7ch 截前 4)与 human(eef3 已 4ch)。**[代码]** `train_video_dit.py:43`。

---

## 6. cond 结构(7ch 逐通道)

**[代码]** `build_can_cond.py`。7ch = flow(3)+ agent(1)+ warp(3),与 DualViewDiTG 一致(`:1-4`)。NOWARP=1 → 4ch(`:25-26`)。输出 (N, V=2, NCH, tL, 16, 16) f16。

| ch | 语义 | 构造 |
|---|---|---|
| 0–1 | flow dx, dy | `DIT.flow_cond(tr0, trt, ef0, eft, vis)` object-flow splat(`:70`) |
| 2 | footprint | 物体点凸包填充 mask(`exp_scel_dualview_dit.py:37-38`) |
| 3 | agent | 三型之一(见下) |
| 4–6 | warp | `G.warp_preview(frame0, tr0, trt, vis)` 像素搬运预览(`:79`) |

**agent 三型**(`AGENT` env,`:22-23`):
- `gmask`(默认):agent 剪影,需 joint(gmask 网络)→ human 无 joint 不可用(`:73-76`)。
- `skel`:FK 骨架线画,需 skel2d sidecar(`skel_chan`,`:31-38`)。
- `eef3`:3 点 eef 三角形简易 hint,只需 eef → **human 可用**(`eef3_chan`,`:41-48`)。这是 human cond 用的(cond_human_eef3)。

**warp 已证冗余 + CCOND=4 已关**:`project_wan_vae_migration`(warp ablation)+ `STRATEGY_2026-07-30` §4:双口径判决 with-warp 天花板帮一点 / pred-flow e2e 下无差(② 噪声盖过)→ **可去简化 4ch**。cond 文件仍存 7ch,NOWARP=1 省掉即可。**[记忆]**

**footprint** = 物体点集凸包填充的二值 mask,给 ③ 物体占据区提示(cube 位置监督不受 vis 清零影响,`project_v3_latent_predict_renderer`)。

**agent 通道 disjoint 问题**:robot cond=skel(9 点机械臂骨架),human cond=eef3(3 点)→ 本身 = domain-ID 泄漏,human agent 迁移不过来的一大根子。retarget human 手→robot 骨架(build_can_cond SRC=human AGENT=skel + retarget sidecar)是候选机制,未落地。**[记忆]** `project_multihead_aux_wm` §agent-skel retarget。

**agent 通道 ablation(③,Wan 线)**:mask vs skel — 收敛 60000 后 **skel 略胜**(pred-flow agent 区 skel 0.1616 < mask 0.1675,7/8 seq)。早先"mask 稳赢"是 skel 欠训假象。**[记忆]** `project_wan_vae_migration` 最终收敛节。

---

## 7. 全实验史 + 结论表

> 时间线主题分组。每条:做了什么 / 结果 / 采用or舍弃 / 为什么。★ = 活/重要。

### 7.1 ② human-helps(核心命题演进)

| 实验 | 结果 | 采用/舍弃 | 出处 |
|---|---|---|---|
| Step 0 flow 可分性 | object-flow probe 0.645 / MMD ratio 1–1.5× vs 像素 1.0/582× | ✅绿灯 object-flow 方向 | `project_object_flow_shared_step0` |
| ② 稀缺 human-helps(v3) | N=50 robot:11.03→4.27(−61%),≈28× 数据效率;robot 充足饱和不帮 | ✅命题成立(**稀缺**) | `FLOW_WM_REPORT` §1 |
| 干净全量 2550 tracks | human_only 钉 7–8px(封顶弱预测器);robot 降到 2.84 足量反害 1.16 | 满量 human 冗余 | `project_humanhelps_full2550_cleantracks` |
| GDINO 干净 track 重跑 | 清干净 track 没复活满量 human-help | ★铁证:满量冗余 = **单场景数据天花板**非 track 噪声 | `project_multihead_aux_wm` §干净tracks判决 |
| ★双向迁移 3×2 | ro/ho/rh × robot/human:动作 disjoint 对称(零 shot 两边崩);加 robot 帮 human(3.60→3.19)不帮 robot(饱和) | 迁移走 object-dynamics 非 action;谁没饱和谁受益 | `project_bidirectional_transfer_3x2` |
| ★seed-verified 干净 L48 | 全量 human-help mp −0.066(横跳,噪声);稀缺 n100 +4.9(≫噪声,真) | ✅**只稀缺真帮** | `STRATEGY_2026-07-30` §1 |

**★headline(定稿)**:human-helps-at-scale **基本证死**(单场景天花板),**只在稀缺(小 demo)真帮 +4.9**。别再写"human 帮训更好的满量 robot WM"。**[记忆]** `STRATEGY_2026-07-30` §0。

### 7.2 ② 动作表示(action rep)

**[记忆]** `SESSION_REPORT_2026-07-29` §4(干净全量 H40 drift_px_cam_high):

| rep | ro | rh | human 帮 | 判 |
|---|---|---|---|---|
| **mp**(中点丢腕+grip 标量) | **2.083** | 2.22 | −0.13 | ✅**赢家** |
| dummy5 | 2.44 | 2.76 | −0.32 | 基线 |
| dummy5rt(几何 retarget) | 2.928 | 3.297 | −0.37 | ❌**判负**(冻死开口抹抓取,probe 0.9994) |
| dummy5rs(breathing) | ≈mp(n100 +4.95 vs mp +5.36) | | | ⚪无净增益,mp 仍赢 |
| cpt(纯接触点) | 2.505 | 3.199 | | ❌丢太多 |

结论:mp 赢因为**开口留成域中性显式标量**(grip 通道)不塞几何再抹掉。★根因诊断:action 端输入支撑 disjoint(probe 1.0)是 human-helps 弱根因,非 Δflow 输出(0.53 同域);杠杆 = 速度/差分(同域)但丢定位精度。**[记忆]** `project_action_dflow_separability` / `project_velocity_action_featurization`。

### 7.3 ② two-head + FiLM(域头)

**[记忆]** `project_dualwm_domain_head` + `SESSION_REPORT_2026-07-29` §2 + `STRATEGY_2026-07-30` §1:
- **two-head 判负**:rh_two 2.314 ≈ rh_single 2.318 ≫ ro 2.083 → 域专属输出头没削全量饱和害;**害是 trunk 级**,输出侧治不了(正中 related-work 预测:没一篇用硬域输出头,OSCAR 把"avoids embodiment-specific heads"当卖点)。
- **FiLM(软域)判负**:全量 +0.030(横跳,噪声);稀缺 n100 +4.45 略输单头 +4.90。
- 2×2 头诊断:两头确实各自分化(对角优于非对角),但不改 trunk 级害。
- 代码 `HEAD_MODE=single|two|film` 保留(`exp_scel_dualview_wm.py`,commit 4e86a8b..6795f4d)。

### 7.4 ② flow-cond > naive(接口价值)

- dual-view DiT:flow-cond 显著赢 naive eef-FiLM(双视角 >2σ);e2e ②→③ 贴 GT-flow 天花板(+0.007/0.016)仍赢 naive。**[记忆]** `DUALVIEW_DIT_REPORT` / `project_dualview_dit_flow_vs_eef` / `project_dualview_dit_formal_done`。
- ★纠正:优势拆成机制(空间注入)+ 内容(物体运动);eefsp(空间注入 eef)cam_high 反超 flow → 净价值靠端到端可预测性。
- keyboard 四列(GT|IWS-naive|ours-noflow|ours):赢 IWS 主因 = **我们 VAE+latent 实现**(ours-noflow ≫ IWS);flow 增益 = cube 精度 2.7×(2.2 vs 6.0px)。**[记忆]** `project_keyboard_3way_ablation`。

### 7.5 ③ 渲染器 latent route(detmem)

**[记忆]** `project_renderer_latent_route`(Wan 迁移前的结论):
- ★latent route 锐度翻盘赢 pixel。赢家 = **detmem-16ch**(16ch VAE + 确定性预测 latent + decode-LPIPS + prev-memory,**不用 flow-matching**)。GT-flow LPIPS 0.138 < pixel temporal-SPADE 0.141。四要素缺一不可。
- 7-arch 条件化 ablation:SPADE 空间 per-pixel 调制极高效(1.8M ≈ 15.5M concat);xattn/film_global 垫底 = 空间对齐是关键;形变各 arch ~7.4 = 单帧固有抖动。
- DELTA-latent 渲染判负(residual 是 ② 跨具身杠杆非 ③ 锐度杠杆)。
- DF 抗 compounding 判负(DF 帮倒忙,形变根因是单帧锐度非缺时间机制)。

### 7.6 ③ Wan 视频迁移(现行 ③)

**[记忆]** `project_wan_vae_migration`(M1–M5 全线打通):
- render LPIPS ~0.055–0.062(旧逐帧 ~0.12 的一半)。e2e ②→③ 0.060 贴天花板 0.055(② 误差只 +0.006)。
- warp ablation:天花板帮一点 / e2e 无差 → 冗余,CCOND=4 关掉。
- agent:收敛后 skel 略胜 mask(早先 mask 赢是 skel 欠训假象)。

### 7.7 ③ human-helps(多头 aux)

**[记忆]** `project_multihead_aux_wm`:
- 新视频 ③ 上 human 帮渲染翻旧 rh≈ro:4-run render-LPIPS 单调 ro_noaux 0.171 > rh_noaux 0.156(+8.6%),**scarce(N100)**。
- ★但随 N:human 帮 ③ N100 +8.7 → N400 +15.5(峰)→ N800 +2.9 → **N1600 −5.9(反害)** → **训练期 human 帮 ③ 根本上 scarce-only,满量消失且反转**。
- 头消融(render-LPIPS n=24):mask +10.6% > hc +9.5% > noaux +8.7% > all +8.4% > **dino +5.8%(最低,与 human 重叠)**。几何头(mask/HC)> DINO。
- ★几何头 @full~2000 判决:mask/hc 满量也不复活 human-help(N1600 mask 帮是 n=8 噪声)→ **表示/几何头/B 对齐撑住满量 human-help 全线证伪**。
- ★120k 公平重比:ro_full 0.0568 ≈ rh_full 0.0566 → **human 混训对 ③ 渲染中性**(40k"human 害"是 rh 欠训假象)。co-train 免费换统一渲染器(rh 渲 human 0.061 ≈ 渲 robot;ro 渲 human 0.128 乱码)。**[记忆]** `project_keyboard_3way_ablation` 2026-07-29 节。
- ★flow-prompt 前提验证:换 cond 的 flow 源 → 物体跟随,human flow 零 eef 跨具身驱动 robot ③ 物体渲染成立(测试期 volume-independent,paper 主线候选)。

### 7.8 keyboard demo(可控接口)

**[记忆]** `project_can_zlift_keyboard` / `project_keyboard_interactive_demo`:
- can ②-驱动 pick-place(真实动作 + 转向,GT 锚,标准 save_combined_gif)。★abs vs rel 定论:相对 dummy5 每个 N 精度都赢,满数据 human-helps 消失,渲染层 0.123 < abs 0.137。
- cube 可控已证;agent 消失 = 只合成 cube 没合成 agent(joint 冻结 + eef 拖远 OOD)非 mask 崩。gripper 开合 GRIP=1 可控但 grasp 判决未破 weld。
- ★真 OOD = keyboard 交互控制(无 GT)非 held-out clip;判据 = controllability + 眼检非 render-LPIPS。**[记忆]** `project_multihead_aux_wm` OOD 重构。

### 7.9 L48 重建 + 泄漏修复(2026-07-29/30)

**[记忆/代码]** `L48_REBUILD_LOG.md` + `SESSION_REPORT_2026-07-29`:
- 泄漏实锤(§2.6):85% heldout 帧泄漏 → episode-split 修(`split_by_episode`,commit 7868fa8)。
- human L24→L48 重建(8 阶段链,3 retrack 脚本参数化 BASE/HCLIP);robot 已 L48。
- ★L48 clean 初步翻转(单种子):L24(leaky)human 害 −0.235 → L48(clean)human 帮 +0.085,two-head +0.039 → 疑 short-clip + 泄漏制造假负。**但 seed 复核后**(`STRATEGY_2026-07-30` §1):全量 −0.066 横跳 = 噪声,回到"满量数据天花板"。稀缺 n100 +4.9 稳。

### 7.10 相关工作对标(paper 定位)

**[记忆]** `project_multihead_aux_wm` §related-work + `reference_flow_repr_survey` + `feedback_ground_in_related_work`:
- FlowWAM(2607.13017):Wan2.2 VAE + 稠密逐像素 flow(agent 本体运动,与我们"删 agent 只留 object-flow"**正好相反**)→ 它无域不变量只有格式统一 = 我们抓手。
- OSCAR(2606.04463):Cosmos VAE + 2D 骨架 + warm-start human 帮渲染(Table3 已做)。
- EgoWAM(2607.08436):多头,负迁移只在 action 通道不在 world 通道;human 价值在 OOD。
- DexWM(2512.13644):分置 ACTION(手 kpt 差分 Δ 同域)vs STATE(DINO 绝对位置走 HC 头);撞对我们 HC 头 + dummy5。
- MaskWAM(2606.13515):条件不配预测监督就被无视(84.9→21.6)→ 辅助预测头是杠杆。
- ★换旗:novelty 从"结果"(会渲染/flow 替 eef/human warm-start 都被插)挪到 **机制 + 接口**:(A) object-flow(删 agent)= 跨具身 invariant 接口;(B) human 帮 render-LPIPS 判的 ③;(C) 多抽象头共享 trunk。

---

## 8. 当前状态 + 开放问题 + 下一步

**[记忆]** `STRATEGY_2026-07-30` §5-7:

**活着的三条腿(paper 该押)**:
1. **flow-cond 优越**(FlowWAM 式,但我们 **object-centric** 差异化 vs 他们 agent-flow)。
2. **可控 object-centric 跨具身接口**(② 预测 object-flow → 驱动 ③;keyboard / human-flow-prompt)。
3. **会渲染**(Wan VAE 视频 ③);human-helps **只在稀缺**老实讲。

**② 定位**:不靠 human-helps 立足(human 在 ② 稀缺帮、不传渲染);立足于 (a) 可控 object-centric 接口(差异化 FlowWAM agent-flow / OSCAR 骨架)、(b) flow-cond 优越 claim 的预测器。独立贡献,保留。

**③ 两件事别混**:(a) ③ 质量(渲染清晰/天花板)= renderer capacity(Wan VAE/分辨率/DiT),**和 human 无关**,纯工程,是提升渲染主力;(b) ③ human-helps = aux 头,满量也数据天花板消失,只稀缺/OOD 可能帮,几何头(mask/HC)> DINO。

**下一步(优先级)**:
1. **flow-cond vs naive-cond 重做**(当前 Wan ③ + L48,照 FlowWAM Fig.4:固定 backbone 只换 cond,加 cube_px 控制 metric)—— paper 第 1 条腿,最该做。
2. FiLM 结果收口头模式全表(single/two/film × 域 × 全量/稀缺,seed-verified)。
3. ③ 稀缺/OOD human-helps(aux 头,几何头优先)—— 唯一还有戏的 human-helps。
4. ③ 渲染质量(renderer capacity,和 human 无关)。
5. learning-based retarget(MT-π 式)/ ③ agent-cond OSCAR 线画(human MANO vs pinch)—— 稀缺/迁移侧。
6. **多场景 human 数据 = 满量 human-helps 唯一真出路**(超出当前单场景数据)。

**开放问题**:
- FiLM 稀缺是否 > 单头?
- flow-cond vs naive 的 cube_px gap 多大(TA 类)?
- ③ aux 头在 OOD(keyboard)上 human 是否帮(唯一没测干净的 human-helps regime)?
- learning-based retarget 在稀缺区能否 > mp?
- 切片 overlap stride(Phase 2,当前 episode-split 已够诚实)。

**当前 running(2026-07-30)**:L48 clean 重训链 recovery(realwrist=55028 → latents → cond → ②{ro/rh/rh_two/稀缺} → ③{ro/rh})。输出 `outputs/cross_embodiment_wm/epsplit_L48/`(②)、`outputs/video_arch_wm/epsplit_L48_mh/`(③)。**[记忆]** `L48_REBUILD_LOG.md` recovery 节。

---

## 附:关键文件索引

| 用途 | 文件 |
|---|---|
| ② WM | `exp_scel_dualview_wm.py`(DualLWC)、`amplify_wm.py`(FlowWM_LWC 基类) |
| ② dual-view DiT(旧③) | `exp_scel_dualview_dit.py` / `exp_scel_dualview_dit_formal.py` |
| ③ Wan 视频 | `video_dit.py`、`train_video_dit.py`、`video_multihead_wm.py`、`train_multihead_wm.py`、`wan_vae.py` |
| cond/latent build | `build_can_cond.py`、`build_can256_latents.py` |
| 数据 gen/retrack | `gen_flow_render_dataset_caneef.py`、`gen_flow_render_dataset.py`、`retrack_sam2_full.py`、`assemble_retrack_clips.py`、`build_human_realwrist_clips.py`、`build_human_wrist_low.py` |
| SS 超参 | `eval_scheduled_sampling.py`(R_SS=16;WM_LR/BS/EPOCHS from `e2e_flow_wm_render.py`),K=4/F=20 常量 `e2e_flow_wm_render.py:30-31` |
| 报告 | `FLOW_WM_REPORT.md`、`DUALVIEW_DIT_REPORT.md`、`RENDERER_GRASP_RESEARCH_LOG.md`、`L48_REBUILD_LOG.md`、`SESSION_REPORT_2026-07-29_leakage_twohead_dataloading.md`、`STRATEGY_2026-07-30_paper_positioning.md` |
| 数据集 | `outputs/flow_render_dataset_can_dual/`(clips + sidecar + retrack)、`outputs/flow_render_dataset_can_dual_L48/`(L48)、`outputs/video_arch_wm/{wan_latents_can_dual,cond_can_dual,aux_targets}/` |
