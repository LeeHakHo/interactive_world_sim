# Keyboard 3D 交互控制(视频架构 ③)设计

> 2026-07-23。把 free-form keyboard 交互从旧 per-frame ③(DualViewDiTG)迁到**新视频架构 ③**(Wan VAE + VideoDiT),
> 并把控制从「2D 图像平面/视角各自偏移」升级为**真世界系 3D**:水平平移(x,y 桌面) + z 轴上下 + grip 功能性抓取。
> 基础脚本 = `exp_scel_keyboard_can_ik.py`(canonical, 别另造轮子);渲染换 `eval_e2e_combined.py` 的 VideoDiT `sample()`。

## 0. 动机 / 要解决什么

- **旧 keyboard(`exp_scel_keyboard_can_ik.py`)的控制是 2D-per-view**:对 cam_high / cam_low **各自**打 image-plane 偏移(`DIRS["lift"]=(0,-1)` 对两视角都是 image-y)。
  这把「z 抬起」和「水平平移」混在一起 → 已知 bug:`lift_high`(纯抬起)时**水平面也漂移**(用户 2026-07-22 抓到,要求架构迁移后修)。
- **③ 是旧 per-frame 架构**,不是刚落地的视频架构(Wan VAE + VideoDiT,render LPIPS ~0.054 是旧逐帧的一半)。

**本设计一次解决两者**:真 3D 控制(投影天然分离 z 和水平,消除漂移) + 视频 ③ 渲染。

## 1. 数据基础(已核实,非猜)

`outputs/flow_render_dataset_can_dual/clips_robot.npz`(2700 clips, L=48):
| key | shape | 含义 |
|---|---|---|
| `eef3d` | (N,48,3,3) | eef 3 点星座,**世界系米制**。coord2=高度(真实抬起 0.096→0.354 ≈26cm);coord0/1=桌面 x,y |
| `tracks3d` | (N,48,48,3) | 物体 48 点 3D(世界系) |
| `tracks3d_valid` | (N,48,48) bool | 3D 有效掩码 |
| `grip` | (N,48) | 夹爪开口宽度。全局 0.0(全闭空)→0.04(全开);≈0.029=夹住罐 |
| `joint` | (N,48,7) | 机械臂关节(供 IK/gmask 参照) |
| `eef`/`eef_low` | (N,48,3,2) | 两视角 2D eef(已由 eef3d 投影得到) |

**标定(REAL, 已验证)**:`calib/rgb_cam_calib_can_REAL.json`(cam_high) + `calib/rgb_cam_calib_can_low_REAL.json`(cam_low)。
自检:eef3d 经两视角 REAL 标定重投影 vs 存储 2D = **mean 0.023px / max 0.063px**(128 crop)。→ 世界系 3D→2D 投影可靠。
标定读法 = `gen_flow_render_dataset_caneef.py:can_rig_KT()`(quat_xyzw + t_world + f,cx,cy);crop-norm = `to_norm`(CROP high=(60,60,390,390) low=(0,0,640,480))。

## 2. 架构 / 管线

```
键盘 → 世界系 3D delta(dx,dy,dz,dgrip)
        │  作用在 eef3d 3 点刚体星座 + grip
        ▼
   世界系 eef3d(t) 轨迹 + grip(t)          ── 功能性抓取:grip 闭且近物体 → 物体 3D 刚体跟随 eef
        │ project(REAL calib, 两视角)
        ▼
   两视角 2D eef 轨迹(efA cam_high, efB cam_low)
        │
        ├─► ② rollout_dual(dummy5)──► predtr(物体 2D flow, 两视角)
        │        (②驱动 + 兜底:抓取相被物体不跟随时用刚体投影覆盖)
        │
        ├─► IK adapter(cam_high eef → joint7)──► FK(pinocchio)──► skel2d(9 kpt, 两视角, REAL 投影)
        │
        ▼
   cond = flow3 + skel1(线画) + warp3  @16×16, tL=12
        │
        ▼
   VideoDiT ③ skel(Wan VAE)sample() ──► 渲染视频(两视角)
```

组件版本(端到端声明):
- ② = `outputs/cross_embodiment_wm/abs_vs_rel_humanhelps/wm_dummy5_rh_N3000.pt`(dummy5, rollout_dual)
- ③ = `outputs/video_arch_wm/m4_ablB_skel/video_dit_ema.pt`(**新视频架构 + skel agent 条件**,消融赢家 7/8,60000 步)
  - agent 条件 = **skel 线画**(9 关键点/8 段,由 IK joint 经 FK 投影得);mask ③ `m4_video_dit` 作 fallback(若合成 joint 下 skel 渲染退化)
- IK = `outputs/cross_embodiment_wm/ik_adapter_can/ik_adapter.pt`(eef→joint7,gmask IoU 0.912)
- FK = pinocchio(`Trossen_Analysis/stationary_ai.urdf`,port 自 `augment_clips_skeleton.py:fk_points`)→ skel2d(REAL 标定投影)
- VAE = 官方 `AutoencoderKLWan`(`wan_vae.py`)

**环境隔离(已核实)**:真 pinocchio FK **只在 conda `phantom`(3.9.0)**;`AutoencoderKLWan` **只在 `.venv_wan`**(iws/.venv_wan 的 pinocchio 都是假 0.1)。
→ **三阶段管线**:A(`.venv_wan`)控制+②+IK+物体轨迹 → B(`phantom`,纯 FK,只 import pinocchio/numpy)joint→skel2d → C(`.venv_wan`)skel cond + VideoDiT 渲染 + gif。中间用 npz 传。

## 3. 控制语义

### 3.1 世界系方向键(修 lift_high 漂移的核心)
| 键 | 世界系 delta | 视角表现(投影后天然) |
|---|---|---|
| left / right | coord0 ∓ (桌面 x) | 两视角水平移动 |
| fwd / back | coord1 ∓ (桌面 y) | 两视角景深/水平移动 |
| **up / down** | **coord2 ±(高度 z)** | cam_high 几乎不动、cam_low 竖直上下 —— **无水平串扰** |

- delta 作用在 eef3d 3 点**刚体星座质心**(星座内部偏移锁死),每按键 `REPEAT` 帧;每帧后世界系工作区 clip(BOX3D,防投影出画/OOD)。
- 纯 up/down = 只改 coord2 → 投影到两视角自动正确 → **lift_high 不再水平漂移**(旧 bug 根因=对两视角各打 image-y,现在是世界系一个 z 分量)。

### 3.2 grip = 功能性抓取
- grip 值域 [0(闭),0.04(开)];键 close→往 0.029(罐宽)减,open→往 0.04 加,每按键 REPEAT 帧线性 ramp。
- **抓取判定**:`grasp = (grip < GRASP_TH) AND (eef 质心到物体质心 3D 距离 < NEAR_TH)`。GRASP_TH≈0.033,NEAR_TH 待标定(用真实抓握帧的 eef-obj 距离分布定)。
- **抓取时**:记录抓取瞬间 `offset3d = obj_centroid3d − eef_centroid3d`;之后每帧 `obj_tracks3d(t) = eef 刚体位姿 · (抓取帧 obj_tracks3d + Δeef)`(平移刚体跟随,本期不转)。
- **释放时**(grip 开过阈值):物体脱离,停在最后 3D 位置(本期不做重力下落)。

### 3.3 物体驱动:②驱动 + 兜底
- 始终跑 ② rollout(2D eef → predtr),保留 ② 价值展示。
- **兜底逻辑**:抓取相内,若 ② 预测物体位移与 eef 位移的跟随比 < FOLLOW_TH(物体没跟着抬/移,合成动作 OOD),
  则用**刚体投影**(§3.2 的 obj_tracks3d 投影到两视角)覆盖 predtr。非抓取相用 ②。
- overlay 同时画 ② 预测(红)与实际用于渲染的轨迹,便于看 ② 是否自己就跟上。

## 4. 视频 ③ 渲染细节

- 照 `eval_e2e_combined.py`(其 skel 分支 m3S):`sample(model, za, cond)`,RF + Diffusion Forcing,首帧 I₀ 锚(za = VAE.encode(GT 首帧))。
- cond: 每 tL=12 个 latent 帧一条,`rf = 0 if k==0 else min(4*k, H-1)`(4× 时序),`cond_v(agent="skel")` = flow_cond3 + skel1(skel2d 线画 via `skel_chan`) + warp3,pool 到 16×16。
- **H=48 单视频块**:keyboard 脚本总帧数 = Σ(keys)·REPEAT 需 = 48(如 REPEAT=4 → 12 次按键)→ 一次 VideoDiT pass。
  更长序列 = chunk-autoregressive(视频块级自回归),列为 follow-up TODO,本期不做。

## 5. 交付物(标准 save_combined_gif,遵守 layout 硬要求)

- **一个 gif = 双视角并列**(cam_high | cam_low 两列),`save_combined_gif`:Rendered 行 + Flow overlay 行 + 列标题 + caption。
  - **Overlay 同时叠 flow + skel**(用户要求):② 预测物体 flow(红)+ GT 参照(绿, 若该 seq 有真实动作)+ **skel 骨架线画(白/彩点,agent 手臂)** + eef(黄)+ **grip 状态**(夹爪开/闭视觉标记)。参照 `eval_e2e_combined.py:ov()`(已支持 flow+skel 同叠)。
  - caption = `{name} | {cmdstr} | grip:{state} | H=48`(信息写标题,短防截断;legend 底部给颜色说明)。
- **脚本集**(SCRIPTS,**多命令组合**,每条 Σn·REPEAT=48):
  - `translate_LR` — 纯水平左右平移。
  - `square_xy` — 桌面正方形(left/fwd/right/back)。
  - `lift_high` — 纯 z 抬起,**验证漂移已修**(眼检 cam_high 物体 x 不动)。
  - `z_wave` — 上下往复(z 可控性)。
  - `grip_cycle` — 开合往复(夹爪视觉)。
  - `pick_place_left` / `pick_place_right` — grip_close→up→left/right→down→grip_open。
  - `carry_square` — 抓起沿方形搬运再放。
  - `lift_translate_lower` — 抓-抬-平移-放-松完整轨迹。核心 demo。
- **中间产物可视化**(遵守 visualize_intermediates):
  - 世界系 3D 控制轨迹 → 两视角投影 2D eef overlay(证明投影正确、纯 z 无水平漂移)。
  - grip 状态 / 抓取判定时间线。
- 独立输出目录 `outputs/video_arch_wm/keyboard_3d/`(gifs/ + README_SETUP.md + summary.txt),别覆盖已有 eval。
- 上传 Drive `iws_evals/2026-07-23_keyboard_3d_video_arch/`。

## 5.5 为什么不把 flow 做成 3D(以及何时该做)

分两层:**物体/eef 表示** vs **③ condition**。
- 本方案**物体/eef 表示已是 3D**(世界系 eef3d/tracks3d,抓取刚体跟随在 3D 里算),只是喂 ③ 的 condition 仍是
  **同一条 3D 轨迹经 REAL 标定投影到两视角的 2D flow**。双视角两次投影唯一确定 3D → z 已被完整编码,③ 无需重训即可渲染 z 抬起。
- **z 控制的正确性来自「一个 3D 轨迹 → 两视角一致投影」**,而非「flow 本身是 3D」。旧 lift_high 漂移=两视角各打独立 image-y(非同一 3D 投影),不是 2D flow 表达不了 z。
- **真 3D flow WM**(② 预测 tracks3d 位移、③ condition 换每点 (dx,dy,dz))技术可行(3D+标定已验证),但需**重训 ② 和 ③**,
  且 human-helps 在 3D 下是否保持需**重新验证**(human 3D 来自 HaMeR/depth 更脏,跨具身 3D 同域性未证)。
  → **列为独立后续研究方向,不并入本 demo**;本 demo 走 3D-控制→2D-投影(z 稳、不重训)。

## 6. 诚实边界 / 已知限制

- 抓取只做平移刚体跟随(不做物体旋转)。
- 释放不做重力下落(物体停在原地)。
- H≤48 单视频块(更长要 chunk-autoregressive,follow-up)。
- ③ 用 mask/gmask agent 条件(skel ③ 质量更好但需合成 skel2d,follow-up)。
- 极限位置/大幅度仍可能 ③ 糊(OOD)→ 幅度 clip + 眼检把关;flow overlay 永远清晰作对照。

## 7. 文件结构

- **Create** `exp_scel_keyboard_3d.py` — 主脚本(世界系 3D 控制 + 投影 + ②/兜底 + IK + VideoDiT ③ + gif)。
- **Reuse** `exp_scel_keyboard_can_ik.py`(script/rollout/IK 结构)、`eval_e2e_combined.py`(VideoDiT sample/cond)、
  `gen_flow_render_dataset_caneef.py:can_rig_KT`(标定)、`wan_vae.py`、`viz_combined.py:save_combined_gif`。
- **Create** `tests/test_keyboard_3d.py` — 投影正确性(纯 z delta → cam_high x 不变)、grip 抓取判定、脚本帧数=48。
- **Create** `outputs/video_arch_wm/keyboard_3d/README_SETUP.md` — 组件版本/ckpt/是否削弱/指标。

## 8. 测试策略(关键正确性,非仅眼检)

1. **投影单元测试**:构造纯 z delta(只改 coord2)→ 投影两视角 → 断言 cam_high 物体质心 x 位移 < ε(证明水平漂移已修);cam_low y 位移显著。
2. **grip 抓取判定**:grip 从 open ramp 到 close 且 eef 近物体 → grasp 触发;远离 → 不触发。
3. **脚本帧数**:每个 SCRIPT 总帧 = 48(单视频块约束)。
4. **端到端眼检**(遵守 measure_real_deliverable_metric):渲染 gif vs 概念预期并排,`lift` 眼检 cam_high 罐子 x 不漂;`pick_place` 抓-移-放全流程物体跟随夹爪。报 ③ 渲染质量(replay 有 GT 的 seq 报 LPIPS)。
