# can dual-view z-lift keyboard 交互 demo — 设计

**2026-07-20**。用户要在 can 上做完整 dual-view keyboard 交互(xy 平移 + z 抬起 + grip 开合),明天开会汇报,同时要一份帮做 slides 的详细文档。

## 背景 / 动机
- v3 那套 keyboard(`exp_scel_keyboard.py`)把 eef **pin 死在固定 z 平面**(z=0.10),只能 2D,且单视角 + 走 IK adapter。
- can 数据**没 pin 平面** + **dual-view**,罐子有真实 z 运动(tracks3d z 抬起 p90 12cm/max 19.5cm)。
- **lift eval 已 de-risk**(2026-07-20):真实抬起片段里 ② replay 能渲染出抬起、pred 红点跟随 → ② 学到了 z。
- can 的 ② `rollout_dual(wm, tracks, ef_high, ef_low, H)` **直接吃双视角 eef**,不用 IK/joint → keyboard 只需合成双视角 eef 序列。

## 架构(新脚本 `exp_scel_keyboard_can.py`,不动 v3 版)
```
keyboard 动作序列 (如 "lift3 right3 drop3")
  → ① 动作模板查表 (action_templates.npz: 每动作 → 双视角 eef 单位Δ)
  → 双视角 eef 序列 (ef_high, ef_low) 逐帧累积
  → ② rollout_dual (align_wm ②, wm_align_wm_rh_N400)
  → 物体 flow (双视角 tracks 预测)
  → ③ dual-view DiT 渲染 (final_warp_100ep 定版)
  → 四列 gif (viz_combined layout)
```

## 组件

### C1. 动作模板提取 (核心, `build_action_templates()`)
从 can `clips_robot.npz` 真实片段统计每个单位动作的**双视角 eef 2D 位移方向**(绕过坏标定,数据驱动):
- `lift`/`drop`:筛 tracks3d z 增大/减小最多的片段,取那段 eef 在 cam_high/cam_low 各自的平均归一化 2D Δ。
- `left/right/up/down`:筛 tracks(2D)对应方向平移的片段,取 eef 双视角 Δ。
- `open/close`:筛 grip 字段增大/减小的片段,取 eef **3 点间距** Δ。
- 输出 `action_templates.npz`:每动作一个 `(2 view, 3 pt, 2 xy)` 单位 Δ + 步长标定(单位步 = 数据里典型单帧位移)。一次性,可复用。

### C2. 动作序列 → eef 序列 (`synth_eef(script, ef0)`)
从起始 eef `ef0`,按脚本逐动作累积模板 Δ,生成 `(H, 2, 3, 2)` 双视角 eef 序列。REPEAT=每按键步数。

### C3. ② rollout (复用 `exp_scel_dualview_wm.rollout_dual`)
align_wm ② 吃合成 eef + 初始 tracks → 预测物体双视角 flow。H = 动作序列总长。

### C4. ③ 渲染 (复用 `render_g_pred` 思路 / DualViewDiTG)
predicted tracks → dual-view ③ (final_warp_100ep) → 双视角 RGB。

### C5. gif (复用 `viz_combined.save_combined_gif`)
合成动作无真实未来 → 加 `GT(start)` 列(起始帧) + 双视角 Rendered 行 + flow overlay 行(物体绿/红 + eef 黄) + 顶部 caption(动作序列) + 自解释列标题。

## 动作集 (8) + 组合脚本
单位动作:`left right up down`(xy) · `lift drop`(z) · `open close`(grip,best-effort)。

| 脚本 | 序列 | 展示 |
|---|---|---|
| 单动作长 gif | 各 `×5` (grip `×4`) | 每动作干净效果 |
| pick_place | lift3 → right3 → drop3 | 抓起→平移→放下 |
| lift_release | lift3 → open2 | 抬起松开 |
| loop_xy | left2 → up2 → right2 → down2 | xy 可控 |
| z_xy_mix | lift2 → left2 → drop2 | z+平面组合 |

**seq eval**:固定 3 个不同起始 seq(罐子初始位置不同),每个跑上述组合(seq 固定保证可比,同 v3 keyboard 惯例)。

## 组件版本 (必须声明)
- ② = `dualview_comb/extra400_s2/wm_align_wm_rh_N400.pt` (LaST-HD align_wm)
- ③ = `dualview_gmask/final_warp_100ep/dvdit_g.pt` (128 定版, objLPIPS 0.119/0.137)
- 数据 = `flow_render_dataset_can_dual/clips_robot.npz` (L48)

## grip 诚实边界
align_wm ② 吃 eef 3 点,grip=3 点间距变化,② **未必敏感**。模板照提、gif 照出,跑出来看响应;不响应就在文档标注"grip 待 grip-aware ②(GripLWC)",不阻塞 xy+z 主线。

## 测试策略 (TDD)
- `build_action_templates`:模板 shape (8 动作各 2×3×2) + lift 的 cam_low Δy 为负(向上)、drop 为正(方向性 sanity)。
- `synth_eef`:脚本 → eef 序列长度/累积正确性;起点 = ef0。
- 端到端 smoke:一个短脚本跑通 ②→③→gif,输出双视角帧 shape + 非崩。

## 交付
1. `exp_scel_keyboard_can.py` + `tests/test_keyboard_can.py`
2. 所有组合 gif(单动作 + 组合 × 3 seq),上传 Drive `2026-07-20_can_zlift_keyboard/`
3. **汇报文档** `docs/superpowers/specs/2026-07-20-can-zlift-keyboard-报告.md`:背景/方法/流程图/de-risk(lift eval 眼检)/结果 gif+数字/Drive 链接/诚实边界 —— 供做 slides
