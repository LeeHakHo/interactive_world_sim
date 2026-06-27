# Keyboard 三列对比:GT | IWS-naive co-train | ours(object-flow)

> 设计文档 / spec。2026-06-27。
> 目标:把 IWS 原生 latent-diffusion world model(naive human+robot co-train,baseline)接进
> keyboard-control rollout,和我们的 object-flow ②③(detmem)+ 真实 GT 三列并排对比,产出
> paper 用的 baseline 对比 figure 与 metric。

## 0. 动机

我们这条线的卖点是 object-flow / structured 跨具身 WM 能精确可控,而**像素级 naive co-train**
(把 human+robot 帧直接混训一个 latent-diffusion WM)是 §0 结论里"像素级 WM 难迁移"的那条 baseline。
本任务用**同一套 keyboard 指令 + 同一段真实 action**,让 GT、IWS-naive、ours 三者在 v3 数据上
同初始帧并排,直观 + 定量地展示二者差异。对标见 `CROSS_EMBODIMENT_WM_ATTEMPTS.md`(§0、§4.1 OSCAR)。

## 1. 模型与接口(已调研确认)

**IWS-naive co-train ckpt** = `checkpoints/epoch=3-step=250000.ckpt`
(`LatentWorldModel` stage-2,`interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py`)。
- 结构:conv `encoder`(6ch→latent 4ch/32×32)→ `dynamics`(`CMLatentDynamics`,7 维 action 经 MLP `action_emd` 条件)→ `decoder`(diffusion UNet + ControlNet,DDPM)decode 成像素。
- **action = 7 维 = right_ee `pos(3) + euler_xyz(3) + gripper_width(1)`**。由 normalizer stats 锁定:
  dim2(z)≈0.1=常数(pin plane,= v3 `z_plane` 0.10024)、dim3/4/5(euler)基本固定(dim5≈1.57)、
  **dim0/1=world x,y 平移(可动)**、**dim6=gripper width 0~0.042(可动=开合)**。
- 别人用**与 ours 相同的 crop**(v3 workspace)训练 → v3 数据对它分布内,可与 ours 同初始帧。
- 关键调用链(复用 `evaluate_latent_metrics.py`):`load_model` → `encoder_forward` → `dynamics_forward(z0, actions)` → `render_img_cm(model, z_pred, ...)`。
- ⚠️ ckpt 裸放在 `checkpoints/`,**无附带 `.hydra/config.yaml`** → load 要手动用
  `configurations/algorithm/latent_world_model.yaml` 重建 algo cfg(`action_dim=7`,对齐 latent/decoder 维度)。

**ours** = object-flow ②(`FlowWM_LWC`/`GripLWC`)+ ③(`DetMemRenderer`),复用 `exp_scel_keyboard.py`
GRIP 模式现有的 keyboard→eef/flow→detmem 管线。

## 2. 数据与 action 构造

- 数据:v3 robot seqs(`human_play_data/play_robot_v3_*`,crop (190,225,210,205)→128),
  与 ours 已用的同批 seq,三列共享同初始帧。
- **replay action**:从 v3 parquet 取 right_ee pos(3)+euler_xyz(3)+gripper_width(1)。
  精确列名(`obs_right_ee_position` / `obs_right_ee_euler_xyz` / observed `width_right`,用 observed 非 command)
  在实现第一步**对照 normalizer min/max 校验**确定(构造的 action 必须落在 stats 区间内)。
- **合成 keyboard action**:从起始真实 action 出发,左右上下→dim0/1 增量、open/close→dim6 ramp,
  其余维持起始值。

## 3. 两类输出

### (a) replay 质量 gif —— 三列 `GT真实未来 | IWS-naive | ours`
- 喂真实 action,有 GT。metric:
  - 像素 vs GT:**PSNR / LPIPS**(逐帧)、**FVD**(序列)。
  - **cube 位置误差**:HSV 检测红块,**必报检测率 + cube 消失记罚**(防 nan trap,见 `feedback_cube_metric_nan_trap`)。
- 存 `summary.txt` + 指标表。

### (b) keyboard 合成可控 gif —— 两列 `IWS-naive | ours`
- 指令集 = 左右上下 + open/close(同现有 demo)。
- **方向校准**:用一段真实移动拟合 world(x,y)↔图像(u,v) 的轴/符号,保证两模型对同一指令朝同方向动。
- metric:cube travel + 方向跟随(实际位移方向 vs 指令方向)。

两类 gif 都加 GT-flow overlay 列/参照(见 `feedback_gif_add_gtflow_column`)。

## 4. 组件边界(单一职责)

1. `iws_wm_runner`:封装 IWS `LatentWorldModel` load + rollout(真实/合成 action)→ 像素帧 (T,128,128,3)。
2. `action_builder`:v3 parquet→7维 action(replay);keyboard→7维 action(合成)+ world↔image 方向校准。
3. `ours_runner`:复用 `exp_scel_keyboard.py` GRIP 管线 → 像素帧。
4. `compare_render`:三/两列拼帧 + GT-flow overlay + 写 gif。
5. `metrics`:PSNR/LPIPS/FVD + cube 检测(检测率+位置)+ 方向跟随 → summary.txt。

新建脚本 `exp_keyboard_3way.py`(不污染现有 `exp_scel_keyboard.py`,见 `feedback_unify_code_no_script_sprawl`:
ours 侧 import 复用而非重写)。输出独立目录 `outputs/cross_embodiment_wm/keyboard_3way_iws/`
(INDEX.md + summary.txt + gifs/),见 `feedback_per_experiment_output_dir`。

## 5. 风险 & 实现第一步(验证优先)

1. **load 验证**:手动重建 algo cfg → `load_from_checkpoint` 成功 → 喂一个 v3 seq replay → decode →
   **眼检像素合理**(agent/cube 在位、不糊成噪声)。若失败,先解决 load/cfg 再继续。
2. **action 构造验证**:构造的 7 维 replay action 落在 normalizer min/max 内;eyeball replay rollout
   是否跟真实未来大致一致(分布内 sanity)。
3. **方向校准**:合成时两模型方向必须一致,否则对比不公平。
4. **cube nan trap**:必检检测率 + 眼检,别只信 px(见 `feedback_cube_metric_nan_trap`)。
5. **held-out 声明**:这些 v3 seq 对 IWS 最好 held-out 更公平;若分不清,在 INDEX/summary 明确声明。
6. **组件版本声明**:gif/summary 注明用的 ② / ③ / IWS ckpt 具体版本(见 `feedback_declare_component_versions`)。

## 6. 里程碑(粗)

- M0:IWS load + 单 seq replay + decode 眼检通过(风险点 1/2)。
- M1:replay 三列 gif + 质量 metric(输出 a)。
- M2:keyboard 合成两列 gif + 方向校准 + 可控 metric(输出 b)。
- M3:INDEX.md + summary.txt + NSEQ≥8 全量,结论存盘。
