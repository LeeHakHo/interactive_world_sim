# v3 数据集 phantom EEF 流水线（human↔robot 同坐标系）+ bbox-gate + 可视化

**日期**：2026-06-07
**分支**：phantom_dynamo
**状态**：设计已批准，待写实现计划

---

## 0. 一句话目标

把新的 12 个 human play 数据集用现成 phantom 流水线抽出 human hand EEF，**EEF 的深度 z 不用 phantom 的噪声单目估计，而是用 URDF 变换把 human 和 robot eef 弄到同一个 robot-world 坐标系、z 钉在 robot 锁定的桌面平面（全局单一常数）**；只跑到 EEF（不 inpaint），打包成 LeRobot 数据集并上传 HF；下游 object-flow WM 用。**bbox 必须先核验正确再铺开全量。**

---

## 1. 数据范围

| 域 | 数据集 | 帧数 | 处理 |
|---|---|---|---|
| human | `aytsaiusc/play_human_v3_{1..12}`（12 个） | 各 ~17930 帧 / 10min@30fps，单 ep | phantom Pass1 + smooth + plane-pin → build → upload |
| robot | `aytsaiusc/play_robot_v3_{1..18}_eef`（18 个） | ~17930 帧 | **仅下载**；已带 `action_right_ee_position/quat/gripper`；用于取 z_plane 常数 |

结构与旧 `play_human_{1,2,3}` 完全一致：`data/chunk-000/episode_000000.parquet` + `videos/chunk-000/observation.images.{cam_high,cam_right_wrist}/episode_000000.mp4` + `meta/info.json`。human 无 eef/action 列。

**已核验的关键事实**
- robot `action_right_ee_position` 的 z 锁在 ≈0.0991–0.1010 m（std~0.0003），x/y 在 ±0.1–0.3 变化 → robot eef 在固定高度平面运动。
- 旧 human phantom eef（cam_high 光学系）：z ∈ [0.72, 2.13]，均值≈0.99m = **离相机的深度**（噪声大，正是要替换的轴）；x/y 由可靠 2D 检测决定。
- 两者 z 不是同一个轴：robot z=桌面高度，human z=相机深度 → **不能在 cam 系直接把 z 写成 0.099**。

---

## 2. 坐标系与变换（已验证）

- 内参 K（`phantom_human_play/intrinsics_cam_high.json`）：fx=fy=381.092, cx=310.085, cy=245.318，480×640，无畸变。
- `T_cam_world`（robot-world / `tabletop_link` → cam_high 光学系），`ot_align.robot_world_to_cam()` 从 URDF FK 算，与已验证 fallback 一致（`Trossen_Analysis/reproject_test.py` 验证投影落在夹爪上）：
  ```
  [[ 0.0,    -1.0,    0.0,    0.0090],
   [-0.9063,  0.0,   -0.4226, 0.1486],
   [ 0.4226,  0.0,   -0.9063, 1.0868],
   [ 0.0,     0.0,    0.0,    1.0   ]]
  ```
- `T_world_cam = inv(T_cam_world)`。
- **前提**：human 与 robot 共用同一台固定 cam_high（同内参同机位），故同一 `T_cam_world` 对 human 成立。这是整个跨具身设定的前提；若 human 录制机位不同则本方案不成立（实现时在 bbox-gate 阶段顺带肉眼确认机位一致）。

---

## 3. 核心算法：plane-pin（human eef → robot-world 平面）

`z_plane`（全局单一常数）= 18 个 robot v3 数据集 **`action_right_ee_position[:,2]`**（指令端，真·锁定平面）的总均值（build 时实算，≈**0.100m**）。

**为什么用 action 而非 obs**（已实测 `play_robot_v3_1_eef`）：`action[t]` 是下一步指令位姿（`action[t]≈obs[t+1]`，dist 0.0166<0.0180m），其 z 锁定 = mean 0.1002 / std 0.00033；`obs_right_ee_position` z = mean 0.0866 / std 0.00584（实际 EE 因柔顺下沉~1.4cm 且抖 17×）。用户说的"锁住 z"= 指令端的锁，故取 action z。注：本流水线只需此单一常数，action 的"下一步"偏移对常数无影响；下游若逐帧比 robot↔human eef 再单独决定 obs vs action。

对每个**检测帧**，输入 = phantom 全 ep smooth 后的 `ee_pts`（cam 系 3D，p_cam）：

1. `(u,v) = project(p_cam, K)` —— eef 像素（可靠，来自 2D 检测/smooth）。
2. 相机射线方向 `d = K^{-1} · [u, v, 1]`（cam 系）。
3. robot 平面 {z_world = z_plane} 表达到 cam 系：法向 `n_cam = R_cam_world · [0,0,1]`，平面点 `x0_cam = T_cam_world · [0,0,z_plane,1]`。
4. 射线∩平面：`t = (n_cam · (x0_cam)) / (n_cam · d)`，`p_cam_pin = t · d`。
5. 变换到 robot-world：`p_world = T_world_cam · p_cam_pin`，构造上 `p_world.z == z_plane`。

输出存 **robot-world 系**：
- `position_right` = `p_world`（x/y 来自检测，z=z_plane 常数）。
- `rotation_matrix_right` = `R_world_cam · ee_oris`（一并转到 robot-world；次要、可靠性低，照转）。
- `width_right` = phantom `ee_widths`（标量，frame-independent，不变）。
- `detected_right` = bool。

**与旧 build_eef 的语义差异**：旧版 eef 在 cam_high 系；v3 在 **robot-world 系**、z 为常数平面。info.json / README 必须注明坐标系变更 + z_plane 数值 + 检测率（实算，非硬编码 85.5%）。

新增模块 `pin_eef_to_plane.py`（被 build 调用）。viz 用时把 `p_cam_pin` 经 K 投影 = 原像素 (u,v)，应落在手上、z 标注恒为 z_plane。

---

## 4. 流水线（每 human ep）

```
下载 → chunk(2/ep) → [BBOX GATE] → Pass1 全量[bbox,hand2d,action]
     → concat+全ep smooth(cam系) → pin_eef_to_plane(→robot-world)
     → build_eef_dataset(v3) → 本地+viz 核验 → push 到 HF
```
**跳过 backplant + Pass2(inpaint)**（只要 EEF）。

- chunk：17930/9000 → **2 chunk/ep**，`demo_num = ep*1000+chunk`（1000,1001,…,12000,12001，纯整数 phantom 可解析）。
- **BBOX GATE（用户强调：bbox 不对 eef 白跑）**：先只在 pilot（如 v3_1 的 chunk 0/1 共 2–3 个 chunk）跑 `mode=[bbox]`（最便宜，只 DINO 不跑 HaMeR），肉眼核 `…/{demo_num}/bbox_processor/video_bboxes.mkv`（左手红框/右手绿框，phantom 自动存）。确认**右手框稳定贴手、无左右混淆/漏检/误检**，必要时调 DINO 阈值 / "a hand" prompt，**通过后才铺开全量 Pass1**。
- Pass1 全量：`--array=0-23`（12ep×2chunk），6h/1GPU/256G，可拆 `0-11`/`12-23` 两批并行抢卡。phantom 用 `conda activate phantom`。
- concat+smooth：CPU，全 ep 一次 GP(pts/widths)+gaussian-SLERP(oris)，cam 系。
- pin：CPU，§3 算法 → robot-world。
- build：读 `processed_play_human_v3_full/{ep}/` + pin → 写本地 `human_play_eef_data/play_human_v3_eef_{N}/`（LeRobot，视频 hardlink，parquet 加 4 列）。
- viz：见 §5。
- push：`yusenluo9z/play_human_v3_{N}_eef`，**上传前用户看 viz 核验放行**。

---

## 5. 可视化（交付给用户看）

phantom 本来就自动存：
- `bbox_processor/video_bboxes.mkv` —— bbox 叠加（左红右绿，ffv1 无损）。
- `hand_processor/` 的 2x2 网格视频（bbox/sam/left-hand/right-hand）。

**bbox-gate 用** = 直接看 `video_bboxes.mkv`。
**eef 终检用** = 用现成 `phantom_human_play/visualize_eef.py`（把 eef 经 K 投影画绿点+三轴+z 标注）对默认 **v3_1 / v3_5 / v3_10** 各取一段 clip 渲染叠加视频；plane-pin 后绿点应仍落在手上、z 标注恒为 z_plane。所有产物**绝对路径**贴给用户，确认后再 build/upload。

---

## 6. 代码组织（避免 script sprawl + 不破坏旧 run）

- 新增 `aytsai_pipeline_config.py`：`{version → {repo_pattern, eps, chunks_per_ep, raw_subdir, processed_chunks_subdir, processed_full_subdir, eef_local_subdir, hf_repo_pattern}}`，含 `v1`（旧 play_human_1/2/3）与 `v3`（新）两套。
- 现有 6 个驱动脚本顶部改为按环境变量 `AYTSAI_VERSION`（默认 `v3`）从配置取参；sbatch 透传。
- v3 用**独立磁盘目录名**（`play_human_v3_chunks` / `processed_play_human_v3_chunks` / `processed_play_human_v3_full` / `play_human_v3_eef_*`），与旧 `play_human_aytsai_*` 产物互不覆盖，旧 run 仍可复现。
- 涉及/新增文件：`download_aytsaiusc_play_human.py`、`download_aytsaiusc_play_robot.py`、`chunk_play_human_aytsai.py`、Pass1 sbatch、`concat_smooth_play_human_aytsai.py`、`build_eef_dataset.py`、`push_eef_dataset.py`；新增 `aytsai_pipeline_config.py`、`pin_eef_to_plane.py`。
- 遵循 working-tree 约定：就地工作，只 `git add` 自己的文件，不加 Co-Authored-By。

---

## 7. 规模/成本

- 24 个 Pass1 GPU job（6h、256G、1GPU）是主开销；bbox-gate pilot 仅 2–3 个轻量 job。
- robot 18 个纯下载（无 GPU）；human 12 个下载。总下载几十 GB 级。
- concat/smooth/pin/build/viz 均 CPU、快。

---

## 8. 环境/坑（沿用 memory）

- phantom 用 `conda activate phantom`；下载/build/pin/viz 用 `iws`（numpy1.26，别用 base：numpy2.x cv2 崩）。
- robot 视频 AV1（cv2 解不了需 PyAV），但本任务**不解码 robot 帧**，无碍。
- `conda run` 缓冲 stdout 到结束，要中途看输出得 print(flush) + 轮询或等完成。
- mask/bbox 质量必须**亲眼看 overlay**，不信覆盖率/描述（[[feedback_mask_quality_must_eyeball]]）。

---

## 8b. 修订（2026-06-08，bbox-gate 后）

### bbox 漏检根因 + crop 修复
v3 首跑 bbox 右手检出仅 ~76%（vs v1 85.5%），最长连续漏 16.7s。诊断（`derisk_bbox_dino_sweep.py` / `derisk_bbox_crop_fullfilter.py`）证：**不是 DINO 检测力/阈值**（"a hand"@0.20 对漏检帧都能检到手），而是**手从画面侧缘伸入带裸前臂 → DINO 返回 hand+arm 细长框 → 被 phantom 单臂 filter `aspect<1.8` 拒（该 filter 是有意保留，只留手框避免拖垮 eef）→ 判漏检**。
**修复**：在 `phantom/phantom/processors/bbox_processor.py` 加 `DETECT_CROP_XYWH=(190,225,210,205)` + `_detect_hand()`：检测前 crop 到工作区 → DINO → 框**偏移映射回全帧坐标** → 裸臂被裁掉、框变方过原 filter。下游 HaMeR/eef 全用全帧不变。实测 chunk1000/1001 检出 75/77% → **85.5/87.5%**，框 aspect p95 1.77、整臂框 0%。filter 一行未改。（phantom 是独立 git 仓且工作树本就含用户未提交 patch，此改动同样留作工作树修改不 commit。）

### 下游 eef 表示：3 点 2D@crop128 + 3D 保留
下游 flow-WM 训练用 **crop(190,225,210,205)→resize 128** 的图像。eef 数据集除存 §3 的 3D robot-world pose 外，**新增 3 个手关键点（wrist=kpt0 / thumb=kpt4 / index=kpt8，取自 `hand_processor/hand_data_right.npz` 的 `kpts_2d`，全帧像素）投影到 crop128 的 2D 像素**列，与 robot 的 base+两爪尖三点同构。
- 投影：`u' = (u-190)*(128/210)`，`v' = (v-225)*(128/205)`。
- **腕（kpt0）常落在 crop 外**（手从右缘进，u≈414>400）→ 原样存（可能 <0 或 >128），配 detected 标志，下游自行 clip/mask。
- eef 的 3D robot-world 值与 crop/分辨率无关、不随下游用 crop 图而变；2D 只是其投影。
- 新增列（每点各一）：`observation.eef.kpt2d_crop128_{wrist,thumb,index}` float32 (2,)（NaN if 未检测）。

## 9. 风险

- **R1（最高）bbox 检测错** → hand2d/eef 全废。缓解 = §4 BBOX GATE，pilot 先核 `video_bboxes.mkv` 再铺开。
- **R2 机位假设不成立**：human 机位若与 robot 不同，`T_cam_world` 失效、plane-pin 错。缓解 = bbox-gate 阶段顺带肉眼确认 human 视频机位与 robot 一致。
- **R3 检测率低**：human 手频繁出框/遮挡导致 detected 比例低，eef 稀疏。缓解 = build 时实算并报告检测率，低于阈值（如 <70%）提示用户。
- **R4 17930 帧 chunk 边界**：2 chunk/ep，全 ep smooth 已消除边界 artifact（concat 后一次 smooth）。
- **R5 orientation 在 robot-world 系可靠性低**：单目手姿态本就糙，标注里如实说明，下游优先用 position+width。
