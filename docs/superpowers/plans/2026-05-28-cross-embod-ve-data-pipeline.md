# Cross-Embodiment Video-Edit Baseline — Plan 1/4: 数据管线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 human (`play_human_aytsai` chunks) 和 robot (`play_robot_eef`) 数据各自切成定长 clip，每个 clip 产出 VACE video-editor 训练所需的一份自包含记录 `{frames, agent_mask, eef(M_s, 统一相机系), obj_traj(O_s, 相机系3D), embodiment_image(C), domain}`，写到磁盘 + 一个 index.json。

**Architecture:** 新建包 `interactive_world_sim/cross_embod_ve/data/`，把可复用的纯函数（EEF 坐标变换、像素→3D lift、mask 合并、crop）做成有单测的小模块，IO 重的部分（SAM2 追踪、VGGT 深度、视频解码）做成 smoke + 可视化 QC。一个 root 脚本 `build_ceve_dataset.py` 把它们串起来产出数据集。大量复用现有代码：`robot_world_to_cam`(ot_align.py)、`human_arm_mask`/`inpaint`(inpaint_gap_test.py)、SAM2 build pattern、`gen_phantom_depth_vggt.py`、相机内参 `phantom_human_play/intrinsics_cam_high.json`。

**Tech Stack:** Python 3 / conda env `iws`；PyTorch；PyAV (`av`) 解 robot AV1；OpenCV 解 human mkv + inpaint；SAM2 (`sam2_hiera_large.pt`)；VGGT (`facebook/VGGT-1B`)；scipy（旋转）；pytest。

**Scope note:** 本计划只产出**磁盘上的训练数据集**，不碰 VACE/训练/下游。Plan 2（VACE 集成 + v0 自重建）、Plan 3（disentangle loss + cross-generation）、Plan 4（下游 co-train WM）见文末 roadmap，待 Plan 1 落地后分别展开。

---

## 数据来源（已核实存在）

- **Human**：`phantom/data/processed_play_human_aytsai_chunks/play_human_aytsai_chunks/{chunk}/`（chunk ∈ {1000..3003, 9000}）
  - `video_rgb_imgs.mkv`：30fps 全分辨率 RGB（cv2 解，BGR→RGB）。
  - `segmentation_processor/masks_arm.npy`：(T,480,640) bool，手部 mask。
  - `smoothing_processor/smoothed_actions_right_single_arm.npz`：`ee_pts`(T,3)、`ee_oris`(T,3,3)、`ee_widths`(T,)，**相机系**（cam_high optical），是 phantom 平滑后的 EEF。
- **Robot**：`play_robot_eef/play_robot_{N}_eef/`（N∈1..12）
  - `videos/chunk-000/observation.images.cam_high/episode_000000.mp4`：AV1，480×640，30fps（**PyAV 解，cv2 解不了**）。
  - `data/chunk-000/episode_000000.parquet`：列 `action_right_ee_position`(3)、`action_right_ee_quat_xyzw`(4)、`action_right_gripper`(1)，**Trossen world 系**。
- **相机内参**：`phantom_human_play/intrinsics_cam_high.json`。
- **复用变换**：`interactive_world_sim/algorithms/latent_decompose/ot_align.py::robot_world_to_cam(urdf_path)` → 4×4 `T_cam_world`（含硬编码 fallback，det≈1 已验证）。

## 统一约定（所有 clip 一致）

- **crop + resize**：先 crop `(x=195, y=195, w=256, h=256)`（排除 robot 外围白 mount），再 resize 到 `RES=256`。与现有 dataset 的 crop 一致。
- **clip 长度**：`T_CLIP=25` 帧，步长 `STRIDE=25`（不重叠）。
- **相机系**：human EEF 本就在 cam_high optical 系；robot EEF 经 `T_cam_world` 转到同一系。**position 为主**（也存 rotation 但下游默认只用 position）。
- **O_s 物体**：共享物体 = 蓝盘 + 红块，`K_OBJ=2`。未检出帧填 NaN。
- **embodiment image C**：从 clip 中间帧，围绕 EEF 像素位置 crop `128×128`。

## 输出格式

每个 clip 写 `data/ceve/{domain}/{source_id}/clip_{idx:04d}.npz`：

| key | shape | dtype | 说明 |
|---|---|---|---|
| `frames` | (T_CLIP, RES, RES, 3) | uint8 | crop+resize 后 RGB |
| `agent_mask` | (T_CLIP, RES, RES) | bool | True=agent 像素（手臂/夹爪） |
| `eef` | (T_CLIP, 8) | float32 | pos(3)+quat_xyzw(4)+gripper(1)，相机系 |
| `obj_traj` | (T_CLIP, K_OBJ, 3) | float32 | 物体 3D 位置（相机系，米），NaN=未检出 |
| `embodiment_image` | (128, 128, 3) | uint8 | 末端静态 crop C |
| `domain` | () | `<U8` | "human" / "robot" |
| `meta` | () | object | dict: source_id, frame_start, fps, crop |

并维护 `data/ceve/index.json`：`[{"path":..., "domain":..., "source_id":..., "n_frames":T_CLIP}, ...]`。

---

## File Structure

- Create: `interactive_world_sim/cross_embod_ve/__init__.py`
- Create: `interactive_world_sim/cross_embod_ve/data/__init__.py`
- Create: `interactive_world_sim/cross_embod_ve/data/geometry.py` — 纯函数：EEF world→cam、pixel+depth→3D、EEF→pixel 投影
- Create: `interactive_world_sim/cross_embod_ve/data/agent_mask.py` — human 臂 mask / robot 夹爪 mask
- Create: `interactive_world_sim/cross_embod_ve/data/object_track.py` — SAM2 追踪 + VGGT 深度 lift → O_s
- Create: `interactive_world_sim/cross_embod_ve/data/embodiment_image.py` — C crop
- Create: `interactive_world_sim/cross_embod_ve/data/video_io.py` — robot(PyAV)/human(cv2) 帧读取 + crop/resize
- Create: `interactive_world_sim/cross_embod_ve/data/clip_record.py` — 单 clip 组装 + 写 npz
- Create: `build_ceve_dataset.py` (repo root) — 串联，遍历 episode/chunk 产出数据集 + index
- Create: `configurations/dataset/cross_embod_ve.yaml` — 路径 / 常量配置
- Test: `tests/cross_embod_ve/__init__.py`
- Test: `tests/cross_embod_ve/test_geometry.py`
- Test: `tests/cross_embod_ve/test_agent_mask.py`
- Test: `tests/cross_embod_ve/test_clip_record.py`
- Test: `tests/cross_embod_ve/smoke_build_one_clip.py` — 端到端跑 1 human + 1 robot clip 并 dump QC 图

---

## Task 1: Scaffold 包 + 配置

**Files:**
- Create: `interactive_world_sim/cross_embod_ve/__init__.py`
- Create: `interactive_world_sim/cross_embod_ve/data/__init__.py`
- Create: `tests/cross_embod_ve/__init__.py`
- Create: `configurations/dataset/cross_embod_ve.yaml`

- [ ] **Step 1: 建空包文件**

三个 `__init__.py` 内容都为空文件（仅占位使包可导入）。

- [ ] **Step 2: 写配置文件**

`configurations/dataset/cross_embod_ve.yaml`:

```yaml
# Cross-embodiment video-edit dataset build config (Plan 1)
out_root: /scr2/yusenluo/interactive_world_sim/data/ceve
res: 256
crop: [195, 195, 256, 256]   # x, y, w, h
t_clip: 25
stride: 25
k_obj: 2                      # 蓝盘 + 红块
emb_crop: 128                 # embodiment image C 边长

human:
  chunks_root: /scr2/yusenluo/interactive_world_sim/phantom/data/processed_play_human_aytsai_chunks/play_human_aytsai_chunks
  chunk_ids: [1000, 1001, 1002, 1003, 2000, 2001, 2002, 2003, 3000, 3001, 3002, 3003, 9000]

robot:
  dataset_root: /scr2/yusenluo/interactive_world_sim/play_robot_eef
  episodes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  urdf: /scr2/yusenluo/interactive_world_sim/Trossen_Analysis/stationary_ai.urdf

intrinsics: /scr2/yusenluo/interactive_world_sim/phantom_human_play/intrinsics_cam_high.json
sam2_cfg: sam2_hiera_l.yaml
sam2_ckpt: /scr2/yusenluo/interactive_world_sim/phantom/submodules/sam2/checkpoints/sam2_hiera_large.pt
```

- [ ] **Step 3: 验证可导入 + 配置可加载**

Run:
```bash
cd /scr2/yusenluo/interactive_world_sim && conda activate iws
python -c "import interactive_world_sim.cross_embod_ve.data as d; print('pkg ok')"
python -c "from omegaconf import OmegaConf; c=OmegaConf.load('configurations/dataset/cross_embod_ve.yaml'); print(c.res, c.t_clip, len(c.human.chunk_ids), len(c.robot.episodes))"
```
Expected: `pkg ok` 然后 `256 25 13 12`

- [ ] **Step 4: Commit**

```bash
git add interactive_world_sim/cross_embod_ve tests/cross_embod_ve configurations/dataset/cross_embod_ve.yaml
git commit -m "feat(ceve): scaffold cross-embodiment video-edit data package + config"
```

---

## Task 2: 几何纯函数（EEF world→cam, pixel→3D, EEF→pixel）

**Files:**
- Create: `interactive_world_sim/cross_embod_ve/data/geometry.py`
- Test: `tests/cross_embod_ve/test_geometry.py`

- [ ] **Step 1: 写失败测试**

`tests/cross_embod_ve/test_geometry.py`:

```python
import numpy as np
from interactive_world_sim.cross_embod_ve.data.geometry import (
    transform_points, lift_pixels_to_3d, project_points_to_pixels, load_intrinsics,
)


def test_transform_points_identity():
    pts = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    out = transform_points(np.eye(4, dtype=np.float32), pts)
    np.testing.assert_allclose(out, pts, atol=1e-6)


def test_transform_points_translation():
    T = np.eye(4, dtype=np.float32); T[:3, 3] = [10, 0, -5]
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = transform_points(T, pts)
    np.testing.assert_allclose(out, [[11.0, 2.0, -2.0]], atol=1e-6)


def test_lift_then_project_roundtrip():
    # 已知内参，点在相机前方 → 投影到像素 → 用同 depth lift 回来应一致
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    pts_cam = np.array([[0.1, -0.05, 0.8], [0.0, 0.0, 1.2]], dtype=np.float32)
    uv = project_points_to_pixels(K, pts_cam)
    depth = pts_cam[:, 2]
    back = lift_pixels_to_3d(K, uv, depth)
    np.testing.assert_allclose(back, pts_cam, atol=1e-4)


def test_lift_pixels_nan_depth_propagates():
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    uv = np.array([[100.0, 100.0]], dtype=np.float32)
    back = lift_pixels_to_3d(K, uv, np.array([np.nan], dtype=np.float32))
    assert np.isnan(back).all()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/cross_embod_ve/test_geometry.py -v`
Expected: FAIL，ImportError（`geometry` 模块不存在）。

- [ ] **Step 3: 实现 geometry.py**

```python
import json
import numpy as np


def load_intrinsics(path: str) -> np.ndarray:
    """读 cam_high 内参 JSON → 3x3 K (float32)。兼容 {fx,fy,cx,cy} 或 {K:[9]} 两种格式。"""
    with open(path) as f:
        d = json.load(f)
    if "K" in d:
        return np.asarray(d["K"], dtype=np.float32).reshape(3, 3)
    return np.array([[d["fx"], 0, d["cx"]],
                     [0, d["fy"], d["cy"]],
                     [0, 0, 1]], dtype=np.float32)


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """T: (4,4) 齐次变换；pts: (N,3) → (N,3)。"""
    pts = np.asarray(pts, dtype=np.float32)
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), np.float32)], axis=1)  # (N,4)
    out = hom @ np.asarray(T, dtype=np.float32).T
    return out[:, :3]


def project_points_to_pixels(K: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    """pts_cam: (N,3) 相机系 → (N,2) 像素 (u,v)。"""
    pts_cam = np.asarray(pts_cam, dtype=np.float32)
    z = pts_cam[:, 2:3]
    uvw = pts_cam @ np.asarray(K, dtype=np.float32).T
    return uvw[:, :2] / z


def lift_pixels_to_3d(K: np.ndarray, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """uv: (N,2) 像素；depth: (N,) 米 → (N,3) 相机系。depth=NaN 则该点全 NaN。"""
    K = np.asarray(K, dtype=np.float32)
    uv = np.asarray(uv, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (uv[:, 0] - cx) / fx * depth
    y = (uv[:, 1] - cy) / fy * depth
    return np.stack([x, y, depth], axis=1)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/cross_embod_ve/test_geometry.py -v`
Expected: 4 passed。

- [ ] **Step 5: 加 robot EEF→cam 的集成校验（用真实 T_cam_world）**

追加到 `tests/cross_embod_ve/test_geometry.py`:

```python
def test_robot_world_to_cam_is_rigid():
    from interactive_world_sim.algorithms.latent_decompose.ot_align import robot_world_to_cam
    T = robot_world_to_cam()  # 默认 URDF 或 fallback
    assert T.shape == (4, 4)
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-3)   # 正交
    assert abs(abs(np.linalg.det(R)) - 1.0) < 1e-3              # det≈±1
```

Run: `pytest tests/cross_embod_ve/test_geometry.py -v`
Expected: 5 passed。

- [ ] **Step 6: Commit**

```bash
git add interactive_world_sim/cross_embod_ve/data/geometry.py tests/cross_embod_ve/test_geometry.py
git commit -m "feat(ceve): geometry pure funcs (transform/lift/project) + robot world->cam check"
```

---

## Task 3: 视频 IO（robot PyAV / human cv2 + crop/resize）

**Files:**
- Create: `interactive_world_sim/cross_embod_ve/data/video_io.py`
- Test: `tests/cross_embod_ve/smoke_build_one_clip.py`（本任务先建骨架，逐步填充）

- [ ] **Step 1: 实现 video_io.py**

```python
import av
import cv2
import numpy as np


def crop_resize(frame_rgb: np.ndarray, crop, res: int) -> np.ndarray:
    """frame_rgb: (H,W,3) uint8 → crop (x,y,w,h) → resize res×res。"""
    x, y, w, h = crop
    c = frame_rgb[y:y + h, x:x + w]
    return cv2.resize(c, (res, res), interpolation=cv2.INTER_AREA)


def read_robot_frames(mp4_path: str) -> np.ndarray:
    """AV1 robot 视频 → (T,H,W,3) uint8 RGB（全分辨率，未 crop）。"""
    container = av.open(mp4_path)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames, axis=0)


def read_human_frames(mkv_path: str) -> np.ndarray:
    """human video_rgb_imgs.mkv → (T,H,W,3) uint8 RGB（全分辨率，未 crop）。"""
    cap = cv2.VideoCapture(mkv_path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0)
```

- [ ] **Step 2: Smoke — 各读 1 个真实视频，验证形状**

新建 `tests/cross_embod_ve/smoke_build_one_clip.py`，先放：

```python
"""Smoke: 读 1 human + 1 robot 视频首帧，验证解码与 crop。直接 `python` 运行。"""
import numpy as np
from omegaconf import OmegaConf
from interactive_world_sim.cross_embod_ve.data.video_io import (
    read_robot_frames, read_human_frames, crop_resize,
)

cfg = OmegaConf.load("configurations/dataset/cross_embod_ve.yaml")


def main():
    rb = f"{cfg.robot.dataset_root}/play_robot_1_eef/videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
    hm = f"{cfg.human.chunks_root}/{cfg.human.chunk_ids[0]}/video_rgb_imgs.mkv"
    rf = read_robot_frames(rb)
    hf = read_human_frames(hm)
    print("robot frames", rf.shape, rf.dtype)
    print("human frames", hf.shape, hf.dtype)
    rc = crop_resize(rf[0], list(cfg.crop), cfg.res)
    hc = crop_resize(hf[0], list(cfg.crop), cfg.res)
    assert rc.shape == (cfg.res, cfg.res, 3) and hc.shape == (cfg.res, cfg.res, 3)
    assert rf.ndim == 4 and hf.ndim == 4
    print("OK: decode + crop_resize")


if __name__ == "__main__":
    main()
```

Run: `python tests/cross_embod_ve/smoke_build_one_clip.py`
Expected: 打印两个 `(T,480,640,3) uint8`，末行 `OK: decode + crop_resize`。

- [ ] **Step 3: Commit**

```bash
git add interactive_world_sim/cross_embod_ve/data/video_io.py tests/cross_embod_ve/smoke_build_one_clip.py
git commit -m "feat(ceve): video IO (PyAV robot / cv2 human) + crop_resize, smoke decode"
```

---

## Task 4: Agent mask（human 臂 / robot 夹爪）

**Files:**
- Create: `interactive_world_sim/cross_embod_ve/data/agent_mask.py`
- Test: `tests/cross_embod_ve/test_agent_mask.py`

> 复用思路源自 `inpaint_gap_test.py::human_arm_mask` / robot SAM2 夹爪种点。这里重写为模块函数：human 用 phantom `masks_arm` 膨胀 + 暗袖连通域 − 红块；robot 用 SAM2 在 crop 内最暗点种点 + 膨胀。

- [ ] **Step 1: 写失败测试（纯逻辑部分：mask 合并/红块剔除）**

`tests/cross_embod_ve/test_agent_mask.py`:

```python
import numpy as np
from interactive_world_sim.cross_embod_ve.data.agent_mask import (
    red_cube_mask, human_arm_mask_from_hand,
)


def test_red_cube_mask_picks_red():
    img = np.zeros((10, 10, 3), np.uint8)
    img[2:5, 2:5] = [200, 20, 20]      # 红块
    img[6:9, 6:9] = [20, 20, 200]      # 蓝盘
    m = red_cube_mask(img)
    assert m[3, 3] and not m[7, 7]


def test_human_arm_mask_dilates_hand_and_drops_cube():
    hand = np.zeros((20, 20), bool); hand[8:12, 8:12] = True   # 手部 mask
    img = np.zeros((20, 20, 3), np.uint8)
    img[2:5, 2:5] = [200, 20, 20]                              # 远处红块
    m = human_arm_mask_from_hand(img, hand, dilate=3)
    assert m[10, 10]                  # 手保留
    assert not m[3, 3]                # 红块被剔除
    assert m.dtype == bool and m.shape == (20, 20)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/cross_embod_ve/test_agent_mask.py -v`
Expected: FAIL，ImportError。

- [ ] **Step 3: 实现 agent_mask.py**

```python
import cv2
import numpy as np


def red_cube_mask(img_rgb: np.ndarray) -> np.ndarray:
    """红块 mask：HSV 双区间红 + 高饱和。返回 (H,W) bool。"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((h < 12) | (h > 168)) & (s > 90) & (v > 60))
    return red


def human_arm_mask_from_hand(img_rgb: np.ndarray, hand_mask: np.ndarray,
                             dilate: int = 15, dark_thr: int = 60) -> np.ndarray:
    """手部 mask → 膨胀 + 并上与手相连的暗袖连通域 − 红块。返回 (H,W) bool。"""
    hand = hand_mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    hand_d = cv2.dilate(hand, k)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    dark = (gray < dark_thr).astype(np.uint8)
    # 只保留与手相连的暗区（连通域）
    num, labels = cv2.connectedComponents(((dark | hand_d) > 0).astype(np.uint8))
    keep = np.zeros_like(dark, bool)
    hand_labels = set(labels[hand_d > 0].tolist()) - {0}
    for lb in hand_labels:
        keep |= (labels == lb)

    arm = keep & ~red_cube_mask(img_rgb)
    return arm.astype(bool)


def build_sam2_predictor(cfg_name: str, ckpt: str, device: str = "cuda"):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor(build_sam2(cfg_name, ckpt, device=device))


def robot_gripper_mask(predictor, img_rgb: np.ndarray, dilate: int = 21) -> np.ndarray:
    """robot 夹爪 mask：在图内最暗点种 SAM2，取面积合理的 mask，膨胀。返回 (H,W) bool。"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    yx = np.unravel_index(np.argmin(gray), gray.shape)
    pt = np.array([[yx[1], yx[0]]], dtype=np.float32)   # (x,y)
    predictor.set_image(img_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=pt, point_labels=np.array([1]), multimask_output=True)
    area = img_rgb.shape[0] * img_rgb.shape[1]
    cand = [(s, m) for s, m in zip(scores, masks)
            if 0.01 * area < m.sum() < 0.5 * area]
    if not cand:
        best = masks[int(np.argmax(scores))]
    else:
        best = max(cand, key=lambda sm: sm[0])[1]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    return cv2.dilate(best.astype(np.uint8), k).astype(bool)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/cross_embod_ve/test_agent_mask.py -v`
Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
git add interactive_world_sim/cross_embod_ve/data/agent_mask.py tests/cross_embod_ve/test_agent_mask.py
git commit -m "feat(ceve): agent masks (human arm via hand+sleeve-cube, robot gripper via SAM2)"
```

---

## Task 5: Object trajectory O_s（SAM2 追踪 + VGGT 深度 lift）

**Files:**
- Create: `interactive_world_sim/cross_embod_ve/data/object_track.py`
- Test: `tests/cross_embod_ve/smoke_build_one_clip.py`（追加 O_s smoke）

> O_s = K_OBJ 个共享物体（蓝盘、红块）的逐帧相机系 3D 位置。做法：每帧用颜色先验得到物体像素 centroid（蓝盘=蓝 HSV，红块=红 HSV），用 SAM2 在 centroid 种点精修 mask，取 mask centroid 的像素 (u,v)；深度从 VGGT depth.npy 取该像素 → `lift_pixels_to_3d` 得 3D。未检出帧填 NaN。**position 为主，不做 6DoF/ICP**。

- [ ] **Step 1: 实现 object_track.py**

```python
import cv2
import numpy as np
from .geometry import lift_pixels_to_3d


def _hsv_centroid(img_rgb, lo, hi, min_px=40):
    """颜色区间 mask 的最大连通域 centroid (u,v)；不足 min_px 返回 None。"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, np.array(lo), np.array(hi))
    num, labels, stats, cents = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
    if num <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < min_px:
        return None
    return cents[idx].astype(np.float32)   # (u,v)


# 蓝盘 / 红块的两段红色 HSV 先验（OpenCV H∈[0,180]）
_BLUE = ([95, 80, 40], [130, 255, 255])
_RED1 = ([0, 90, 60], [12, 255, 255])
_RED2 = ([168, 90, 60], [180, 255, 255])


def object_pixels_one_frame(img_rgb):
    """返回 (K_OBJ,2) 像素 centroid，未检出为 NaN。顺序: [blue_plate, red_cube]。"""
    blue = _hsv_centroid(img_rgb, *_BLUE)
    r1 = _hsv_centroid(img_rgb, *_RED1)
    r2 = _hsv_centroid(img_rgb, *_RED2)
    red = r1 if r1 is not None else r2
    out = np.full((2, 2), np.nan, np.float32)
    if blue is not None:
        out[0] = blue
    if red is not None:
        out[1] = red
    return out


def object_traj_3d(frames_rgb, depth_clip, K):
    """frames_rgb:(T,H,W,3) full-res；depth_clip:(T,H,W) 米；K:(3,3)。
    返回 (T,K_OBJ,3) 相机系 3D（米），未检出 NaN。"""
    T = frames_rgb.shape[0]
    out = np.full((T, 2, 3), np.nan, np.float32)
    for t in range(T):
        uv = object_pixels_one_frame(frames_rgb[t])   # (2,2)
        for k in range(2):
            if np.isnan(uv[k]).any():
                continue
            u, v = int(round(uv[k, 0])), int(round(uv[k, 1]))
            if 0 <= v < depth_clip.shape[1] and 0 <= u < depth_clip.shape[2]:
                d = float(depth_clip[t, v, u])
                out[t, k] = lift_pixels_to_3d(K, uv[k:k+1], np.array([d], np.float32))[0]
    return out
```

> 注：`depth_clip` 为 full-res（480×640）深度，与 `frames_rgb` 像素对齐；O_s 在 full-res 像素上算，再 lift。crop/resize 只作用于存盘的 `frames`，不影响 O_s 的 3D 值。robot 暂无 VGGT 深度 → 见 Task 7 的降级策略（robot 用 EEF 深度近似或跳过 O_s，置 NaN）。

- [ ] **Step 2: Smoke — human 一段 clip 抽 O_s，dump QC overlay**

在 `tests/cross_embod_ve/smoke_build_one_clip.py` 的 `main()` 末尾追加（human depth 路径见下，若无则跳过 3D、只验证像素 centroid）：

```python
    # --- O_s smoke (human) ---
    from interactive_world_sim.cross_embod_ve.data.object_track import object_pixels_one_frame
    uv = object_pixels_one_frame(hf[0])
    print("object pixels (blue, red):", uv)
    vis = hf[0].copy()
    for k, color in enumerate([(0, 255, 0), (255, 255, 0)]):
        if not np.isnan(uv[k]).any():
            cv2.circle(vis, (int(uv[k, 0]), int(uv[k, 1])), 8, color, 2)
    import os; os.makedirs("outputs/ceve_qc", exist_ok=True)
    cv2.imwrite("outputs/ceve_qc/obj_centroids.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print("wrote outputs/ceve_qc/obj_centroids.png")
```

Run: `python tests/cross_embod_ve/smoke_build_one_clip.py`
Expected: 打印 object pixels（至少红块或蓝盘非 NaN），生成 `outputs/ceve_qc/obj_centroids.png`。**人工看图**确认圈圈落在物体上。

- [ ] **Step 3: Commit**

```bash
git add interactive_world_sim/cross_embod_ve/data/object_track.py tests/cross_embod_ve/smoke_build_one_clip.py
git commit -m "feat(ceve): object trajectory O_s via HSV centroid + depth lift, QC overlay"
```

---

## Task 6: Embodiment image C + 单 clip 记录组装/写盘

**Files:**
- Create: `interactive_world_sim/cross_embod_ve/data/embodiment_image.py`
- Create: `interactive_world_sim/cross_embod_ve/data/clip_record.py`
- Test: `tests/cross_embod_ve/test_clip_record.py`

- [ ] **Step 1: 写失败测试（C crop 边界 + 记录组装形状）**

`tests/cross_embod_ve/test_clip_record.py`:

```python
import numpy as np
from interactive_world_sim.cross_embod_ve.data.embodiment_image import crop_around
from interactive_world_sim.cross_embod_ve.data.clip_record import assemble_clip


def test_crop_around_clamps_to_bounds():
    img = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
    c = crop_around(img, center_uv=(5, 5), size=128)   # 角落，需 clamp
    assert c.shape == (128, 128, 3)
    c2 = crop_around(img, center_uv=(320, 240), size=128)
    assert c2.shape == (128, 128, 3)


def test_assemble_clip_shapes():
    T, RES = 25, 256
    rec = assemble_clip(
        frames=np.zeros((T, RES, RES, 3), np.uint8),
        agent_mask=np.zeros((T, RES, RES), bool),
        eef=np.zeros((T, 8), np.float32),
        obj_traj=np.full((T, 2, 3), np.nan, np.float32),
        embodiment_image=np.zeros((128, 128, 3), np.uint8),
        domain="human",
        meta={"source_id": "chunk_1000", "frame_start": 0, "fps": 30, "crop": [195, 195, 256, 256]},
    )
    assert rec["frames"].shape == (T, RES, RES, 3)
    assert rec["eef"].shape == (T, 8)
    assert rec["obj_traj"].shape == (T, 2, 3)
    assert str(rec["domain"]) == "human"
    assert rec["meta"].item()["source_id"] == "chunk_1000"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/cross_embod_ve/test_clip_record.py -v`
Expected: FAIL，ImportError。

- [ ] **Step 3: 实现 embodiment_image.py**

```python
import numpy as np


def crop_around(img_rgb: np.ndarray, center_uv, size: int) -> np.ndarray:
    """围绕像素 center_uv=(u,v) crop size×size，越界则 clamp 到图内。返回 (size,size,3)。"""
    H, W = img_rgb.shape[:2]
    half = size // 2
    u, v = int(round(center_uv[0])), int(round(center_uv[1]))
    x0 = int(np.clip(u - half, 0, W - size))
    y0 = int(np.clip(v - half, 0, H - size))
    return img_rgb[y0:y0 + size, x0:x0 + size].copy()
```

- [ ] **Step 4: 实现 clip_record.py**

```python
import json
import os
import numpy as np


def assemble_clip(frames, agent_mask, eef, obj_traj, embodiment_image, domain, meta) -> dict:
    """打包成 npz-ready dict，统一 dtype。"""
    return {
        "frames": np.asarray(frames, np.uint8),
        "agent_mask": np.asarray(agent_mask, bool),
        "eef": np.asarray(eef, np.float32),
        "obj_traj": np.asarray(obj_traj, np.float32),
        "embodiment_image": np.asarray(embodiment_image, np.uint8),
        "domain": np.asarray(domain),
        "meta": np.asarray(meta, dtype=object),
    }


def write_clip(rec: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **rec)


def append_index(index_path: str, entry: dict):
    idx = []
    if os.path.exists(index_path):
        with open(index_path) as f:
            idx = json.load(f)
    idx.append(entry)
    with open(index_path, "w") as f:
        json.dump(idx, f, indent=0)
```

- [ ] **Step 5: 跑测试确认通过**

Run: `pytest tests/cross_embod_ve/test_clip_record.py -v`
Expected: 2 passed。

- [ ] **Step 6: Commit**

```bash
git add interactive_world_sim/cross_embod_ve/data/embodiment_image.py interactive_world_sim/cross_embod_ve/data/clip_record.py tests/cross_embod_ve/test_clip_record.py
git commit -m "feat(ceve): embodiment-image crop + clip record assemble/write/index"
```

---

## Task 7: 端到端 build 脚本 + EEF 加载 + 小规模产出与 QC

**Files:**
- Create: `build_ceve_dataset.py` (repo root)
- Modify: `tests/cross_embod_ve/smoke_build_one_clip.py`（接通完整单 clip 组装）

> EEF (M_s) 加载：
> - **human**：`smoothing_processor/smoothed_actions_right_single_arm.npz` 的 `ee_pts`(T,3)+`ee_oris`(T,3,3)+`ee_widths`(T,)，相机系。rot 矩阵→quat 用 `scipy.spatial.transform.Rotation`。已在相机系，**不做变换**。
> - **robot**：parquet `action_right_ee_position`+`action_right_ee_quat_xyzw`+`action_right_gripper`，world 系 → `transform_points(robot_world_to_cam(), pos)` 转相机系（position-only；quat 暂保留 world 朝向，下游默认只用 position）。
> - EEF→像素（给 C 定位）：`project_points_to_pixels(K, eef_pos_cam)`。

- [ ] **Step 1: 实现 build_ceve_dataset.py**

```python
"""Build cross-embodiment video-edit dataset (Plan 1).
Usage:
  python build_ceve_dataset.py --domain human --limit 2     # 小规模试跑
  python build_ceve_dataset.py --domain robot --limit 2
  python build_ceve_dataset.py --domain all                 # 全量
"""
import argparse
import os
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from interactive_world_sim.cross_embod_ve.data.video_io import (
    read_robot_frames, read_human_frames, crop_resize)
from interactive_world_sim.cross_embod_ve.data.geometry import (
    load_intrinsics, transform_points, project_points_to_pixels)
from interactive_world_sim.cross_embod_ve.data.agent_mask import (
    human_arm_mask_from_hand, build_sam2_predictor, robot_gripper_mask)
from interactive_world_sim.cross_embod_ve.data.object_track import object_traj_3d
from interactive_world_sim.cross_embod_ve.data.embodiment_image import crop_around
from interactive_world_sim.cross_embod_ve.data.clip_record import (
    assemble_clip, write_clip, append_index)
from interactive_world_sim.algorithms.latent_decompose.ot_align import robot_world_to_cam

CFG = OmegaConf.load("configurations/dataset/cross_embod_ve.yaml")
K = load_intrinsics(CFG.intrinsics)
INDEX = os.path.join(CFG.out_root, "index.json")


def _clip_bounds(n):
    return [(s, s + CFG.t_clip) for s in range(0, n - CFG.t_clip + 1, CFG.stride)]


def build_human_chunk(chunk_id, predictor=None):
    base = f"{CFG.human.chunks_root}/{chunk_id}"
    frames_full = read_human_frames(f"{base}/video_rgb_imgs.mkv")          # (T,480,640,3)
    masks_arm = np.load(f"{base}/segmentation_processor/masks_arm.npy")     # (T,480,640) bool
    act = np.load(f"{base}/smoothing_processor/smoothed_actions_right_single_arm.npz")
    ee_pts, ee_oris, ee_w = act["ee_pts"], act["ee_oris"], act["ee_widths"]
    n = min(len(frames_full), len(masks_arm), len(ee_pts))
    depth_path = f"{base}/depth.npy"
    depth = np.load(depth_path) if os.path.exists(depth_path) else None
    for ci, (a, b) in enumerate(_clip_bounds(n)):
        fr = frames_full[a:b]
        quat = Rotation.from_matrix(ee_oris[a:b]).as_quat()                 # xyzw
        eef = np.concatenate([ee_pts[a:b], quat, ee_w[a:b, None]], axis=1).astype(np.float32)
        amask = np.stack([human_arm_mask_from_hand(fr[t], masks_arm[a + t]) for t in range(b - a)])
        obj = (object_traj_3d(fr, depth[a:b], K) if depth is not None
               else np.full((b - a, CFG.k_obj, 3), np.nan, np.float32))
        mid = (b - a) // 2
        uv_mid = project_points_to_pixels(K, eef[mid:mid + 1, :3])[0]
        cimg = crop_around(fr[mid], uv_mid, CFG.emb_crop)
        frames = np.stack([crop_resize(f, list(CFG.crop), CFG.res) for f in fr])
        amask_rs = np.stack([crop_resize(m.astype(np.uint8) * 255, list(CFG.crop), CFG.res) > 127
                             for m in amask])
        rec = assemble_clip(frames, amask_rs, eef, obj, cimg, "human",
                            {"source_id": f"chunk_{chunk_id}", "frame_start": int(a),
                             "fps": 30, "crop": list(CFG.crop)})
        out = f"{CFG.out_root}/human/chunk_{chunk_id}/clip_{ci:04d}.npz"
        write_clip(rec, out)
        append_index(INDEX, {"path": out, "domain": "human",
                             "source_id": f"chunk_{chunk_id}", "n_frames": int(b - a)})
        print("wrote", out)


def build_robot_episode(ep, predictor):
    base = f"{CFG.robot.dataset_root}/play_robot_{ep}_eef"
    mp4 = f"{base}/videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
    pq = f"{base}/data/chunk-000/episode_000000.parquet"
    frames_full = read_robot_frames(mp4)
    df = pd.read_parquet(pq)
    pos = np.stack(df["action_right_ee_position"].to_numpy())               # (T,3) world
    quat = np.stack(df["action_right_ee_quat_xyzw"].to_numpy())             # (T,4)
    grip = np.stack(df["action_right_gripper"].to_numpy()).reshape(-1, 1)
    T_cw = robot_world_to_cam(CFG.robot.urdf)
    pos_cam = transform_points(T_cw, pos)
    n = min(len(frames_full), len(pos_cam))
    for ci, (a, b) in enumerate(_clip_bounds(n)):
        fr = frames_full[a:b]
        eef = np.concatenate([pos_cam[a:b], quat[a:b], grip[a:b]], axis=1).astype(np.float32)
        amask = np.stack([robot_gripper_mask(predictor, fr[t]) for t in range(b - a)])
        obj = np.full((b - a, CFG.k_obj, 3), np.nan, np.float32)            # robot 无 VGGT 深度 → NaN
        mid = (b - a) // 2
        uv_mid = project_points_to_pixels(K, eef[mid:mid + 1, :3])[0]
        cimg = crop_around(fr[mid], uv_mid, CFG.emb_crop)
        frames = np.stack([crop_resize(f, list(CFG.crop), CFG.res) for f in fr])
        amask_rs = np.stack([crop_resize(m.astype(np.uint8) * 255, list(CFG.crop), CFG.res) > 127
                             for m in amask])
        rec = assemble_clip(frames, amask_rs, eef, obj, cimg, "robot",
                            {"source_id": f"robot_{ep}", "frame_start": int(a),
                             "fps": 30, "crop": list(CFG.crop)})
        out = f"{CFG.out_root}/robot/robot_{ep}/clip_{ci:04d}.npz"
        write_clip(rec, out)
        append_index(INDEX, {"path": out, "domain": "robot",
                             "source_id": f"robot_{ep}", "n_frames": int(b - a)})
        print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["human", "robot", "all"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个 source")
    args = ap.parse_args()
    os.makedirs(CFG.out_root, exist_ok=True)
    predictor = build_sam2_predictor(CFG.sam2_cfg, CFG.sam2_ckpt)
    if args.domain in ("human", "all"):
        for cid in list(CFG.human.chunk_ids)[: args.limit]:
            build_human_chunk(cid, predictor)
    if args.domain in ("robot", "all"):
        for ep in list(CFG.robot.episodes)[: args.limit]:
            build_robot_episode(ep, predictor)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 小规模试跑 human（1 chunk）**

Run:
```bash
cd /scr2/yusenluo/interactive_world_sim && conda activate iws
python build_ceve_dataset.py --domain human --limit 1
```
Expected: 打印若干 `wrote .../human/chunk_1000/clip_0000.npz`；`data/ceve/index.json` 出现 human 条目。
如 `smoothed_actions_*.npz` 的 key 名与假设不符 → 先 `python -c "import numpy as np; print(list(np.load('<path>').keys()))"` 核实 key，再据实改 `build_human_chunk`。

- [ ] **Step 3: 小规模试跑 robot（1 episode）**

Run: `python build_ceve_dataset.py --domain robot --limit 1`
Expected: 打印 `wrote .../robot/robot_1/clip_0000.npz`。
如 parquet 列是标量而非 list-of-array，`np.stack(df[col].to_numpy())` 会报错 → 用 `np.asarray(df[col].tolist())` 兜底。

- [ ] **Step 4: QC — 一个 clip 的 frames / mask / C 可视化**

新建临时验证（可并入 smoke 脚本）：读回一个 npz，dump 第 0 帧的 `frames`、`agent_mask`(红覆盖)、`embodiment_image` 三联图到 `outputs/ceve_qc/clip_qc.png`。

```bash
python - <<'PY'
import numpy as np, cv2, glob, os
p = sorted(glob.glob("data/ceve/human/*/clip_0000.npz"))[0]
d = np.load(p, allow_pickle=True)
fr = d["frames"][0]; m = d["agent_mask"][0]; c = d["embodiment_image"]
ov = fr.copy(); ov[m] = (0.5*ov[m] + 0.5*np.array([255,0,0])).astype(np.uint8)
cimg = cv2.resize(c, (fr.shape[1], fr.shape[0]))
row = np.concatenate([fr, ov, cimg], axis=1)
os.makedirs("outputs/ceve_qc", exist_ok=True)
cv2.imwrite("outputs/ceve_qc/clip_qc.png", cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
print("eef[0]=", d["eef"][0], "obj_traj[0]=", d["obj_traj"][0], "domain=", d["domain"])
print("wrote outputs/ceve_qc/clip_qc.png")
PY
```
Expected: 生成三联图；**人工看**：raw 正常、mask 红覆盖恰好盖住手臂/夹爪、C 是末端特写。eef[0] 三个 position 数值量级合理（米级）。

- [ ] **Step 5: Commit**

```bash
git add build_ceve_dataset.py tests/cross_embod_ve/smoke_build_one_clip.py
git commit -m "feat(ceve): end-to-end build script (human+robot clips) + QC, small-scale verified"
```

- [ ] **Step 6: 全量产出（确认 QC 通过后）**

Run:
```bash
python build_ceve_dataset.py --domain all 2>&1 | tee outputs/ceve_build.log
python -c "import json; idx=json.load(open('data/ceve/index.json')); from collections import Counter; print(Counter(e['domain'] for e in idx))"
```
Expected: 打印 `Counter({'robot': <多>, 'human': <若干>})`，clip 总数与 episode 时长一致。
> robot 12 episode + human 13 chunk，全量 SAM2 较慢 → 可挂 `sbatch`（参考 `sbatch/phase0_stage1_mixed_emb_film.sbatch` 改 GPU=1、调用本脚本）。本步可后置，先用小规模数据推进 Plan 2。

---

## Self-Review（对照 spec 检查）

**Spec coverage（spec 第「数据准备」节）：**
- M_s（EEF 统一相机系）→ Task 2（geometry）+ Task 7（human 不变换 / robot `robot_world_to_cam`）。✅
- O_s（SAM2+VGGT 深度 lift，position 为主）→ Task 5 + Task 7。⚠️ **已知缺口**：robot 无 VGGT 深度，O_s 置 NaN（Task 7 Step 1 注明）。这与 spec「position 为主」一致，但 robot 侧 O_s 缺失会削弱 z_task 的物体信息——记入 Plan 2 风险，必要时补 robot 单目深度。
- C（末端静态图）→ Task 6 + Task 7（EEF 投影定位 crop）。✅
- masked video（复用现有 mask）→ Task 4（human 臂 / robot 夹爪）。✅ 存的是 `agent_mask`，masked 输入交给 Plan 2 的 VACE dataloader（更灵活）。
- crop(195,195,256,256)+resize、T_CLIP、相机系统一 → 配置 + Task 7。✅

**Placeholder scan：** 无 TBD/TODO；所有 step 含真实代码或具体命令 + 期望输出。两处「降级兜底」（npz key 名、parquet 列类型）给了具体排查命令，非占位。

**Type consistency：** `assemble_clip(frames, agent_mask, eef, obj_traj, embodiment_image, domain, meta)` 签名在 Task 6 定义、Task 7 调用一致；`crop_resize(frame, crop, res)`、`lift_pixels_to_3d(K, uv, depth)`、`robot_world_to_cam(urdf)`、`build_sam2_predictor`/`robot_gripper_mask` 全程一致。输出 npz 字段与「输出格式」表一致。

**已知偏差（非阻塞）：** human EEF 来自 `smoothing_processor` 的 npz（相机系、已平滑），而非 `play_human_eef` parquet——因为前者与 `masks_arm`/`video_rgb_imgs.mkv` 同源同帧索引，避免跨数据集对齐。Plan 2 训练读 `data/ceve/index.json`，与 play_human_eef 解耦。

---

## Roadmap：后续 3 个计划（待 Plan 1 落地后分别用 writing-plans 展开）

> 下面只列**范围 + 消费的接口 + 主风险**，不是 bite-sized 任务。

### Plan 2/4：VACE 集成 + v0 自重建训练
- **消费**：`data/ceve/index.json` 的 clip（frames/agent_mask/eef/obj_traj/embodiment_image/domain）。
- **范围**：下载/加载 **Wan2.1-VACE-1.3B**（HF，`.hf_cache`）并冻结；挂 **LoRA(rank16–64) on DiT**；`E_task`（M_s+O_s→token）、`E_emb`（C 过 frozen VACE-CLIP→token）注入 VACE VCU 条件流；masked video = frames⊙(1−agent_mask) 作 VACE inpainting 条件。损失只 `L_FM`（flow-matching）。bf16 + grad checkpointing，batch1/卡×2，~25 帧、256 分辨率。产物：能自重建 robot 视频的 v0。
- **主风险**：A6000 显存/吞吐；VACE 是否「认识」Trossen 夹爪（靠 robot 自重建教）；LoRA 注入点与 VACE VCU API 对接（需读 VACE 源码定 cross-attn 注入位置）。

### Plan 3/4：disentangle loss（v1）+ cross-generation
- **消费**：Plan 2 的 v0 训练栈。
- **范围**：加 **CLUB**（q_φ(z_emb|z_task)，每10步更新，minimize MI）+ **emb-contrast InfoNCE**（同 agent 正样本）；丢 task-contrast。disentangle 诊断复用 `diagnostics/linear_probe.py`。然后 **cross-generation**：human clip 的 z_task + robot 的 z_emb + human masked video → 生成 robot 视频，批量产出 `data/ceve_gen_robot/`。
- **主风险**：disentangle 与重建打架（沿用 v4 dual_head/CLUB 经验，手动双优化器）；生成形态/伪影质量（FVD/LPIPS + 人工 QC 把关，不达标不进下游）。

### Plan 4/4：下游 co-train latent WM
- **消费**：`data/ceve_gen_robot/`（生成 robot 视频）+ 真实 robot。
- **范围**：仿 `MixedPlayEEFDataset` 加第三源（gen_robot），或直接把生成视频转成 `play_*_eef` 兼容格式喂现有 `main.py` 训练栈（`algorithm=latent_world_model dataset=...`）。对照 **robot-only vs robot+gen_robot**，指标 rollout PSNR / 多步预测误差。回答 §6 方向1。
- **主风险**：生成伪影成为下游噪声源（这正是要测量的）；新数据源接入 datamodule 的 domain_label/sample_ratio 改动（explore 已定位扩展点）。
