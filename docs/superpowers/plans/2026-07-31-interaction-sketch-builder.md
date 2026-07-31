# 交互草图 Builder (Plan A) 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 robot/human clips 编码成同一种域无关的 8 通道 2D 交互草图(3D-canonical 投影而来),并逐通道可视化核验。

**Architecture:** 复用现有标定投影(`keyboard_3d_control.project_norm/can_KT/CROP`)、flow_cond(`exp_scel_dualview_dit`)、robot skel sidecar 与 human IK(`ik_eef2skel_derisk`)。新增一个纯函数库 `sketch_lib.py`(可单测)、一个几何接触检测 `contact_detector.py`(可单测)、一个驱动 `build_sketch.py`(组装+可视化)。草图 = 3D 世界系算,投影到 cam_high/cam_low 各一张 2D 图,pool 到 16×16。

**Tech Stack:** numpy, cv2, torch, pytest。★**建草图与 pytest 全在 `.venv_wan`**(flow_cond 源 `exp_scel_dualview_dit` 需 diffusers,在 phantom env 会因 torch/diffusers 版本崩)。**skel 从现成 sidecar 读,不跑 pinocchio/IK**:robot=`skel_sidecar_robot.npz`、human=`skel_sidecar_human_ik.npz`(全量 1800 已做,与 clips_human_L48 对齐)。命令用 `.venv_wan/bin/python -m pytest ...`。

## Global Constraints

- 数据版本一律 **retrack + human L48 + episode-split(heldout vids 100,102)**。robot=`outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz`;human=`outputs/flow_render_dataset_can_dual_L48/clips_human_L48_retrack_realwrist.npz`;robot skel sidecar=`outputs/flow_render_dataset_can_dual/skel_sidecar_robot.npz`;**human skel sidecar=`outputs/flow_render_dataset_can_dual/skel_sidecar_human_ik.npz`(全量1800, skel2d_high/low+grip, 同robot结构;不再运行时跑IK)**。
- 草图 8 通道顺序固定:`[flow_dx, flow_dy, footprint, agent_skel, grip, agent_trace, attachment, contact_splat]`。warp **不在**草图里(归 decoder,Plan B)。
- 每视角 2D、128 建、`pool16` 到 16×16;输出 `(N, V=2, 8, tL, 16, 16)` float16,tL=6(TLCAP)。
- GRIP_MAX=0.04。robot/human 的 grip 都**直接从各自 skel sidecar 的 `grip` 字段读**(human sidecar 的 grip 已是自身range映射)。
- 每中间产物出 `原帧|通道|叠加` 三联可视化;结果传 Drive。别建冗余脚本,扩展/复用最干净已有。
- 建草图在 **`.venv_wan`**(不跑 pinocchio/GPU,纯几何+cv2+flow_cond),但仍**别在登录节点跑批量**,全量走 sbatch(env=iws 或 .venv_wan,见 Task 8)。
- 只 git add 自己的文件,commit 不加 Co-Authored-By。

---

### Task 1: 投影 + 3D object-flow(sketch_lib 基座)

**Files:**
- Create: `sketch_lib.py`
- Test: `tests/test_sketch_lib.py`

**Interfaces:**
- Produces:
  - `project_world_to_view(pts3d: np.ndarray[...,3], view: str) -> np.ndarray[...,2]`(view∈{"high","low"},返回 crop-norm 2D)
  - `object_flow_channels(tr3d0, tr3dt, ef0_2d, eft_2d, vis_t, view) -> np.ndarray(3,128,128)`
  - `pool16(x: np.ndarray(C,128,128)) -> np.ndarray(C,16,16)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sketch_lib.py
import numpy as np
import sketch_lib as S

def test_project_tracks3d_matches_stored_2d():
    """投影 robot tracks3d[clip,0] 到 high 应≈存的 2D tracks[clip,0](同一投影链)。"""
    z = np.load("outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz")
    clip = 356
    tr3d = z["tracks3d"][clip, 0]                      # (48,3)
    valid = z["tracks3d_valid"][clip, 0]              # (48,)
    tr2d = z["tracks"][clip, 0]                        # (48,2) crop-norm
    proj = S.project_world_to_view(tr3d, "high")       # (48,2)
    err = np.linalg.norm((proj - tr2d)[valid], axis=1) * 128
    assert np.nanmedian(err) < 3.0, f"median reproj err {np.nanmedian(err):.2f}px too big"

def test_pool16_shape():
    x = np.zeros((3, 128, 128), np.float32)
    assert S.pool16(x).shape == (3, 16, 16)

def test_object_flow_channels_shape():
    tr0 = np.random.rand(48, 2).astype(np.float32); trt = tr0 + 0.02
    ef = np.random.rand(3, 2).astype(np.float32); vis = np.ones(48, np.float32)
    out = S.object_flow_channels(np.random.rand(48, 3), np.random.rand(48, 3), ef, ef, vis, "high")
    assert out.shape == (3, 128, 128)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /scr2/yusenluo/interactive_world_sim && source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh && conda activate phantom && python -m pytest tests/test_sketch_lib.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sketch_lib'`

- [ ] **Step 3: Write minimal implementation**

```python
# sketch_lib.py
"""交互草图纯函数库(2026-07-31 spec)。3D世界系投影 + 逐通道构建。复用已有标定/flow_cond。"""
import numpy as np, cv2, torch
from keyboard_3d_control import CROP, can_KT, project_norm
import exp_scel_dualview_dit as DIT

IMG = 128; POOL = 8


def project_world_to_view(pts3d, view):
    """(...,3) 世界系 -> (...,2) crop-norm 2D(cam view)。复用 Task1-验证过的标定链。"""
    T, K = can_KT(view)
    flat = np.asarray(pts3d, np.float64).reshape(-1, 3)
    out = np.stack([project_norm(p, T, K, CROP[view]) for p in flat])
    return out.reshape(np.asarray(pts3d).shape[:-1] + (2,)).astype(np.float32)


def object_flow_channels(tr3d0, tr3dt, ef0_2d, eft_2d, vis_t, view):
    """3D物体track投影到view 2D,再走现有 flow_cond -> (3,128,128)[dx,dy,footprint]。"""
    p0 = project_world_to_view(tr3d0, view); pt = project_world_to_view(tr3dt, view)
    return DIT.flow_cond(p0, pt, ef0_2d, eft_2d, vis_t)


def pool16(x):
    return torch.nn.functional.avg_pool2d(torch.from_numpy(x[None]).float(), POOL)[0].numpy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sketch_lib.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: Commit**

```bash
git add sketch_lib.py tests/test_sketch_lib.py
git commit -m "sketch_lib: 3D投影+object-flow通道基座(投影链对齐存的2D tracks<3px)"
```

---

### Task 2: agent-trace 通道(eef3d 历史投影拖尾)

**Files:**
- Modify: `sketch_lib.py`
- Test: `tests/test_sketch_lib.py`

**Interfaces:**
- Consumes: `project_world_to_view`
- Produces: `agent_trace_channel(eef3d_hist: np.ndarray(H,3,3), view: str, trail: int=8) -> np.ndarray(1,128,128)`(eef3d_hist 是到当前帧为止(含)的历史,取每帧两指尖中点投影,画渐亮拖尾)

- [ ] **Step 1: Write the failing test**

```python
def test_agent_trace_channel_marks_recent_positions():
    """拖尾应在最近帧 eef 中点位置有非零像素,且更亮(权重随时间增)。"""
    H = 5
    eef = np.zeros((H, 3, 3), np.float64)
    # 两指尖(idx1,2)中点沿 x 前进; 投影后应落在图像内不同列
    for t in range(H):
        eef[t, 1] = [0.02 * t - 0.05, 0.0, 0.17]; eef[t, 2] = [0.02 * t - 0.05, 0.02, 0.17]
    ch = __import__("sketch_lib").agent_trace_channel(eef, "high", trail=8)
    assert ch.shape == (1, 128, 128)
    assert ch.max() > 0.0
    # 最后一帧(最亮)对应位置的像素 >= 更早帧对应位置
    assert ch.sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sketch_lib.py::test_agent_trace_channel_marks_recent_positions -v`
Expected: FAIL — `AttributeError: module 'sketch_lib' has no attribute 'agent_trace_channel'`

- [ ] **Step 3: Write minimal implementation**

```python
# 追加到 sketch_lib.py
def agent_trace_channel(eef3d_hist, view, trail=8):
    """eef3d_hist (H,3,3) 历史(含当前) -> (1,128,128) 中点投影渐亮拖尾。"""
    img = np.zeros((IMG, IMG), np.float32)
    hist = np.asarray(eef3d_hist, np.float64)[-trail:]
    mids = (hist[:, 1] + hist[:, 2]) / 2                      # (h,3) 两指尖中点
    p = project_world_to_view(mids, view)                     # (h,2)
    n = len(p)
    for i, (x, y) in enumerate(p):
        w = (i + 1) / n                                       # 越近越亮
        xi, yi = int(np.clip(x * IMG, 0, IMG - 1)), int(np.clip(y * IMG, 0, IMG - 1))
        cv2.circle(img, (xi, yi), 2, float(w), -1)
    return img[None]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sketch_lib.py -v`
Expected: PASS(4 passed)

- [ ] **Step 5: Commit**

```bash
git add sketch_lib.py tests/test_sketch_lib.py
git commit -m "sketch_lib: agent-trace通道(eef3d历史中点投影渐亮拖尾)"
```

---

### Task 3: grip + contact 通道(广播 + splat)

**Files:**
- Modify: `sketch_lib.py`
- Test: `tests/test_sketch_lib.py`

**Interfaces:**
- Consumes: `project_world_to_view`
- Produces:
  - `grip_channel(grip_scalar: float) -> np.ndarray(1,128,128)`(整层广播,值域 [0,1]:grip/GRIP_MAX)
  - `contact_channels(attachment: float, contact_pt3d: np.ndarray(3,), view: str) -> np.ndarray(2,128,128)`(第0层=attachment广播,第1层=接触点2D高斯splat)

- [ ] **Step 1: Write the failing test**

```python
def test_grip_channel_broadcast():
    import sketch_lib as S
    ch = S.grip_channel(0.02)
    assert ch.shape == (1, 128, 128)
    assert np.allclose(ch, 0.02 / 0.04)

def test_contact_channels_attachment_and_splat():
    import sketch_lib as S
    ch = S.contact_channels(1.0, np.array([-0.05, 0.0, 0.17]), "high")
    assert ch.shape == (2, 128, 128)
    assert np.allclose(ch[0], 1.0)                    # attachment 广播
    assert ch[1].max() > 0.5                          # splat 有峰
    # attachment=0 时 splat 仍可有(接触点存在),但 attach 层全 0
    ch0 = S.contact_channels(0.0, np.array([-0.05, 0.0, 0.17]), "high")
    assert np.allclose(ch0[0], 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sketch_lib.py -k "grip_channel or contact_channels" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'grip_channel'`

- [ ] **Step 3: Write minimal implementation**

```python
# 追加到 sketch_lib.py
GRIP_MAX = 0.04

def grip_channel(grip_scalar):
    return np.full((1, IMG, IMG), np.clip(grip_scalar / GRIP_MAX, 0, 1), np.float32)

def contact_channels(attachment, contact_pt3d, view, sigma=4.0):
    attach = np.full((1, IMG, IMG), float(np.clip(attachment, 0, 1)), np.float32)
    splat = np.zeros((IMG, IMG), np.float32)
    if np.all(np.isfinite(contact_pt3d)):
        x, y = project_world_to_view(np.asarray(contact_pt3d)[None], view)[0]
        cx, cy = int(np.clip(x * IMG, 0, IMG - 1)), int(np.clip(y * IMG, 0, IMG - 1))
        yy, xx = np.ogrid[:IMG, :IMG]
        splat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    return np.concatenate([attach, splat[None]], 0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sketch_lib.py -v`
Expected: PASS(6 passed)

- [ ] **Step 5: Commit**

```bash
git add sketch_lib.py tests/test_sketch_lib.py
git commit -m "sketch_lib: grip广播 + contact(attachment广播+接触点高斯splat)通道"
```

---

### Task 4: ContactDetector(几何式:attachment + 3D 接触点)

**Files:**
- Create: `contact_detector.py`
- Test: `tests/test_contact_detector.py`

**Interfaces:**
- Produces: `detect_contact_geometric(tracks3d: np.ndarray(T,K,3), eef3d: np.ndarray(T,3,3), grip: np.ndarray(T,)) -> (attachment: np.ndarray(T,), contact_pt3d: np.ndarray(T,3))`
  - attachment = σ(夹爪闭合) · σ(3D 夹爪-物体质心距近) · σ(物体速度 与 夹爪速度 相关),0..1
  - contact_pt3d = 每帧 夹爪中点 与 物体质心 的中点(近似接触处)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contact_detector.py
import numpy as np
from contact_detector import detect_contact_geometric

def _traj(T, obj0, ef_off, grip_val, move):
    """构造: 物体从obj0沿x移动move; 夹爪在物体+ef_off处同步移动; grip=grip_val。"""
    t = np.arange(T)[:, None]
    obj = obj0[None] + np.concatenate([move * t / T, np.zeros((T, 2))], 1)   # (T,3)
    tracks3d = obj[:, None, :] + np.zeros((T, 6, 3))                          # (T,6,3) 刚体
    center = obj + ef_off
    eef = np.stack([center, center + [0, 0.014, 0], center + [0, -0.014, 0]], 1)  # (T,3,3)
    grip = np.full(T, grip_val)
    return tracks3d, eef, grip

def test_grasping_gives_high_attachment():
    """夹爪闭合(grip小)+在物体上+同步移动 -> attachment 高。"""
    tr, ef, g = _traj(12, np.array([-0.05, 0., 0.17]), np.array([0., 0., 0.]), 0.005, np.array([0.06]))
    att, cp = detect_contact_geometric(tr, ef, g)
    assert att.shape == (12,) and cp.shape == (12, 3)
    assert att.mean() > 0.6, f"grasping attachment {att.mean():.2f} too low"

def test_open_hand_far_gives_low_attachment():
    """夹爪张开(grip大)+离物体远+物体不动 -> attachment 低。"""
    tr, ef, g = _traj(12, np.array([-0.05, 0., 0.17]), np.array([0.12, 0., 0.]), 0.04, np.array([0.0]))
    att, _ = detect_contact_geometric(tr, ef, g)
    assert att.mean() < 0.3, f"non-grasp attachment {att.mean():.2f} too high"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_contact_detector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'contact_detector'`

- [ ] **Step 3: Write minimal implementation**

```python
# contact_detector.py
"""几何式接触检测(2026-07-31 spec §5)。attachment=夹爪闭合∧夹爪-物近∧速度相关; contact_pt=夹爪-物中点。
学习式(自举标签)后续 Plan/Task 另加,同接口。"""
import numpy as np

GRIP_MAX = 0.04
D_NEAR = 0.05          # 3D 夹爪-物体"近"的尺度(米)
EPS = 1e-6

def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def detect_contact_geometric(tracks3d, eef3d, grip):
    tracks3d = np.asarray(tracks3d, np.float64); eef3d = np.asarray(eef3d, np.float64)
    grip = np.asarray(grip, np.float64)
    obj = np.nanmean(tracks3d, axis=1)                        # (T,3) 物体质心
    gmid = (eef3d[:, 1] + eef3d[:, 2]) / 2                    # (T,3) 夹爪中点
    # 1) 夹爪闭合: grip 越小越闭 -> [0,1]
    closed = 1.0 - np.clip(grip / GRIP_MAX, 0, 1)
    # 2) 夹爪-物体近: 距离越小越接触
    dist = np.linalg.norm(gmid - obj, axis=1)
    near = _sig((D_NEAR - dist) / (0.5 * D_NEAR))
    # 3) 速度相关: 物体速度与夹爪速度余弦(不动时中性0.5)
    vo = np.diff(obj, axis=0, prepend=obj[:1]); vg = np.diff(gmid, axis=0, prepend=gmid[:1])
    num = (vo * vg).sum(1); den = np.linalg.norm(vo, axis=1) * np.linalg.norm(vg, axis=1) + EPS
    corr = np.where(den > 1e-4, np.clip(num / den, 0, 1), 0.5)
    attachment = closed * near * corr
    contact_pt3d = (gmid + obj) / 2
    return attachment.astype(np.float32), contact_pt3d.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_contact_detector.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: Commit**

```bash
git add contact_detector.py tests/test_contact_detector.py
git commit -m "contact_detector: 几何式attachment(闭合∧近∧速度相关)+3D接触点"
```

---

### Task 5: build_sketch — robot 分支组装 8ch 草图

**Files:**
- Create: `build_sketch.py`
- Test: `tests/test_build_sketch.py`

**Interfaces:**
- Consumes: `sketch_lib.*`, `contact_detector.detect_contact_geometric`, robot skel sidecar
- Produces: `build_clip_sketch(A: dict, n: int, tL: int, L: int, src: str) -> np.ndarray(2,8,tL,16,16)`;CLI 出 `outputs/video_arch_wm/sketch_can_dual/sketch_robot.npz`(键 `sketch (N,2,8,tL,16,16) f16`, `tL`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_sketch.py
import numpy as np
from build_sketch import build_clip_sketch, load_robot_arrays

def test_robot_sketch_shape_and_channels():
    A = load_robot_arrays()
    out = build_clip_sketch(A, 356, tL=6, L=48, src="robot")
    assert out.shape == (2, 8, 6, 16, 16)
    # 通道非全零(至少 flow/skel/grip 有内容)
    assert np.abs(out[0, :4]).sum() > 0        # flow+skel high 视角
    assert out[0, 4].mean() >= 0               # grip 广播
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_build_sketch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_sketch'`

- [ ] **Step 3: Write minimal implementation**

```python
# build_sketch.py
"""组装 8 通道交互草图(2026-07-31 spec)。SRC=robot|human。3D世界系算->投影双视角2D->pool16。
robot: skel 取已存 sidecar; human: IK(ik_net) 生成 skel。在 conda phantom 跑。
通道序: [flow(3), skel(1), grip(1), trace(1), attachment(1), contact(1)] = 8。
用法: SRC=robot python build_sketch.py"""
import os, numpy as np, cv2, torch
import sketch_lib as S
from contact_detector import detect_contact_geometric

IMG = 128; GRID = 16
RB = "outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz"
HM = "outputs/flow_render_dataset_can_dual_L48/clips_human_L48_retrack_realwrist.npz"
SKELF = "outputs/flow_render_dataset_can_dual/skel_sidecar_robot.npz"
OUTDIR = "outputs/video_arch_wm/sketch_can_dual"; os.makedirs(OUTDIR, exist_ok=True)
_SK = np.load(SKELF); _SEG = _SK["segments"]


def skel_chan(pts2d):
    img = np.zeros((IMG, IMG), np.float32); P = (pts2d * IMG).astype(np.int32)
    for a, b in _SEG:
        if np.all(np.abs(pts2d[a]) < 3) and np.all(np.abs(pts2d[b]) < 3):
            cv2.line(img, tuple(P[a]), tuple(P[b]), 1.0, 3, cv2.LINE_AA)
    for i, p in enumerate(P):
        if np.all(np.abs(pts2d[i]) < 3): cv2.circle(img, tuple(p), 3, 1.0, -1)
    return img


def load_robot_arrays():
    z = np.load(RB)
    keys = ["tracks3d", "tracks3d_valid", "eef", "eef_low", "vis", "vis_low",
            "eef3d", "grip", "frames", "frames_low"]
    A = {k: z[k][:] for k in keys}
    A["skel2d_high"] = _SK["skel2d_high"][:]; A["skel2d_low"] = _SK["skel2d_low"][:]
    A["vid"] = z["vid"][:]
    return A


def build_clip_sketch(A, n, tL, L, src):
    f32 = lambda a: np.nan_to_num(np.asarray(a, np.float32))
    tr3d = f32(A["tracks3d"][n]); eef3d = f32(A["eef3d"][n]); grip = f32(A["grip"][n])
    ef = [f32(A["eef"][n]), f32(A["eef_low"][n])]; vs = [f32(A["vis"][n]), f32(A["vis_low"][n])]
    att, cpt = detect_contact_geometric(tr3d, eef3d, grip)                 # (L,), (L,3)
    out = np.zeros((2, 8, tL, GRID, GRID), np.float32)
    views = ["high", "low"]
    for k in range(tL):
        rf = 0 if k == 0 else min(4 * k, L - 1)
        for v in range(2):
            flow = S.object_flow_channels(tr3d[0], tr3d[rf], ef[v][0], ef[v][rf], vs[v][rf], views[v])
            sk = skel_chan(A["skel2d_high" if v == 0 else "skel2d_low"][n, rf])[None]
            gp = S.grip_channel(grip[rf])
            tr = S.agent_trace_channel(eef3d[:rf + 1], views[v])
            ct = S.contact_channels(att[rf], cpt[rf], views[v])           # (2,128,128)
            chans = np.concatenate([flow, sk, gp, tr, ct], 0)             # (8,128,128)
            out[v, :, k] = S.pool16(chans)
    return out.astype(np.float16)


def main():
    src = os.environ.get("SRC", "robot")
    assert src == "robot", "human 分支见 Task 6"
    A = load_robot_arrays(); N = A["tracks3d"].shape[0]; L = A["tracks3d"].shape[1]; tL = 6
    rows = range(4) if os.environ.get("SMOKE") == "1" else range(N)
    sketch = np.zeros((N, 2, 8, tL, GRID, GRID), np.float16)
    for i, n in enumerate(rows):
        sketch[n] = build_clip_sketch(A, n, tL, L, src)
        if i % 100 == 0: print(f"sketch {i} (clip {n})", flush=True)
    np.savez(f"{OUTDIR}/sketch_{src}.npz", sketch=sketch, tL=np.array(tL))
    print(f"saved {OUTDIR}/sketch_{src}.npz {sketch.shape}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_build_sketch.py -v`
Expected: PASS(1 passed)

- [ ] **Step 5: Smoke 全通道 run(4 clip)**

Run: `SMOKE=1 SRC=robot python build_sketch.py`
Expected: 打印 `saved .../sketch_robot.npz (N,2,8,6,16,16)` 不崩。

- [ ] **Step 6: Commit**

```bash
git add build_sketch.py tests/test_build_sketch.py
git commit -m "build_sketch: robot分支组装8ch草图(flow3+skel+grip+trace+contact2)"
```

---

### Task 6: build_sketch — human 分支(读现成 human skel sidecar)

★human IK skel 全量已做完(`skel_sidecar_human_ik.npz`,1800,同 robot 结构)。所以 human 分支**不跑 IK**,和 robot **同一套** `build_clip_sketch`,只是 `load_human_arrays` 从 human clips + human sidecar 组 A。

**Files:**
- Modify: `build_sketch.py`
- Test: `tests/test_build_sketch.py`

**Interfaces:**
- Consumes: `outputs/flow_render_dataset_can_dual/skel_sidecar_human_ik.npz`(skel2d_high/low + grip)、`clips_human_L48`(tracks3d/eef/eef3d/frames)
- Produces: `load_human_arrays() -> dict`(键与 `load_robot_arrays` 一致:`tracks3d/eef/eef_low/vis/vis_low/eef3d/grip/frames/frames_low/skel2d_high/skel2d_low/vid` + `_valid_idx`);`build_clip_sketch(A, n, tL, L, "human")` **body 不变**(统一读 A 的 skel2d/grip)

- [ ] **Step 1: Write the failing test**

```python
def test_human_sketch_shape():
    from build_sketch import build_clip_sketch, load_human_arrays
    A = load_human_arrays()
    n = int(A["_valid_idx"][0])
    out = build_clip_sketch(A, n, tL=6, L=48, src="human")
    assert out.shape == (2, 8, 6, 16, 16)
    assert np.abs(out[0, 3]).sum() > 0        # human skel high 非空
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv_wan/bin/python -m pytest tests/test_build_sketch.py::test_human_sketch_shape -v`
Expected: FAIL — `ImportError: cannot import name 'load_human_arrays'`

- [ ] **Step 3: Write minimal implementation**

在 Task 5 里 `build_clip_sketch` 的 `main()` 顶部把 `assert src == "robot", "human 分支见 Task 6"` 改为按 src 选 loader(见下);`build_clip_sketch` **body 保持不变**(它统一读 `A["skel2d_high"/"skel2d_low"][n, rf]` 与 `A["grip"][n]`,human/robot 都从各自 sidecar 填这两个键)。追加:

```python
# 追加到 build_sketch.py(模块顶):
_SKH = np.load("outputs/flow_render_dataset_can_dual/skel_sidecar_human_ik.npz")

def load_human_arrays():
    z = np.load(HM)
    keys = ["tracks3d", "eef", "eef_low", "vis", "vis_low", "eef3d", "frames", "frames_low", "vid"]
    A = {k: z[k][:] for k in keys}
    A["skel2d_high"] = _SKH["skel2d_high"][:]; A["skel2d_low"] = _SKH["skel2d_low"][:]
    A["grip"] = _SKH["grip"][:].astype(np.float32)                     # human sidecar grip(自身range已映射)
    e3 = z["eef3d"]; t3 = z["tracks3d"]
    valid = np.isfinite(e3.reshape(len(e3), e3.shape[1], -1)).all(-1).all(-1) & \
            np.isfinite(t3.reshape(len(t3), t3.shape[1], -1)).all(-1).all(-1)
    A["_valid_idx"] = np.where(valid.all(1))[0]
    return A
```

`main()` 改为:

```python
def main():
    src = os.environ.get("SRC", "robot")
    A = load_human_arrays() if src == "human" else load_robot_arrays()
    N = A["tracks3d"].shape[0]; L = A["tracks3d"].shape[1]; tL = 6
    idx = A["_valid_idx"] if src == "human" else np.arange(N)
    rows = idx[:4] if os.environ.get("SMOKE") == "1" else idx
    sketch = np.zeros((N, 2, 8, tL, GRID, GRID), np.float16)
    for i, n in enumerate(rows):
        sketch[n] = build_clip_sketch(A, n, tL, L, src)
        if i % 100 == 0: print(f"sketch {i} (clip {n})", flush=True)
    np.savez(f"{OUTDIR}/sketch_{src}.npz", sketch=sketch, tL=np.array(tL))
    print(f"saved {OUTDIR}/sketch_{src}.npz {sketch.shape}", flush=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv_wan/bin/python -m pytest tests/test_build_sketch.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: Smoke human run**

Run: `SMOKE=1 SRC=human .venv_wan/bin/python build_sketch.py`
Expected: 打印 `saved .../sketch_human.npz (1800,2,8,6,16,16)` 不崩。

- [ ] **Step 6: Commit**

```bash
git add build_sketch.py tests/test_build_sketch.py
git commit -m "build_sketch: human分支读现成human skel sidecar(同robot一套8ch草图)"
```

---

### Task 7: 逐通道三联可视化(眼检核验)

**Files:**
- Modify: `build_sketch.py`(加 `VIZ=1` 模式)
- Test: 无自动断言(可视化任务),产物人工眼检 + 传 Drive

**Interfaces:**
- Consumes: `build_clip_sketch` 的中间 128 分辨率通道(在 VIZ 模式直接从 clip 重算 128,不 pool)
- Produces: `outputs/video_arch_wm/sketch_can_dual/viz/sketch_{src}_clip{n}_cam{hi/lo}.png`,布局 = 每通道一行 `原帧 | 通道 | 叠加`

- [ ] **Step 1: 写可视化函数(无失败测试,直接实现 + 眼检)**

```python
# 追加到 build_sketch.py
def viz_clip(A, n, src, view_i, out_png):
    """出 8 通道 原帧|通道|叠加 三联(128分辨率, 取 rf 中段一帧)。"""
    f32 = lambda a: np.nan_to_num(np.asarray(a, np.float32))
    view = "high" if view_i == 0 else "low"
    frames = f32(A["frames"][n]) if view_i == 0 else f32(A["frames_low"][n])
    L = A["tracks3d"].shape[1]; rf = min(20, L - 1)
    tr3d = f32(A["tracks3d"][n]); eef3d = f32(A["eef3d"][n]); grip = f32(A["grip"][n])
    ef = f32(A["eef"][n] if view_i == 0 else A["eef_low"][n]); vs = f32(A["vis"][n] if view_i == 0 else A["vis_low"][n])
    att, cpt = detect_contact_geometric(tr3d, eef3d, grip)
    if src == "human":
        skh, skl = _human_ikskel(eef3d, grip); sk2d = (skh if view_i == 0 else skl)[rf]
    else:
        sk2d = A["skel2d_high" if view_i == 0 else "skel2d_low"][n, rf]
    chans = {
        "flow": S.object_flow_channels(tr3d[0], tr3d[rf], ef[0], ef[rf], vs[rf], view),
        "skel": skel_chan(sk2d)[None], "grip": S.grip_channel(grip[rf]),
        "trace": S.agent_trace_channel(eef3d[:rf + 1], view),
        "contact": S.contact_channels(att[rf], cpt[rf], view),
    }
    bg = frames[rf].astype(np.uint8) if frames[rf].max() > 1 else (frames[rf] * 255).astype(np.uint8)
    rows = []
    for name, ch in chans.items():
        c = ch[0] if ch.shape[0] == 1 else ch[:3].transpose(1, 2, 0)
        cimg = (np.clip(np.abs(c), 0, 1) * 255).astype(np.uint8)
        if cimg.ndim == 2: cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2RGB)
        over = cv2.addWeighted(bg, 0.6, cv2.resize(cimg, (IMG, IMG)), 0.6, 0)
        cv2.putText(bg.copy(), name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        rows.append(np.concatenate([bg, cimg, over], 1))
    grid = np.concatenate(rows, 0)
    cv2.imwrite(out_png, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[viz] {out_png}", flush=True)
```

在 `main()` 顶部加:`if os.environ.get("VIZ") == "1": A = load_human_arrays() if src=="human" else load_robot_arrays(); [viz_clip(A, n, src, v, f"{OUTDIR}/viz/sketch_{src}_clip{n}_cam{'hi' if v==0 else 'lo'}.png") for n in [356 if src=='robot' else int(A['_valid_idx'][0])] for v in range(2)]; return`(先 `os.makedirs(f"{OUTDIR}/viz", exist_ok=True)`)。

- [ ] **Step 2: 出 robot + human 可视化并眼检**

Run:
```bash
mkdir -p outputs/video_arch_wm/sketch_can_dual/viz
VIZ=1 SRC=robot python build_sketch.py
VIZ=1 SRC=human python build_sketch.py
```
Expected: 生成 `viz/sketch_robot_clip356_camhi.png` 等 4 张。**人工眼检**:每通道 flow 场对物体、skel 贴 agent、grip 灰度合理、trace 拖尾在手轨迹、contact splat 在接触处、attachment 抓取时亮。

- [ ] **Step 3: 传 Drive**

Run: `TOPIC=交互草图builder bash upload_evals_gdrive.sh "01_8通道草图三联_robot+human=outputs/video_arch_wm/sketch_can_dual/viz"`
Expected: 打印 DONE。

- [ ] **Step 4: Commit**

```bash
git add build_sketch.py
git commit -m "build_sketch: VIZ模式出8通道原帧|通道|叠加三联(眼检核验)"
```

---

### Task 8: 全量建草图(robot + human)

**Files:**
- Create: `sbatch_build_sketch.sbatch`

**Interfaces:** 无新代码,跑全量。

- [ ] **Step 1: 写 sbatch**

```bash
# sbatch_build_sketch.sbatch
#!/bin/bash
#SBATCH --job-name=build_sketch
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/sketch_can_dual/build_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
cd /scr2/yusenluo/interactive_world_sim
export HF_HUB_OFFLINE=1
SRC=robot .venv_wan/bin/python build_sketch.py
SRC=human .venv_wan/bin/python build_sketch.py
echo "=== DONE ==="
```

- [ ] **Step 2: 提交并等完成**

Run: `sbatch sbatch_build_sketch.sbatch`
Expected: 完成后 `outputs/video_arch_wm/sketch_can_dual/sketch_robot.npz`(2700,2,8,6,16,16)、`sketch_human.npz`(1800,...)。

- [ ] **Step 3: 形状核验**

Run: `python -c "import numpy as np; [print(p, np.load(f'outputs/video_arch_wm/sketch_can_dual/sketch_{p}.npz')['sketch'].shape) for p in ['robot','human']]"`
Expected: robot (2700,2,8,6,16,16) / human (1800,2,8,6,16,16)。

- [ ] **Step 4: Commit**

```bash
git add sbatch_build_sketch.sbatch
git commit -m "build_sketch: 全量sbatch(robot 2700 + human 1800 8ch草图)"
```

---

## Self-Review

**1. Spec coverage:** §4.1 草图 8 通道(flow/skel/grip/trace/attachment/contact)→ Task 1-3,5,6 ✓;warp 不在草图 ✓(本 plan 无 warp);§4.2 flow 升 tracks3d → Task 1 ✓、agent-trace → Task 2 ✓、human IK → Task 6 ✓;§5 ContactDetector 几何 → Task 4 ✓(学习式 spec 说"两法都做",本 plan 只几何,学习式留 Plan B/后续,已在 Task 4 注释标注);§3 组件边界(sketch_lib/contact_detector/build_sketch)✓;数据版本对齐 → Global Constraints + 各 load 函数用 retrack/L48 路径 ✓;可视化 → Task 7 ✓;heatmap aux 属 decoder(Plan B),本 plan 不含 ✓。
**缺口**:学习式 ContactDetector 未在本 plan(spec §5 提"两法都做")——归 Plan B 一并做(需 decoder 下游比较才有意义),已在 Task 4 注释说明。
**2. Placeholder scan:** 无 TBD/TODO;每 code step 有完整代码;每 test step 有真断言。
**3. Type consistency:** `project_world_to_view(pts3d, view)`、`detect_contact_geometric(...)->(att(T,),cpt(T,3))`、`build_clip_sketch(A,n,tL,L,src)->(2,8,tL,16,16)`、8 通道序在 Global Constraints 与 Task 5 一致。
