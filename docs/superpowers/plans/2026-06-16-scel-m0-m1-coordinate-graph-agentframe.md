# SCEL M0–M1: 坐标图表示 + agent-frame 相对坐标 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭起 SCEL 的结构化坐标表示（object 节点 [cx,cy,scale] + agent grasp frame + per-domain 速度归一化），并落地主推改造"agent-frame 相对坐标"，验证它降 ② flow 误差且 human-helps 不输现状。

**Architecture:** agent-frame 相对坐标做成套在现有 ② (`amplify_wm.FlowWM_LWC` + `train_lwc_ss`) 外的**可逆坐标变换层**——训练前把 object tracks 逐帧变换到 grasp frame 坐标系，rollout 后用已知 grasp frame（eef 是已知 action）逆变换回世界系。不改 ② 内部、不碰 ③ 渲染器（object 仍 48 点）。表示工具集中在新纯函数模块 `scel_repr.py`，强 TDD。

**Tech Stack:** Python, NumPy, PyTorch, pytest；conda env `iws`。复用 `amplify_wm.py`(LWC ②)、`eval_scheduled_sampling.py`(SS)、`exp_v3_human_helps_flow.py`(下游口径)、数据 `outputs/flow_render_dataset_v3/clips_{robot,human}.npz`。

**范围**：本计划只做 spec 的 **M0（表示工具）+ M1（agent-frame 相对坐标，主推改造）**。后续独立计划：M2 其余改造消融（action 加厚 / agent 多关键点 / transition-delta 一致性正则 / SS 加长）、M3 ③ 渲染加厚（GAN / latent decode / mask-then-color）、备选 plan B（object-occupancy）、图节点重构（object/agent 节点 transformer，牵连 ③）。

**数据约定**（`clips_{robot,human}.npz`）：`tracks(N,L,48,2)` cube 点 [0,1]、`eef(N,L,3,2)` = [base, +finger, −finger] [0,1]、`vis(N,L,48)`、`joint`(robot only, N,L,7)、`vid(N,)`、`frames(N,L,128,128,3)u8`。常量 `K=4, F=20, L=24`（`e2e_flow_wm_render`）。

---

## File Structure

- **Create** `scel_repr.py` — 纯函数表示工具：`object_node`, `grasp_frame`, `to_agent_frame`, `from_agent_frame`, `fit_vel_stats`, `normalize_vel`, `denormalize_vel`。无副作用、无 IO、无 CUDA 依赖（NumPy/torch tensor in→out），便于单元测试。
- **Create** `tests/test_scel_repr.py` — `scel_repr` 的单元测试。
- **Create** `viz_scel_repr.py` — M0 眼检脚本：在真实帧上叠加 object 节点(质心+scale 圈) + grasp frame(位置+朝向箭头+开合)，robot/human 各几帧，存 overlay png + `summary.txt`。
- **Create** `exp_scel_agentframe.py` — M1 实验：agent-frame 变换层包住 `train_lwc_ss`/`rollout_lwc`，训 baseline-LWC vs agentframe-LWC，比 ② ADE + 下游 human-helps，存 `summary.txt` + 曲线。
- **Modify** 无（M1 不改 `amplify_wm.py` 内部——agent-frame 是外层变换）。

---

## Task 1: `scel_repr.object_node` — object 节点 [cx, cy, scale]

**Files:**
- Create: `scel_repr.py`
- Test: `tests/test_scel_repr.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scel_repr.py
import numpy as np
import torch
import scel_repr as S


def test_object_node_centroid_and_scale():
    # 4 点正方形，中心 (0.5,0.5)，半边 0.1 -> 角点距中心 0.1*sqrt2
    pts = np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]], np.float32)
    tracks = pts[None, None]                      # (N=1, L=1, P=4, 2)
    node = S.object_node(tracks)                  # (N, L, 3) = [cx, cy, scale]
    assert node.shape == (1, 1, 3)
    assert np.allclose(node[0, 0, :2], [0.5, 0.5], atol=1e-6)
    # scale = RMS radius = sqrt(mean(||p-c||^2)) = 0.1*sqrt2
    assert np.isclose(node[0, 0, 2], 0.1 * np.sqrt(2), atol=1e-6)


def test_object_node_batch_shapes():
    tracks = np.random.rand(5, 24, 48, 2).astype(np.float32)
    node = S.object_node(tracks)
    assert node.shape == (5, 24, 3)
    assert (node[..., 2] >= 0).all()              # scale 非负
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n iws python -m pytest tests/test_scel_repr.py -k object_node -v`
Expected: FAIL — `AttributeError: module 'scel_repr' has no attribute 'object_node'`

- [ ] **Step 3: Write minimal implementation**

```python
# scel_repr.py
"""SCEL structured representation helpers (pure functions, no IO/side-effects).

object node  = [cx, cy, scale]            质心 + RMS 散布半径（无朝向：2D 近方形 cube 旋转不可靠）
grasp frame  = [gx, gy, cosφ, sinφ, w]    base 位置 + base→指尖中点朝向 + 两指开合
agent-frame  = object 在 grasp frame 坐标系下的可逆变换（rollout 时 grasp 已知 -> 逆变换回世界）
"""
import numpy as np

EPS = 1e-8


def object_node(tracks):
    """tracks (..., P, 2) -> (..., 3) = [cx, cy, scale]. scale = RMS radius about centroid."""
    c = tracks.mean(axis=-2)                                   # (...,2)
    d = tracks - c[..., None, :]                               # (...,P,2)
    scale = np.sqrt((d ** 2).sum(-1).mean(-1) + EPS)           # (...) RMS radius
    return np.concatenate([c, scale[..., None]], axis=-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n iws python -m pytest tests/test_scel_repr.py -k object_node -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scel_repr.py tests/test_scel_repr.py
git commit -m "feat(scel): object_node [cx,cy,scale] pure helper + tests"
```

---

## Task 2: `scel_repr.grasp_frame` — agent grasp frame [gx, gy, cosφ, sinφ, w]

**Files:**
- Modify: `scel_repr.py`
- Test: `tests/test_scel_repr.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_scel_repr.py
def test_grasp_frame_known_geometry():
    # base=(0.5,0.5), +finger=(0.5,0.7), -finger=(0.5,0.3)
    # 指尖中点=(0.5,0.5)? 重合 -> 朝向退化; 用非退化布局:
    # base=(0.5,0.5), fingers 在 base 右侧, 中点=(0.7,0.5) -> 朝向 +x; 开合=|f+ - f-|
    eef = np.array([[0.5, 0.5], [0.7, 0.6], [0.7, 0.4]], np.float32)
    eef = eef[None, None]                              # (1,1,3,2)
    gf = S.grasp_frame(eef)                            # (1,1,5)
    assert gf.shape == (1, 1, 5)
    assert np.allclose(gf[0, 0, :2], [0.5, 0.5], atol=1e-6)   # gx,gy = base
    # finger_mid=(0.7,0.5), base->mid = (+0.2,0) -> cosφ=1, sinφ=0
    assert np.allclose(gf[0, 0, 2:4], [1.0, 0.0], atol=1e-6)
    # width = ||(0.7,0.6)-(0.7,0.4)|| = 0.2
    assert np.isclose(gf[0, 0, 4], 0.2, atol=1e-6)


def test_grasp_frame_orientation_unit_norm():
    eef = np.random.rand(3, 24, 3, 2).astype(np.float32)
    gf = S.grasp_frame(eef)
    n = np.sqrt(gf[..., 2] ** 2 + gf[..., 3] ** 2)
    assert np.allclose(n, 1.0, atol=1e-5)             # (cosφ,sinφ) 单位长
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n iws python -m pytest tests/test_scel_repr.py -k grasp_frame -v`
Expected: FAIL — `AttributeError: ... 'grasp_frame'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to scel_repr.py
def grasp_frame(eef):
    """eef (...,3,2) = [base, +finger, -finger] -> (...,5) = [gx,gy,cosφ,sinφ,w].
    朝向 = base -> 指尖中点方向（退化时 fallback +x）；w = 两指距离。"""
    base = eef[..., 0, :]                                      # (...,2)
    f1, f2 = eef[..., 1, :], eef[..., 2, :]
    mid = 0.5 * (f1 + f2)
    d = mid - base                                            # (...,2)
    n = np.sqrt((d ** 2).sum(-1) + EPS)[..., None]
    dirv = np.where(n > 1e-4, d / n, np.broadcast_to(np.array([1.0, 0.0], np.float32), d.shape))
    w = np.sqrt(((f1 - f2) ** 2).sum(-1) + EPS)[..., None]    # (...,1)
    return np.concatenate([base, dirv, w], axis=-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n iws python -m pytest tests/test_scel_repr.py -k grasp_frame -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scel_repr.py tests/test_scel_repr.py
git commit -m "feat(scel): grasp_frame [pos,orient,width] helper + tests"
```

---

## Task 3: `scel_repr.to_agent_frame` / `from_agent_frame` — 可逆坐标变换

**Files:**
- Modify: `scel_repr.py`
- Test: `tests/test_scel_repr.py`

agent-frame: 给定 grasp frame (gx,gy,cosφ,sinφ,·)，把世界系 object 点 `p` 变换到 grasp 坐标系
`p_rel = R(φ)^T @ (p - g)`；逆变换 `p = R(φ) @ p_rel + g`。R(φ)=[[c,-s],[s,c]]。

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_scel_repr.py
def test_agent_frame_known_value():
    # grasp at (1,1), φ=90° (cos=0,sin=1). world point (2,1).
    # p-g=(1,0). R(φ)^T@(1,0): R^T=[[0,1],[-1,0]] -> (0,-1)
    gf = np.array([1.0, 1.0, 0.0, 1.0, 0.2], np.float32)[None, None]   # (1,1,5)
    p = np.array([2.0, 1.0], np.float32)[None, None, None]             # (1,1,1,2)
    rel = S.to_agent_frame(p, gf)
    assert np.allclose(rel[0, 0, 0], [0.0, -1.0], atol=1e-6)


def test_agent_frame_roundtrip_identity():
    rng = np.random.default_rng(0)
    p = rng.random((4, 24, 48, 2)).astype(np.float32)
    eef = rng.random((4, 24, 3, 2)).astype(np.float32)
    gf = S.grasp_frame(eef)
    rel = S.to_agent_frame(p, gf)
    back = S.from_agent_frame(rel, gf)
    assert np.allclose(back, p, atol=1e-5)            # 往返恒等
    assert rel.shape == p.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n iws python -m pytest tests/test_scel_repr.py -k agent_frame -v`
Expected: FAIL — `AttributeError: ... 'to_agent_frame'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to scel_repr.py
def _rot(gf):
    """gf (...,>=4) -> cos,sin broadcastable to (...,1)."""
    return gf[..., 2], gf[..., 3]


def to_agent_frame(p, gf):
    """p (...,P,2) world points, gf (...,5) grasp frame -> p_rel (...,P,2) = R(φ)^T (p-g)."""
    g = gf[..., None, :2]                                     # (...,1,2)
    c, s = _rot(gf); c = c[..., None, None]; s = s[..., None, None]
    d = p - g                                                # (...,P,2)
    x, y = d[..., 0:1], d[..., 1:2]
    xr = c * x + s * y                                       # R^T row0 = [ c,  s]
    yr = -s * x + c * y                                      # R^T row1 = [-s,  c]
    return np.concatenate([xr, yr], axis=-1)


def from_agent_frame(p_rel, gf):
    """inverse of to_agent_frame: p = R(φ) p_rel + g."""
    g = gf[..., None, :2]
    c, s = _rot(gf); c = c[..., None, None]; s = s[..., None, None]
    x, y = p_rel[..., 0:1], p_rel[..., 1:2]
    xw = c * x - s * y                                       # R row0 = [c, -s]
    yw = s * x + c * y                                       # R row1 = [s,  c]
    return np.concatenate([xw, yw], axis=-1) + g
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n iws python -m pytest tests/test_scel_repr.py -k agent_frame -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scel_repr.py tests/test_scel_repr.py
git commit -m "feat(scel): invertible to/from_agent_frame transform + tests"
```

---

## Task 4: `scel_repr.fit_vel_stats` / `normalize_vel` / `denormalize_vel` — per-domain 速度 z-score

**Files:**
- Modify: `scel_repr.py`
- Test: `tests/test_scel_repr.py`

per-domain（robot=1/human=0）速度归一化：human 手比 robot 夹爪快 ~2×（probe 0.987），按域 z-score
抵消速度幅度指纹。仅对 agent 量用（thick-latent 验证口径）；object 不归一化（保物理位移）。

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_scel_repr.py
def test_vel_stats_per_domain_zscore():
    # 域1 速度小，域0 速度大
    v = np.concatenate([np.linspace(0, 0.02, 50), np.linspace(0, 0.2, 50)]).astype(np.float32)[:, None]
    dom = np.concatenate([np.ones(50, int), np.zeros(50, int)])
    stats = S.fit_vel_stats(v, dom)                   # {0:(mean,std), 1:(mean,std)}
    out = S.normalize_vel(v, dom, stats)
    for d in (0, 1):
        m = out[dom == d]
        assert abs(m.mean()) < 1e-4
        assert abs(m.std() - 1.0) < 0.05


def test_vel_normalize_roundtrip():
    rng = np.random.default_rng(1)
    v = rng.standard_normal((100, 2)).astype(np.float32)
    dom = (rng.random(100) < 0.5).astype(int)
    stats = S.fit_vel_stats(v, dom)
    back = S.denormalize_vel(S.normalize_vel(v, dom, stats), dom, stats)
    assert np.allclose(back, v, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n iws python -m pytest tests/test_scel_repr.py -k vel -v`
Expected: FAIL — `AttributeError: ... 'fit_vel_stats'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to scel_repr.py
def fit_vel_stats(vel, dom):
    """vel (N,...,D) flattened over leading dims per sample; dom (N,) in {0,1}.
    -> {d: (mean(D,), std(D,))} per domain over all but last axis."""
    vel = np.asarray(vel); dom = np.asarray(dom)
    D = vel.shape[-1]
    flat = vel.reshape(-1, D); dflat = np.repeat(dom, flat.shape[0] // len(dom)) if flat.shape[0] != len(dom) else dom
    stats = {}
    for d in (0, 1):
        sel = flat[dflat == d]
        if len(sel) == 0:
            stats[d] = (np.zeros(D, np.float32), np.ones(D, np.float32))
        else:
            stats[d] = (sel.mean(0).astype(np.float32), (sel.std(0) + 1e-6).astype(np.float32))
    return stats


def _apply(vel, dom, stats, fwd):
    vel = np.asarray(vel).copy(); dom = np.asarray(dom)
    out = vel.reshape(len(dom), -1, vel.shape[-1])
    for d in (0, 1):
        m, s = stats[d]
        idx = dom == d
        out[idx] = (out[idx] - m) / s if fwd else out[idx] * s + m
    return out.reshape(vel.shape)


def normalize_vel(vel, dom, stats):   return _apply(vel, dom, stats, True)
def denormalize_vel(vel, dom, stats): return _apply(vel, dom, stats, False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n iws python -m pytest tests/test_scel_repr.py -v`
Expected: PASS (all tests, 8 total)

- [ ] **Step 5: Commit**

```bash
git add scel_repr.py tests/test_scel_repr.py
git commit -m "feat(scel): per-domain velocity z-score (fit/normalize/denormalize) + tests"
```

---

## Task 5: `viz_scel_repr.py` — M0 眼检 overlay（mask 质量铁律：必须眼检）

**Files:**
- Create: `viz_scel_repr.py`

产出 robot/human 各 4 帧的 overlay：质心点 + scale 圆 + grasp 位置 + 朝向箭头 + 开合标注，
人工核验"质心在 cube 上、scale 圈住 cube、grasp 箭头指向夹爪/手开口方向"。**这是 gate，不是自动断言**
（memory: mask/几何质量必须亲眼看 overlay，别信数字）。

- [ ] **Step 1: Write the script**

```python
# viz_scel_repr.py
"""M0 eyeball: overlay object_node (centroid+scale circle) and grasp_frame (pos+orient arrow+width)
on real frames, robot & human. Output: outputs/cross_embodiment_wm/scel_m0_viz/."""
import os, numpy as np, cv2, scel_repr as S

DS = "outputs/flow_render_dataset_v3"
OUT = "outputs/cross_embodiment_wm/scel_m0_viz"; os.makedirs(OUT, exist_ok=True)
IMG = 128


def draw(frame, node, gf):
    im = frame.copy()
    cx, cy, sc = node
    p = (int(cx * IMG), int(cy * IMG))
    cv2.circle(im, p, max(2, int(sc * IMG)), (0, 255, 0), 1)      # scale circle (green)
    cv2.circle(im, p, 2, (0, 255, 0), -1)                         # centroid
    gx, gy, c, s, w = gf
    gp = (int(gx * IMG), int(gy * IMG))
    tip = (int((gx + 0.15 * c) * IMG), int((gy + 0.15 * s) * IMG))
    cv2.arrowedLine(im, gp, tip, (0, 0, 255), 1, tipLength=0.3)   # grasp orient (red)
    cv2.putText(im, f"w{w:.2f}", (gp[0] + 3, gp[1]), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 1)
    return im


def run(tag):
    z = np.load(f"{DS}/clips_{tag}.npz")
    fr, tr, ef = z["frames"], z["tracks"].astype(np.float32), z["eef"].astype(np.float32)
    nodes = S.object_node(tr); gfs = S.grasp_frame(ef)
    rng = np.random.default_rng(0); picks = rng.choice(len(fr), 4, replace=False)
    rows = []
    for i in picks:
        t = tr.shape[1] // 2
        rows.append(draw(fr[i, t].astype(np.uint8), nodes[i, t], gfs[i, t]))
    grid = np.concatenate(rows, 1)
    cv2.imwrite(f"{OUT}/overlay_{tag}.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return f"{tag}: {len(fr)} clips, drew {len(picks)} frames"


if __name__ == "__main__":
    lines = [run("robot"), run("human")]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True); print(f"saved {OUT}/", flush=True)
```

- [ ] **Step 2: Run it**

Run: `conda run -n iws python viz_scel_repr.py`
Expected: 打印 `robot: ... drew 4 frames` / `human: ...`，生成
`outputs/cross_embodiment_wm/scel_m0_viz/overlay_{robot,human}.png` + `summary.txt`

- [ ] **Step 3: 眼检 gate（人工，必须做）**

打开 `outputs/cross_embodiment_wm/scel_m0_viz/overlay_robot.png` 和 `overlay_human.png`，确认：
质心落在 cube 上、绿圈大致圈住 cube、红箭头从夹爪/手 base 指向开口方向、`w` 数值随开合变化合理。
**附绝对路径给用户、请用户确认通过后再进 Task 6**（memory: 产出图主动给路径 + 几何质量必须眼检）。

- [ ] **Step 4: Commit**

```bash
git add viz_scel_repr.py
git commit -m "feat(scel): M0 eyeball overlay for object_node + grasp_frame"
```

---

## Task 6: `exp_scel_agentframe.py` — agent-frame 变换层 + LWC 训练/rollout（smoke）

**Files:**
- Create: `exp_scel_agentframe.py`

agent-frame 包在现有 LWC 外：训练前把 object tracks 逐帧 `to_agent_frame`（用每帧 grasp frame），
喂 `amplify_wm.train_lwc_ss`；rollout 用 `amplify_wm.rollout_lwc` 在 agent-frame 预测，每步用**已知** grasp
frame `from_agent_frame` 回世界系算 ADE。eef（=已知 action）在两系都可得，故逆变换合法、无泄漏。

> ⚠️ declare component versions：② = `amplify_wm.FlowWM_LWC` + `train_lwc_ss`（采用版 LWC，horizon-robust），
> 非 `eval_scheduled_sampling` 的回归 `FlowWM`。对照 baseline 用**同** LWC，仅坐标系不同（公平：同模型同设置）。
>
> ⚠️ **执行踩坑 1（vel_half 窗口）**：LWC 有界窗口默认 `vel_half=0.12` 是为**世界系**单步速度设的。agent-frame 下
> object 相对 grasp 的速度分布可能不同（grasp 静止时 object 跟随→相对速度小；grasp 动而 object 静→相对速度 ≈ −grasp
> 速度，可能 > 0.12 被 clamp 失真）。先打印 agent-frame 单步速度的分位数；若 99% 分位 > 0.12，则给 `train_lwc_ss`
> 传更大的 `vel_half`（如 0.20），baseline 与 agentframe **用同一 vel_half** 保持公平。
>
> ⚠️ **执行踩坑 2（速度归一化范围）**：Task 4 的 `normalize_vel` 是给 **agent 节点/action 加厚**（后续 M2）用的，
> 本 M1 agent-frame 实验**不集成**它（M1 只变换 object 坐标系，不改 agent 表示）。不要在 Task 6 里调用它。

- [ ] **Step 1: Write the script (with a built-in SMOKE self-check)**

```python
# exp_scel_agentframe.py
"""M1: agent-frame relative coordinate as an invertible wrapper around LWC ②.
Compares world-frame LWC (baseline) vs agent-frame LWC on robot held-out ADE (px@224)
and downstream human-helps (robot-only vs robot+human). Output:
outputs/cross_embodiment_wm/scel_m1_agentframe/."""
import os, numpy as np, torch
import scel_repr as S
import amplify_wm as A
import eval_scheduled_sampling as SSm
from amplify_wm import K, F, device

SMOKE = os.environ.get("SMOKE", "0") == "1"
DS = "outputs/flow_render_dataset_v3"
OUT = "outputs/cross_embodiment_wm/scel_m1_agentframe"; os.makedirs(OUT, exist_ok=True)
HELDOUT = 150; H = 16
N_ROB = 100                                  # scarcity point where human helped


def transform_tracks(tracks, eef, fwd=True):
    """tracks (N,L,48,2) world<->agent-frame using per-FRAME grasp frame from eef (N,L,3,2)."""
    gf = S.grasp_frame(eef)                                   # (N,L,5)
    fn = S.to_agent_frame if fwd else S.from_agent_frame
    return fn(tracks, gf).astype(np.float32)


def ade_world(model, tracks, eef, idx, agentframe):
    """rollout in (world|agent) frame; always score in WORLD px@224."""
    buf = torch.from_numpy(tracks[idx, :K]).float().to(device)
    ef = torch.from_numpy(eef[idx]).float().to(device)
    pred = A.rollout_lwc(model, buf, ef, H).cpu().numpy()     # (B,H,48,2) in train frame
    if agentframe:                                            # back to world with KNOWN grasp frames
        gf = S.grasp_frame(eef[idx])[:, K:K + H]              # (B,H,5)
        pred = S.from_agent_frame(pred, gf)
    gt = tracks_world[idx][:, K:K + H]
    vis = vis_all[idx][:, K:K + H]
    err = np.linalg.norm(pred - gt, axis=-1) * 224.0          # world px
    return float((err * vis).sum() / (vis.sum() + 1e-6))


def train_one(tracks, vis, eef, idx, agentframe, seed=0):
    tr = transform_tracks(tracks, eef) if agentframe else tracks
    return A.train_lwc_ss(tr, vis, eef, idx, seed=seed)


def main():
    global tracks_world, vis_all
    if SMOKE:
        SSm.WM_EPOCHS = 2; SSm.R_SS = 4
    zr = np.load(f"{DS}/clips_robot.npz"); zh = np.load(f"{DS}/clips_human.npz")
    r_tr, r_ef, r_vs = (zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32), zr["vis"].astype(np.float32))
    h_tr, h_ef, h_vs = (zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32), zh["vis"].astype(np.float32))
    Nr = len(r_tr); perm = np.random.default_rng(0).permutation(Nr)
    ho, pool = perm[:HELDOUT], perm[HELDOUT:]
    tracks_world, vis_all = r_tr, r_vs                        # scored in world frame

    mtr = np.concatenate([r_tr, h_tr]); mef = np.concatenate([r_ef, h_ef]); mvs = np.concatenate([r_vs, h_vs])
    hi = torch.arange(Nr, Nr + len(h_tr))
    sub = pool[np.random.default_rng(100).choice(len(pool), min(N_ROB, len(pool)), replace=False)]
    ri = torch.from_numpy(sub)

    lines = [f"SCEL M1 agent-frame vs world | N_rob={N_ROB} | LWC(amplify_wm) | held-out robot ADE px@224",
             f"{'model':>22} | {'ADE':>7}"]
    res = {}
    for agentframe in (False, True):
        tag = "agentframe" if agentframe else "world"
        # NOTE: mixed-domain train tensors must be transformed with their OWN eef
        wm_ro = train_one(mtr, mvs, mef, ri, agentframe)
        wm_rh = train_one(mtr, mvs, mef, torch.cat([ri, hi]), agentframe)
        ade_ro = ade_world(wm_ro, r_tr, r_ef, ho, agentframe)
        ade_rh = ade_world(wm_rh, r_tr, r_ef, ho, agentframe)
        res[tag] = (ade_ro, ade_rh)
        lines += [f"{tag+' robot-only':>22} | {ade_ro:7.2f}",
                  f"{tag+' robot+human':>22} | {ade_rh:7.2f}",
                  f"{tag+' human-helps Δ':>22} | {ade_ro-ade_rh:+7.2f}"]
    # gates
    w_help = res['world'][0] - res['world'][1]
    a_help = res['agentframe'][0] - res['agentframe'][1]
    lines += ["",
              f"GATE1 agent-frame ADE not worse: agentframe rh {res['agentframe'][1]:.2f} vs world rh {res['world'][1]:.2f}",
              f"GATE2 human-helps preserved/stronger: agentframe Δ {a_help:+.2f} vs world Δ {w_help:+.2f}"]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run SMOKE to verify it executes end-to-end**

Run: `cd /scr2/yusenluo/interactive_world_sim && SMOKE=1 conda run -n iws python exp_scel_agentframe.py`
Expected: 跑完不报错，打印 4 个 model 的 ADE + GATE1/GATE2 行，生成
`outputs/cross_embodiment_wm/scel_m1_agentframe/summary.txt`。SMOKE 数值不可信，只验管线打通
（往返变换、rollout 维度、world 评分对齐）。

- [ ] **Step 3: Add a transform-roundtrip assertion to the test file**

```python
# append to tests/test_scel_repr.py
def test_transform_tracks_roundtrip():
    import exp_scel_agentframe as X
    rng = np.random.default_rng(2)
    tr = rng.random((3, 24, 48, 2)).astype(np.float32)
    ef = rng.random((3, 24, 3, 2)).astype(np.float32)
    rel = X.transform_tracks(tr, ef, fwd=True)
    back = X.transform_tracks(rel, ef, fwd=False)
    assert np.allclose(back, tr, atol=1e-5)
```

- [ ] **Step 4: Run the roundtrip test**

Run: `conda run -n iws python -m pytest tests/test_scel_repr.py -k transform_tracks_roundtrip -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add exp_scel_agentframe.py tests/test_scel_repr.py
git commit -m "feat(scel): M1 agent-frame wrapper around LWC + smoke + roundtrip test"
```

---

## Task 7: M1 full run + human-helps gate

**Files:**
- Create: `sbatch/scel_m1_agentframe.sbatch`

- [ ] **Step 1: Write the sbatch (full, non-SMOKE)**

```bash
# sbatch/scel_m1_agentframe.sbatch
#!/bin/bash
#SBATCH --job-name=scel_m1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=outputs/cross_embodiment_wm/scel_m1_agentframe/slurm_%j.out
mkdir -p outputs/cross_embodiment_wm/scel_m1_agentframe
source ~/.bashrc
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
python exp_scel_agentframe.py
```

- [ ] **Step 2: Submit**

Run: `cd /scr2/yusenluo/interactive_world_sim && mkdir -p outputs/cross_embodiment_wm/scel_m1_agentframe && sbatch sbatch/scel_m1_agentframe.sbatch`
Expected: 打印 `Submitted batch job <id>`

- [ ] **Step 3: Wait + read result**

Run: `cat outputs/cross_embodiment_wm/scel_m1_agentframe/summary.txt`
Expected: 4 个 ADE + GATE1/GATE2 行。

- [ ] **Step 4: Evaluate gates (judgment, report to user)**

- **GATE1（不退化）**：agentframe robot+human ADE **≤** world robot+human ADE（+~10% 容差内即过；
  agent-frame 主要图增 invariant/降漂移，绝对 ADE 不应明显变差）。
- **GATE2（human-helps 保留/变强）**：agentframe 的 human-helps Δ（ro−rh）**≥** world 的 Δ。
- 把 `summary.txt` 绝对路径 + 两个 gate 结论报给用户。**任一 gate 不过**→不算失败，按"probe 不必要、看下游"
  原则记录现象，转 M2 用别的改造（action 加厚 / transition-consistency）对比，不强推 agent-frame。

- [ ] **Step 5: Commit**

```bash
git add sbatch/scel_m1_agentframe.sbatch outputs/cross_embodiment_wm/scel_m1_agentframe/summary.txt
git commit -m "feat(scel): M1 full run sbatch + agent-frame human-helps result"
```

---

## Self-Review notes（已对照 spec）

- **覆盖**：M0（object_node/grasp_frame/速度归一化）= Task 1/2/4；agent-frame 相对坐标（spec 候选 2 主推）
  = Task 3/6/7；M0 眼检 gate（mask 质量铁律）= Task 5。
- **明确不在本计划**（spec 已注后续独立计划）：刚体平移约束(候选1)、action 加厚(候选3)、SS 加长(候选4)、
  agent 多关键点(候选5)、transition-delta 一致性正则(候选6)、③ 渲染加厚(M3)、备选 plan B、图节点重构。
- **类型一致**：`object_node`→(...,3)、`grasp_frame`→(...,5)、`to/from_agent_frame`(...,P,2)↔(...,P,2)、
  `transform_tracks` 用 per-frame grasp frame；rollout 在训练所用坐标系、评分统一 world px@224。
- **judgment gates**（GATE1/GATE2）非自动断言：ML 结果按 spec"下游 human-helps 拍板、probe 仅诊断"处理。
- **guardrails 落地**：declare component versions（Task 6 注 LWC 版本）、同序列同设置对照、眼检 gate、
  summary.txt 存盘 + 给绝对路径、per-experiment 独立目录。
