# Keyboard 3D 交互控制(视频架构 ③)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 free-form keyboard 交互接到新视频架构 ③(Wan VAE + VideoDiT),用真世界系 3D 控制实现水平平移 + z 轴上下(修 lift_high 漂移)+ grip 功能性抓取。

**Architecture:** 键盘→世界系 3D delta 作用于 eef3d 星座 + grip → REAL 标定投影到两视角 2D → ② rollout(+抓取刚体兜底)→ IK gmask → VideoDiT ③ sample() 渲染 → 标准 save_combined_gif 双视角 gif。

**Tech Stack:** Python, PyTorch, 官方 AutoencoderKLWan(`.venv_wan`), numpy/cv2/imageio。复用 `exp_scel_keyboard_can_ik.py`(script/rollout/IK)、`eval_e2e_combined.py`(VideoDiT sample/cond)、`gen_flow_render_dataset_caneef.py:can_rig_KT`、`viz_combined.py`。

## Global Constraints

- **环境隔离(已核实)**:`AutoencoderKLWan` 只在 `.venv_wan`;真 pinocchio FK 只在 conda `phantom`(3.9.0);iws/.venv_wan pinocchio 是假 0.1。
  → **三阶段管线**:A(`.venv_wan`)控制+②+IK+物体 → B(`phantom`,纯 FK)joint→skel2d → C(`.venv_wan`)skel cond+渲染+gif。npz 传递。
- GPU job 走 **sbatch**(partition-1),不 nohup。sbatch 内按阶段切 `.venv_wan/bin/python` 与 `conda run -n phantom python`。
- 数据 = `outputs/flow_render_dataset_can_dual/clips_robot.npz`(L=48);标定 = `calib/rgb_cam_calib_can_{REAL,low_REAL}.json`。
- 组件 ckpt(端到端声明):② `wm_dummy5_rh_N3000.pt`;③ **`outputs/video_arch_wm/m4_ablB_skel/video_dit_ema.pt`(skel agent,消融赢家,60000)**;IK `ik_adapter_can/ik_adapter.pt`。mask ③ `m4_video_dit` 作 fallback。
- H=48 单视频块(tL=12,rf=min(4k,H-1));**脚本 = 运动帧(Σn·REPEAT)+ hold-pad 补满 48**(运动可 <48,末尾保持最后位姿)。
- **控制标度**:`DELTA=0.012`/帧(世界米),`REPEAT=3`;`BOX3D=(x_lo,x_hi,y_lo,y_hi,z_lo,z_hi)`,z 上限≈z0+0.28(cam_low 在 +0.5m 以上投影发散)。lift 幅度 ≤ ~0.25m。
- **相机几何(实测,勿假设 top-down)**:cam_high 光轴偏世界竖直 24.1°、cam_low 42.4°。纯 z 抬起在两视角以竖直为主 + **小的透视正确水平分量**(非零)。真 3D 的修复=两视角是同一 3D 轨迹的一致投影,不是「cam_high x 零位移」。
- CROP:cam_high=(60,60,390,390),cam_low=(0,0,640,480);GRID=16,POOL=8。
- 交付走标准 `save_combined_gif`(双视角并列,Rendered 行 + Flow overlay 行 + caption);**overlay 同叠 flow+skel**(参照 `eval_e2e_combined.py:ov()`);禁自造 PIL 布局。
- 输出独立目录 `outputs/video_arch_wm/keyboard_3d/`,不覆盖已有 eval。中间产物必须可视化。
- torch.load 需 register WM/VideoDiT 类到 `__main__`(见 eval_e2e_combined.py:17-18)。

---

### Task 1: 世界系 3D 控制 + 双视角投影

**Files:**
- Create: `keyboard_3d_control.py`(控制原语模块,纯 numpy,可单测无 GPU)
- Test: `tests/test_keyboard_3d.py`

**Interfaces:**
- Produces:
  - `can_KT(view) -> (T_cw(4,4), K(3,3))` — 读 REAL 标定(view∈{"high","low"})。
  - `project_norm(p3(3,), T, K, crop) -> (2,)` — 世界 3D→crop-norm 2D。
  - `WORLD_DIRS: dict[str,(3,)]` — 键→世界系单位方向(left/right=x∓, fwd/back=y∓, up/down=z±)。
  - `script_eef3d(eef3d0(3,3), grip0, script, DELTA, GRIP_DELTA, REPEAT, F, BOX3D) -> (eef3d_traj(S+F,3,3), grip_traj(S+F,), H)` — 复合脚本→世界系 eef3d 轨迹 + grip 轨迹,尾 pad F。
  - `project_eef_dual(eef3d_traj) -> (efA(T,3,2), efB(T,3,2))` — 两视角 crop-norm 2D eef 轨迹。

- [ ] **Step 1: 写失败测试(投影自洽 + 纯 z 无水平漂移)**

```python
# tests/test_keyboard_3d.py
import numpy as np, pytest
from keyboard_3d_control import can_KT, project_norm, project_eef_dual, script_eef3d, WORLD_DIRS
CROP = {"high": (60,60,390,390), "low": (0,0,640,480)}

def _clip():
    z = np.load("outputs/flow_render_dataset_can_dual/clips_robot.npz")
    return z

def test_reproj_matches_stored():
    z = _clip(); si = 332
    e3 = z["eef3d"][si].astype(np.float64)      # (48,3,3)
    for view, ek in [("high","eef"), ("low","eef_low")]:
        T, K = can_KT(view); e2 = z[ek][si].astype(np.float32)
        errs = []
        for t in range(48):
            for k in range(3):
                if np.all(np.isfinite(e2[t,k])):
                    uv = project_norm(e3[t,k], T, K, CROP[view])
                    errs.append(np.linalg.norm(uv - e2[t,k]))
        assert np.mean(errs)*128 < 0.2, f"{view} reproj {np.mean(errs)*128:.3f}px"

def _triangulate(uvA, uvB):
    """两视角 crop-norm 2D -> 世界 3D (pixel-space DLT, float32 下稳). 验'纯z投影三角化回来纯竖直'。"""
    from keyboard_3d_control import can_KT, CROP
    rows = []
    for uv, view in [(uvA,"high"), (uvB,"low")]:
        T,K = can_KT(view); P = K @ T[:3]            # 3x4 world->pixel
        x,y,w,h = CROP[view]; px = np.array([uv[0]*w+x, uv[1]*h+y])  # norm->pixel
        rows.append(px[0]*P[2]-P[0]); rows.append(px[1]*P[2]-P[1])
    _,_,Vt = np.linalg.svd(np.stack(rows).astype(np.float64)); X = Vt[-1]; return X[:3]/X[3]

def test_pure_z_triangulates_to_vertical():
    z = _clip(); si = 332
    e30 = z["eef3d"][si,0].astype(np.float64)   # (3,3)
    # 现实幅度 lift (~0.22m): up 6键*REPEAT3*DELTA0.012
    z0 = e30[:,2].mean()
    traj, grip, H = script_eef3d(e30, 0.029, [("up",6)], DELTA=0.012, GRIP_DELTA=0.005, REPEAT=3, F=0,
                                 BOX3D=(-1,1,-1,1,0.0,z0+0.28))
    efA, efB = project_eef_dual(traj)           # (H,3,2)
    # 三角化 eef 质心 首帧 vs 末帧 -> 世界位移应为纯 +z
    w0 = _triangulate(efA[0].mean(0), efB[0].mean(0))
    wT = _triangulate(efA[-1].mean(0), efB[-1].mean(0))
    d = wT - w0
    assert d[2] > 0.15, f"世界z位移应显著为正 {d[2]:.3f}"
    assert abs(d[0]) < 0.02 and abs(d[1]) < 0.02, f"世界水平位移应≈0 (dx{d[0]:.3f} dy{d[1]:.3f})"
    # 且两视角图像运动以竖直为主(透视水平分量小)
    dyA = abs(efA[-1,:,1].mean()-efA[0,:,1].mean()); dxA = abs(efA[-1,:,0].mean()-efA[0,:,0].mean())
    assert dyA > 2*dxA, f"cam_high 应竖直为主 dy{dyA*128:.1f} dx{dxA*128:.1f}px"
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv_wan/bin/python -m pytest tests/test_keyboard_3d.py -x -q`
Expected: FAIL(ModuleNotFoundError: keyboard_3d_control)

- [ ] **Step 3: 实现 keyboard_3d_control.py**

```python
"""世界系 3D keyboard 控制原语:脚本→eef3d轨迹+grip, REAL标定投影两视角。纯numpy,无GPU,可单测。"""
import numpy as np, json
from scipy.spatial.transform import Rotation

_CAL = {"high": "calib/rgb_cam_calib_can_REAL.json", "low": "calib/rgb_cam_calib_can_low_REAL.json"}
CROP = {"high": (60,60,390,390), "low": (0,0,640,480)}
# 世界系单位方向: coord0=x(桌面), coord1=y(桌面), coord2=z(高度). up=+z.
WORLD_DIRS = {"left":(-1,0,0),"right":(1,0,0),"fwd":(0,-1,0),"back":(0,1,0),"up":(0,0,1),"down":(0,0,-1)}
_AR = {"left":"<","right":">","fwd":"^f","back":"vb","up":"Uz","down":"Dz","gopen":"O","gclose":"C"}

def can_KT(view):
    r = json.load(open(_CAL[view])); R = Rotation.from_quat(r["quat_xyzw"]).as_matrix()
    t = np.asarray(r["t_world"]); T = np.eye(4); T[:3,:3] = R.T; T[:3,3] = -R.T @ t
    K = np.array([[r["f"],0,r["cx"]],[0,r["f"],r["cy"]],[0,0,1.0]])
    return T, K

def project_norm(p3, T, K, crop):
    pc = (T @ np.append(p3,1.0))[:3]; uv = K @ pc; uv = uv[:2]/uv[2]
    x,y,w,h = crop; return np.array([(uv[0]-x)/w, (uv[1]-y)/h], np.float32)

def script_eef3d(eef3d0, grip0, script, DELTA, GRIP_DELTA, REPEAT, F, BOX3D):
    base = eef3d0.mean(0); off = eef3d0 - base       # 刚体星座内偏移锁死
    c = base.astype(np.float64).copy(); g = float(grip0); cs=[]; gs=[]
    for cmd, n in script:
        if cmd in ("gopen","gclose"):
            step = GRIP_DELTA if cmd=="gopen" else -GRIP_DELTA
            for _ in range(n*REPEAT):
                g = float(np.clip(g+step, 0.0, 0.04)); cs.append(c.copy()); gs.append(g)
        else:
            d = np.asarray(WORLD_DIRS[cmd], np.float64) * DELTA
            for _ in range(n*REPEAT):
                c = c + d; c[0]=np.clip(c[0],BOX3D[0],BOX3D[1]); c[1]=np.clip(c[1],BOX3D[2],BOX3D[3]); c[2]=np.clip(c[2],BOX3D[4],BOX3D[5])
                cs.append(c.copy()); gs.append(g)
    cs = np.stack(cs); gs = np.asarray(gs, np.float32)
    traj = (cs[:,None] + off[None]).astype(np.float64)      # (S,3,3)
    H = len(traj)
    traj = np.concatenate([traj, np.repeat(traj[-1:],F,0)],0); gs = np.concatenate([gs, np.repeat(gs[-1:],F)])
    return traj, gs, H
# 注:渲染阶段(Task4 precompute)把 eef/grip 轨迹 hold-pad 到恰好 RENDER_H=48(运动 H≤48,末尾保持最后位姿);
#     script_eef3d 只产运动帧(+F 滑窗未来 pad),不负责补到 48。

def project_eef_dual(eef3d_traj):
    Th,Kh = can_KT("high"); Tl,Kl = can_KT("low"); T = len(eef3d_traj)
    efA = np.stack([[project_norm(eef3d_traj[t,k],Th,Kh,CROP["high"]) for k in range(3)] for t in range(T)]).astype(np.float32)
    efB = np.stack([[project_norm(eef3d_traj[t,k],Tl,Kl,CROP["low"]) for k in range(3)] for t in range(T)]).astype(np.float32)
    return efA, efB
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv_wan/bin/python -m pytest tests/test_keyboard_3d.py -x -q`
Expected: PASS(2 tests)

- [ ] **Step 5: Commit**

```bash
git add keyboard_3d_control.py tests/test_keyboard_3d.py
git commit -m "feat: 世界系3D keyboard控制原语+双视角REAL标定投影(修lift_high水平漂移)"
```

---

### Task 2: 功能性抓取 + 物体 3D 刚体跟随

**Files:**
- Modify: `keyboard_3d_control.py`(加抓取/物体跟随)
- Test: `tests/test_keyboard_3d.py`(加抓取用例)

**Interfaces:**
- Consumes: `script_eef3d`, `project_norm`, `can_KT`(Task 1)。
- Produces:
  - `NEAR_TH`(常量,从真实抓握帧 eef-obj 3D 距离分布定)、`GRASP_TH=0.033`。
  - `grasp_timeline(eef3d_traj, grip_traj, obj3d0(48,3)) -> grasp(T,) bool` — 每帧是否抓取(grip<GRASP_TH 且 eef 质心近 obj 质心)。
  - `object_track_dual(eef3d_traj, grip_traj, obj3d0, valid0(48,)) -> (objA(T,48,2), objB(T,48,2), grasp(T,))` — 抓取相物体刚体跟随 eef(平移),投影两视角;非抓取相物体停最后位置。

- [ ] **Step 1: 写失败测试(抓取判定 + z 抬起物体跟随)**

```python
def test_grasp_and_object_follow():
    from keyboard_3d_control import object_track_dual, script_eef3d
    z = _clip(); si = 332
    e30 = z["eef3d"][si,0].astype(np.float64)
    obj3d0 = z["tracks3d"][si,0].astype(np.float64)      # (48,3)
    valid0 = z["tracks3d_valid"][si,0]
    # 抓紧(grip 低)后抬起
    traj, grip, H = script_eef3d(e30, 0.029, [("up",6)], DELTA=0.03, GRIP_DELTA=0.01, REPEAT=4, F=0, BOX3D=(-1,1,-1,1,0,1))
    objA, objB, grasp = object_track_dual(traj, grip, obj3d0, valid0)
    assert grasp.mean() > 0.8, "grip低+近物体应判定抓取"
    # 抓取抬起 -> cam_low 物体质心竖直上移(跟随 eef)
    dy_low = objB[-1,valid0,1].mean() - objB[0,valid0,1].mean()
    assert abs(dy_low) > 0.03, f"抓取抬起物体应在cam_low竖直动 {dy_low:.3f}"
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv_wan/bin/python -m pytest tests/test_keyboard_3d.py::test_grasp_and_object_follow -x -q`
Expected: FAIL(ImportError: object_track_dual)

- [ ] **Step 3: 实现抓取 + 跟随**(先跑一次性脚本从真实抓握帧标定 NEAR_TH)

先定 NEAR_TH:
```bash
.venv_wan/bin/python -c "
import numpy as np
z=np.load('outputs/flow_render_dataset_can_dual/clips_robot.npz')
g=z['grip']; e3=z['eef3d']; t3=z['tracks3d']; tv=z['tracks3d_valid']
d=[]
for si in range(0,2700,7):
    for t in range(48):
        if g[si,t]<0.033 and tv[si,t].any():
            d.append(np.linalg.norm(e3[si,t].mean(0)-t3[si,t][tv[si,t]].mean(0)))
d=np.array(d); print('抓握帧 eef-obj 3D dist: p50%.3f p90%.3f p99%.3f'%(np.percentile(d,50),np.percentile(d,90),np.percentile(d,99)))
"
# 用 p90 作 NEAR_TH 写入常量
```

```python
# 追加到 keyboard_3d_control.py
GRASP_TH = 0.033
NEAR_TH = 0.12   # ← 用上面 p90 实测值替换

def grasp_timeline(eef3d_traj, grip_traj, obj3d0):
    ec = eef3d_traj.mean(1); oc = obj3d0.mean(0)      # 简化:物体初始质心
    near = np.linalg.norm(ec - oc[None], axis=1) < NEAR_TH
    return (grip_traj < GRASP_TH) & near

def object_track_dual(eef3d_traj, grip_traj, obj3d0, valid0):
    Th,Kh = can_KT("high"); Tl,Kl = can_KT("low"); T = len(eef3d_traj)
    grasp = grasp_timeline(eef3d_traj, grip_traj, obj3d0)
    ec = eef3d_traj.mean(1)
    obj3d = np.repeat(obj3d0[None], T, 0).astype(np.float64)   # (T,48,3)
    g0 = None
    for t in range(1, T):
        if grasp[t]:
            if g0 is None: g0 = t                                # 抓取起点:锁 offset
            obj3d[t] = obj3d[g0] + (ec[t] - ec[g0])[None]        # 平移刚体跟随
        else:
            g0 = None; obj3d[t] = obj3d[t-1]                     # 释放:停在原地
    objA = np.stack([[project_norm(obj3d[t,p],Th,Kh,CROP["high"]) for p in range(48)] for t in range(T)]).astype(np.float32)
    objB = np.stack([[project_norm(obj3d[t,p],Tl,Kl,CROP["low"]) for p in range(48)] for t in range(T)]).astype(np.float32)
    return objA, objB, grasp
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv_wan/bin/python -m pytest tests/test_keyboard_3d.py -x -q`
Expected: PASS(3 tests)

- [ ] **Step 5: Commit**

```bash
git add keyboard_3d_control.py tests/test_keyboard_3d.py
git commit -m "feat: 功能性抓取判定+物体3D刚体跟随投影两视角"
```

---

### Task 3: ②驱动 + 抓取兜底 的物体轨迹

**Files:**
- Create: `exp_scel_keyboard_3d.py`(主脚本骨架;本任务只做 ② rollout + fallback 合成 predtr,不渲染)
- Test: `tests/test_keyboard_3d.py`(加 fallback 用例,mock ②)

**Interfaces:**
- Consumes: `project_eef_dual`, `object_track_dual`(Task 1/2);`exp_scel_dualview_wm.rollout_dual`。
- Produces:
  - `FOLLOW_TH=0.4` — ② 物体位移/eef 位移比阈值。
  - `blend_predtr(pred2_A, pred2_B, objA, objB, grasp, efA, efB) -> (trajA(H,48,2), trajB(H,48,2), used_fallback(H,) bool)` — 抓取相 ② 跟随不足则用刚体投影覆盖。

- [ ] **Step 1: 写失败测试(② 不跟随时兜底触发)**

```python
def test_fallback_when_under_follow():
    from exp_scel_keyboard_3d import blend_predtr
    H = 24
    # ② 预测物体几乎不动(under-follow), eef 抬起明显
    pA = np.zeros((H,48,2), np.float32); pB = np.zeros((H,48,2), np.float32)
    objA = np.zeros((H,48,2), np.float32)
    objB = np.tile(np.linspace(0.6,0.4,H)[:,None,None],(1,48,2)).astype(np.float32)  # 刚体跟随:cam_low 上移
    grasp = np.ones(H, bool)
    efA = np.zeros((H,3,2),np.float32); efB = np.tile(np.linspace(0.6,0.4,H)[:,None,None],(1,3,2)).astype(np.float32)
    tA,tB,fb = blend_predtr(pA,pB,objA,objB,grasp,efA,efB)
    assert fb.mean() > 0.5, "抓取相②不跟随应触发兜底"
    assert abs(tB[-1,:,1].mean() - 0.4) < 0.05, "兜底应采用刚体投影(cam_low上移)"
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv_wan/bin/python -m pytest tests/test_keyboard_3d.py::test_fallback_when_under_follow -x -q`
Expected: FAIL(ModuleNotFoundError: exp_scel_keyboard_3d)

- [ ] **Step 3: 实现 blend_predtr(+主脚本骨架)**

```python
# exp_scel_keyboard_3d.py (骨架, 后续 Task 4/5 续)
import os, sys, numpy as np
FOLLOW_TH = float(os.environ.get("FOLLOW_TH", "0.4"))

def blend_predtr(pred2_A, pred2_B, objA, objB, grasp, efA, efB):
    H = len(pred2_A); trajA = pred2_A.copy(); trajB = pred2_B.copy(); fb = np.zeros(H, bool)
    for h in range(H):
        if grasp[h]:
            ef_disp = np.linalg.norm(np.concatenate([efA[h].mean(0)-efA[0].mean(0), efB[h].mean(0)-efB[0].mean(0)]))
            pr_disp = np.linalg.norm(np.concatenate([pred2_A[h].mean(0)-pred2_A[0].mean(0), pred2_B[h].mean(0)-pred2_B[0].mean(0)]))
            if ef_disp > 1e-3 and pr_disp / ef_disp < FOLLOW_TH:
                trajA[h] = objA[h]; trajB[h] = objB[h]; fb[h] = True
    return trajA, trajB, fb
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv_wan/bin/python -m pytest tests/test_keyboard_3d.py -x -q`
Expected: PASS(4 tests)

- [ ] **Step 5: Commit**

```bash
git add exp_scel_keyboard_3d.py tests/test_keyboard_3d.py
git commit -m "feat: ②驱动+抓取兜底 blend_predtr(②不跟随用刚体投影覆盖)"
```

---

### Task 4: 阶段 A 预计算(控制+②+IK+物体)+ 阶段 B FK skel2d

**Files:**
- Modify: `exp_scel_keyboard_3d.py`(加 `precompute_script` + main 的 `--stage precompute`)
- Create: `fk_skel2d.py`(阶段 B,phantom env,纯 pinocchio:joint→skel2d,port 自 `augment_clips_skeleton.py:fk_points`)

**Interfaces:**
- Consumes: `blend_predtr`(Task 3);`object_track_dual`(Task 2);`rollout_dual`;IK adapter。
- Produces:
  - `precompute_script(si, script, R) -> dict`(efA,efB,trajA,trajB,joint(H,7),grip,grasp,fb,f0_high,f0_low,H)存 npz。
  - `fk_skel2d.py`:读 `precompute_*.npz` 的 joint → `skel2d_high(H,9,2)`,`skel2d_low(H,9,2)`,`segments(8,2)` 存 npz。

- [ ] **Step 1: 写测试(precompute 产物形状 + z 段 cam_high 物体不漂;FK skel2d 形状+贴合)**

```python
# tests/test_keyboard_3d_stage.py
import os, numpy as np, pytest

def test_precompute_shapes_and_no_drift():
    from exp_scel_keyboard_3d import precompute_script, load_control
    R = load_control()            # ②+IK, 无 VAE/③
    d = precompute_script(332, [("gclose",1),("up",5),("left",3),("down",2),("gopen",1)], R)
    assert d["H"] == 48 and d["joint"].shape == (48,7) and d["efA"].shape == (48,3,2)
    dx = np.abs(d["trajA"][4:24,:,0].mean(1) - d["trajA"][4,:,0].mean()).max()
    assert dx < 0.03, f"cam_high z段水平漂移 {dx*128:.2f}px"

@pytest.mark.skipif(not os.path.exists("Trossen_Analysis/stationary_ai.urdf"), reason="no urdf")
def test_fk_skel2d_shape():
    # 需 phantom env 跑; 这里只断言函数可 import + 单帧输出形状
    from fk_skel2d import fk_skel2d_dual
    joint = np.zeros((3,7), np.float32)
    sh, sl, seg = fk_skel2d_dual(joint)
    assert sh.shape == (3,9,2) and sl.shape == (3,9,2) and seg.shape == (8,2)
```

- [ ] **Step 2: 运行确认失败**

Run(precompute): `.venv_wan/bin/python -m pytest tests/test_keyboard_3d_stage.py::test_precompute_shapes_and_no_drift -x -q` → FAIL(ImportError)
Run(FK): `conda run -n phantom python -m pytest tests/test_keyboard_3d_stage.py::test_fk_skel2d_shape -x -q` → FAIL(ImportError)

- [ ] **Step 3a: 实现 precompute_script + load_control**(照 exp_scel_keyboard_can_ik.py 的 rollout/IK)

关键点:`load_control()` 只 load ②(wm_dummy5_rh_N3000)+ IK(ik_adapter_can),register WM 类到 `__main__`。
`precompute_script`:script→`script_eef3d`→`project_eef_dual`(efA/efB)→`trDseq=repeat(初始物体 K 帧)`→`rollout_dual(wm,trDseq,efA,efB,H)`→pred2_A/B;
`object_track_dual`→objA/objB/grasp;`blend_predtr`→trajA/trajB/fb;每帧 `joint[h]=ik(efA[f])`;
存 f0_high/f0_low(GT 首帧, `gt256` 逻辑)。存 `outputs/video_arch_wm/keyboard_3d/precompute/{name}_si{si}.npz`。

- [ ] **Step 3b: 实现 fk_skel2d.py**(phantom env,port fk_points,REAL 标定投影)

```python
"""阶段B(phantom env): IK joint(H,7) -> pinocchio FK -> skel2d 两视角. port自 augment_clips_skeleton.fk_points.
跑: conda run -n phantom python fk_skel2d.py  (遍历 precompute/*.npz 补 skel2d)"""
import os, glob, json, numpy as np, pinocchio as pin
from scipy.spatial.transform import Rotation
_CAL={"high":"calib/rgb_cam_calib_can_REAL.json","low":"calib/rgb_cam_calib_can_low_REAL.json"}
CROP={"high":(60,60,390,390),"low":(0,0,640,480)}
_M=pin.buildModelFromUrdf("Trossen_Analysis/stationary_ai.urdf"); _D=_M.createData()
_JIDX=[_M.joints[_M.getJointId(f"follower_right_joint_{j}")].idx_q for j in range(6)]
# ↓ FRAMES/segments 照 augment_clips_skeleton.py 的 9-kpt 链(实现时逐字对照该文件的 fk_points 取哪些 frame origin)
def _KT(view):
    r=json.load(open(_CAL[view])); R=Rotation.from_quat(r["quat_xyzw"]).as_matrix(); t=np.asarray(r["t_world"])
    T=np.eye(4); T[:3,:3]=R.T; T[:3,3]=-R.T@t; K=np.array([[r["f"],0,r["cx"]],[0,r["f"],r["cy"]],[0,0,1.]]); return T,K
def _proj(P,T,K,crop):
    Pc=(T@np.concatenate([P,np.ones((len(P),1))],1).T).T[:,:3]; uv=(K@Pc.T).T; uv=uv[:,:2]/uv[:,2:]
    x,y,w,h=crop; return np.stack([(uv[:,0]-x)/w,(uv[:,1]-y)/h],1).astype(np.float32)
def fk_points(joints6):     # (6,)->(9,3) frame origins  [逐字对照 augment_clips_skeleton]
    q=pin.neutral(_M)
    for i,ji in enumerate(_JIDX): q[ji]=joints6[i]
    pin.forwardKinematics(_M,_D,q)
    return np.stack([...])  # ← 照 augment 选的 9 个 frame origin
def fk_skel2d_dual(joint):  # (H,7)->(H,9,2)x2, segments(8,2)
    P=np.stack([fk_points(joint[h,:6]) for h in range(len(joint))])   # (H,9,3) world
    Th,Kh=_KT("high"); Tl,Kl=_KT("low")
    sh=np.stack([_proj(P[h],Th,Kh,CROP["high"]) for h in range(len(P))])
    sl=np.stack([_proj(P[h],Tl,Kl,CROP["low"]) for h in range(len(P))])
    seg=np.load("outputs/flow_render_dataset_can_dual/skel_sidecar_robot.npz")["segments"]
    return sh,sl,seg
```
FK 正确性眼检:对某真实 clip 用其 `joint` 跑 `fk_skel2d_dual`,和 `skel_sidecar_robot.npz` 存的 skel2d 比 median<2px(证明 port 无误)。

- [ ] **Step 4: 运行确认通过**

Run: `.venv_wan/bin/python -m pytest tests/test_keyboard_3d_stage.py::test_precompute_shapes_and_no_drift -x -q` → PASS
Run: `conda run -n phantom python -m pytest tests/test_keyboard_3d_stage.py::test_fk_skel2d_shape -x -q` → PASS
+ FK vs sidecar median<2px 眼检脚本。

- [ ] **Step 5: Commit**

```bash
git add exp_scel_keyboard_3d.py fk_skel2d.py tests/test_keyboard_3d_stage.py
git commit -m "feat: 阶段A预计算(控制+②+IK+物体)+阶段B pinocchio FK skel2d(phantom env)"
```

---

### Task 5: 阶段 C skel ③ 渲染 + flow+skel overlay + 多组合交付

**Files:**
- Modify: `exp_scel_keyboard_3d.py`(`--stage render`:读 precompute+skel2d → skel cond → VideoDiT 渲染 → 标准 gif)
- Create: `viz_keyboard_3d_intermediate.py`(中间产物:3D 控制→两视角投影+skel+grip 时间线)
- Create: `outputs/video_arch_wm/keyboard_3d/README_SETUP.md`
- Create: `sbatch_keyboard_3d.sbatch`(三阶段多环境)

**Interfaces:**
- Consumes: precompute npz + skel2d npz(Task 4);`eval_e2e_combined` 的 `sample`/`cond_v(agent="skel")`/`skel_chan`/`ov`;`wan_vae.WanVAE`;`viz_combined.save_combined_gif`。

- [ ] **Step 1: 渲染冒烟测试(skel ③ 出图 shape + H=48)**

```python
# tests/test_keyboard_3d_render.py (.venv_wan + GPU)
import os, numpy as np, pytest
pytestmark = pytest.mark.skipif(not os.path.exists("outputs/video_arch_wm/m4_ablB_skel/video_dit_ema.pt"), reason="no skel ③")
def test_render_from_precompute():
    from exp_scel_keyboard_3d import render_from_precompute
    rend = render_from_precompute("outputs/video_arch_wm/keyboard_3d/precompute/lift_high_si332.npz",
                                  "outputs/video_arch_wm/keyboard_3d/skel2d/lift_high_si332.npz")
    assert rend.shape == (2,48,128,128,3)
```

- [ ] **Step 2: 运行确认失败** → FAIL(ImportError: render_from_precompute)

- [ ] **Step 3: 实现 render_from_precompute + SCRIPTS(多组合)+ 标准 gif(flow+skel overlay)**

SCRIPTS(REPEAT=3,每条 Σn·REPEAT ≤48 的运动帧,precompute 再 hold-pad 到 48;Σn=12→36 运动+12 hold):
```python
SCRIPTS = {
  "translate_LR":         [("left",6),("right",6)],
  "square_xy":            [("left",3),("fwd",3),("right",3),("back",3)],
  "lift_high":            [("up",6),("up",6)],                                   # 纯z, 验漂移已修
  "z_wave":               [("up",3),("down",3),("up",3),("down",3)],
  "grip_cycle":           [("gclose",3),("gopen",3),("gclose",3),("gopen",3)],
  "pick_place_left":      [("gclose",2),("up",3),("left",3),("down",2),("gopen",2)],
  "pick_place_right":     [("gclose",2),("up",3),("right",3),("down",2),("gopen",2)],
  "carry_square":         [("gclose",2),("up",2),("left",2),("fwd",2),("down",2),("gopen",2)],
  "lift_translate_lower": [("gclose",1),("up",3),("left",4),("down",3),("gopen",1)],  # ★核心
}
```
render:cond=`cond_v(agent="skel")`(flow3+skel1 via `skel_chan(skel2d)`+warp3),`sample(m3S,za,cond)`→decode 两视角。
gif:每视角 overlay 用 `ov(bg, gtobj, predobj, eef, sk=skel2d[v])`(**flow 红/绿 + skel 白线 + eef 黄**),
`save_combined_gif(..., ["cam_high","cam_low"], [None,None], K, caption=f"{name} | {cmdstr} | grip:{state} | H=48")`。
grip 状态:caption 标 + eef 3 点间距按 grip 缩放画开/闭。

- [ ] **Step 4: 中间产物 viz + README + sbatch(三阶段)**

- `viz_keyboard_3d_intermediate.py`:列=[cam_high 投影 eef+obj+skel | cam_low 同 | grip 时间线];重点 `lift_high` 眼检 cam_high 罐子 x 不动。
- `README_SETUP.md`:组件版本(② wm_dummy5_rh_N3000 / ③ **m4_ablB_skel skel** / IK ik_adapter_can / FK pinocchio phantom)、三阶段管线、抓取=功能性刚体跟随、兜底逻辑、H=48、诚实边界、replay 有 GT 的 seq 报 ③ render LPIPS。
- `sbatch_keyboard_3d.sbatch`:
  ```bash
  cd /scr2/yusenluo/interactive_world_sim
  .venv_wan/bin/python exp_scel_keyboard_3d.py --stage precompute      # A
  conda run -n phantom python fk_skel2d.py                              # B
  .venv_wan/bin/python exp_scel_keyboard_3d.py --stage render          # C
  .venv_wan/bin/python viz_keyboard_3d_intermediate.py
  ```

- [ ] **Step 5: 跑 + 眼检 + 上传 + Commit**

```bash
sbatch sbatch_keyboard_3d.sbatch
# 眼检: lift_high cam_high 罐子x不漂; pick/carry 抓-移-放物体跟随夹爪+skel贴臂; ③不糊
bash upload_evals_gdrive.sh outputs/video_arch_wm/keyboard_3d 2026-07-23_keyboard_3d_video_arch
git add exp_scel_keyboard_3d.py viz_keyboard_3d_intermediate.py sbatch_keyboard_3d.sbatch outputs/video_arch_wm/keyboard_3d/README_SETUP.md
git commit -m "feat: keyboard 3D demo交付(9脚本双视角gif, flow+skel overlay, skel③视频架构)"
```

---

## Self-Review

- **Spec coverage**:水平平移(translate_LR/square_xy)✓ z上下(lift_high/z_wave,Task1 漂移测试✓)grip 功能性抓取(Task2✓)②驱动+兜底(Task3✓)视频skel③(Task4 阶段A/B + Task5 阶段C✓)**flow+skel overlay**(Task5 `ov`✓)**多命令组合**(9 脚本✓)标准gif+中间产物(Task5✓)。
- **Placeholder scan**:NEAR_TH 有 Task2 Step3 实测标定;`fk_points` 的 9-kpt 选取标注「逐字对照 augment_clips_skeleton.py」(实现时照抄,不臆造);其余阈值给了默认值。
- **Type consistency**:efA/efB (T,3,2);objA/objB/trajA/trajB (T,48,2);joint (H,7);skel2d_high/low (H,9,2);segments (8,2);rend (2,H,128,128,3);贯穿一致。
- **环境**:pinocchio FK 隔离在 phantom(Task4 阶段B),渲染/VAE 在 .venv_wan(Task5 阶段C),npz 传递;sbatch 分阶段切环境。
- **风险**:skel ③ 用合成 IK joint 的 skel2d 未验证 → Task4 有「FK vs sidecar median<2px」眼检 + mask ③ fallback(Global Constraints)。
- **依赖**:rollout_dual/gmask/sample/save_combined_gif 全来自现有已验证脚本,不新造架构。
