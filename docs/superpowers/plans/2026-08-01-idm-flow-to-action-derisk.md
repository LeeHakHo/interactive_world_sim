# IDM (object-flow + agent-trace → Δjoint) Step 0 De-risk — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 训一个 inverse-dynamics 模型,从 `(object-flow + agent接触trace + grip)` 的双向时间窗预测 robot 动作 `Δjoint(6)+grip`,并用 offline 闭环(FK→前向②看 object-flow 复现)判它在 held-out robot demo 上站不站得住。

**Architecture:** VPT 式双向窗口 MLP:中心帧 t 取窗口 `[t-KP, t+FF]` 的输入通道(可 config 切,支持消融),回归中心帧 `Δjoint(6)+grip`。闭环判据 = 从 GT `joint_0` 积分 IDM 动作 → pinocchio FK 得 3 点 eef → mp 星座 → 前向 ②(mp robot WM)→ object-flow,和 GT 比(自由段/接触段分开)。

**Tech Stack:** numpy, torch(iws env 训练), pinocchio(phantom env FK), 复用 `exp_scel_dualview_wm`(mp_constellation / rollout_dual / DualLWC)与 `fk_skel2d.fk_skel2d_dual`。pytest 单测。

## Global Constraints

- 数据:`outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz`,键 `tracks`(2700,48,48,2)/`tracks_low`/`eef`(2700,48,3,2,float16)/`eef3d`/`joint`(2700,48,**7**)/`grip`(2700,48)/`vid`/`low_valid`/`fidx`/`vis`/`vis_low`。
- 动作监督:`Δjoint[t]=joint[t+1,:6]-joint[t,:6]`(6-dof 臂);grip 目标 = `grip[t]`(绝对开合,0-1 量级)。**不用第 7 维 joint 当动作**(它是 gripper,用 grip 键代替)。
- Split:episode-split,held-out `vid ∈ {100,102}`,其余训练;只用 `low_valid` 为 True 的 clip。**无帧泄漏**(和 ②/③ 一致)。
- 动作输出 **Δjoint 不出 Δeef**(避 6DoF IK 死路)。
- proprio = eef-pose,**由 agent-trace 携带,不喂 joint-angle 输入**(跨具身)。
- 输入集**可 config 切**(§ 消融:A0 只 flow / A1 +agent-trace / A2 +grip / A3 因果vs双向窗口)。
- 前向 ②(闭环评估用):`outputs/cross_embodiment_wm/epsplit_L48/mp_r_all/wm_dual.pt`(robot-only mp)。
- FK:`fk_skel2d.fk_skel2d_dual(joint7, grip)→(skel2d_high,skel2d_low,segs)`,9 关节 crop-norm 2D 双视角;eef 3 点索引 = `[5(link_6=腕),6(carriage_left),7(carriage_right)]`(Task 3 校验)。
- 窗口默认 `KP=4, FF=4`(IDM 只需少量未来帧看"效果";**不复用 ② 的 F=20**),边界 edge-repeat pad。
- 环境:训练 `iws`;闭环 eval `phantom`(需 pinocchio + torch,IK de-risk 已验此 env 有 torch)。
- 输出目录:`outputs/idm_derisk/<ARM>/`(每输入臂独立,别覆盖)。gif 传 Drive(`upload_evals_gdrive.sh`)。
- 不加 Co-Authored-By;只 `git add` 本计划新建的文件。

---

### Task 1: IDM 数据模块 `idm_data.py`

**Files:**
- Create: `idm_data.py`
- Test: `tests/test_idm_data.py`

**Interfaces:**
- Produces:
  - `INPUT_SPECS: dict[str, dict]` — 消融臂配置,如 `{"A1": {"flow": True, "trace": True, "grip": False, "future": True}, ...}`。
  - `build_windows(z, spec, KP=4, FF=4) -> (X, Y, meta)`:`X` float32 `(Nwin, Din)` 展平输入;`Y` float32 `(Nwin, 7)` = `[Δjoint(6), grip(1)]`;`meta` dict 含 `clip_idx`(Nwin,)、`t`(Nwin,)、`contact`(Nwin,) bool。`Din` 由 spec 决定(见下)。
  - `episode_split(z) -> (train_idx, ho_idx)`:clip 级 index,held-out vid∈{100,102} 且 low_valid。
  - `CONTACT_FN(z, ci, t) -> bool`:接触判据 = `grip[ci,t] < GRIP_OPEN_THRESH` 或该帧物体质心位移 > MOVE_THRESH(见常量)。

**输入通道构造(每中心帧 t,窗口 w∈[t-KP, t+FF],edge-repeat pad,W=KP+FF+1=9):**
- `flow`:两视角物体 track 相对中心帧质心的位移。view v: `tracks_v[ci, w] - tracks_v[ci, t].mean(0)` → `(W,48,2)`;两视角 concat → `(W,96,2)`。
- `trace`:两视角 mp 星座(用 `exp_scel_dualview_wm.mp_constellation` 从 `eef`/`eef_low` 3 点建,5 点/视角)→ `(W,10,2)`。**注:`eef` 是 (3,2) 只有 high 视角;low 视角用 `eef_low` 若存在,否则复用 high**(Task 内 assert 键)。
- `grip`(仅当 spec["grip"]):`grip[ci, w]` → `(W,1)`。
- `future`=False(A3 因果臂)时窗口改 `[t-KP, t]`(FF=0,W=KP+1=5),其余通道同构。
- 展平所有启用通道并 concat → `X[i]`。

**Constants(文件顶部):**
```python
GRIP_OPEN_THRESH = 0.6      # grip<0.6 视为夹爪在闭合/接触(grip 0-1, robot 张开≈1)
MOVE_THRESH = 0.01          # 物体质心相邻帧位移>0.01(crop-norm)视为在动=接触相关帧
HELDOUT_VIDS = (100, 102)
```

- [ ] **Step 1: 写失败测试** `tests/test_idm_data.py`

```python
import numpy as np, idm_data as D
Z = np.load("outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz")

def test_split_no_leak():
    tr, ho = D.episode_split(Z)
    vids = Z["vid"]
    assert set(vids[ho].tolist()) <= set(D.HELDOUT_VIDS)
    assert set(vids[tr].tolist()).isdisjoint(D.HELDOUT_VIDS)
    assert len(tr) > 0 and len(ho) > 0

def test_action_is_delta_joint():
    X, Y, meta = D.build_windows(Z, D.INPUT_SPECS["A1"])
    # 找一个非边界窗口, 验 Y[:6]==joint[t+1,:6]-joint[t,:6]
    i = np.where((meta["t"] > 5) & (meta["t"] < 40))[0][0]
    ci, t = meta["clip_idx"][i], meta["t"][i]
    exp = Z["joint"][ci, t+1, :6] - Z["joint"][ci, t, :6]
    assert np.allclose(Y[i, :6], exp, atol=1e-4)
    assert np.isclose(Y[i, 6], Z["grip"][ci, t], atol=1e-4)

def test_input_dim_matches_spec():
    for arm in ["A0", "A1", "A2"]:
        X, Y, meta = D.build_windows(Z, D.INPUT_SPECS[arm])
        assert X.ndim == 2 and X.shape[0] == Y.shape[0]
        assert X.shape[1] == D.input_dim(D.INPUT_SPECS[arm])  # 解析式维度一致
    # A0(只flow)维度 < A1(+trace) < A2(+grip)
    d0 = D.input_dim(D.INPUT_SPECS["A0"]); d1 = D.input_dim(D.INPUT_SPECS["A1"]); d2 = D.input_dim(D.INPUT_SPECS["A2"])
    assert d0 < d1 < d2

def test_causal_window_smaller():
    _, _, m_bi = D.build_windows(Z, D.INPUT_SPECS["A1"])        # future=True
    _, _, m_ca = D.build_windows(Z, D.INPUT_SPECS["A3"])        # future=False
    assert D.input_dim(D.INPUT_SPECS["A3"]) < D.input_dim(D.INPUT_SPECS["A1"])
```

- [ ] **Step 2: 跑测试确认失败** `conda run -n iws python -m pytest tests/test_idm_data.py -q` → FAIL(`No module named idm_data`)。

- [ ] **Step 3: 实现 `idm_data.py`**

```python
"""IDM 数据: 从 robot clips 建 (窗口输入, Δjoint+grip) 张量 + episode-split。输入集可 config 切(消融)。"""
import numpy as np, sys
sys.path.insert(0, ".")
from exp_scel_dualview_wm import mp_constellation
import torch

GRIP_OPEN_THRESH = 0.6
MOVE_THRESH = 0.01
HELDOUT_VIDS = (100, 102)

INPUT_SPECS = {
    "A0": {"flow": True,  "trace": False, "grip": False, "future": True},
    "A1": {"flow": True,  "trace": True,  "grip": False, "future": True},
    "A2": {"flow": True,  "trace": True,  "grip": True,  "future": True},
    "A3": {"flow": True,  "trace": True,  "grip": False, "future": False},  # causal 窗口
}

def _win(KP, FF, spec):
    return KP + (FF if spec["future"] else 0) + 1

def input_dim(spec, KP=4, FF=4):
    W = _win(KP, FF, spec); d = 0
    if spec["flow"]:  d += W * 96 * 2         # 双视角 48 track
    if spec["trace"]: d += W * 10 * 2         # 双视角 mp 星座 5 点
    if spec["grip"]:  d += W * 1
    return d

def episode_split(z):
    vid = z["vid"]; ok = z["low_valid"]
    idx = np.where(ok)[0]
    ho = np.array([i for i in idx if vid[i] in HELDOUT_VIDS])
    tr = np.array([i for i in idx if vid[i] not in HELDOUT_VIDS])
    return tr, ho

def _mp_trace(eef3):   # (L,3,2) -> (L,5,2) mp 星座, view0 常量
    t = torch.from_numpy(np.nan_to_num(eef3.astype(np.float32), nan=0.5))[None]  # (1,L,3,2)
    return mp_constellation(t, 0)[0].numpy()

def CONTACT_FN(z, ci, t):
    if z["grip"][ci, t] < GRIP_OPEN_THRESH: return True
    c = z["tracks"][ci].astype(np.float32).mean(1)   # (L,2) 物体质心
    if t + 1 < len(c) and np.linalg.norm(c[t+1] - c[t]) > MOVE_THRESH: return True
    return False

def build_windows(z, spec, KP=4, FF=4, clips=None):
    tracks = z["tracks"].astype(np.float32); tracks_lo = z["tracks_low"].astype(np.float32)
    eef = z["eef"].astype(np.float32); eef_lo = z["eef_low"].astype(np.float32) if "eef_low" in z.files else eef
    joint = z["joint"].astype(np.float32); grip = z["grip"].astype(np.float32)
    L = tracks.shape[1]; ff = FF if spec["future"] else 0
    clips = range(len(tracks)) if clips is None else clips
    Xs, Ys, ci_l, t_l, con_l = [], [], [], [], []
    for ci in clips:
        tr0 = _mp_trace(eef[ci]); tr1 = _mp_trace(eef_lo[ci])       # (L,5,2) each view
        oc0 = tracks[ci]; oc1 = tracks_lo[ci]
        for t in range(L - 1):                                     # 需 t+1 有 Δjoint
            widx = np.clip(np.arange(t - KP, t + ff + 1), 0, L - 1) # edge-repeat pad
            feats = []
            if spec["flow"]:
                f0 = oc0[widx] - oc0[t].mean(0); f1 = oc1[widx] - oc1[t].mean(0)
                feats.append(np.concatenate([f0, f1], 1).reshape(-1))
            if spec["trace"]:
                feats.append(np.concatenate([tr0[widx], tr1[widx]], 1).reshape(-1))
            if spec["grip"]:
                feats.append(grip[ci][widx].reshape(-1))
            Xs.append(np.concatenate(feats).astype(np.float32))
            Ys.append(np.concatenate([joint[ci, t+1, :6] - joint[ci, t, :6], [grip[ci, t]]]).astype(np.float32))
            ci_l.append(ci); t_l.append(t); con_l.append(CONTACT_FN(z, ci, t))
    meta = {"clip_idx": np.array(ci_l), "t": np.array(t_l), "contact": np.array(con_l, bool)}
    return np.stack(Xs), np.stack(Ys), meta
```

- [ ] **Step 4: 跑测试确认通过** `conda run -n iws python -m pytest tests/test_idm_data.py -q` → 4 passed。

- [ ] **Step 5: Commit** `git add idm_data.py tests/test_idm_data.py && git commit -m "feat(idm): 数据模块 build_windows+episode_split+输入spec(消融可切)"`

---

### Task 2: IDM 模型 `idm_model.py`

**Files:**
- Create: `idm_model.py`
- Test: `tests/test_idm_model.py`

**Interfaces:**
- Produces: `class IDM(torch.nn.Module)`,`__init__(din, hidden=512, dout=7)`;`forward(x)->(B,7)`。`predict_clip(model, z, spec, ci, KP, FF)->(L-1,7)`(对一个 clip 逐中心帧出 Δjoint+grip)。

- [ ] **Step 1: 写失败测试** `tests/test_idm_model.py`

```python
import torch, idm_model as M
def test_forward_shape():
    m = M.IDM(din=100); y = m(torch.randn(8, 100)); assert y.shape == (8, 7)
def test_overfits_one_batch():
    torch.manual_seed(0); m = M.IDM(din=32); opt = torch.optim.Adam(m.parameters(), 1e-3)
    x = torch.randn(16, 32); y = torch.randn(16, 7)
    for _ in range(400): opt.zero_grad(); l = ((m(x)-y)**2).mean(); l.backward(); opt.step()
    assert l.item() < 1e-3   # 能过拟合小 batch = 网络通
```

- [ ] **Step 2: 跑测试确认失败** → FAIL(`No module named idm_model`)。

- [ ] **Step 3: 实现 `idm_model.py`**

```python
"""IDM 网络: 展平双向窗口输入 -> Δjoint(6)+grip(1)。小 MLP。"""
import numpy as np, torch, torch.nn as nn
import idm_data as D

class IDM(nn.Module):
    def __init__(s, din, hidden=512, dout=7):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(din, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                              nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, dout))
    def forward(s, x): return s.net(x)

def predict_clip(model, z, spec, ci, KP=4, FF=4):
    X, _, _ = D.build_windows(z, spec, KP, FF, clips=[ci])
    with torch.no_grad():
        return model(torch.from_numpy(X).float()).numpy()   # (L-1,7)
```

- [ ] **Step 4: 跑测试确认通过** → 2 passed。

- [ ] **Step 5: Commit** `git add idm_model.py tests/test_idm_model.py && git commit -m "feat(idm): IDM MLP 网络+predict_clip"`

---

### Task 3: FK 工具 `idm_fk.py`(预测 joint → 3 点 eef,索引校验)

**Files:**
- Create: `idm_fk.py`
- Test: `tests/test_idm_fk.py`(**phantom env** 跑,需 pinocchio)

**Interfaces:**
- Produces: `EEF_SKEL_IDX = [5, 6, 7]`(link_6 腕 / carriage_left / carriage_right);`fk_eef2d(joint7, grip) -> (eef_high, eef_low)` 各 `(T,3,2)` crop-norm。

- [ ] **Step 1: 写失败测试** `tests/test_idm_fk.py`(校验索引正确 = FK(GT joint) 的 3 点 ≈ GT eef)

```python
import numpy as np, idm_fk as FK
Z = np.load("outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz")
def test_fk_gt_joint_matches_gt_eef():
    ci = 0
    j = np.concatenate([Z["joint"][ci], np.zeros((Z["joint"].shape[1],0))], 1)  # 已 7 维
    eh, el = FK.fk_eef2d(Z["joint"][ci], Z["grip"][ci])   # (48,3,2)
    gt = np.nan_to_num(Z["eef"][ci].astype(np.float32))    # GT eef high (48,3,2)
    # 有效帧上, FK 出的腕/两指 vs GT eef 3 点像素误差应小(证 EEF_SKEL_IDX 对)
    ok = np.all(np.abs(gt) < 3, axis=(-1,-2))
    err = np.linalg.norm(eh[ok] - gt[ok], axis=-1).mean() * 128
    assert err < 8.0, f"FK-eef vs GT-eef 误差 {err:.1f}px 太大 → EEF_SKEL_IDX 可能不对"
```

- [ ] **Step 2: 跑测试确认失败** `conda run -n phantom python -m pytest tests/test_idm_fk.py -q` → FAIL(`No module named idm_fk`)。

- [ ] **Step 3: 实现 `idm_fk.py`**

```python
"""预测 joint -> 3 点 eef(2D 双视角)via fk_skel2d_dual。EEF_SKEL_IDX 由测试校验。"""
import numpy as np
from fk_skel2d import fk_skel2d_dual
EEF_SKEL_IDX = [5, 6, 7]   # link_6(腕), carriage_left, carriage_right

def fk_eef2d(joint7, grip):
    """joint7 (T,7), grip (T,) -> (eef_high, eef_low) 各 (T,3,2) crop-norm。"""
    j = np.asarray(joint7, np.float64)
    if j.shape[-1] == 6:
        j = np.concatenate([j, np.zeros((*j.shape[:-1], 1))], -1)
    sh, sl, _ = fk_skel2d_dual(j, np.asarray(grip, np.float64))   # (T,9,2) x2
    return sh[:, EEF_SKEL_IDX], sl[:, EEF_SKEL_IDX]
```

- [ ] **Step 4: 跑测试确认通过** `conda run -n phantom python -m pytest tests/test_idm_fk.py -q` → 1 passed。**若 err≥8px:改 `EEF_SKEL_IDX`(试 `[8,6,7]` ee_gripper 当腕)重跑,直到 <8px。**

- [ ] **Step 5: Commit** `git add idm_fk.py tests/test_idm_fk.py && git commit -m "feat(idm): FK预测joint->3点eef, EEF_SKEL_IDX测试校验vs GT eef"`

---

### Task 4: 训练脚本 `idm_train.py`

**Files:**
- Create: `idm_train.py`
- Create: `tests/test_idm_train_smoke.py`

**Interfaces:**
- Consumes: `idm_data.{build_windows,episode_split,INPUT_SPECS,input_dim}`,`idm_model.IDM`。
- Produces: ckpt `outputs/idm_derisk/<ARM>/idm.pt`(含 `state`,`spec`,`KP`,`FF`,`x_mean`,`x_std`,`y_mean`,`y_std`),`train_summary.txt`。env:`ARM`(默认 A1),`EPOCHS`(默认 200),`OUT_DIR`。

- [ ] **Step 1: 写 smoke 测试** `tests/test_idm_train_smoke.py`

```python
import os, subprocess, numpy as np
def test_train_smoke(tmp_path):
    out = str(tmp_path / "A1")
    env = {**os.environ, "ARM": "A1", "EPOCHS": "3", "SMOKE": "1", "OUT_DIR": out}
    r = subprocess.run(["python", "idm_train.py"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    import torch; ck = torch.load(f"{out}/idm.pt", map_location="cpu", weights_only=False)
    assert ck["spec"]["trace"] is True and "state" in ck
```

- [ ] **Step 2: 跑测试确认失败** `conda run -n iws python -m pytest tests/test_idm_train_smoke.py -q` → FAIL。

- [ ] **Step 3: 实现 `idm_train.py`**

```python
"""IDM 训练: supervised regression on robot Δjoint+grip。输入臂由 ARM env 切(消融)。"""
import os, numpy as np, torch
import idm_data as D, idm_model as M
ARM = os.environ.get("ARM", "A1"); EPOCHS = int(os.environ.get("EPOCHS", "200"))
SMOKE = os.environ.get("SMOKE") == "1"
OUT = os.environ.get("OUT_DIR", f"outputs/idm_derisk/{ARM}"); os.makedirs(OUT, exist_ok=True)
KP, FF = 4, 4
z = np.load("outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz")
spec = D.INPUT_SPECS[ARM]
tr, ho = D.episode_split(z)
if SMOKE: tr = tr[:20]
X, Y, meta = D.build_windows(z, spec, KP, FF, clips=tr)
xm, xs = X.mean(0), X.std(0) + 1e-6; ym, ys = Y.mean(0), Y.std(0) + 1e-6
Xn = torch.tensor((X - xm) / xs); Yn = torch.tensor((Y - ym) / ys)
m = M.IDM(din=X.shape[1]); opt = torch.optim.Adam(m.parameters(), 1e-3)
bs = 4096; n = len(Xn)
for ep in range(EPOCHS):
    perm = torch.randperm(n); tot = 0.
    for i in range(0, n, bs):
        idx = perm[i:i+bs]; opt.zero_grad()
        loss = torch.nn.functional.smooth_l1_loss(m(Xn[idx]), Yn[idx]); loss.backward(); opt.step()
        tot += loss.item() * len(idx)
    if ep % 50 == 0 or ep == EPOCHS-1: print(f"ep{ep} loss {tot/n:.4f}", flush=True)
torch.save({"state": m.state_dict(), "spec": spec, "KP": KP, "FF": FF,
            "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys, "din": X.shape[1]}, f"{OUT}/idm.pt")
open(f"{OUT}/train_summary.txt", "w").write(f"ARM={ARM} din={X.shape[1]} nwin_train={n} epochs={EPOCHS}\n")
print(f"[ckpt] {OUT}/idm.pt", flush=True)
```

- [ ] **Step 4: 跑测试确认通过** → 1 passed。

- [ ] **Step 5: Commit** `git add idm_train.py tests/test_idm_train_smoke.py && git commit -m "feat(idm): 训练脚本(ARM env切输入臂, 标准化, Huber)"`

---

### Task 5: 闭环评估 `idm_eval.py`

**Files:**
- Create: `idm_eval.py`
- Create: `tests/test_idm_eval_smoke.py`

**Interfaces:**
- Consumes: `idm_model.{IDM,predict_clip}`,`idm_fk.fk_eef2d`,`idm_data`,`exp_scel_dualview_wm.{rollout_dual,load_action_tokens,mp_constellation,DualLWC}`。
- Produces: `outputs/idm_derisk/<ARM>/eval_summary.txt`(eef-recon mm/px + object-flow-recon px,自由/接触分开)+ overlay gif `.../eval_overlay.gif`。env:`ARM`,`OUT_DIR`,`SEQS`(默认 held-out 前若干 clip)。
- **闭环**:IDM 出 Δjoint → 从 GT `joint_0` 积分得 joint 轨迹 → `fk_eef2d` 得 3 点 eef(双视角)→ `mp_constellation` 建 mp 星座 → `rollout_dual`(mp_r_all)得预测 object-flow → 对 GT object-flow 算 ADE。同时 FK-eef vs GT eef 算 recon。**天花板列** = 用 GT joint 走同一 FK→mp→② 管线的误差。

- [ ] **Step 1: 写 smoke 测试** `tests/test_idm_eval_smoke.py`

```python
import os, subprocess
def test_eval_smoke(tmp_path):
    # 依赖 Task4 已产出 A1 ckpt(smoke 训练); 这里用真 ckpt 路径若存在, 否则跳过
    ck = "outputs/idm_derisk/A1/idm.pt"
    if not os.path.exists(ck): import pytest; pytest.skip("需先训 A1")
    env = {**os.environ, "ARM": "A1", "SEQS": "0", "OUT_DIR": "outputs/idm_derisk/A1"}
    r = subprocess.run(["python", "idm_eval.py"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert os.path.exists("outputs/idm_derisk/A1/eval_summary.txt")
```

- [ ] **Step 2: 跑测试确认失败(或 skip)** `conda run -n phantom python -m pytest tests/test_idm_eval_smoke.py -q`。

- [ ] **Step 3: 实现 `idm_eval.py`**(phantom env:torch+pinocchio)

```python
"""IDM 闭环评估: IDM出Δjoint→积分→FK→mp星座→前向②→object-flow, 对GT比(自由/接触分开)+overlay gif。"""
import os, sys, numpy as np, torch, cv2, imageio
sys.path.insert(0, "."); os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ["ACTION"] = "mp"
import exp_scel_dualview_wm as W
import idm_data as D, idm_model as M, idm_fk as FK
setattr(sys.modules["__main__"], "DualLWC", W.DualLWC)
ARM = os.environ.get("ARM", "A1"); OUT = os.environ.get("OUT_DIR", f"outputs/idm_derisk/{ARM}")
K, P = W.K, 48
z = np.load("outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz")
ck = torch.load(f"{OUT}/idm.pt", map_location="cpu", weights_only=False)
m = M.IDM(din=ck["din"]); m.load_state_dict(ck["state"]); m.eval()
spec, KP, FF = ck["spec"], ck["KP"], ck["FF"]
xm, xs, ym, ys = ck["x_mean"], ck["x_std"], ck["y_mean"], ck["y_std"]
wm2 = torch.load("outputs/cross_embodiment_wm/epsplit_L48/mp_r_all/wm_dual.pt", map_location="cpu", weights_only=False).eval()
tr, ho = D.episode_split(z)
SEQS = [int(x) for x in os.environ.get("SEQS", ",".join(map(str, ho[:4]))).split(",")]

def idm_joint_traj(ci):
    X, _, _ = D.build_windows(z, spec, KP, FF, clips=[ci])
    with torch.no_grad():
        Yp = m(torch.tensor((X - xm) / xs).float()).numpy() * ys + ym   # (L-1,7)
    dj, gp = Yp[:, :6], Yp[:, 6]
    jt = np.zeros((len(dj)+1, 7)); jt[0, :6] = z["joint"][ci, 0, :6]     # 从GT joint_0积分
    for t in range(len(dj)): jt[t+1, :6] = jt[t, :6] + dj[t]
    grip = np.concatenate([gp, gp[-1:]])
    return jt, grip

def obj_flow_from_eef(eh, el):   # (L,3,2)x2 -> 预测 object-flow via mp②
    a0 = W.mp_constellation(torch.from_numpy(eh[None]).float(), 0)  # 走 _act_pts? 用 load_action_tokens 口径
    trg = [np.nan_to_num(z["tracks"][ci].astype(np.float32)), np.nan_to_num(z["tracks_low"][ci].astype(np.float32))]
    # eef 三点 -> mp action tokens(与训练口径一致: eef+grip 第4槽)
    efA = np.concatenate([eh, np.stack([np.stack([np.linalg.norm(eh[:,1]-eh[:,2],-1)/ (np.linalg.norm(eh[:,1]-eh[:,2],-1).max()+1e-6), np.zeros(len(eh))],-1)[:,None]],1)], 1)
    efB = np.concatenate([el, np.stack([np.stack([np.linalg.norm(el[:,1]-el[:,2],-1)/(np.linalg.norm(el[:,1]-el[:,2],-1).max()+1e-6), np.zeros(len(el))],-1)[:,None]],1)], 1)
    trD = np.concatenate([trg[0][:K], trg[1][:K]], 1)[None]
    with torch.no_grad():
        pr = W.rollout_dual(wm2, torch.from_numpy(trD).float(), torch.from_numpy(efA[None]).float(),
                            torch.from_numpy(efB[None]).float(), 48-K)[0].numpy()
    return np.concatenate([trg[0][:K], pr[:, :P]], 0)   # (48,48,2) cam_high 预测物体

summ = [f"IDM 闭环 ARM={ARM} (eef-recon px@128 / object-flow-recon px, 自由/接触分开)\n"]
frames_all = []
for ci in SEQS:
    jt, grip = idm_joint_traj(ci)
    eh, el = FK.fk_eef2d(jt, grip)                              # 预测 eef(48,3,2)
    gh = np.nan_to_num(z["eef"][ci].astype(np.float32))         # GT eef
    _, _, meta = D.build_windows(z, spec, KP, FF, clips=[ci]); con = np.append(meta["contact"], meta["contact"][-1])
    okp = np.all(np.abs(gh) < 3, axis=(-1,-2))
    eef_err = np.linalg.norm(eh - gh, axis=-1).mean(-1) * 128   # (48,) 每帧
    free_e = np.nanmean(eef_err[okp & ~con]); con_e = np.nanmean(eef_err[okp & con])
    pred_obj = obj_flow_from_eef(eh, el); gt_obj = np.nan_to_num(z["tracks"][ci].astype(np.float32))
    flow_err = np.linalg.norm(pred_obj - gt_obj, axis=-1).mean(-1) * 128
    ff, cf = np.nanmean(flow_err[K:][~con[K:]]), np.nanmean(flow_err[K:][con[K:]])
    summ.append(f"seq{ci}: eef-recon 自由 {free_e:.1f} 接触 {con_e:.1f} | object-flow-recon 自由 {ff:.1f} 接触 {cf:.1f}")
open(f"{OUT}/eval_summary.txt", "w").write("\n".join(summ) + "\n")
print("\n".join(summ), flush=True)
```

  **注(实现者)**:overlay gif 用 `viz_combined.save_combined_gif` 标准 protocol(Flow 行:绿 GT object / 红 pred object / 黄 eef;列 = `GT-action天花板 | IDM`)——**别手搓**(见 memory `feedback_gif_eval_layout`)。天花板列 = 用 GT joint 走同一 FK→mp→② 管线。若 `obj_flow_from_eef` 的 mp token 口径与 `load_action_tokens("mp")` 不符,以后者为准(直接调它)。

- [ ] **Step 4: 跑 smoke 确认通过 + 眼检数值合理**(接触段 flow-recon 应接近天花板;自由段可能大)。

- [ ] **Step 5: Commit** `git add idm_eval.py tests/test_idm_eval_smoke.py && git commit -m "feat(idm): 闭环评估 积分+FK+前向②object-flow复现(自由/接触分开)+标准protocol gif"`

---

### Task 6: 输入消融 + 判决 `sbatch_idm.sbatch`

**Files:**
- Create: `sbatch_idm.sbatch`
- Create: `idm_ablation_table.py`

**Interfaces:**
- Consumes: `idm_train.py`,`idm_eval.py` 的 `eval_summary.txt`。
- Produces: `outputs/idm_derisk/ABLATION.txt`(A0/A1/A3[+A2] × eef-recon/flow-recon,自由/接触),Drive 上 gif。

- [ ] **Step 1: 写 `sbatch_idm.sbatch`**(核心臂 A0/A1/A3 必跑;A2 可选)

```bash
#!/bin/bash
#SBATCH --job-name=idm
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/idm_derisk/logs/idm_%j.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem=48G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
cd /scr2/yusenluo/interactive_world_sim; mkdir -p outputs/idm_derisk/logs
for ARM in A0 A1 A3; do
  echo "=== 训练 $ARM ==="; ARM=$ARM EPOCHS=200 OUT_DIR=outputs/idm_derisk/$ARM conda run -n iws python idm_train.py
  echo "=== 评估 $ARM ==="; ARM=$ARM OUT_DIR=outputs/idm_derisk/$ARM conda run -n phantom python idm_eval.py
done
conda run -n iws python idm_ablation_table.py
echo "=== DONE ==="
```

- [ ] **Step 2: 写 `idm_ablation_table.py`**(汇总各臂 eval_summary → 一张表)

```python
"""汇总 A0/A1/A3(+A2) 的 eval_summary -> ABLATION.txt, 挑最小够用输入集。"""
import os, glob
rows = []
for arm in ["A0", "A1", "A2", "A3"]:
    f = f"outputs/idm_derisk/{arm}/eval_summary.txt"
    if os.path.exists(f):
        lines = [l for l in open(f).read().splitlines() if l.startswith("seq")]
        rows.append(f"[{arm}]\n" + "\n".join("  " + l for l in lines))
open("outputs/idm_derisk/ABLATION.txt", "w").write("\n\n".join(rows) + "\n")
print(open("outputs/idm_derisk/ABLATION.txt").read())
```

- [ ] **Step 3: 提交 sbatch 跑全量** `sbatch sbatch_idm.sbatch`;完成后 `cat outputs/idm_derisk/ABLATION.txt`。

- [ ] **Step 4: 判决 + 传 Drive** `bash upload_evals_gdrive.sh outputs/idm_derisk 2026-08-01_idm_derisk`。**读表下判据**:A0 自由段应崩(证病态);A1 是否救回自由段;A3(因果)vs A1(双向)看未来窗口价值;若接触段 flow-recon 接近天花板 → PASS 进 Step 1。把判决写入 `outputs/idm_derisk/VERDICT.txt`。

- [ ] **Step 5: Commit** `git add sbatch_idm.sbatch idm_ablation_table.py && git commit -m "feat(idm): 消融sbatch(A0/A1/A3)+汇总表+判决"`

---

## Self-Review notes(作者已核)

- **Spec 覆盖**:§3 数据流→Task1;§4 网络→Task2;§4.5 消融→Task1(specs)+Task6;§5 闭环判据→Task5;FK→Task3。全覆盖。
- **类型一致**:`build_windows` 返回 `(X,Y,meta)` 三处一致;`Y` 恒 `[Δjoint(6),grip(1)]`=7 维,IDM `dout=7` 一致;`fk_eef2d` 返回 `(T,3,2)×2` 与 eval 用法一致;ckpt 键 `state/spec/KP/FF/x_mean/x_std/y_mean/y_std/din` 训练存↔eval读一致。
- **已知留给实现者的经验点(非 placeholder,是校验步)**:Task3 `EEF_SKEL_IDX` 用测试校验(<8px)、Task5 mp token 口径以 `load_action_tokens("mp")` 为准、CONTACT_FN 阈值可按接触段占比微调。
