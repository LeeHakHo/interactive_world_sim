# 结合 skel+world action featurization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 A1(加性 residual)/align(LaST-HD 对齐)/warm(两阶段课程)三个结合机制,在 held-out robot ADE 上同时拿到 human-helps(Δ>world 的 +0.68)与最终精度(rh≤world 的 8.93),突破 velocity_action 线的 Pareto 前沿。

**Architecture:** 新 `CombLWC(FlowWM_LWC)` 双 action 头(`act_shared`=skel 同域主体 / `act_world`=world 绝对残差),`forward` 收可选 `is_h` mask 让 human 样本只走 shared;评价代码零改动(缺省全 robot,world 残差全开)。三机制共用一个 `train_comb` SS trainer,`combine` 参数切换。协议(数据/split/ade_world)逐字复用现有权威实现。

**Tech Stack:** PyTorch;iws conda env `/scr/yusenluo/anaconda3/envs/iws/bin/python`(numpy 1.26);复用 `amplify_wm`(FlowWM_LWC/rollout_lwc/vel_to_class)、`exp_scel_velocity_action`(ACT_DIM/协议)、`exp_scel_agentframe`(DS/HELDOUT/ade_world)、`exp_scel_latent_detmem`(③ 渲染)。

## Global Constraints

- 协议严格复用 `exp_scel_agentframe`:数据 `flow_render_dataset_v3`;HELDOUT=150;held-out robot ADE px@224(H=16,`X.ade_world`);ro(N_rob robot)vs rh(+1800 human);Δ=ro−rh;N∈{20,50,100,400}×seeds{0,1,2}。**不新造评价口径。**
- SS trainer 超参逐字复用:`SSm.WM_EPOCHS/WM_BS/WM_LR/R_SS`,teacher prob 1.0→0.3 退火,`VEL_HALF=0.12`,`W=15`,`Dm=384`,`layers=3`。
- **不改动已验证代码**:`exp_scel_velocity_action.py` / `amplify_wm.py` / `exp_scel_agentframe.py` / `exp_scel_latent_detmem.py` 只 import 复用,不修改。
- `is_h` 判定:batch 内 `b >= Nr` 为 human(`Nr=len(r_tr)`,human clips 拼在 robot 后)。
- **A1 防残差独吞是成败手**:`act_world` 小初始化(weight ×0.1, bias 0)+ α=0.5 + `λ_res` L2 惩罚;消融 λ_res∈{0,1e-3,1e-2}。
- 运行环境:iws env;跑前 `nvidia-smi` 选空 GPU 用 `CUDA_VISIBLE_DEVICES`。
- git:就地工作,只 `git add` 本计划新建/修改的文件,commit 不加 Co-Authored-By。
- 输出目录 `outputs/cross_embodiment_wm/combine_action/`;落盘 `COMBINE_ACTION_LOG.md`;重要可视化上传 drive(`upload_evals_gdrive.sh`)。

---

### Task 1: CombLWC 双头模型 + is_h domain-mask(A1/align 前向)

**Files:**
- Create: `exp_scel_combine_action.py`
- Test: `tests/test_combine_action.py`

**Interfaces:**
- Consumes: `amplify_wm.FlowWM_LWC`(`.inp/.act/.tf/.head/.Dm/.W/.vel_half/.expected_vel`),`K,F,device`;`exp_scel_velocity_action.ACT_DIM`(`"skel"`=Lw·10,`"world"`=Lw·6),`.VEL_HALF`。
- Produces:
  - `feat_world(eef3, objc) -> (B, Lw*6)`,`feat_skel(eef3, objc) -> (B, Lw*10)`(纯函数)。
  - `CombLWC(P, combine="a1", alpha=0.5, lam_res=1e-3, lam_align=0.3, **kw)`;属性 `.combine/.alpha/.lam_res/.lam_align/.warm_stage/.aux`(dict)。
  - `CombLWC.trunk(hist, eef3, is_h=None) -> (x, anchor)`;`CombLWC.forward(hist, eef3, is_h=None) -> (logits (B,P,F,W*W), anchor (B,P,2))`。副作用:填 `s.aux["res_sq"]`(a1/warm)或 `s.aux["align"]`(align)。

- [ ] **Step 1: 写失败测试**

`tests/test_combine_action.py`:
```python
import torch, numpy as np
import exp_scel_combine_action as C
from amplify_wm import K, F, device

def _dummy(B=8, P=48):
    hist = torch.randn(B, P, K, 2, device=device) * 0.05 + 0.5
    eef3 = torch.randn(B, K + F, 3, 2, device=device) * 0.05 + 0.5
    return hist, eef3

def test_forward_shape():
    hist, eef3 = _dummy()
    m = C.CombLWC(48, combine="a1").to(device)
    logits, anchor = m(hist, eef3)
    assert logits.shape == (8, 48, F, m.W * m.W)
    assert anchor.shape == (8, 48, 2)

def test_human_only_shared_a1():
    # human 样本 (is_h=True) 的 world 残差必须被 mask 成 0 -> 与纯 shared 前向一致
    hist, eef3 = _dummy(B=4)
    m = C.CombLWC(48, combine="a1", alpha=0.5).to(device).eval()
    is_h = torch.tensor([False, False, True, True], device=device)
    with torch.no_grad():
        m(hist, eef3, is_h)
        res = m.aux["res_sq"]  # residual^2 mean over ALL samples; human rows contribute 0
        # 直接核验:强制全 human -> 残差全 0
        m(hist, eef3, torch.ones(4, dtype=torch.bool, device=device))
        assert m.aux["res_sq"].item() < 1e-12
        # 全 robot -> 残差非 0
        m(hist, eef3, torch.zeros(4, dtype=torch.bool, device=device))
        assert m.aux["res_sq"].item() > 0

def test_default_is_robot():
    # 评价路径 model(hist,eef3) 不传 is_h -> 视作全 robot(world 残差全开)
    hist, eef3 = _dummy(B=4)
    m = C.CombLWC(48, combine="a1").to(device).eval()
    with torch.no_grad():
        m(hist, eef3)
        assert m.aux["res_sq"].item() > 0

def test_align_produces_loss():
    hist, eef3 = _dummy(B=4)
    m = C.CombLWC(48, combine="align").to(device)
    is_h = torch.tensor([False, False, True, True], device=device)
    m(hist, eef3, is_h)
    assert torch.isfinite(m.aux["align"]) and m.aux["align"].item() >= 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/scr/yusenluo/anaconda3/envs/iws/bin/python -m pytest tests/test_combine_action.py -x -q`
Expected: FAIL(`ModuleNotFoundError: exp_scel_combine_action` 或 `AttributeError`)。

- [ ] **Step 3: 写 `exp_scel_combine_action.py`(模型部分)**

```python
"""结合 skel(同域主体,human 帮) + world(绝对残差,robot 精度)的 ② 世界模型。
突破 velocity_action 线的 Pareto(单一 featurization 不可两全)。机制:A1 加性 residual /
align(LaST-HD 对齐) / warm(两阶段课程)。精度来源纯 robot,human 只从 shared 通道帮进来。
协议逐字复用 exp_scel_agentframe;SS trainer 复用 amplify_wm/velocity 超参。见 spec
docs/superpowers/specs/2026-07-12-combine-skel-world-action-design.md。
Output: outputs/cross_embodiment_wm/combine_action/"""
import os, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import amplify_wm as A
import eval_scheduled_sampling as SSm
import exp_scel_velocity_action as V
import exp_scel_agentframe as X
from amplify_wm import K, F, vel_to_class, device

Lw = K + F


def feat_world(eef3, objc):                                   # (B, Lw*6) 绝对星座(接触几何,disjoint)
    B = eef3.shape[0]
    return (eef3 - objc[:, :, None]).reshape(B, -1)


def feat_skel(eef3, objc):                                    # (B, Lw*10) OSCAR 骨架(同域)
    B = eef3.shape[0]
    wrist = eef3[:, :, 0]
    dwr = torch.cat([torch.zeros_like(wrist[:, :1]), wrist[:, 1:] - wrist[:, :-1]], 1)
    b1 = eef3[:, :, 1] - wrist; b2 = eef3[:, :, 2] - wrist
    db1 = torch.cat([torch.zeros_like(b1[:, :1]), b1[:, 1:] - b1[:, :-1]], 1)
    db2 = torch.cat([torch.zeros_like(b2[:, :1]), b2[:, 1:] - b2[:, :-1]], 1)
    return torch.cat([dwr, b1, b2, db1, db2], -1).reshape(B, -1)


class CombLWC(A.FlowWM_LWC):
    """双 action 头:act_shared(skel 主体) + act_world(world 残差)。forward 收可选 is_h;
    human 样本只走 shared(a1/warm world 残差置 0)。评价路径不传 is_h -> 全 robot,残差全开。"""
    def __init__(s, P, combine="a1", alpha=0.5, lam_res=1e-3, lam_align=0.3, **kw):
        super().__init__(P, **kw)
        s.combine, s.alpha, s.lam_res, s.lam_align = combine, alpha, lam_res, lam_align
        s.act_shared = nn.Linear(V.ACT_DIM["skel"], s.Dm)
        s.act_world = nn.Linear(V.ACT_DIM["world"], s.Dm)
        with torch.no_grad():                                 # 小初始化残差,防独吞
            s.act_world.weight.mul_(0.1); s.act_world.bias.zero_()
        s.warm_stage = 1
        s.aux = {}

    def _obj_tokens(s, hist):
        B, P = hist.shape[:2]
        anchor = hist[:, :, -1, :]
        obj = s.inp(torch.cat([(hist - anchor[:, :, None]).reshape(B, P, 2 * K), anchor], -1))
        objc = anchor.mean(1, keepdim=True)
        return obj, anchor, objc

    def trunk(s, hist, eef3, is_h=None):
        B, P = hist.shape[:2]
        obj, anchor, objc = s._obj_tokens(hist)
        if is_h is None:
            is_h = torch.zeros(B, dtype=torch.bool, device=hist.device)
        act_s = s.act_shared(feat_skel(eef3, objc))           # (B,Dm) 主体
        act_w = s.act_world(feat_world(eef3, objc))           # (B,Dm) 残差
        if s.combine in ("a1", "warm"):
            use_w = (~is_h).float()[:, None]                  # human -> 0
            if s.combine == "warm" and s.warm_stage == 1:
                use_w = use_w * 0.0                           # stage1 纯 shared
            residual = s.alpha * act_w * use_w
            s.aux["res_sq"] = residual.pow(2).mean()
            act = (act_s + residual)[:, None, :]
            x = s.tf(torch.cat([obj, act], 1))[:, :P]
            return x, anchor
        if s.combine == "align":
            x_s = s.tf(torch.cat([obj, act_s[:, None]], 1))[:, :P]
            x_w = s.tf(torch.cat([obj, act_w[:, None]], 1))[:, :P]
            robot = (~is_h).float()                            # (B,) align 只在 robot
            per = ((x_w - x_s.detach()) ** 2).mean((1, 2))     # (B,)
            s.aux["align"] = (per * robot).sum() / (robot.sum() + 1e-6)
            x = torch.where(is_h[:, None, None], x_s, x_w)      # robot 走 world(精度),human 走 shared
            return x, anchor
        raise ValueError(s.combine)

    def forward(s, hist, eef3, is_h=None):
        x, anchor = s.trunk(hist, eef3, is_h)
        B, P = x.shape[:2]
        logits = s.head(x).reshape(B, P, F, s.W * s.W)
        return logits, anchor
```

- [ ] **Step 4: 运行测试确认通过**

Run: `/scr/yusenluo/anaconda3/envs/iws/bin/python -m pytest tests/test_combine_action.py -x -q`
Expected: 4 passed。若无 GPU 空闲,先 `CUDA_VISIBLE_DEVICES=<free>`。

- [ ] **Step 5: Commit**

```bash
git add exp_scel_combine_action.py tests/test_combine_action.py
git commit -m "feat(combine): CombLWC dual-head (shared skel + world residual) with is_h domain mask"
```

---

### Task 2: train_comb SS trainer(A1/align/warm)+ SMOKE 跑通

**Files:**
- Modify: `exp_scel_combine_action.py`(追加 `train_comb`)
- Test: `tests/test_combine_action.py`(追加训练 smoke 测试)

**Interfaces:**
- Consumes: Task 1 的 `CombLWC`;`SSm.WM_EPOCHS/WM_BS/WM_LR/R_SS`;`V.VEL_HALF`。
- Produces: `train_comb(tracks, vis, eef, idx, combine, Nr, seed=0, alpha=0.5, lam_res=1e-3, lam_align=0.3) -> CombLWC(eval)`。`idx` 为 torch.LongTensor 样本索引;`Nr` 用于判 is_h。warm 内部两阶段(stage1 rh backbone / stage2 冻 backbone 训 world 残差 robot-only)。

- [ ] **Step 1: 写训练 smoke 测试**

追加到 `tests/test_combine_action.py`:
```python
def test_train_comb_smoke():
    import eval_scheduled_sampling as SSm
    SSm.WM_EPOCHS, SSm.R_SS, SSm.WM_BS = 2, 4, 16
    import exp_scel_combine_action as C
    rng = np.random.default_rng(0)
    Nr = 20
    tr = (rng.random((30, 24, 48, 2)).astype(np.float32))
    vs = np.ones((30, 24, 48), np.float32)
    ef = (rng.random((30, 24, 3, 2)).astype(np.float32))
    idx = torch.arange(30)  # 0..19 robot, 20..29 human
    for comb in ["a1", "align", "warm"]:
        m = C.train_comb(tr, vs, ef, idx, comb, Nr, seed=0)
        logits, anchor = m(torch.from_numpy(tr[:4, :K]).permute(0, 2, 1, 3).float().to(device),
                           torch.from_numpy(ef[:4]).float().to(device))
        assert torch.isfinite(logits).all()
```

- [ ] **Step 2: 运行确认失败**

Run: `/scr/yusenluo/anaconda3/envs/iws/bin/python -m pytest tests/test_combine_action.py::test_train_comb_smoke -x -q`
Expected: FAIL(`AttributeError: train_comb`)。

- [ ] **Step 3: 追加 `train_comb`**

```python
def _run_stage(m, tracks, vis, eef, idx, Nr, stage, epochs, seed):
    m.warm_stage = stage
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=SSm.WM_LR)
    g = torch.Generator().manual_seed(seed)
    tr = torch.from_numpy(tracks).float(); vs = torch.from_numpy(vis).float(); ef = torch.from_numpy(eef).float()
    for ep in range(epochs):
        p = 1.0 + (0.3 - 1.0) * ep / max(epochs - 1, 1)
        m.train(); pe = idx[torch.randperm(len(idx), generator=g)]
        for i in range(0, len(pe), SSm.WM_BS):
            b = pe[i:i + SSm.WM_BS]
            is_h = (b >= Nr).to(device)
            G = tr[b].to(device); Vv = vs[b].to(device); Ef = ef[b].to(device)
            buf = G[:, :K].clone(); losses = []; regs = []
            for h in range(SSm.R_SS):
                logits, anchor = m(buf[:, -K:].permute(0, 2, 1, 3), Ef, is_h)
                lg0 = logits[:, :, 0, :]
                gt_vel = G[:, K + h] - buf[:, -1]
                cls = vel_to_class(gt_vel, m.W, m.vel_half)
                w = (Vv[:, K + h] * Vv[:, K - 1])
                ce = nn.functional.cross_entropy(lg0.reshape(-1, m.W * m.W), cls.reshape(-1), reduction="none")
                losses.append((ce * w.reshape(-1)).sum() / (w.sum() + 1e-6))
                if m.combine in ("a1", "warm"): regs.append(m.aux["res_sq"] * m.lam_res)
                elif m.combine == "align": regs.append(m.aux["align"] * m.lam_align)
                nxt = buf[:, -1] + m.expected_vel(lg0)
                use_gt = (torch.rand(len(b), 1, 1, device=device) < p)
                buf = torch.cat([buf, torch.where(use_gt, G[:, K + h], nxt.detach())[:, None]], 1)
            loss = torch.stack(losses).mean() + torch.stack(regs).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def train_comb(tracks, vis, eef, idx, combine, Nr, seed=0, alpha=0.5, lam_res=1e-3, lam_align=0.3):
    torch.manual_seed(seed); P = tracks.shape[2]
    m = CombLWC(P, combine=combine, alpha=alpha, lam_res=lam_res, lam_align=lam_align,
               Dm=384, layers=3, W=15, vel_half=V.VEL_HALF).to(device)
    if combine == "warm":
        _run_stage(m, tracks, vis, eef, idx, Nr, 1, SSm.WM_EPOCHS, seed)   # stage1: shared, robot+human
        for pm in list(m.inp.parameters()) + list(m.tf.parameters()) + list(m.act_shared.parameters()):
            pm.requires_grad_(False)                                       # freeze backbone
        r_idx = idx[idx < Nr]                                             # robot-only for精度
        _run_stage(m, tracks, vis, eef, r_idx, Nr, 2, SSm.WM_EPOCHS, seed)
    else:
        _run_stage(m, tracks, vis, eef, idx, Nr, 1, SSm.WM_EPOCHS, seed)
    return m.eval()
```

- [ ] **Step 4: 运行确认通过**

Run: `CUDA_VISIBLE_DEVICES=<free> /scr/yusenluo/anaconda3/envs/iws/bin/python -m pytest tests/test_combine_action.py -x -q`
Expected: 5 passed。

- [ ] **Step 5: Commit**

```bash
git add exp_scel_combine_action.py tests/test_combine_action.py
git commit -m "feat(combine): train_comb SS trainer for a1/align/warm (+residual/align reg, warm 2-stage)"
```

---

### Task 3: main sweep + 保存/绘图 + A1 full run

**Files:**
- Modify: `exp_scel_combine_action.py`(追加 `main`/`_save`/`_plot`)

**Interfaces:**
- Consumes: Task 2 的 `train_comb`;`V.train_feat`(锚 world/skel);`X.ade_world`,`X.DS`,`X.H`,`X.tracks_world`,`X.vis_all`。
- Produces: `main()`;结果 `outputs/cross_embodiment_wm/combine_action/summary.txt` + `pareto.png`。env:`COMBINES`(默认 `a1,align,warm`)、`N_ROB_LIST`、`SEEDS`、`OUT`、`SMOKE`。

- [ ] **Step 1: 追加 main/_save/_plot**

```python
OUT = os.environ.get("OUT", "outputs/cross_embodiment_wm/combine_action"); os.makedirs(OUT, exist_ok=True)
DS = os.environ.get("DS", X.DS)
N_ROB_LIST = [int(x) for x in os.environ.get("N_ROB_LIST", "20,50,100,400").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
ANCHORS = ["world", "skel"]                                   # 上下界锚,走 V.train_feat
COMBINES = os.environ.get("COMBINES", "a1,align,warm").split(",")
METHODS = ANCHORS + COMBINES


def _agg(res, meth, N):
    arr = np.array(res[meth][N])
    if len(arr) == 0: return None
    ro, rh = arr[:, 0], arr[:, 1]; d = ro - rh
    return ro.mean(), rh.mean(), d.mean(), d.std()


def _save(res, header):
    lines = [header, f"{'method':>10} {'N':>5} | {'ro':>7} {'rh':>7} | {'Δ(help)':>8} {'±std':>6}"]
    for meth in METHODS:
        for N in N_ROB_LIST:
            a = _agg(res, meth, N)
            if a is None: continue
            lines.append(f"{meth:>10} {N:>5} | {a[0]:7.2f} {a[1]:7.2f} | {a[2]:+8.2f} {a[3]:6.2f}")
        lines.append("")
    lines.append("BREAKTHROUGH check (判据1: rh <= world.rh AND Δ >> world.Δ):")
    for N in N_ROB_LIST:
        w = _agg(res, "world", N)
        if w is None: continue
        for meth in COMBINES:
            a = _agg(res, meth, N)
            if a is None: continue
            hit = "★BREAK" if (a[1] <= w[1] + 1e-6 and a[2] > w[2] + 0.3) else ""
            lines.append(f"  N={N:>4} {meth:>6}: rh {a[1]:.2f} (world {w[1]:.2f})  Δ {a[2]:+.2f} (world {w[2]:+.2f}) {hit}")
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")


def _plot(res):
    fig, ax = plt.subplots(figsize=(7, 6))
    N = max([n for n in N_ROB_LIST if _agg(res, "world", n)], default=N_ROB_LIST[0])
    for meth in METHODS:
        a = _agg(res, meth, N)
        if a is None: continue
        ax.scatter(a[1], a[2], s=90); ax.annotate(meth, (a[1], a[2]), fontsize=10,
                   xytext=(4, 4), textcoords="offset points")
    w = _agg(res, "world", N)
    if w:                                                     # 目标区 = world 左上方
        ax.axvline(w[1], color="gray", ls="--", lw=.8); ax.axhline(w[2], color="gray", ls="--", lw=.8)
        ax.annotate("目标区\n(rh≤world, Δ大)", (w[1] - .3, w[2] + .5), fontsize=9, color="green")
    ax.set_xlabel("最终精度 rh (px@224, ↓好)"); ax.set_ylabel("human-helps Δ=ro-rh (↑好)")
    ax.set_title(f"结合机制 vs Pareto 锚 (N={N})"); ax.grid(alpha=.3); ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(f"{OUT}/pareto.png", dpi=130); plt.close(fig)


def main():
    if os.environ.get("SMOKE", "0") == "1":
        SSm.WM_EPOCHS, SSm.R_SS = 2, 4
        N_ROB_LIST[:] = [20, 100]; SEEDS[:] = [0]
    zr = np.load(f"{DS}/clips_robot.npz"); zh = np.load(f"{DS}/clips_human.npz")
    r_tr, r_ef, r_vs = (zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32), zr["vis"].astype(np.float32))
    h_tr, h_ef, h_vs = (zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32), zh["vis"].astype(np.float32))
    Nr = len(r_tr); perm = np.random.default_rng(0).permutation(Nr)
    ho, pool = perm[:X.HELDOUT], perm[X.HELDOUT:]
    X.tracks_world, X.vis_all = r_tr, r_vs                    # ade_world reads these globals
    mtr = np.concatenate([r_tr, h_tr]); mef = np.concatenate([r_ef, h_ef]); mvs = np.concatenate([r_vs, h_vs])
    hi = torch.arange(Nr, Nr + len(h_tr))
    res = {m: {N: [] for N in N_ROB_LIST} for m in METHODS}
    header = (f"COMBINE action-featurization | held-out robot ADE px@224 (H={X.H}) | data={DS} | seeds={SEEDS}\n"
              f"methods={METHODS}  (world/skel=Pareto 锚; a1/align/warm=结合机制)\n"
              "Δ = ro - rh (>0 human helps); 判据1: rh<=world.rh 且 Δ 明显>world\n")
    print(header, flush=True)
    for N in N_ROB_LIST:
        for seed in SEEDS:
            sub = pool[np.random.default_rng(100 + seed).choice(len(pool), min(N, len(pool)), replace=False)]
            ri = torch.from_numpy(sub); rih = torch.cat([ri, hi])
            for meth in METHODS:
                if meth in ANCHORS:
                    wm_ro = V.train_feat(mtr, mvs, mef, ri, meth, seed=seed)
                    wm_rh = V.train_feat(mtr, mvs, mef, rih, meth, seed=seed)
                else:
                    wm_ro = train_comb(mtr, mvs, mef, ri, meth, Nr, seed=seed)
                    wm_rh = train_comb(mtr, mvs, mef, rih, meth, Nr, seed=seed)
                a_ro = X.ade_world(wm_ro, r_tr, r_ef, ho, False)
                a_rh = X.ade_world(wm_rh, r_tr, r_ef, ho, False)
                res[meth][N].append((a_ro, a_rh))
                print(f"  N={N:>4} seed={seed} {meth:>8} | ro {a_ro:6.2f}  rh {a_rh:6.2f}  Δ {a_ro-a_rh:+6.2f}", flush=True)
        _save(res, header)
    _save(res, header); _plot(res)
    print(f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: SMOKE 验证 main 跑通(先只 a1)**

Run: `CUDA_VISIBLE_DEVICES=<free> SMOKE=1 COMBINES=a1 /scr/yusenluo/anaconda3/envs/iws/bin/python exp_scel_combine_action.py`
Expected: 打印 world/skel/a1 各 N/seed 的 ro/rh/Δ,`=== DONE ===`,`summary.txt`+`pareto.png` 生成,无 nan。

- [ ] **Step 3: Commit**

```bash
git add exp_scel_combine_action.py
git commit -m "feat(combine): main sweep (world/skel anchors + a1/align/warm) + Pareto plot + breakthrough check"
```

- [ ] **Step 4: A1 full run(后台)**

先 `nvidia-smi` 选空 GPU。Run(后台):
```bash
CUDA_VISIBLE_DEVICES=<free> COMBINES=a1 OUT=outputs/cross_embodiment_wm/combine_action/a1 \
  nohup /scr/yusenluo/anaconda3/envs/iws/bin/python exp_scel_combine_action.py \
  > outputs/cross_embodiment_wm/combine_action/a1.log 2>&1 &
```
完成后 Read `a1/summary.txt`,看 BREAKTHROUGH check:a1 是否 rh≤world 且 Δ 明显>world。达标/接近即里程碑 2 成立。

- [ ] **Step 5: 记录 A1 结果**

把 a1 的 rh/Δ/是否突破写入新建 `COMBINE_ACTION_LOG.md`(option 表:机制/rh/Δ/突破?/为什么),commit。

---

### Task 4: align + warm full sweep + λ 消融

**Files:**
- Modify: `COMBINE_ACTION_LOG.md`(追加结果)

**Interfaces:**
- Consumes: Task 3 的 `main`(env `COMBINES`/`OUT`);已实现的 align/warm。

- [ ] **Step 1: align + warm full run(后台)**

```bash
CUDA_VISIBLE_DEVICES=<free> COMBINES=align,warm OUT=outputs/cross_embodiment_wm/combine_action/align_warm \
  nohup /scr/yusenluo/anaconda3/envs/iws/bin/python exp_scel_combine_action.py \
  > outputs/cross_embodiment_wm/combine_action/align_warm.log 2>&1 &
```
Expected: `align_warm/summary.txt` + `pareto.png`,含 world/skel/align/warm。

- [ ] **Step 2: A1 λ_res 消融(若 A1 未突破或疑似残差独吞)**

`train_comb` 的 lam_res 已是参数;加临时 env 分支或直接跑三档比较(N=100 seed=0 快检):
```bash
CUDA_VISIBLE_DEVICES=<free> SMOKE=0 N_ROB_LIST=100 SEEDS=0 COMBINES=a1 \
  OUT=outputs/cross_embodiment_wm/combine_action/a1_lam0 \
  /scr/yusenluo/anaconda3/envs/iws/bin/python -c \
"import os,exp_scel_combine_action as C; C.train_comb.__defaults__" # 见下:改为脚本内 sweep lam_res
```
实作:在 `main` 支持 `LAM_RES` env 透传给 `train_comb`(一行:`lam_res=float(os.environ.get("LAM_RES","1e-3"))`,`train_comb(..., lam_res=lam_res)`),跑 LAM_RES∈{0,1e-3,1e-2} 三次,比较 a1 的 rh/Δ,确认正则是否防住独吞(λ=0 应看到 Δ 塌向 world)。

- [ ] **Step 3: 合表 + 判定**

汇总 world/skel/a1/align/warm(+λ 消融)到一张 Pareto 表 + `pareto.png`。判定哪个机制落到 world 左上方(判据 1)或最接近(起步门槛)。写入 `COMBINE_ACTION_LOG.md`。

- [ ] **Step 4: Commit**

```bash
git add COMBINE_ACTION_LOG.md
git commit -m "docs(combine): align/warm sweep + lam_res ablation results, Pareto verdict"
```

---

### Task 5: 突破点/最优点渲染补齐(4 列 gif)+ 落盘 + 上传

**Files:**
- Modify: `exp_scel_combine_action.py`(追加 `render_best`)
- Modify: `COMBINE_ACTION_LOG.md`;memory `project_velocity_action_featurization.md`(追加 point 8)

**Interfaces:**
- Consumes: `exp_scel_latent_detmem as D`(`train_detmem`/`render_detmem`/`load_gmask`),`amplify_wm.rollout_lwc`(CombLWC forward 兼容,缺省全 robot),`viz_combined.build_flow_cols/save_combined_gif`,`exp_v3_human_helps_pixels.cube_pos_err`。CombLWC.forward(hist,eef3) 不传 is_h → 全 robot,与评价一致。
- Produces: `render_best(best_comb)`;4 列 gif `GT | GT-flow→③ | world-② | <best>-②`,`outputs/cross_embodiment_wm/combine_action/render/`。

- [ ] **Step 1: 追加 render_best(复用 velocity_render 结构)**

参照 `exp_scel_velocity_render.py`(同结构:训 detmem③ on v3、选 NSEQ、rollout_lwc、render_detmem、save_combined_gif 4 列),仅把 challenger 列的 ② 从 `V.train_feat("skel")` 换成 `train_comb(best_comb)`;world 列仍 `V.train_feat("world")`;GT-flow 天花板列不变。列标题 `["GT","GT-flow→③","world-②","<best>-②"]`。metrics:② flow ADE、③ cube_err(nanmean + 检测率,别信消失)、LPIPS。EPOCHS=60 detmem。

- [ ] **Step 2: 运行渲染**

Run: `CUDA_VISIBLE_DEVICES=<free> /scr/yusenluo/anaconda3/envs/iws/bin/python -c "import exp_scel_combine_action as C; C.render_best('<best_comb>')"`
Expected: `render/gifs/seq*.gif` + `render/montage.png` + `render/summary.txt`。

- [ ] **Step 3: 眼检 montage**

Read `render/montage.png`,亲眼核验 best-② 列 cube 是否比 skel 跟得紧、是否接近 world/GT-flow(遵 mask 眼检 + cube nan 陷阱规矩)。结论写文字。

- [ ] **Step 4: 上传 drive + 落盘 memory**

```bash
TOPIC=combine_skel_world_action渲染 bash upload_evals_gdrive.sh \
  "结合机制_world对best_4列gif=outputs/cross_embodiment_wm/combine_action/render" \
  "Pareto表_summary=outputs/cross_embodiment_wm/combine_action/align_warm"
```
更新 memory `project_velocity_action_featurization.md`(追加 point 8:结合机制 A1/align/warm、Pareto 是否突破、渲染眼检、脚本名、链接 [[reference_human2any_egoengine]] LaST-HD)。

- [ ] **Step 5: Commit**

```bash
git add exp_scel_combine_action.py COMBINE_ACTION_LOG.md
git commit -m "feat(combine): render best combine ② vs world (4-col gif) + drive upload + log"
```

---

## 自审记录

- **Spec 覆盖**:§1 判据→Task 3 `_save` BREAKTHROUGH check;§3 A1/align/warm→Task 1/2;防独吞 λ_res→Task 2 参数 + Task 4 消融;§5 评价/Pareto/渲染→Task 3/5;§5 落盘/drive→Task 4/5;§6 兜底(负结果入 log)→Task 4 Step 3。全覆盖。
- **类型一致**:`train_comb(...,Nr,...)`、`CombLWC(P,combine,alpha,lam_res,lam_align)`、`forward(hist,eef3,is_h=None)`、`aux["res_sq"]`/`aux["align"]` 跨 Task 一致。评价 `X.ade_world(model,r_tr,r_ef,ho,False)` 与 velocity 一致。
- **无 placeholder**:除 `<free>`(运行时选 GPU)、`<best_comb>`(依 Task4 结果定)为必要运行时值,其余代码完整。
