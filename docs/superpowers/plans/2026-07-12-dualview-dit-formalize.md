# Dual-View DiT ③ 正式化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把"flow 域不变接口 > naive eef action 条件化"在 dual-view DiT ③ 上做成 paper-ready:controlled 三方消融(同 backbone,多 seed)、cross-view 消融、human-helps 数据混合消融(robot-only vs robot+human)、端到端 ② pred-flow、真 IWS stage2(DF 训练 CMLatentDynamics)external baseline、可复现 REPORT。

**Architecture:** 正式脚本 `exp_scel_dualview_dit_formal.py` 从探索脚本 `exp_scel_dualview_dit.py` **import 复用**(flow_cond/eef_point/load_dual/latcache/DualViewDiT),子类 `DualViewDiTFormal` 加第三条件臂 eefsp + cross-view attention-mask;IWS baseline 单独脚本 `exp_dualview_iws_stage2.py` 用 IWS 原生 `CMLatentDynamics` 模块 + diffusion-forcing 训练/采样,同 frozen VAE 同 eval 口径(用户 2026-07-12 拍板)。

**Tech Stack:** PyTorch + iws conda env(`/scr/yusenluo/anaconda3/envs/iws/bin/python`),frozen 16ch VAE(ostris/vae-kl-f8-d16),数据 `outputs/flow_render_dataset_can_dual/`(robot 2700×L48,human 1800×L24,双视角),② ckpt `outputs/cross_embodiment_wm/dualview_wm/wm_dual.pt`。

## Global Constraints

- 分支 `phantom_dynamo`,**就地工作不开 worktree**(仓库大量未提交代码);`git add` 只加自己的文件;commit 不加 Co-Authored-By([[feedback_iws_untracked_working_tree]])。
- 一律 iws env:`PY=/scr/yusenluo/anaconda3/envs/iws/bin/python`,从仓库根运行;跑 GPU 前 `nvidia-smi` 选空卡(现在 0,2-7 空,每张 49GB)。
- 输出树:`outputs/cross_embodiment_wm/dualview_dit_formal/`,每 run 独立子目录 `{cond}_cv{0|1}_s{seed}[_ronly]/`,**不覆盖**;每 run 落 `metrics.json` + `summary.txt`([[feedback_per_experiment_output_dir]])。
- **human-helps 消融纳入(用户 2026-07-12 推翻 spec 旧固定项)**:MIX ∈ {rh(co-train 默认), r(robot-only)},M1c = {flow, eeffilm} × {rh, r} × seed{0,1,2}(cv 固定 1);对比各臂 human 增益 Δ=LPIPS(r)−LPIPS(rh),诚实报(③ 层先验判过 +0.03 天花板,不帮也是结果)。
- **单一权威 eval**:`obj_lpips_audit`(footprint<5 点的帧记无效并计入 det-rate,不悄悄跳过)+ PSNR;所有臂(含 IWS baseline)同一函数同一 eval seq 集([[feedback_cube_metric_nan_trap]] / [[feedback_unify_code_no_script_sprawl]])。
- **gif protocol**(用户 2026-07-12 重申):一律 `save_combined_gif`(Rendered 行 + Flow overlay 行 + 列标题 + legend + 顶部 caption),replay 有 GT 列,eval seq 固定不每次变;重要 gif 最后统一 `upload_evals_gdrive.sh` 传 drive(OAuth 可能需用户重授权);比较 metric 全部存盘保留([[feedback_gif_eval_layout]] / [[feedback_upload_gifs_gdrive]])。
- 每份 summary/report 声明用的组件版本(哪个 ② ckpt、哪个 VAE、哪个 ③ 变体)([[feedback_declare_component_versions]])。
- 训练超参不动探索版配方(BS=8, LR=2e-4, PREV_DF=0.3, LAM_LPIPS=1.0, DIM=384/DEPTH=8/HEADS=6, co-train robot latent-MSE+decode-LPIPS / human obj-region pixel MSE),唯一变量 = 条件注入 / crossview / seed / EPOCHS=60。
- heldout split 固定 `rng(0)` 且**独立于训练 seed**;eval seqs = heldout 中 motion 最大的 24 条,持久化 `eval_seqs.json`,所有 run assert 一致。

**关键常量(已核实 ground truth):**
- `K=4, F=20`(amplify_wm);`H=20`(渲染 horizon);`GRID=16`;`IMG=128`;`FLOW_SCALE=10.0`。
- npz keys:robot `frames/frames_low (2700,48,128,128,3) u8`,`tracks/tracks_low (2700,48,48,2) f16`,`eef/eef_low (2700,48,3,2) f16`,`vis/vis_low`,`low_valid (2700,) bool`,`grip/joint`(本正式化不用);human 同构 1800×L24。
- `splat128(pos0, post, vis) -> (4,128,128)`:ch0/1 = frame0 锚定 dx,dy(×FLOW_SCALE),ch2 = vis 密度,ch3 = 目标位置密度。
- 探索版已有结果(30ep 单 seed):flow v0_lp 0.326 / v1_lp 0.342;eef v0_lp 0.393 / v1_lp 0.427。latcache 已存在可复用:`outputs/cross_embodiment_wm/dualview_dit/latcache_{robot,human}.npy`。
- ② rollout 真名 = `exp_scel_dualview_wm.rollout_dual`(spec 里写的 rollout_dualview 是笔误);`wm_dual.pt` 已训好。
- IWS stage2 = `CMLatentDynamics.forward(x (B,C,T,H,W), noise_levels (T,B), stop_noise_levels (T,B), external_cond (T,B,D))`,原生 `action_emd` MLP → 每个 ResnetBlock `cond_emb_layers` FiLM;推理 = AR chunk=1 + 滑窗 + 多步去噪(`latent_world_model.py:1190`)。

**三条件臂定义(M1 的"三方"):**
| 臂 | 内容 | 注入 | 隔离什么 |
|---|---|---|---|
| `flow` | 物体 flow splat + footprint (3ch) | spatial-add(`ce`→`emb_cond`) | ours |
| `eefsp` | **只 splat 3 个 eef 点**(无物体 flow,3ch 同构) | spatial-add(同 flow 臂) | 内容变量:flow 赢是因为"物体运动信息"还是"spatial 注入机制" |
| `eeffilm` | 单 eef 点 4-dim 向量 | adaLN-FiLM(=IWS stage2 `action_emd`→`cond_emb_layers` 同构) | naive action 接口(探索版 `eef` 臂原样) |

---

### Task 1: `Block` 支持 attn_mask(共享代码最小改动)+ 单测

**Files:**
- Modify: `exp_scel_latent_dit.py:67-75`(`Block.forward` 加可选 `attn_mask`)
- Test: `tests/test_dualview_formal.py`(新建)

**Interfaces:**
- Produces: `Block.forward(x, c=None, attn_mask=None)` — `attn_mask` 为 float additive mask `(L,L)`,直接透传 `nn.MultiheadAttention`。向后兼容:所有现有调用 `b(x, cvec)` 不受影响。

- [ ] **Step 1: 写失败单测**

```python
# tests/test_dualview_formal.py
import os, sys
sys.path.insert(0, "/scr2/yusenluo/interactive_world_sim")
os.environ.setdefault("VAE_NAME", "ostris/vae-kl-f8-d16"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import pytest


def _block_mask(n, half):
    m = torch.full((n, n), float("-inf")); m[:half, :half] = 0; m[half:, half:] = 0
    return m


def test_block_attn_mask_blocks_info():
    """mask 阻断后,前半 token 输出对后半 token 扰动不变(信息不泄漏)。"""
    from exp_scel_latent_dit import Block
    torch.manual_seed(0)
    b = Block(32, 4, ada=False).eval()
    n, half = 8, 4
    mask = _block_mask(n, half)
    x1 = torch.randn(2, n, 32); x2 = x1.clone(); x2[:, half:] = torch.randn(2, half, 32)
    with torch.no_grad():
        y1 = b(x1, attn_mask=mask); y2 = b(x2, attn_mask=mask)
    assert torch.allclose(y1[:, :half], y2[:, :half], atol=1e-6)
    with torch.no_grad():
        y3 = b(x2)                                   # 无 mask 应受影响(对照)
    assert not torch.allclose(y1[:, :half], y3[:, :half], atol=1e-4)
```

- [ ] **Step 2: 跑单测确认失败**

Run: `cd /scr2/yusenluo/interactive_world_sim && /scr/yusenluo/anaconda3/envs/iws/bin/python -m pytest tests/test_dualview_formal.py -v -x`
Expected: FAIL `TypeError: forward() got an unexpected keyword argument 'attn_mask'`

- [ ] **Step 3: 最小实现 — `Block.forward` 加 attn_mask**

`exp_scel_latent_dit.py` 的 `Block.forward` 改为(仅两处 `s.att(...)` 调用加参数):

```python
    def forward(s, x, c=None, attn_mask=None):
        if s.ada:
            sh1, sc1, g1, sh2, sc2, g2 = s.mod(c)[:, None].chunk(6, -1)
            h = s.n1(x) * (1 + sc1) + sh1; x = x + g1 * s.att(h, h, h, need_weights=False, attn_mask=attn_mask)[0]
            h = s.n2(x) * (1 + sc2) + sh2; x = x + g2 * s.mlp(h)
        else:
            h = s.n1(x); x = x + s.att(h, h, h, need_weights=False, attn_mask=attn_mask)[0]
            x = x + s.mlp(s.n2(x))
        return x
```

- [ ] **Step 4: 跑单测确认通过**

Run: 同 Step 2。Expected: PASS(2 个 assert 都过)。

- [ ] **Step 5: 回归 — 现有调用方不破**

Run: `/scr/yusenluo/anaconda3/envs/iws/bin/python -c "import exp_scel_latent_dit; import torch; b=exp_scel_latent_dit.Block(32,4,ada=False); print(b(torch.randn(1,4,32)).shape)"`
Expected: `torch.Size([1, 4, 32])`

- [ ] **Step 6: Commit**

```bash
git add exp_scel_latent_dit.py tests/test_dualview_formal.py
git commit -m "feat(dualview-formal): Block optional attn_mask passthrough + non-leakage unit test"
```

---

### Task 2: `exp_scel_dualview_dit_formal.py` — 模型(三臂 × crossview)+ cond builders + 单测

**Files:**
- Create: `exp_scel_dualview_dit_formal.py`(本 task 先写模型/cond 部分,Task 3 补 trainer/eval)
- Test: `tests/test_dualview_formal.py`(追加)

**Interfaces:**
- Consumes: `exp_scel_dualview_dit.{DualViewDiT, flow_cond, eef_point, load_dual, latcache, _fp, u8, psnr}`;`exp_scel_latent_dit.{Block, GRID}`;`exp_v3_human_helps_pixels.{splat128, K, IMG, device}`。
- Produces:
  - `eefsp_cond(ef0, eft) -> np.ndarray (3,128,128)`
  - `build_conds_formal(D, j, t, mode) -> (cond (2,3,128,128)|(2,4), mask (2,128,128))`,`mode ∈ {"flow","eefsp","eeffilm"}`
  - `DualViewDiTFormal(mode, crossview: bool, D=384, depth=8, heads=6)`,`forward(z0 (B,2,Cz,16,16), prev 同, cond) -> (B,2,Cz,16,16)`

- [ ] **Step 1: 写失败单测(追加到 tests/test_dualview_formal.py)**

```python
def _mk(mode, crossview):
    from exp_scel_dualview_dit_formal import DualViewDiTFormal
    torch.manual_seed(0)
    return DualViewDiTFormal(mode=mode, crossview=crossview, D=64, depth=2, heads=2).eval()


def _rand_inputs(mode, Cz):
    z0 = torch.randn(2, 2, Cz, 16, 16); prev = torch.randn(2, 2, Cz, 16, 16)
    cond = torch.randn(2, 2, 3, 128, 128) if mode in ("flow", "eefsp") else torch.randn(2, 2, 4)
    return z0, prev, cond


@pytest.mark.parametrize("mode", ["flow", "eefsp", "eeffilm"])
def test_formal_forward_shapes(mode):
    from exp_scel_latent_renderer import latent_ch
    Cz = latent_ch()
    m = _mk(mode, crossview=True)
    z0, prev, cond = _rand_inputs(mode, Cz)
    with torch.no_grad():
        out = m(z0, prev, cond)
    assert out.shape == (2, 2, Cz, 16, 16) and torch.isfinite(out).all()


def test_crossview_off_no_leak_flow():
    """crossview=False 时 view0 输出对 view1 的 z0/prev/cond 扰动完全不变(spec §5 判定单测)。"""
    from exp_scel_latent_renderer import latent_ch
    Cz = latent_ch()
    m = _mk("flow", crossview=False)
    z0, prev, cond = _rand_inputs("flow", Cz)
    z0b, prevb, condb = z0.clone(), prev.clone(), cond.clone()
    z0b[:, 1] = torch.randn_like(z0b[:, 1]); prevb[:, 1] = torch.randn_like(prevb[:, 1])
    condb[:, 1] = torch.randn_like(condb[:, 1])
    with torch.no_grad():
        y1 = m(z0, prev, cond); y2 = m(z0b, prevb, condb)
    assert torch.allclose(y1[:, 0], y2[:, 0], atol=1e-6)
    assert not torch.allclose(y1[:, 1], y2[:, 1], atol=1e-4)


def test_crossview_off_no_leak_eeffilm_tokens():
    """eeffilm 臂:视觉 token 不泄漏;cond 向量经 mean(both views) 全局 FiLM 属 action 级设计,只扰动 z0/prev。"""
    from exp_scel_latent_renderer import latent_ch
    Cz = latent_ch()
    m = _mk("eeffilm", crossview=False)
    z0, prev, cond = _rand_inputs("eeffilm", Cz)
    z0b, prevb = z0.clone(), prev.clone()
    z0b[:, 1] = torch.randn_like(z0b[:, 1]); prevb[:, 1] = torch.randn_like(prevb[:, 1])
    with torch.no_grad():
        y1 = m(z0, prev, cond); y2 = m(z0b, prevb, cond)
    assert torch.allclose(y1[:, 0], y2[:, 0], atol=1e-6)


def test_crossview_on_does_leak():
    from exp_scel_latent_renderer import latent_ch
    Cz = latent_ch()
    m = _mk("flow", crossview=True)
    z0, prev, cond = _rand_inputs("flow", Cz)
    z0b = z0.clone(); z0b[:, 1] = torch.randn_like(z0b[:, 1])
    with torch.no_grad():
        y1 = m(z0, prev, cond); y2 = m(z0b, prev, cond)
    assert not torch.allclose(y1[:, 0], y2[:, 0], atol=1e-4)


def test_eefsp_cond_shape():
    import numpy as np
    from exp_scel_dualview_dit_formal import eefsp_cond
    ef0 = np.random.rand(3, 2).astype(np.float32); eft = ef0 + 0.05
    c = eefsp_cond(ef0, eft)
    assert c.shape == (3, 128, 128) and np.isfinite(c).all() and c[2].max() > 0
```

- [ ] **Step 2: 跑单测确认失败**

Run: `/scr/yusenluo/anaconda3/envs/iws/bin/python -m pytest tests/test_dualview_formal.py -v`
Expected: 新增 test 全 FAIL(`ModuleNotFoundError: exp_scel_dualview_dit_formal`),Task 1 的 test PASS。

- [ ] **Step 3: 实现模型/cond 部分**

`exp_scel_dualview_dit_formal.py`(第一部分):

```python
"""DUAL-VIEW DiT ③ 正式化 (spec docs/superpowers/specs/2026-07-12-dualview-dit-formalize-design.md).
controlled 三方 (flow | eefsp | eeffilm) x crossview {1,0} x seed, 复用探索脚本组件 (import, 不复制).
MODE=train: 训一个 (COND, CROSSVIEW, SEED, EPOCHS) 格子 -> OUT/{cond}_cv{cv}_s{seed}/
MODE=eval : 只重跑 eval (需已有 dvdit.pt)
MODE=gif  : replay 对比 gif (GT | flow | eefsp | eeffilm, seed0 cv1)
MODE=e2e  : ② pred-flow 端到端 (Task 5)
Env: MODE, COND(flow|eefsp|eeffilm), CROSSVIEW(1|0), SEED, EPOCHS(60), SMOKE, NEVAL(24)
iws env + GPU. 输出根 outputs/cross_embodiment_wm/dualview_dit_formal/"""
import json
import os

os.environ.setdefault("VAE_NAME", "ostris/vae-kl-f8-d16"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch, torch.nn as nn, cv2
import exp_scel_dualview_dit as dv
from exp_scel_dualview_dit import (DualViewDiT, flow_cond, eef_point, load_dual, latcache,
                                   _fp, u8, psnr, FLOW_SCALE, H, BS, LR, PREV_DF, LAM, DIM, DEPTH, HEADS)
from exp_scel_latent_renderer import enc, dec, latent_ch
from exp_scel_latent_dit import GRID
from exp_v3_human_helps_pixels import splat128, K, IMG, device

SMOKE = os.environ.get("SMOKE", "0") == "1"
MODE = os.environ.get("MODE", "train")
CONDM = os.environ.get("COND", "flow")                     # flow | eefsp | eeffilm
CROSSVIEW = os.environ.get("CROSSVIEW", "1") == "1"
MIX = os.environ.get("MIX", "rh")                          # rh = co-train robot+human | r = robot-only (M1c)
SEED = int(os.environ.get("SEED", "0"))
EPOCHS = 2 if SMOKE else int(os.environ.get("EPOCHS", "60"))
NEVAL = int(os.environ.get("NEVAL", "24"))
ROOT = "outputs/cross_embodiment_wm/dualview_dit_formal"
RUN = f"{CONDM}_cv{int(CROSSVIEW)}_s{SEED}" + ("_ronly" if MIX == "r" else "") + ("_smoke" if SMOKE else "")
OUT = f"{ROOT}/{RUN}"; os.makedirs(f"{OUT}/gifs", exist_ok=True)
VERSIONS = ("VAE=ostris/vae-kl-f8-d16(frozen) | backbone=add-DiT D384x8 (project_detmem_dit winner) | "
            "data=flow_render_dataset_can_dual | 2ckpt=dualview_wm/wm_dual.pt (e2e only)")


def eefsp_cond(ef0, eft):
    """第三臂 eefsp: 只 splat 3 个 eef 点 (无物体 flow), 与 flow 臂同构 3ch [dx,dy,eef 目标位置密度]."""
    fm = splat128(ef0, eft, np.ones(len(ef0), np.float32))
    return np.stack([fm[0], fm[1], fm[3]]).astype(np.float32)


def build_conds_formal(D, j, t, mode):
    """clip j frame t 的 per-view cond + loss mask (物体 footprint, cond 无关, 三臂同)."""
    conds, masks = [], []
    for v in range(2):
        tr, ef, vs = D["tr"][v], D["ef"][v], D["vs"][v]
        if mode == "flow":
            c = flow_cond(tr[j, 0], tr[j, t], ef[j, 0], ef[j, t], vs[j, t])
        elif mode == "eefsp":
            c = eefsp_cond(ef[j, 0], ef[j, t])
        else:                                              # eeffilm = 探索版 eef 臂原样
            c = eef_point(ef[j, 0], ef[j, t])
        conds.append(c); masks.append(_fp(tr[j, t]))
    return np.stack(conds), np.stack(masks)


class DualViewDiTFormal(DualViewDiT):
    """探索版 DualViewDiT + (a) 第三臂 eefsp 走 flow 的 spatial-add 通路 (b) crossview=False 时
    attention-mask 阻断 view0<->view1 token (参数量不变, spec M1a)."""
    def __init__(s, mode=CONDM, crossview=CROSSVIEW, **kw):
        assert mode in ("flow", "eefsp", "eeffilm")
        super().__init__(cond=("eef" if mode == "eeffilm" else "flow"), **kw)
        s.mode, s.crossview = mode, crossview
        n = 4 * GRID * GRID                                # [z0_v0,prev_v0,z0_v1,prev_v1] = 1024
        if not crossview:
            m = torch.full((n, n), float("-inf")); half = n // 2
            m[:half, :half] = 0; m[half:, half:] = 0
            s.register_buffer("attn_mask", m, persistent=False)
        else:
            s.attn_mask = None

    def forward(s, z0, prev, cond):
        B = z0.shape[0]; toks = []; cvec = None
        for v in range(2):
            t0 = s.emb_z0(s._tok(z0[:, v]))
            if s.cond == "flow":
                cf = s.ce(cond[:, v]); t0 = t0 + s.emb_cond(s._tok(cf))
            t0 = t0 + s.pos + s.view_emb[v] + s.typ[0]
            tp = s.emb_prev(s._tok(prev[:, v])) + s.pos + s.view_emb[v] + s.typ[1]
            toks += [t0, tp]
        if s.cond == "eef":
            cvec = s.act_emd(cond.reshape(B, 2, 4).mean(1))
        x = torch.cat(toks, 1)
        for b in s.blocks: x = b(x, cvec, attn_mask=s.attn_mask)
        zs = []
        for v in range(2):
            f = s.norm(x[:, v * 2 * GRID * GRID: v * 2 * GRID * GRID + GRID * GRID])
            zs.append(s.out(f).transpose(1, 2).reshape(B, s.zdim, GRID, GRID))
        return torch.stack(zs, 1)
```

注意:token 布局是 `[z0_v0(256), prev_v0(256), z0_v1(256), prev_v1(256)]`(父类 `toks += [t0, tp]` 逐 view),所以 mask 以 512 为界正确;输出读取 `v*2*GRID*GRID` 起的 256 个 token,与父类一致。

- [ ] **Step 4: 跑单测确认通过**

Run: `/scr/yusenluo/anaconda3/envs/iws/bin/python -m pytest tests/test_dualview_formal.py -v`
Expected: 全 PASS(含 Task 1 的)。

- [ ] **Step 5: Commit**

```bash
git add exp_scel_dualview_dit_formal.py tests/test_dualview_formal.py
git commit -m "feat(dualview-formal): DualViewDiTFormal 3-arm (flow/eefsp/eeffilm) + crossview attn-mask + leak tests"
```

---

### Task 3: formal trainer + 权威 eval(det-rate)+ SMOKE 走通 + 计时

**Files:**
- Modify: `exp_scel_dualview_dit_formal.py`(追加 trainer/eval/main)
- Test: `tests/test_dualview_formal.py`(追加 obj_lpips_audit 单测)

**Interfaces:**
- Produces:
  - `obj_lpips_audit(lp, pred (T,128,128,3) f32, gt 同, objm (T,128,128)) -> (mean_lpips: float, n_valid: int, n_total: int)`
  - `eval_seqs(R, ho, n) -> list[int]`(motion top-n,持久化/assert)
  - `train_formal(R, Hh, idx_r, idx_h, latR, latH, mode, crossview, seed, epochs) -> model`
  - 每 run 产物:`{OUT}/dvdit.pt`、`{OUT}/metrics.json`(keys: `v0_ps,v0_lp,v1_ps,v1_lp,det_rate_v0,det_rate_v1,cond,crossview,seed,epochs,wall_min,n_eval`)、`{OUT}/summary.txt`

- [ ] **Step 1: 写失败单测(det-rate 语义)**

```python
def test_obj_lpips_audit_counts_invalid():
    """footprint<5 点的帧记无效计入分母, 不悄悄跳 (cube-nan trap)."""
    import numpy as np
    from exp_scel_dualview_dit_formal import obj_lpips_audit

    class FakeLP:
        def __call__(self, a, b): return torch.tensor(0.5)

    T = 4
    pred = np.random.rand(T, 128, 128, 3).astype(np.float32); gt = pred.copy()
    objm = np.zeros((T, 128, 128), np.float32)
    objm[0, 60:70, 60:70] = 1.0; objm[1, 60:70, 60:70] = 1.0      # 只有 2 帧有效
    mean, n_valid, n_total = obj_lpips_audit(FakeLP(), pred, gt, objm)
    assert n_valid == 2 and n_total == 4 and abs(mean - 0.5) < 1e-6
```

- [ ] **Step 2: 跑单测确认失败**

Run: `/scr/yusenluo/anaconda3/envs/iws/bin/python -m pytest tests/test_dualview_formal.py::test_obj_lpips_audit_counts_invalid -v`
Expected: FAIL(`ImportError: obj_lpips_audit`)

- [ ] **Step 3: 实现 trainer/eval/main(追加到 formal 脚本)**

```python
def obj_lpips_audit(lp, pred, gt, objm):
    """单一权威 obj-region LPIPS: footprint 质心 64x64 crop; <5 点帧记无效并计数 (det-rate)."""
    outs = []; n_total = len(pred)
    for h in range(n_total):
        ys, xs = np.where(objm[h] > 0.5)
        if len(xs) < 5: continue
        cx, cy = int(xs.mean()), int(ys.mean())
        x1 = min(max(cx - 32, 0) + 64, IMG); y1 = min(max(cy - 32, 0) + 64, IMG)
        x0, y0 = x1 - 64, y1 - 64
        a = torch.from_numpy(pred[h, y0:y1, x0:x1]).permute(2, 0, 1)[None].float().to(device) * 2 - 1
        b = torch.from_numpy(gt[h, y0:y1, x0:x1]).permute(2, 0, 1)[None].float().to(device) * 2 - 1
        outs.append(float(lp(a, b)))
    return (float(np.mean(outs)) if outs else float("nan")), len(outs), n_total


def eval_seqs(R, ho, n=NEVAL):
    """heldout 中 motion top-n, 固定持久化; 所有 run assert 同一集合 (gif protocol: seq 固定)."""
    mot = np.array([np.linalg.norm(np.diff(R["tr"][0][si, K:K + H].mean(1), axis=0), axis=-1).sum() for si in ho])
    chosen = sorted(int(x) for x in ho[np.argsort(-mot)[:n]])
    p = f"{ROOT}/eval_seqs.json"; os.makedirs(ROOT, exist_ok=True)
    if os.path.exists(p):
        prev = json.load(open(p))
        assert prev == chosen, f"eval seq set drifted: {prev[:5]} vs {chosen[:5]}"
    else:
        json.dump(chosen, open(p, "w"))
    return chosen


def train_formal(R, Hh, idx_r, idx_h, latR, latH, mode, crossview, seed, epochs):
    """探索版 train() 配方原样, 唯一变化: seed 线程化 + DualViewDiTFormal + build_conds_formal."""
    torch.manual_seed(seed)
    m = DualViewDiTFormal(mode=mode, crossview=crossview).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    print(f"formal {RUN} params={sum(p.numel() for p in m.parameters())/1e6:.1f}M", flush=True)
    from interactive_world_sim.algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity
    from exp_scel_latent_lpips import _decode_grad
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device).eval()
    for p in lp.parameters(): p.requires_grad_(False)
    rng = np.random.default_rng(seed)
    samples = np.array([("r", int(j)) for j in idx_r] + [("h", int(j)) for j in idx_h], dtype=object)
    fr01 = lambda a: a.astype(np.float32).transpose(2, 0, 1) / 255.0
    for ep in range(epochs):
        m.train(); order = rng.permutation(len(samples)); tot = 0; nb = 0
        for i in range(0, len(order), BS):
            bidx = order[i:i + BS]
            z0i, z1i, pvi, gti, cdi, mki, doms = [], [], [], [], [], [], []
            for si in bidx:
                dom, j = samples[si]; D = R if dom == "r" else Hh; lc = latR if dom == "r" else latH
                t = int(rng.integers(9, D["fr"][0].shape[1]))
                cond, mask = build_conds_formal(D, j, t, mode)
                pt = 0 if rng.random() < 0.15 else t - 1
                z0i.append(lc[j, 0].astype(np.float32)); z1i.append(lc[j, t].astype(np.float32)); pvi.append(lc[j, pt].astype(np.float32))
                gti.append([fr01(D["fr"][v][j, t]) for v in range(2)])
                cdi.append(cond); mki.append(mask); doms.append(0 if dom == "r" else 1)
            f2 = lambda a: torch.from_numpy(np.stack(a).astype(np.float32)).to(device)
            z0 = f2(z0i); z1 = f2(z1i); pv = f2(pvi); GT = f2(gti)
            cond = f2(cdi); mask = f2(mki)[:, :, None]; dom = torch.tensor(doms, device=device)
            a = torch.rand(len(bidx), 1, 1, 1, 1, device=device) * PREV_DF; pv = (1 - a) * pv + a * torch.randn_like(pv)
            pred = m(z0, pv, cond)
            img = _decode_grad(pred.reshape(-1, latent_ch(), GRID, GRID)).clamp(0, 1).reshape(len(bidx), 2, 3, IMG, IMG)
            rs = (dom == 0); hs = (dom == 1)
            loss = 0.0 * pred.sum()
            if rs.any():
                loss = loss + ((pred[rs] - z1[rs]) ** 2).mean()
                loss = loss + LAM * lp(img[rs].reshape(-1, 3, IMG, IMG) * 2 - 1, GT[rs].reshape(-1, 3, IMG, IMG) * 2 - 1)
            if hs.any():
                om = mask[hs]; loss = loss + (((img[hs] - GT[hs]) ** 2) * om).sum() / (om.sum() * 3 + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        if ep % 10 == 0 or ep == epochs - 1: print(f"  {RUN} ep{ep} loss={tot/nb:.4f}", flush=True)
    return m.eval()


@torch.no_grad()
def render_formal(m, R, si, mode, pred_tr=None):
    """render H 帧双视角. pred_tr=None: replay GT 条件; 否则 (2,H,P,2) 用 ② 预测 tracks 建 flow cond
    (vis 用最后观测帧 K-1, 可部署口径)."""
    fr01 = lambda a: torch.from_numpy(a.astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to(device)
    z0 = torch.stack([enc(fr01(R["fr"][v][si, 0]))[0] for v in range(2)])[None]
    prev = z0.clone(); outs = [[], []]
    for h in range(H):
        t = K + h
        if pred_tr is None:
            cond, _ = build_conds_formal(R, si, t, mode)
        else:
            assert mode == "flow"
            cs = [flow_cond(R["tr"][v][si, 0], pred_tr[v][h], R["ef"][v][si, 0], R["ef"][v][si, t],
                            R["vs"][v][si, K - 1]) for v in range(2)]
            cond = np.stack(cs)
        cond = torch.from_numpy(cond[None].astype(np.float32)).to(device)
        pred = m(z0, prev, cond); prev = pred
        for v in range(2): outs[v].append(dec(pred[:, v])[0].cpu().numpy())
    return np.stack([np.stack(outs[v]).transpose(0, 2, 3, 1) for v in range(2)])


def run_eval(m, R, chosen, mode, out=None, pred_tr_fn=None):
    """单一权威 eval: 每 seq render + per-view PSNR/obj-LPIPS/det-rate. -> res dict"""
    from interactive_world_sim.algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device).eval()
    agg = {f"v{v}_{k}": [] for v in range(2) for k in ["ps", "lp"]}; det = {0: [0, 0], 1: [0, 0]}
    for si in chosen:
        rr = render_formal(m, R, si, mode, pred_tr=None if pred_tr_fn is None else pred_tr_fn(si))
        if out is not None: np.save(f"{out}/gifs/seq{si}_render.npy", u8(rr))
        for v in range(2):
            gtf = R["fr"][v][si, K:K + H].astype(np.float32) / 255.0
            objm = np.stack([_fp(R["tr"][v][si, K + h]) for h in range(H)])
            agg[f"v{v}_ps"].append(np.mean([psnr(rr[v, h], gtf[h]) for h in range(H)]))
            lpv, nv, nt = obj_lpips_audit(lp, rr[v].astype(np.float32), gtf, objm)
            agg[f"v{v}_lp"].append(lpv); det[v][0] += nv; det[v][1] += nt
    res = {f"v{v}_{k}": float(np.nanmean(agg[f"v{v}_{k}"])) for v in range(2) for k in ["ps", "lp"]}
    for v in range(2): res[f"det_rate_v{v}"] = det[v][0] / max(det[v][1], 1)
    return res


def main():
    import time
    t0 = time.time()
    R, Hh = load_dual()
    okr = np.where(R["ok"])[0]; okh = np.where(Hh["ok"])[0]
    perm = np.random.default_rng(0).permutation(okr)                # split 固定, 独立于 SEED
    ho, pool = perm[:150], perm[150:]
    if MIX == "r": okh = okh[:0]                                    # M1c robot-only 臂 (human-helps 消融)
    if SMOKE: pool = pool[:80]; okh = okh[:80]
    chosen = eval_seqs(R, ho, n=(4 if SMOKE else NEVAL))
    print(f"=== formal {RUN} | robot {len(pool)} + human {len(okh)} | eval n={len(chosen)} ===", flush=True)
    latR = latcache(R, "robot"); latH = latcache(Hh, "human")       # 复用探索版 cache
    m = train_formal(R, Hh, pool, okh, latR, latH, CONDM, CROSSVIEW, SEED, EPOCHS)
    torch.save(m, f"{OUT}/dvdit.pt")
    res = run_eval(m, R, chosen, CONDM, out=OUT)
    res.update({"cond": CONDM, "crossview": int(CROSSVIEW), "mix": MIX, "seed": SEED, "epochs": EPOCHS,
                "wall_min": round((time.time() - t0) / 60, 1), "n_eval": len(chosen)})
    json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=2)
    lines = [f"DUAL-VIEW DiT FORMAL {RUN} | {VERSIONS}",
             f"cam_high: PSNR {res['v0_ps']:.2f} | obj-LPIPS {res['v0_lp']:.3f} | det {res['det_rate_v0']:.2f}",
             f"cam_low : PSNR {res['v1_ps']:.2f} | obj-LPIPS {res['v1_lp']:.3f} | det {res['det_rate_v1']:.2f}",
             f"epochs={EPOCHS} seed={SEED} wall={res['wall_min']}min"]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines) + "\n=== DONE ===", flush=True)


if __name__ == "__main__":
    if MODE == "train": main()
    elif MODE == "e2e": run_e2e()                                   # Task 5
    elif MODE == "gif": run_gif()                                   # Task 4 Step 5
```

- [ ] **Step 4: 单测通过 + SMOKE 走通**

Run: `/scr/yusenluo/anaconda3/envs/iws/bin/python -m pytest tests/test_dualview_formal.py -v`
Expected: 全 PASS。

Run(选空卡,如 GPU2): `cd /scr2/yusenluo/interactive_world_sim && CUDA_VISIBLE_DEVICES=2 SMOKE=1 COND=flow /scr/yusenluo/anaconda3/envs/iws/bin/python exp_scel_dualview_dit_formal.py 2>&1 | tail -20`
Expected: 打印 params、2 个 epoch loss、summary 三行;`outputs/cross_embodiment_wm/dualview_dit_formal/flow_cv1_s0_smoke/{metrics.json,summary.txt,dvdit.pt}` 存在;`det_rate_v*` ∈ (0,1]。

同样 SMOKE 跑 `COND=eefsp` 和 `COND=eeffilm CROSSVIEW=0` 各一遍确认三臂+mask 路径都走通。

- [ ] **Step 5: 记录每 epoch 耗时,推算全量成本**

从 SMOKE 输出估算 min/epoch(全量样本数 ≈ SMOKE 的 (2550+1650)/160 ≈ 26 倍)→ 60ep 单 run 小时数写进 `docs/superpowers/plans/2026-07-12-dualview-dit-formalize.md` 本行下方,若单 run > 12h,与用户确认降 EPOCHS 或减 DEPTH 前不擅自改。实测: ______ min/ep,单 run ≈ ______ h。

- [ ] **Step 6: Commit**

```bash
git add exp_scel_dualview_dit_formal.py tests/test_dualview_formal.py
git commit -m "feat(dualview-formal): seeded trainer + authoritative eval (obj-LPIPS det-rate) + smoke"
```

---

### Task 4: M1b sweep(variance-first 门控)+ 聚合分析 + replay 对比 gif

**Files:**
- Create: `run_dualview_formal_sweep.sh`
- Create: `agg_dualview_formal.py`
- Modify: `exp_scel_dualview_dit_formal.py`(追加 `run_gif()`)

**Interfaces:**
- Consumes: Task 3 的每 run `metrics.json`;`viz_combined.save_combined_gif / build_flow_cols`。
- Produces: `{ROOT}/ablation_table.md`(mean±std 表 + 门控判定);`{ROOT}/compare/gifs/seq{si}_cam{high,low}.gif`。

- [ ] **Step 1: sweep 脚本**

```bash
#!/bin/bash
# run_dualview_formal_sweep.sh — STAGE=A: variance-first (flow/eeffilm cv1 rh x seed0-2, 6 runs)
#                                STAGE=B: 其余 controlled 格子 (eefsp cv1 + 三臂 cv0, 12 runs)
#                                STAGE=C: M1c human-helps (flow/eeffilm cv1 robot-only x seed0-2, 6 runs)
# 格子编码 cond:cv:seed:mix   用法: GPUS="2 3 4 5 6 7" STAGE=A bash run_dualview_formal_sweep.sh
set -e
cd /scr2/yusenluo/interactive_world_sim
PY=/scr/yusenluo/anaconda3/envs/iws/bin/python
GPUS=(${GPUS:-2 3 4 5 6 7}); STAGE=${STAGE:-A}
if [ "$STAGE" = "A" ]; then
  GRID_RUNS=(flow:1:0:rh flow:1:1:rh flow:1:2:rh eeffilm:1:0:rh eeffilm:1:1:rh eeffilm:1:2:rh)
elif [ "$STAGE" = "B" ]; then
  GRID_RUNS=(eefsp:1:0:rh eefsp:1:1:rh eefsp:1:2:rh flow:0:0:rh flow:0:1:rh flow:0:2:rh \
             eefsp:0:0:rh eefsp:0:1:rh eefsp:0:2:rh eeffilm:0:0:rh eeffilm:0:1:rh eeffilm:0:2:rh)
else  # C: M1c human-helps 消融 (robot-only 对照, cv1)
  GRID_RUNS=(flow:1:0:r flow:1:1:r flow:1:2:r eeffilm:1:0:r eeffilm:1:1:r eeffilm:1:2:r)
fi
mkdir -p outputs/cross_embodiment_wm/dualview_dit_formal/logs
i=0
for r in "${GRID_RUNS[@]}"; do
  IFS=: read -r cond cv seed mix <<< "$r"
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  suf=$([ "$mix" = "r" ] && echo "_ronly" || echo "")
  log=outputs/cross_embodiment_wm/dualview_dit_formal/logs/${cond}_cv${cv}_s${seed}${suf}.log
  echo "launch $cond cv$cv s$seed mix=$mix -> GPU$gpu ($log)"
  CUDA_VISIBLE_DEVICES=$gpu COND=$cond CROSSVIEW=$cv SEED=$seed MIX=$mix \
    nohup $PY exp_scel_dualview_dit_formal.py > "$log" 2>&1 &
  i=$((i + 1))
  # 每张卡同时只放一个 run: 塞满一轮就等
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
done
wait
echo "STAGE $STAGE done"
```

- [ ] **Step 2: 聚合/门控脚本**

```python
"""agg_dualview_formal.py — 聚合 metrics.json -> mean±std 消融表 + variance-first 门控判定.
门控 (spec M1): 两视角都要 |mean(flow)-mean(eeffilm)| > 2*pooled_std 才可下 'flow 赢' 结论;
否则输出 EXTEND (加 seed 3,4 或 EPOCHS 90). 用法: python agg_dualview_formal.py"""
import glob
import json

import numpy as np

ROOT = "outputs/cross_embodiment_wm/dualview_dit_formal"
runs = {}
for p in sorted(glob.glob(f"{ROOT}/*/metrics.json")):
    r = json.load(open(p))
    if "smoke" in p: continue
    runs.setdefault((r["cond"], r["crossview"], r.get("mix", "rh")), []).append(r)

lines = ["# dual-view DiT formal — ablation table", "",
         "| cond | crossview | mix | n_seed | v0 obj-LPIPS | v1 obj-LPIPS | v0 PSNR | v1 PSNR | det v0/v1 |",
         "|---|---|---|---|---|---|---|---|---|"]
stat = {}
for (cond, cv, mix), rs in sorted(runs.items()):
    g = lambda k: np.array([x[k] for x in rs])
    stat[(cond, cv, mix)] = {k: (g(k).mean(), g(k).std(ddof=1) if len(rs) > 1 else float("nan"))
                             for k in ["v0_lp", "v1_lp", "v0_ps", "v1_ps"]}
    s = stat[(cond, cv, mix)]
    lines.append(f"| {cond} | {cv} | {mix} | {len(rs)} | {s['v0_lp'][0]:.4f}±{s['v0_lp'][1]:.4f} "
                 f"| {s['v1_lp'][0]:.4f}±{s['v1_lp'][1]:.4f} | {s['v0_ps'][0]:.2f}±{s['v0_ps'][1]:.2f} "
                 f"| {s['v1_ps'][0]:.2f}±{s['v1_ps'][1]:.2f} "
                 f"| {np.mean(g('det_rate_v0')):.2f}/{np.mean(g('det_rate_v1')):.2f} |")

lines.append("")
if ("flow", 1, "rh") in stat and ("eeffilm", 1, "rh") in stat:
    for v in ["v0_lp", "v1_lp"]:
        mf, sf = stat[("flow", 1, "rh")][v]; me, se = stat[("eeffilm", 1, "rh")][v]
        pooled = float(np.sqrt(np.nanmean([sf ** 2, se ** 2])))
        d = me - mf                                        # LPIPS 低好: d>0 = flow 赢
        verdict = "PASS" if (not np.isnan(pooled) and abs(d) > 2 * pooled) else "EXTEND(加seed/epoch)"
        lines.append(f"- gate {v}: flow {mf:.4f} vs eeffilm {me:.4f} | d={d:+.4f} pooled_std={pooled:.4f} -> **{verdict}**")

lines.append("")
for cond in ["flow", "eeffilm"]:                           # M1c human-helps: Δ = LPIPS(r-only) - LPIPS(r+h), >0 = human 帮
    if (cond, 1, "rh") in stat and (cond, 1, "r") in stat:
        for v in ["v0_lp", "v1_lp"]:
            mrh, srh = stat[(cond, 1, "rh")][v]; mr, sr = stat[(cond, 1, "r")][v]
            pooled = float(np.sqrt(np.nanmean([srh ** 2, sr ** 2])))
            dd = mr - mrh
            sig = "显著" if (not np.isnan(pooled) and abs(dd) > 2 * pooled) else "不显著(如实报)"
            lines.append(f"- human-helps {cond} {v}: r-only {mr:.4f} vs r+h {mrh:.4f} | Δ={dd:+.4f} "
                         f"pooled_std={pooled:.4f} -> {sig}")
out = "\n".join(lines) + "\n"
open(f"{ROOT}/ablation_table.md", "w").write(out); print(out)
```

- [ ] **Step 3: `run_gif()` 追加到 formal 脚本(gif protocol)**

```python
def run_gif():
    """replay 对比 gif: GT | flow | eefsp | eeffilm (seed0 cv1), 两视角, save_combined_gif protocol."""
    from viz_combined import save_combined_gif, build_flow_cols
    R, _ = load_dual()
    okr = np.where(R["ok"])[0]; perm = np.random.default_rng(0).permutation(okr); ho = perm[:150]
    chosen = eval_seqs(R, ho)[:6]
    models = {}
    for c in ["flow", "eefsp", "eeffilm"]:
        p = f"{ROOT}/{c}_cv1_s0/dvdit.pt"
        models[c] = torch.load(p, map_location=device, weights_only=False).eval()
    outc = f"{ROOT}/compare"; os.makedirs(f"{outc}/gifs", exist_ok=True)
    vname = {0: "high", 1: "low"}
    for si in chosen:
        rr = {c: render_formal(m, R, si, c) for c, m in models.items()}
        for v in range(2):
            gt = R["fr"][v][si, K:K + H].astype(np.uint8)
            cols = np.stack([gt] + [u8(rr[c][v]) for c in ["flow", "eefsp", "eeffilm"]])
            fc = build_flow_cols(gt, R["tr"][v][si, K:K + H], [None] * 4, R["ef"][v][si, K:K + H])
            save_combined_gif(f"{outc}/gifs/seq{si}_cam{vname[v]}.gif", cols, fc,
                              [f"GT {vname[v]}", "flow(ours)", "eefsp", "eeffilm(IWS-style)"], [None] * 4, K,
                              caption=f"formal 3-arm replay | cam_{vname[v]} | seed0 cv1")
    print(f"saved -> {outc}/gifs/", flush=True)
```

- [ ] **Step 4: launch Stage A(6 runs)并等完**

Run: `GPUS="2 3 4 5 6 7" STAGE=A bash run_dualview_formal_sweep.sh`(后台,数小时;用 `tail -f outputs/cross_embodiment_wm/dualview_dit_formal/logs/*.log` 盯)
Expected: 6 个目录各有 metrics.json。

- [ ] **Step 5: 跑聚合,按门控决策**

Run: `/scr/yusenluo/anaconda3/envs/iws/bin/python agg_dualview_formal.py`
Expected: ablation_table.md 生成;gate 两视角 PASS → launch STAGE=B;EXTEND → 先加 seed{3,4}(改 sweep GRID_RUNS)重跑门控,**老实报"不显著"也是结果,别用单点差声称赢**。

- [ ] **Step 6: Stage B + Stage C(M1c human-helps)完成后再跑聚合 + gif**

Run: `STAGE=B bash run_dualview_formal_sweep.sh && STAGE=C bash run_dualview_formal_sweep.sh && /scr/yusenluo/anaconda3/envs/iws/bin/python agg_dualview_formal.py && MODE=gif /scr/yusenluo/anaconda3/envs/iws/bin/python exp_scel_dualview_dit_formal.py`
Expected: 全表 8 行(3 cond × 2 cv rh + flow/eeffilm cv1 r-only)+ human-helps Δ 判定行 + 12 个 gif。Δ 不显著或为负照实进表(③ 层先验 +0.03 天花板,"不帮在③、帮在②"本身就是 paper 论点的一部分)。

- [ ] **Step 7: Commit**

```bash
git add run_dualview_formal_sweep.sh agg_dualview_formal.py exp_scel_dualview_dit_formal.py
git commit -m "feat(dualview-formal): M1b sweep (variance-first gate) + aggregation table + 3-arm replay gifs"
```

---

### Task 5: M2 端到端 ② pred-flow(4 列 + ② ADE)

**Files:**
- Modify: `exp_scel_dualview_dit_formal.py`(追加 `run_e2e()`)
- Test: `tests/test_dualview_formal.py`(追加 ADE 单测)

**Interfaces:**
- Consumes: `exp_scel_dualview_wm.{rollout_dual, DualLWC}`(import 该模块提供 torch.load 反序列化的类定义;注意它 import 时会 `os.chdir` 到仓库根,可接受);`outputs/cross_embodiment_wm/dualview_wm/wm_dual.pt`;Task 4 训好的 `flow_cv1_s0` 与 `eeffilm_cv1_s0`。
- Produces: `ade_px(pred (2,H,P,2), gt (2,H,P,2)) -> (float, float)` per-view ADE;`{ROOT}/e2e/{metrics.json,summary.txt,gifs/seq*_cam*.gif}`(4 列:GT | GT-flow③ | ②pred-flow③ | eef-FiLM③)。

- [ ] **Step 1: ADE 单测(失败)**

```python
def test_ade_px():
    import numpy as np
    from exp_scel_dualview_dit_formal import ade_px
    pred = np.zeros((2, 3, 4, 2), np.float32); gt = pred.copy()
    gt[0] += 1.0 / 128                                  # view0 每点错 1px
    a0, a1 = ade_px(pred, gt)
    assert abs(a0 - 1.0) < 1e-5 and a1 < 1e-6
```

Run: `... -m pytest tests/test_dualview_formal.py::test_ade_px -v` → FAIL(ImportError)。

- [ ] **Step 2: 实现 `run_e2e`(追加到 formal 脚本)**

```python
def ade_px(pred, gt):
    """per-view 平均位移误差 px. pred/gt (2,H,P,2) crop-norm."""
    e = np.linalg.norm(pred - np.nan_to_num(gt, nan=0.5), axis=-1) * IMG
    return float(e[0].mean()), float(e[1].mean())


def run_e2e():
    """M2: ② pred-flow 驱动 ③. 4 列 GT | GT-flow③(③天花板) | ②pred-flow③(端到端) | eef-FiLM③(naive).
    ② = dualview_wm/wm_dual.pt (rollout_dual), 动作输入 = GT eef (replay actions, 允许)."""
    from viz_combined import save_combined_gif, build_flow_cols
    import exp_scel_dualview_wm as DW
    R, _ = load_dual()
    okr = np.where(R["ok"])[0]; perm = np.random.default_rng(0).permutation(okr); ho = perm[:150]
    chosen = eval_seqs(R, ho)
    wm = torch.load("outputs/cross_embodiment_wm/dualview_wm/wm_dual.pt",
                    map_location=device, weights_only=False).eval()
    mf = torch.load(f"{ROOT}/flow_cv1_s0/dvdit.pt", map_location=device, weights_only=False).eval()
    me = torch.load(f"{ROOT}/eeffilm_cv1_s0/dvdit.pt", map_location=device, weights_only=False).eval()
    P = R["tr"][0].shape[2]
    outd = f"{ROOT}/e2e"; os.makedirs(f"{outd}/gifs", exist_ok=True)
    from interactive_world_sim.algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device).eval()
    cols_lp = {c: {0: [], 1: []} for c in ["gtflow", "e2e", "eef"]}; ades = []
    vname = {0: "high", 1: "low"}
    for gi, si in enumerate(chosen):
        trD = np.concatenate([R["tr"][0][si], np.nan_to_num(R["tr"][1][si], nan=0.5)], 1)[None]  # (1,L,2P,2)
        pr = DW.rollout_dual(wm, torch.from_numpy(trD).float().to(device),
                             torch.from_numpy(R["ef"][0][si][None]).float().to(device),
                             torch.from_numpy(R["ef"][1][si][None]).float().to(device), H).cpu().numpy()[0]
        pred_tr = np.stack([pr[:, :P], pr[:, P:]])                                    # (2,H,P,2)
        gt_tr = np.stack([R["tr"][v][si, K:K + H] for v in range(2)])
        ades.append(ade_px(pred_tr, gt_tr))
        rr = {"gtflow": render_formal(mf, R, si, "flow"),
              "e2e": render_formal(mf, R, si, "flow", pred_tr=pred_tr),
              "eef": render_formal(me, R, si, "eeffilm")}
        for v in range(2):
            gtf = R["fr"][v][si, K:K + H].astype(np.float32) / 255.0
            objm = np.stack([_fp(R["tr"][v][si, K + h]) for h in range(H)])
            for c in rr:
                lpv, _, _ = obj_lpips_audit(lp, rr[c][v].astype(np.float32), gtf, objm)
                cols_lp[c][v].append(lpv)
            if gi < 6:                                                                # gif 只出前 6 条固定 seq
                gt = R["fr"][v][si, K:K + H].astype(np.uint8)
                cols = np.stack([gt, u8(rr["gtflow"][v]), u8(rr["e2e"][v]), u8(rr["eef"][v])])
                fc = build_flow_cols(gt, R["tr"][v][si, K:K + H], [None] * 4, R["ef"][v][si, K:K + H])
                save_combined_gif(f"{outd}/gifs/seq{si}_cam{vname[v]}.gif", cols, fc,
                                  [f"GT {vname[v]}", "GT-flow(③ceiling)", "②pred-flow(e2e)", "eef-FiLM(naive)"],
                                  [None] * 4, K, caption=f"end-to-end ②→③ | cam_{vname[v]} | {VERSIONS[:60]}")
    a = np.array(ades)
    res = {f"{c}_v{v}_lp": float(np.nanmean(cols_lp[c][v])) for c in cols_lp for v in range(2)}
    res.update({"ade_v0_px": float(a[:, 0].mean()), "ade_v1_px": float(a[:, 1].mean()), "n_eval": len(chosen)})
    json.dump(res, open(f"{outd}/metrics.json", "w"), indent=2)
    lines = [f"E2E ②→③ | {VERSIONS}",
             f"② ADE: cam_high {res['ade_v0_px']:.2f}px | cam_low {res['ade_v1_px']:.2f}px",
             f"obj-LPIPS v0: GT-flow {res['gtflow_v0_lp']:.3f} | e2e {res['e2e_v0_lp']:.3f} | eef {res['eef_v0_lp']:.3f}",
             f"obj-LPIPS v1: GT-flow {res['gtflow_v1_lp']:.3f} | e2e {res['e2e_v1_lp']:.3f} | eef {res['eef_v1_lp']:.3f}"]
    open(f"{outd}/summary.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines) + "\n=== DONE ===", flush=True)
```

- [ ] **Step 3: 单测通过 + e2e 跑通**

Run: `pytest tests/test_dualview_formal.py -v` → 全 PASS。
Run: `CUDA_VISIBLE_DEVICES=2 MODE=e2e /scr/yusenluo/anaconda3/envs/iws/bin/python exp_scel_dualview_dit_formal.py`
Expected: summary 4 行;② ADE ≲ 4px(该 ② 在 H=40 时 drift ~2.9/3.1px,H=20 应更小,大很多则查接线);判据 = e2e 列 obj-LPIPS 仍 < eef 列,GT-flow 列为上界。若 ② 太差拉爆 e2e:如实入 report("接口在 GT-flow 上界成立、端到端受 ② 限"),不粉饰。

- [ ] **Step 4: Commit**

```bash
git add exp_scel_dualview_dit_formal.py tests/test_dualview_formal.py
git commit -m "feat(dualview-formal): M2 end-to-end pred-flow (4-col gifs + ADE + per-col obj-LPIPS)"
```

---

### Task 6: M2.5 external baseline — IWS stage2 CMLatentDynamics,DF 训练(用户拍板方案)

**Files:**
- Create: `exp_dualview_iws_stage2.py`
- Test: `tests/test_dualview_formal.py`(追加 3 个单测)

**Interfaces:**
- Consumes: `interactive_world_sim.algorithms.latent_dynamics.models.cm_latent_dynamics.CMLatentDynamics`(原样,不改);formal 脚本的 `obj_lpips_audit / eval_seqs / render 同口径`;同 latcache/VAE。
- Produces: `outputs/cross_embodiment_wm/dualview_iws_stage2/s{seed}/{iws_dyn.pt,metrics.json,summary.txt,gifs/}`。
- **是什么/不是什么(写进 summary + report)**:保留 IWS stage2 的 Conv3d 时空 backbone + 原生 `action_emd`→per-block FiLM 注入 + diffusion-forcing 噪声训练与 AR 多步去噪采样;latent 换成我们的 frozen VAE(同数据同 eval 可比),监督配方与我们的臂一致(robot latent-MSE+decode-LPIPS / human obj-region);**没有**复刻 IWS 的 CTM 蒸馏细节与 conv encoder → 非 controlled、注明 backbone+接口两变量混杂,paper 表作 external baseline 行。

- [ ] **Step 1: 单测(失败)**

```python
def test_iws_sched_monotonic():
    from exp_dualview_iws_stage2 import make_sched
    ab = make_sched(1000)
    assert ab.shape == (1000,) and ab[0] > 0.99 and ab[-1] < 0.01
    assert bool((ab[1:] <= ab[:-1] + 1e-8).all())


def test_iws_frame_actions_shape():
    import numpy as np
    from exp_dualview_iws_stage2 import frame_actions
    D = {"ef": [np.random.rand(5, 48, 3, 2).astype(np.float32) for _ in range(2)]}
    a = frame_actions(D, 3, np.arange(8))
    assert a.shape == (8, 8) and np.isfinite(a).all()


def test_iws_rollout_shape():
    from exp_dualview_iws_stage2 import rollout_iws, make_sched
    from interactive_world_sim.algorithms.latent_dynamics.models.cm_latent_dynamics import CMLatentDynamics
    torch.manual_seed(0)
    m = CMLatentDynamics(latent_dim=8, action_dim=8, dim=16, dim_mults=[1, 2],
                         attn_resolutions=[1], attn_heads=2, attn_dim_head=8).eval()
    z0 = torch.randn(1, 8, 1, 16, 16); acts = torch.randn(1, 6, 8)
    out = rollout_iws(m, z0, acts, Hn=5, sched=make_sched(100), infer_steps=3, t_win=4, device="cpu")
    assert out.shape == (1, 8, 5, 16, 16) and torch.isfinite(out).all()
```

Run: `pytest tests/test_dualview_formal.py -k iws -v` → FAIL(ModuleNotFoundError)。

- [ ] **Step 2: 实现 `exp_dualview_iws_stage2.py`**

```python
"""M2.5 external baseline: 真 IWS stage2 dynamics (CMLatentDynamics: Conv3d 时空 backbone + 原生
action_emd->per-block FiLM 注入) 在 can_dual 上 diffusion-forcing 训练 + AR 多步去噪采样.
latent = 我们的 frozen 16ch VAE (双视角 stack 成 32ch, 同 IWS 双 view 32ch 惯例), 同数据同 eval 口径
(exp_scel_dualview_dit_formal.obj_lpips_audit / eval_seqs). 监督配方与 formal 臂一致.
非 controlled (backbone+接口两变量), paper 表 external baseline 行. 用户 2026-07-12 拍板 DF 方案.
Env: SEED, EPOCHS(40), SMOKE, INFER_STEPS(10), T_WIN(8). iws env + GPU.
-> outputs/cross_embodiment_wm/dualview_iws_stage2/s{SEED}/"""
import json
import os
import time

os.environ.setdefault("VAE_NAME", "ostris/vae-kl-f8-d16"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch, torch.nn as nn
from exp_scel_latent_renderer import enc, dec, latent_ch
from exp_scel_dualview_dit import load_dual, latcache, psnr, _fp, u8, FLOW_SCALE, H
from exp_scel_dualview_dit_formal import obj_lpips_audit, eval_seqs, ROOT as FORMAL_ROOT
from exp_v3_human_helps_pixels import K, IMG, device
from interactive_world_sim.algorithms.latent_dynamics.models.cm_latent_dynamics import CMLatentDynamics

SMOKE = os.environ.get("SMOKE", "0") == "1"
SEED = int(os.environ.get("SEED", "0"))
EPOCHS = 2 if SMOKE else int(os.environ.get("EPOCHS", "40"))
T_WIN = int(os.environ.get("T_WIN", "8")); NLEV = 1000
INFER_STEPS = int(os.environ.get("INFER_STEPS", "10"))
BS = 8; LR = 2e-4; LAM = 1.0
OUT = f"outputs/cross_embodiment_wm/dualview_iws_stage2/s{SEED}" + ("_smoke" if SMOKE else "")
os.makedirs(f"{OUT}/gifs", exist_ok=True)
VERSIONS = ("IWS stage2 CMLatentDynamics (Conv3d+action_emd FiLM, DF train/sample) | "
            "VAE=ostris/vae-kl-f8-d16(frozen, 2view stack 32ch) | data=can_dual | 监督配方同 formal 臂")


def make_sched(n=NLEV):
    """cosine alpha_bar (Nichol&Dhariwal), 单调减."""
    t = np.linspace(0, 1, n)
    ab = np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2 / np.cos(np.array(0.008) / 1.008 * np.pi / 2) ** 2
    return torch.tensor(ab, dtype=torch.float32)


def frame_actions(D, j, ts):
    """clip j 的帧集 ts -> (len(ts), 8) per-frame action: 每 view [x,y,(dx,dy)*FLOW_SCALE step 速度], wrist 点."""
    outs = []
    for v in range(2):
        w = np.asarray(D["ef"][v][j][:, 0], np.float32)               # (L,2) wrist
        prev = np.maximum(np.asarray(ts) - 1, 0)
        outs.append(np.stack([w[ts, 0], w[ts, 1],
                              (w[ts, 0] - w[prev, 0]) * FLOW_SCALE,
                              (w[ts, 1] - w[prev, 1]) * FLOW_SCALE], -1))
    return np.concatenate(outs, -1).astype(np.float32)


def _noise(x0, lv, ab):
    """x0 (B,C,T,16,16), lv (B,T) long -> x_t, 按 per-frame 独立噪声等级 (diffusion forcing)."""
    a = ab.to(x0.device)[lv][:, None, :, None, None]                  # (B,1,T,1,1)
    return a.sqrt() * x0 + (1 - a).sqrt() * torch.randn_like(x0)


def train_iws(R, Hh, pool, okh, latR, latH, sched):
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    m = CMLatentDynamics(latent_dim=2 * latent_ch(), action_dim=8, dim=64).to(device)
    print(f"IWS-DF params={sum(p.numel() for p in m.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    from interactive_world_sim.algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity
    from exp_scel_latent_lpips import _decode_grad
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device).eval()
    for p in lp.parameters(): p.requires_grad_(False)
    samples = np.array([("r", int(j)) for j in pool] + [("h", int(j)) for j in okh], dtype=object)
    fr01 = lambda a: a.astype(np.float32).transpose(2, 0, 1) / 255.0
    for ep in range(EPOCHS):
        m.train(); order = rng.permutation(len(samples)); tot = 0; nb = 0
        for i in range(0, len(order), BS):
            bidx = order[i:i + BS]
            x0s, acts, gts, mks, doms, fsel = [], [], [], [], [], []
            for si in bidx:
                dom, j = samples[si]; D = R if dom == "r" else Hh; lc = latR if dom == "r" else latH
                L = D["fr"][0].shape[1]
                t0 = int(rng.integers(0, L - T_WIN + 1)); ts = np.arange(t0, t0 + T_WIN)
                z = lc[j, ts].astype(np.float32)                       # (T,2,Cz,16,16)
                x0s.append(z.reshape(T_WIN, -1, z.shape[-2], z.shape[-1]).transpose(1, 0, 2, 3))  # (32,T,16,16)
                acts.append(frame_actions(D, j, ts))
                fs = int(rng.integers(0, T_WIN)); fsel.append(fs)      # 像素损失只解码 1 帧/样本 (省算)
                gts.append([fr01(D["fr"][v][j, ts[fs]]) for v in range(2)])
                mks.append(_fp(np.asarray(D["tr"][0][j][ts[fs]], np.float32)) if dom == "h" else np.zeros((IMG, IMG), np.float32))
                doms.append(0 if dom == "r" else 1)
            f2 = lambda a: torch.from_numpy(np.stack(a).astype(np.float32)).to(device)
            x0 = f2(x0s); act = f2(acts); GT = f2(gts); mk = f2(mks)[:, None]
            dom = torch.tensor(doms, device=device); fs = torch.tensor(fsel, device=device)
            lv = torch.randint(0, NLEV, (len(bidx), T_WIN), device=device)
            xt = _noise(x0, lv, sched)
            x0p = m(xt, lv.T, torch.zeros_like(lv.T), act.transpose(0, 1))            # x0-pred, stop=0
            rs = (dom == 0); hs = (dom == 1)
            loss = ((x0p[rs] - x0[rs]) ** 2).mean() if rs.any() else 0.0 * x0p.sum()
            zi = x0p[torch.arange(len(bidx)), :, fs]                                  # (B,32,16,16) 选中帧
            img = _decode_grad(zi.reshape(len(bidx) * 2, latent_ch(), 16, 16)).clamp(0, 1).reshape(len(bidx), 2, 3, IMG, IMG)
            if rs.any():
                loss = loss + LAM * lp(img[rs].reshape(-1, 3, IMG, IMG) * 2 - 1, GT[rs].reshape(-1, 3, IMG, IMG) * 2 - 1)
            if hs.any():
                om = mk[hs][:, :, None] if mk[hs].dim() == 3 else mk[hs]
                om = mk[hs].unsqueeze(2) if mk[hs].dim() == 3 else mk[hs]
                om = mk[hs][:, None] if False else mk[hs].unsqueeze(1).expand(-1, 2, -1, -1).unsqueeze(2)  # (Bh,2,1,H,W)
                loss = loss + (((img[hs] - GT[hs]) ** 2) * om).sum() / (om.sum() * 3 + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        if ep % 5 == 0 or ep == EPOCHS - 1: print(f"  iws-df ep{ep} loss={tot/nb:.4f}", flush=True)
    return m.eval()


@torch.no_grad()
def rollout_iws(m, z0, acts, Hn, sched, infer_steps=INFER_STEPS, t_win=T_WIN, device=device):
    """IWS dynamics_forward 同构 AR 采样: chunk=1, 滑窗 t_win, 每帧 infer_steps 步 x0-pred DDIM 去噪.
    z0 (1,C,1,16,16) 干净首帧; acts (1,Tmax,8). -> (1,C,Hn,16,16)"""
    ab = sched.to(device); xs = z0.to(device).clone()
    for h in range(Hn):
        xs = torch.cat([xs, torch.randn_like(xs[:, :, :1])], 2)
        start = max(0, xs.shape[2] - t_win)
        win = xs[:, :, start:].clone(); Tw = win.shape[2]
        act_w = acts[:, start:start + Tw].transpose(0, 1).to(device)               # (Tw,1,8)
        levels = torch.linspace(NLEV - 1, 0, infer_steps + 1).long()
        for i in range(infer_steps):
            lv = torch.zeros(Tw, 1, dtype=torch.long, device=device); lv[-1] = levels[i]
            x0p = m(win, lv, torch.zeros_like(lv), act_w)
            s_i = int(levels[i + 1])
            if s_i > 0:
                a = ab[s_i]
                win[:, :, -1:] = a.sqrt() * x0p[:, :, -1:] + (1 - a).sqrt() * torch.randn_like(x0p[:, :, -1:])
            else:
                win[:, :, -1:] = x0p[:, :, -1:]
        xs[:, :, start:] = win
    return xs[:, :, 1:]


@torch.no_grad()
def render_iws(m, R, si, sched):
    """同 formal render 口径: z0=frame0 双视角 stack, AR 到 K+H-1, 取帧 K..K+H-1 解码. -> (2,H,128,128,3)"""
    fr01 = lambda a: torch.from_numpy(a.astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to(device)
    z0 = torch.cat([enc(fr01(R["fr"][v][si, 0])) for v in range(2)], 1)[:, :, None]   # (1,32,1,16,16)
    ts = np.arange(0, K + H)
    acts = torch.from_numpy(frame_actions(R, si, ts)[None])                           # (1,K+H,8)
    zs = rollout_iws(m, z0, acts, K + H - 1, sched)[0]                                # (32,K+H-1,16,16)
    outs = [[], []]
    for h in range(H):
        z = zs[:, K + h - 1]                                                          # 帧 K+h (rollout 从帧1计)
        for v in range(2):
            outs[v].append(dec(z[v * latent_ch():(v + 1) * latent_ch()][None])[0].cpu().numpy())
    return np.stack([np.stack(outs[v]).transpose(0, 2, 3, 1) for v in range(2)])


def main():
    t0 = time.time(); sched = make_sched()
    R, Hh = load_dual()
    okr = np.where(R["ok"])[0]; okh = np.where(Hh["ok"])[0]
    perm = np.random.default_rng(0).permutation(okr); ho, pool = perm[:150], perm[150:]
    if SMOKE: pool = pool[:80]; okh = okh[:80]
    chosen = eval_seqs(R, ho, n=(4 if SMOKE else 24))
    latR = latcache(R, "robot"); latH = latcache(Hh, "human")
    m = train_iws(R, Hh, pool, okh, latR, latH, sched)
    torch.save(m, f"{OUT}/iws_dyn.pt")
    from interactive_world_sim.algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device).eval()
    agg = {f"v{v}_{k}": [] for v in range(2) for k in ["ps", "lp"]}; det = {0: [0, 0], 1: [0, 0]}
    for si in chosen:
        rr = render_iws(m, R, si, sched)
        np.save(f"{OUT}/gifs/seq{si}_render.npy", u8(rr))
        for v in range(2):
            gtf = R["fr"][v][si, K:K + H].astype(np.float32) / 255.0
            objm = np.stack([_fp(R["tr"][v][si, K + h]) for h in range(H)])
            agg[f"v{v}_ps"].append(np.mean([psnr(rr[v, h], gtf[h]) for h in range(H)]))
            lpv, nv, nt = obj_lpips_audit(lp, rr[v].astype(np.float32), gtf, objm)
            agg[f"v{v}_lp"].append(lpv); det[v][0] += nv; det[v][1] += nt
    res = {f"v{v}_{k}": float(np.nanmean(agg[f"v{v}_{k}"])) for v in range(2) for k in ["ps", "lp"]}
    for v in range(2): res[f"det_rate_v{v}"] = det[v][0] / max(det[v][1], 1)
    res.update({"seed": SEED, "epochs": EPOCHS, "t_win": T_WIN, "infer_steps": INFER_STEPS,
                "wall_min": round((time.time() - t0) / 60, 1), "n_eval": len(chosen)})
    json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=2)
    lines = [f"IWS stage2 DF external baseline | {VERSIONS}",
             f"cam_high: PSNR {res['v0_ps']:.2f} | obj-LPIPS {res['v0_lp']:.3f} | det {res['det_rate_v0']:.2f}",
             f"cam_low : PSNR {res['v1_ps']:.2f} | obj-LPIPS {res['v1_lp']:.3f} | det {res['det_rate_v1']:.2f}"]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines) + "\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
```

**实现注意(踩坑预防):**
- human 像素损失的 mask 维度那三行冗余是草稿,实现时收敛成一行:`om = mk[hs].unsqueeze(1).expand(-1, 2, -1, -1).unsqueeze(2)` 错了——`img[hs]` 是 `(Bh,2,3,H,W)`,`om` 应为 `(Bh,1,1,H,W)` 直接广播:`om = mk[hs][:, None]`(mk 已是 `(B,1,H,W)`,取 hs 后 `[:, None]` 得 `(Bh,1,1,H,W)`)。以单测/SMOKE 验证 shape。
- human 的 loss mask 用 cam_high footprint 同时罩两视角是近似;更准是 per-view `_fp(D["tr"][v]...)`,实现时按 per-view 做(与 formal 臂一致)。
- `render_iws` 的帧对齐:rollout 产出的第 i 帧对应原始帧 i+1,取原始帧 K+h ⇒ index K+h-1;SMOKE 时肉眼比对 gif 第一列确认没有 off-by-one。
- CMLatentDynamics 默认 `dim_mults=[1,2,4,8]`、attn 全分辨率,params 较大;若 OOM 或过慢,先 `dim=48` 再 `dim_mults=[1,2,4]`,改动记 summary。

- [ ] **Step 3: 单测通过 + SMOKE**

Run: `pytest tests/test_dualview_formal.py -k iws -v` → 3 PASS。
Run: `CUDA_VISIBLE_DEVICES=3 SMOKE=1 /scr/yusenluo/anaconda3/envs/iws/bin/python exp_dualview_iws_stage2.py 2>&1 | tail -8`
Expected: params 打印、2 epoch loss 下降、summary 三行、`s0_smoke/` 产物齐;**肉眼看一条 seq 的 render npy 不是纯噪声**(quick decode 检查)。

- [ ] **Step 4: 全量训练(seed0,可选 seed1)**

Run: `CUDA_VISIBLE_DEVICES=3 nohup /scr/yusenluo/anaconda3/envs/iws/bin/python exp_dualview_iws_stage2.py > outputs/cross_embodiment_wm/dualview_iws_stage2/train_s0.log 2>&1 &`
Expected: metrics.json 落盘。空卡富余时补 `SEED=1`。

- [ ] **Step 5: Commit**

```bash
git add exp_dualview_iws_stage2.py tests/test_dualview_formal.py
git commit -m "feat(dualview-formal): M2.5 external baseline - IWS CMLatentDynamics DF-trained on can_dual, same eval"
```

---

### Task 7: M3 — REPORT + 终版 gif(含 IWS 列)+ drive 上传

**Files:**
- Create: `DUALVIEW_DIT_REPORT.md`(仓库根)
- Modify: `agg_dualview_formal.py`(表里追加 IWS baseline 行,读 `dualview_iws_stage2/s*/metrics.json`,标注 `external, non-controlled`)
- Modify: `exp_scel_dualview_dit_formal.py` `run_gif()`(追加第 5 列 IWS-stage2:直接读 Task 6 存的 `seq{si}_render.npy`)

**Interfaces:**
- Consumes: 全部 metrics.json、ablation_table.md、e2e/summary、gif 目录。
- Produces: 终版对照表(3 controlled 臂 × cv × seed mean±std + e2e 行 + IWS external 行)、终版 gif(GT | flow | eefsp | eeffilm | IWS-stage2 两视角)、drive 上传。

- [ ] **Step 1: agg 加 IWS 行 + run_gif 加第 5 列**

agg 追加(读 `outputs/cross_embodiment_wm/dualview_iws_stage2/s*/metrics.json`,同格式一行,首列 `IWS-stage2(DF, external)`);run_gif 里 `cols` 增加 `np.load(f"outputs/cross_embodiment_wm/dualview_iws_stage2/s0/gifs/seq{si}_render.npy")[v]`,label `"IWS-stage2(external)"`(该 npy 已是 u8)。若某 seq npy 缺失,跳过该列并 log,不悄悄断言。

- [ ] **Step 2: 写 REPORT(自包含)**

`DUALVIEW_DIT_REPORT.md` 章节骨架(每节写实测数字与 gif 绝对路径,不留 TBD):

```markdown
# Dual-View DiT ③ 正式化 REPORT(flow 接口 vs naive action 条件化)
## 0. 一句话结论
## 1. Setup(数据/VAE/backbone/组件版本/训练配方——VERSIONS 字符串展开)
## 2. Controlled 三方消融(M1):表(mean±std, det-rate)+ variance 门控判定 + 解读
   - flow vs eeffilm(接口价值主判据)/ flow vs eefsp(内容 vs 注入机制)/ crossview ON vs OFF(联合注意力价值)
## 2.5 Human-helps(M1c):{flow, eeffilm} × {r+h, r-only} Δ 表 + 解读
   - 回答"flow 接口是否更能利用 human 数据";若 ③ 层不帮(先验 +0.03 天花板)如实写,并接 §3 论证 human-helps 主战场在 ②(flow WM 1.7-2×,project_human_helps_end2end)
## 3. 端到端 ②→③(M2):② ADE + 4 列 obj-LPIPS + 判读(e2e 是否仍赢 naive;②误差吃掉多少)
## 4. External baseline(M2.5):IWS stage2 CMLatentDynamics DF 行 + "是什么/不是什么"声明
## 5. 风险与诚实声明(不显著处如实写;eefsp/eeffilm 若与 flow 差距 < 2σ 必须标注)
## 6. 复现命令 + 全部产物路径(metrics.json / gif / ckpt)
```

- [ ] **Step 3: drive 上传 + 路径交付**

Run: `bash upload_evals_gdrive.sh outputs/cross_embodiment_wm/dualview_dit_formal/compare/gifs outputs/cross_embodiment_wm/dualview_dit_formal/e2e/gifs`(脚本用法先 `head -20 upload_evals_gdrive.sh` 核对;目标 `mygoogle:iws_evals/<日期>/`)
Expected: 上传成功;**若 OAuth 失效,停下来让用户重授权**([[feedback_upload_gifs_gdrive]]),不要绕过。
最后向用户交付:ablation_table.md 内容 + 所有 gif/metrics **绝对路径**([[feedback_give_figure_paths]])。

- [ ] **Step 4: Commit + 记忆更新**

```bash
git add DUALVIEW_DIT_REPORT.md agg_dualview_formal.py exp_scel_dualview_dit_formal.py
git commit -m "docs(dualview-formal): M3 report + final 5-col gifs + drive upload"
```

更新 memory `project_dualview_dit_formalize.md`(现状=完成/结论/产物路径)并同步 HANDOFF.md 与 RENDERER_GRASP_RESEARCH_LOG.md(所有 option 含舍弃记录,[[feedback_log_all_options]])。

---

## Self-Review 结果

1. **Spec 覆盖**:M1a(Task 1-2)、M1b 多 seed+门控(Task 3-4)、M1c human-helps 混合消融(Task 3 MIX 轴 + Task 4 Stage C/Δ 判定,用户 2026-07-12 加回)、eval 审计 det-rate(Task 3)、M2 端到端+ADE(Task 5)、M2.5 external baseline DF 方案(Task 6)、M3 report/gif/drive(Task 7)、单测防 mask 泄漏(Task 1-2)——全覆盖。spec 里 `rollout_dualview` 为笔误,实名 `rollout_dual`,plan 已用实名。
2. **Placeholder 扫描**:Task 3 Step 5 的耗时空格是**实测后回填项**(设计如此);Task 6 Step 2 草稿里三行 mask 冗余已在"实现注意"钦定唯一正确写法。无 TBD。
3. **类型/签名一致性**:`obj_lpips_audit(lp,pred,gt,objm)->(float,int,int)`、`eval_seqs(R,ho,n)->list[int]`、`render_formal(m,R,si,mode,pred_tr)`、`rollout_iws(m,z0,acts,Hn,sched,infer_steps,t_win,device)`、`frame_actions(D,j,ts)->(len(ts),8)` 在各 task 间引用一致。
