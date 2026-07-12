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
from exp_scel_dualview_dit import (DualViewDiT, flow_cond, load_dual, latcache,
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
    """第三臂 eefsp: 只 splat 3 个 eef 点 (无物体 flow), 与 flow 臂同构 3ch [dx,dy,eef 目标位置密度].
    ≈ OSCAR (2606.04463) 最强行 '2D skeleton 空间渲染条件' 的轻量类比."""
    fm = splat128(ef0, eft, np.ones(len(ef0), np.float32))
    return np.stack([fm[0], fm[1], fm[3]]).astype(np.float32)


def eef_vec(ef0, eft):
    """eeffilm 臂全量 eef 向量: 3 点(wrist+2指尖) x [x,y,dx,dy] -> (12,). 指尖隐含 grip 开合.
    对齐 IWS stage2 原生 '完整动作向量(pos+euler+grip)进 action_emd' 的精神; 单 wrist 点是 strawman."""
    d = (eft - ef0) * FLOW_SCALE
    return np.concatenate([eft, d], -1).reshape(-1).astype(np.float32)   # (3,4)->(12,)


def build_conds_formal(D, j, t, mode):
    """clip j frame t 的 per-view cond + loss mask (物体 footprint, cond 无关, 三臂同)."""
    conds, masks = [], []
    for v in range(2):
        tr, ef, vs = D["tr"][v], D["ef"][v], D["vs"][v]
        if mode == "flow":
            c = flow_cond(tr[j, 0], tr[j, t], ef[j, 0], ef[j, t], vs[j, t])
        elif mode == "eefsp":
            c = eefsp_cond(ef[j, 0], ef[j, t])
        else:                                              # eeffilm: 全量 12-dim/view
            c = eef_vec(ef[j, 0], ef[j, t])
        conds.append(c); masks.append(_fp(tr[j, t]))
    return np.stack(conds), np.stack(masks)


class DualViewDiTFormal(DualViewDiT):
    """探索版 DualViewDiT + (a) 第三臂 eefsp 走 flow 的 spatial-add 通路 (b) crossview=False 时
    attention-mask 阻断 view0<->view1 token (参数量不变, spec M1a)."""
    def __init__(s, mode=CONDM, crossview=CROSSVIEW, **kw):
        assert mode in ("flow", "eefsp", "eeffilm")
        super().__init__(cond=("eef" if mode == "eeffilm" else "flow"), **kw)
        s.mode, s.crossview = mode, crossview
        if mode == "eeffilm":                              # 全量 24-dim (2view x 12) 换掉父类 4-dim act_emd
            D_ = s.D
            s.act_emd = nn.Sequential(nn.Linear(24, 128), nn.SiLU(), nn.Linear(128, D_))
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
            cvec = s.act_emd(cond.reshape(B, 24))          # 双视角 concat 全量向量 (非 mean, IWS 式单一动作向量)
        x = torch.cat(toks, 1)
        for b in s.blocks: x = b(x, cvec, attn_mask=s.attn_mask)
        zs = []
        for v in range(2):
            f = s.norm(x[:, v * 2 * GRID * GRID: v * 2 * GRID * GRID + GRID * GRID])
            zs.append(s.out(f).transpose(1, 2).reshape(B, s.zdim, GRID, GRID))
        return torch.stack(zs, 1)
