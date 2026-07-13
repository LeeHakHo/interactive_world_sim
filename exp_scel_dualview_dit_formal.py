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
    p = f"{ROOT}/eval_seqs_n{n}.json"; os.makedirs(ROOT, exist_ok=True)   # 按 n 分文件, SMOKE(n=4)不污染全量(n=24)
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


def run_gif():
    """replay 对比 gif: GT | flow | eefsp | eeffilm | IWS-stage2(external) (seed0 cv1), 两视角, save_combined_gif protocol.
    第5列读 Task 6 存的 dualview_iws_stage2/s0/gifs/seq{si}_render.npy (u8, (2,H,128,128,3)); 缺失则该 seq 只出 4 列并 log."""
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
    iws_dir = "outputs/cross_embodiment_wm/dualview_iws_stage2/s0/gifs"
    for si in chosen:
        rr = {c: render_formal(m, R, si, c) for c, m in models.items()}
        iws_p = f"{iws_dir}/seq{si}_render.npy"
        iws = np.load(iws_p) if os.path.exists(iws_p) else None
        if iws is None:
            print(f"run_gif: missing {iws_p}, seq{si} rendered with 4 cols (no IWS-stage2 col)", flush=True)
        for v in range(2):
            gt = R["fr"][v][si, K:K + H].astype(np.uint8)
            arm_cols = [gt] + [u8(rr[c][v]) for c in ["flow", "eefsp", "eeffilm"]]
            labels = [f"GT {vname[v]}", "flow(ours)", "eefsp", "eeffilm(IWS-style)"]
            if iws is not None:
                arm_cols.append(iws[v]); labels.append("IWS-stage2(external)")
            cols = np.stack(arm_cols)
            fc = build_flow_cols(gt, R["tr"][v][si, K:K + H], [None] * len(arm_cols), R["ef"][v][si, K:K + H])
            save_combined_gif(f"{outc}/gifs/seq{si}_cam{vname[v]}.gif", cols, fc, labels,
                              [None] * len(arm_cols), K,
                              caption=f"formal 3-arm + IWS-external replay | cam_{vname[v]} | seed0 cv1")
    print(f"saved -> {outc}/gifs/", flush=True)


def ade_px(pred, gt):
    """per-view 平均位移误差 px. pred/gt (2,H,P,2) crop-norm."""
    e = np.linalg.norm(pred - np.nan_to_num(gt, nan=0.5), axis=-1) * IMG
    return float(e[0].mean()), float(e[1].mean())


def run_e2e():
    """M2: ② pred-flow 驱动 ③. 4 列 GT | GT-flow③(③天花板) | ②pred-flow③(端到端) | eef-FiLM③(naive).
    ② = dualview_wm/wm_dual.pt (rollout_dual), 动作输入 = GT eef (replay actions, 允许)."""
    from viz_combined import save_combined_gif, build_flow_cols
    import exp_scel_dualview_wm as DW
    import __main__ as _m                                   # wm_dual.pt 由 exp_scel_dualview_wm 作为 __main__ 保存
    _m.DualLWC = DW.DualLWC                                 # unpickle 需在 __main__ 找到类 (该脚本唯一自定义类, 其余均模块级 import 可解析)
    R, _ = load_dual()
    # ② 须吃 raw view1 tracks(带 NaN)填 0.5, 匹配 wm_dual 训练口径 (exp_scel_dualview_wm.main);
    # load_dual 的 R["tr"][1] 已被 nan_to_num 填 0.0, 直接喂 ② = train/inference 分布错配.
    trB_raw = np.load(f"{dv.DS}/clips_robot.npz")["tracks_low"].astype(np.float32)
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
        trD = np.concatenate([R["tr"][0][si], np.nan_to_num(trB_raw[si], nan=0.5)], 1)[None]  # (1,L,2P,2)
        pr = DW.rollout_dual(wm, torch.from_numpy(trD).float().to(device),
                             torch.from_numpy(R["ef"][0][si][None]).float().to(device),
                             torch.from_numpy(R["ef"][1][si][None]).float().to(device), H).cpu().numpy()[0]
        pred_tr = np.stack([pr[:, :P], pr[:, P:]])                                    # (2,H,P,2)
        gt_tr = np.stack([R["tr"][0][si, K:K + H], trB_raw[si, K:K + H]])  # view1 raw NaN, ade_px 内填 0.5 同 ② eval 口径
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


if __name__ == "__main__":
    if MODE == "train": main()
    elif MODE == "e2e": run_e2e()                                   # Task 5
    elif MODE == "gif": run_gif()                                   # Task 4 Step 5
