"""② DUAL-VIEW combine action featurization:把 combine 机制(skel 主体/a1 残差/align_wm
LaST-HD 教师对齐)搬到双视角联合预测基础架构上(用户 2026-07-12:正式方法=双视角+DiT)。

在 exp_scel_dualview_wm.DualLWC(2P token+view emb,联合两视角 48pt flow)上加:
  dummy5   = 双视角 DexWM 星座(绝对 rel objc,精度锚,=现有 wm_dual 的 action rep)
  skel     = 双视角 OSCAR 骨架(同域,迁移锚)
  a1       = act_shared(skel) + α·masked(act_world(dummy5)),human 残差置 0
  align_wm = 主路径纯 dummy5 + 冻结混训 skel 教师 cosine latent 对齐(LaST-HD 2606.23685)
数据 can_dual L24(robot 2700 + human 1800,均 L24 统一批);协议同 exp_scel_combine_action:
held-out 150 robot,ro(N_rob) vs rh(+human),Δ=ro−rh,每视角 drift px@128(H=20)。
Env: COMBINES/N_ROB_LIST/SEEDS/SMOKE/OUT。iws env,GPU。
Output: outputs/cross_embodiment_wm/dualview_comb/"""
import os
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

os.chdir("/scr2/yusenluo/interactive_world_sim")
import eval_scheduled_sampling as SSm
import exp_scel_dualview_wm as W
from amplify_wm import K, F, vel_to_class

device = "cuda" if torch.cuda.is_available() else "cpu"
Lw = K + F
IMG = 128; VEL_HALF = 0.06; HELDOUT = 150; H_EVAL = 20
SMOKE = os.environ.get("SMOKE", "0") == "1"
DS = os.environ.get("DS", "outputs/flow_render_dataset_can_dual")
OUT = os.environ.get("OUT", "outputs/cross_embodiment_wm/dualview_comb"); os.makedirs(OUT, exist_ok=True)
N_ROB_LIST = [int(x) for x in os.environ.get("N_ROB_LIST", "50,100,400").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
ANCHORS = [a for a in os.environ.get("ANCHORS", "dummy5,skel").split(",") if a]
COMBINES = [c for c in os.environ.get("COMBINES", "a1,align_wm").split(",") if c]
METHODS = ANCHORS + COMBINES


def feat_skel(eef3):
    """(B,Lw,3,2) -> (B,Lw*10) OSCAR 骨架(同域,objc-free)。同 exp_scel_combine_action.feat_skel。"""
    B = eef3.shape[0]
    wrist = eef3[:, :, 0]
    dwr = torch.cat([torch.zeros_like(wrist[:, :1]), wrist[:, 1:] - wrist[:, :-1]], 1)
    b1 = eef3[:, :, 1] - wrist; b2 = eef3[:, :, 2] - wrist
    db1 = torch.cat([torch.zeros_like(b1[:, :1]), b1[:, 1:] - b1[:, :-1]], 1)
    db2 = torch.cat([torch.zeros_like(b2[:, :1]), b2[:, 1:] - b2[:, :-1]], 1)
    return torch.cat([dwr, b1, b2, db1, db2], -1).reshape(B, -1)


class DualCombLWC(W.DualLWC):
    """DualLWC + combine action 机制。act_world=基类 dummy5 头(双视角绝对);act_shared=双视角 skel。
    fwd_dual 收可选 is_h;缺省全 robot(评价/rollout_dual 兼容)。"""
    def __init__(s, P, combine="dummy5", alpha=0.5, lam_res=1e-3, lam_align=0.3, teacher=None, **kw):
        super().__init__(P, **kw)
        s.combine, s.alpha, s.lam_res, s.lam_align = combine, alpha, lam_res, lam_align
        s.act_shared = nn.Linear(Lw * 10 * 2, s.Dm)
        s.act_stem_h = nn.Linear(Lw * 20, s.Dm)                # stems:human 专属编码器(EgoWAM 式,dummy5 入)
        if combine == "a1":
            with torch.no_grad():                              # 小初始化残差,防独吞
                s.act.weight.mul_(0.1); s.act.bias.zero_()
        s.teacher = teacher                                    # align_wm:冻结混训 skel 教师
        if teacher is not None:
            teacher.eval()
            for pm in teacher.parameters(): pm.requires_grad_(False)
        s.aux = {}

    def _trunk_dual(s, hist, eef_a, eef_b, is_h=None):
        B, P2 = hist.shape[:2]
        anchor = hist[:, :, -1, :]
        obj = s.inp(torch.cat([(hist - anchor[:, :, None]).reshape(B, P2, 2 * K), anchor], -1))
        obj = obj + torch.cat([s.view_emb[0].expand(B, s.P1, s.Dm),
                               s.view_emb[1].expand(B, s.P1, s.Dm)], 1)
        oc_a = anchor[:, :s.P1].mean(1, keepdim=True); oc_b = anchor[:, s.P1:].mean(1, keepdim=True)
        d5a = W.dummy5(eef_a) - oc_a[:, :, None]; d5b = W.dummy5(eef_b) - oc_b[:, :, None]
        d5 = torch.cat([d5a.reshape(B, -1), d5b.reshape(B, -1)], -1)
        if is_h is None:
            is_h = torch.zeros(B, dtype=torch.bool, device=hist.device)
        if s.combine == "dummy5":
            act = s.act(d5)
        elif s.combine == "skel":
            act = s.act_shared(torch.cat([feat_skel(eef_a), feat_skel(eef_b)], -1))
        elif s.combine == "a1":
            act_s = s.act_shared(torch.cat([feat_skel(eef_a), feat_skel(eef_b)], -1))
            residual = s.alpha * s.act(d5) * (~is_h).float()[:, None]      # human -> 0
            s.aux["res_sq"] = residual.pow(2).mean()
            act = act_s + residual
        elif s.combine == "stems":                             # EgoWAM/HPT:每具身独立编码器,无显式对齐
            act = torch.where(is_h[:, None], s.act_stem_h(d5), s.act(d5))
        elif s.combine == "align_wm":                          # LaST-HD:主路径纯世界,latent 拉向冻结教师
            act = s.act(d5)
        else:
            raise ValueError(s.combine)
        x = s.tf(torch.cat([obj, act[:, None]], 1))[:, :P2]
        if s.combine == "align_wm" and s.training and s.teacher is not None:
            with torch.no_grad():
                xt, _ = s.teacher._trunk_dual(hist, eef_a, eef_b)
            s.aux["align"] = (1 - nn.functional.cosine_similarity(x, xt, dim=-1)).mean()
        return x, anchor

    def fwd_dual(s, hist, eef_a, eef_b, is_h=None):
        x, anchor = s._trunk_dual(hist, eef_a, eef_b, is_h)
        B, P2 = x.shape[:2]
        logits = s.head(x).reshape(B, P2, F, s.W * s.W)
        return logits, anchor


def train_dcomb(trD, vsD, efA, efB, idx, combine, Nr, seed=0, teacher=None, alpha=0.5,
                lam_res=1e-3, lam_align=0.3):
    """W.train_dual 协议 + is_h domain mask + combine regs。R_SS=SSm.R_SS(16,L24 上限 20)。"""
    if combine == "align_wm":
        assert teacher is not None, "align_wm 需要冻结混训 skel 教师(同 idx 同 seed)"
    torch.manual_seed(seed); P2 = trD.shape[2]
    m = DualCombLWC(P2 // 2, combine=combine, alpha=alpha, lam_res=lam_res, lam_align=lam_align,
                    teacher=teacher, Dm=384, layers=3, W=15, vel_half=VEL_HALF).to(device)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=SSm.WM_LR)
    g = torch.Generator().manual_seed(seed)
    trT = torch.from_numpy(trD).float(); vsT = torch.from_numpy(vsD).float()
    eAT = torch.from_numpy(efA).float(); eBT = torch.from_numpy(efB).float()
    epochs = 2 if SMOKE else SSm.WM_EPOCHS
    R_SS = 4 if SMOKE else SSm.R_SS
    for ep in range(epochs):
        pteach = 1.0 + (0.3 - 1.0) * ep / max(epochs - 1, 1)
        m.train(); pe = idx[torch.randperm(len(idx), generator=g)]
        for i in range(0, len(pe), SSm.WM_BS):
            b = pe[i:i + SSm.WM_BS]
            is_h = (b >= Nr).to(device)
            G = trT[b].to(device); Vv = vsT[b].to(device)
            Ea = eAT[b].to(device); Eb = eBT[b].to(device)
            buf = G[:, :K].clone(); losses = []; regs = []
            for h in range(R_SS):
                wa = Ea[:, h:h + Lw]; wb = Eb[:, h:h + Lw]
                if wa.shape[1] < Lw:
                    pad = Lw - wa.shape[1]
                    wa = torch.cat([wa, wa[:, -1:].repeat(1, pad, 1, 1)], 1)
                    wb = torch.cat([wb, wb[:, -1:].repeat(1, pad, 1, 1)], 1)
                logits, _ = m.fwd_dual(buf[:, -K:].permute(0, 2, 1, 3), wa, wb, is_h)
                lg0 = logits[:, :, 0, :]
                gt_vel = G[:, K + h] - buf[:, -1]
                cls = vel_to_class(gt_vel, m.W, m.vel_half)
                w = (Vv[:, K + h] * Vv[:, K - 1])
                ce = nn.functional.cross_entropy(lg0.reshape(-1, m.W * m.W), cls.reshape(-1),
                                                 reduction="none")
                losses.append((ce * w.reshape(-1)).sum() / (w.sum() + 1e-6))
                if combine == "a1": regs.append(m.aux["res_sq"] * m.lam_res)
                elif combine == "align_wm": regs.append(m.aux["align"] * m.lam_align)
                nxt = buf[:, -1] + m.expected_vel(lg0)
                use_gt = (torch.rand(len(b), 1, 1, device=device) < pteach)
                buf = torch.cat([buf, torch.where(use_gt, G[:, K + h], nxt.detach())[:, None]], 1)
            loss = torch.stack(losses).mean() + (torch.stack(regs).mean() if regs else 0.0)
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def ade_dual(m, trD, efA, efB, ho):
    """held-out robot rollout H_EVAL -> (drift px cam_high, cam_low)。"""
    pr = W.rollout_dual(m, torch.from_numpy(trD[ho]).float().to(device),
                        torch.from_numpy(efA[ho]).float().to(device),
                        torch.from_numpy(efB[ho]).float().to(device), H_EVAL).cpu().numpy()
    P = trD.shape[2] // 2
    gt = trD[ho][:, K:K + H_EVAL]
    eA = np.linalg.norm(pr[:, :, :P] - gt[:, :, :P], axis=-1) * IMG
    eB = np.linalg.norm(pr[:, :, P:] - gt[:, :, P:], axis=-1) * IMG
    return float(eA.mean()), float(eB.mean())


def _agg(res, meth, N):
    arr = np.array(res[meth][N])                               # (n_seed, 4) roH roL rhH rhL
    if len(arr) == 0: return None
    ro = arr[:, :2].mean(1); rh = arr[:, 2:].mean(1)           # 两视角平均
    d = ro - rh
    return ro.mean(), rh.mean(), d.mean(), d.std(), arr.mean(0)


def _save(res, header):
    lines = [header, f"{'method':>9} {'N':>5} | {'ro':>6} {'rh':>6} | {'Δ':>6} {'±std':>5} | roH/roL rhH/rhL"]
    for meth in METHODS:
        for N in N_ROB_LIST:
            a = _agg(res, meth, N)
            if a is None: continue
            v = a[4]
            lines.append(f"{meth:>9} {N:>5} | {a[0]:6.2f} {a[1]:6.2f} | {a[2]:+6.2f} {a[3]:5.2f} | "
                         f"{v[0]:.2f}/{v[1]:.2f} {v[2]:.2f}/{v[3]:.2f}")
        lines.append("")
    lines.append("BREAKTHROUGH check (rh <= dummy5.rh AND Δ > dummy5.Δ,两视角平均):")
    for N in N_ROB_LIST:
        w = _agg(res, "dummy5", N)
        if w is None: continue
        for meth in [m for m in METHODS if m != "dummy5"]:
            a = _agg(res, meth, N)
            if a is None: continue
            hit = "★BREAK" if (a[1] <= w[1] + 1e-6 and a[2] > w[2] + 0.1) else ""
            lines.append(f"  N={N:>4} {meth:>9}: rh {a[1]:.2f} (dummy5 {w[1]:.2f})  Δ {a[2]:+.2f} (dummy5 {w[2]:+.2f}) {hit}")
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    import json
    json.dump({m: {str(N): v for N, v in d.items()} for m, d in res.items()},
              open(f"{OUT}/res.json", "w"))                   # 供跨 seed-job 合并


def _plot(res):
    fig, ax = plt.subplots(figsize=(7, 6))
    N = max([n for n in N_ROB_LIST if _agg(res, "dummy5", n)], default=N_ROB_LIST[0])
    for meth in METHODS:
        a = _agg(res, meth, N)
        if a is None: continue
        ax.scatter(a[1], a[2], s=90); ax.annotate(meth, (a[1], a[2]), fontsize=10,
                   xytext=(4, 4), textcoords="offset points")
    w = _agg(res, "dummy5", N)
    if w:
        ax.axvline(w[1], color="gray", ls="--", lw=.8); ax.axhline(w[2], color="gray", ls="--", lw=.8)
    ax.set_xlabel("rh drift px@128 (down=better)"); ax.set_ylabel("human-helps D=ro-rh (up=better)")
    ax.set_title(f"dual-view combine (N={N}, mean of 2 views)"); ax.grid(alpha=.3); ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(f"{OUT}/pareto.png", dpi=130); plt.close(fig)


def main():
    global N_ROB_LIST, SEEDS
    if SMOKE:
        N_ROB_LIST = [50]; SEEDS = [0]
    zr = np.load(f"{DS}/clips_robot_L24.npz"); zh = np.load(f"{DS}/clips_human_L24.npz")
    def unpack(z):
        trD = np.concatenate([np.nan_to_num(z["tracks"].astype(np.float32)),
                              np.nan_to_num(z["tracks_low"].astype(np.float32), nan=0.5)], 2)
        vsD = np.concatenate([z["vis"].astype(np.float32), z["vis_low"].astype(np.float32)], 2)
        efA = np.nan_to_num(z["eef"].astype(np.float32))
        efB = np.nan_to_num(z["eef_low"].astype(np.float32), nan=0.5)
        return trD, vsD, efA, efB, z["low_valid"]
    r_tr, r_vs, r_eA, r_eB, r_ok = unpack(zr)
    h_tr, h_vs, h_eA, h_eB, h_ok = unpack(zh)
    Nr = len(r_tr)
    perm = np.random.default_rng(0).permutation(Nr)
    ho = np.array([i for i in perm[:HELDOUT] if r_ok[i]])
    pool = np.array([i for i in perm[HELDOUT:] if r_ok[i]])
    hi_np = Nr + np.where(h_ok)[0]
    if SMOKE: pool = pool[:100]; ho = ho[:24]; hi_np = hi_np[:100]
    hi = torch.from_numpy(hi_np)
    mtr = np.concatenate([r_tr, h_tr]); mvs = np.concatenate([r_vs, h_vs])
    meA = np.concatenate([r_eA, h_eA]); meB = np.concatenate([r_eB, h_eB])
    res = {m: {N: [] for N in N_ROB_LIST} for m in METHODS}
    header = (f"DUAL-VIEW combine | can_dual L24 | held-out robot drift px@{IMG} (H={H_EVAL}) | seeds={SEEDS}\n"
              f"methods={METHODS} (dummy5/skel=锚; a1/align_wm=结合机制) | robot pool {len(pool)} human {len(hi)}\n"
              "Δ = ro − rh (>0 human helps,两视角平均)\n")
    print(header, flush=True)
    for N in N_ROB_LIST:
        for seed in SEEDS:
            sub = pool[np.random.default_rng(100 + seed).choice(len(pool), min(N, len(pool)), replace=False)]
            ri = torch.from_numpy(sub); rih = torch.cat([ri, hi])
            teachers = {}                                      # align_wm 教师=本轮 skel 锚(同 idx 同 seed)
            for meth in METHODS:
                t_ro = teachers.get("ro") if meth == "align_wm" else None
                t_rh = teachers.get("rh") if meth == "align_wm" else None
                wm_ro = train_dcomb(mtr, mvs, meA, meB, ri, meth, Nr, seed=seed, teacher=t_ro)
                wm_rh = train_dcomb(mtr, mvs, meA, meB, rih, meth, Nr, seed=seed, teacher=t_rh)
                if meth == "skel": teachers = {"ro": wm_ro, "rh": wm_rh}
                roH, roL = ade_dual(wm_ro, r_tr, r_eA, r_eB, ho)
                rhH, rhL = ade_dual(wm_rh, r_tr, r_eA, r_eB, ho)
                res[meth][N].append((roH, roL, rhH, rhL))
                if meth in ("dummy5", "align_wm"):              # e2e 配对 ckpt:存 rh 臂(最后 seed 覆盖)
                    try:
                        torch.save(wm_rh, f"{OUT}/wm_{meth}_rh_N{N}.pt")
                    except Exception as e:                      # 磁盘满等:别让 ckpt 保存毁掉 metrics
                        print(f"  [WARN] ckpt save failed: {e}", flush=True)
                print(f"  N={N:>4} seed={seed} {meth:>9} | ro {(roH+roL)/2:6.2f} rh {(rhH+rhL)/2:6.2f} "
                      f"Δ {(roH+roL)/2-(rhH+rhL)/2:+6.2f} | H/L ro {roH:.2f}/{roL:.2f} rh {rhH:.2f}/{rhL:.2f}", flush=True)
            _save(res, header)
    _save(res, header); _plot(res)
    print(f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
