"""结合 skel(同域主体,human 帮) + world(绝对残差,robot 精度)的 ② 世界模型。
突破 velocity_action 线的 Pareto(单一 featurization 不可两全)。机制:A1 加性 residual /
align(同网络自蒸馏简化) / align_wm(LaST-HD 2606.23685 faithful:冻结混训 skel-WM 教师 +
cosine latent 对齐,主路径纯 world) / warm(两阶段课程)。精度来源纯 robot,human 只从共享
通道(skel 特征或混训教师)帮进来。
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
    def __init__(s, P, combine="a1", alpha=0.5, lam_res=1e-3, lam_align=0.3, teacher=None, **kw):
        super().__init__(P, **kw)
        s.combine, s.alpha, s.lam_res, s.lam_align = combine, alpha, lam_res, lam_align
        s.act_shared = nn.Linear(V.ACT_DIM["skel"], s.Dm)
        s.act_world = nn.Linear(V.ACT_DIM["world"], s.Dm)
        if combine in ("a1", "warm"):
            with torch.no_grad():                             # 小初始化残差,防独吞
                s.act_world.weight.mul_(0.1); s.act_world.bias.zero_()
        s.teacher = teacher                                   # align_wm:冻结混训 skel-WM(latent 目标)
        if teacher is not None:
            teacher.eval()
            for pm in teacher.parameters(): pm.requires_grad_(False)
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
        if s.combine == "align_wm":                           # LaST-HD faithful:主路径纯 world 保精度,
            x = s.tf(torch.cat([obj, act_w[:, None]], 1))[:, :P]   # latent 拉向冻结教师的前向动力学特征
            if s.training and s.teacher is not None:
                with torch.no_grad():
                    xt, _ = s.teacher.trunk(hist, eef3)
                s.aux["align"] = (1 - nn.functional.cosine_similarity(x, xt, dim=-1)).mean()
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
                elif m.combine in ("align", "align_wm"): regs.append(m.aux["align"] * m.lam_align)
                nxt = buf[:, -1] + m.expected_vel(lg0)
                use_gt = (torch.rand(len(b), 1, 1, device=device) < p)
                buf = torch.cat([buf, torch.where(use_gt, G[:, K + h], nxt.detach())[:, None]], 1)
            loss = torch.stack(losses).mean() + torch.stack(regs).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def train_comb(tracks, vis, eef, idx, combine, Nr, seed=0, alpha=0.5, lam_res=1e-3, lam_align=0.3,
               teacher=None):
    if combine == "align_wm":
        assert teacher is not None, "align_wm 需要冻结的混训 skel-WM 教师(与本臂同 idx 同 seed)"
    torch.manual_seed(seed); P = tracks.shape[2]
    m = CombLWC(P, combine=combine, alpha=alpha, lam_res=lam_res, lam_align=lam_align,
               teacher=teacher, Dm=384, layers=3, W=15, vel_half=V.VEL_HALF).to(device)
    if combine == "warm":
        _run_stage(m, tracks, vis, eef, idx, Nr, 1, SSm.WM_EPOCHS, seed)   # stage1: shared, robot+human
        for pm in list(m.inp.parameters()) + list(m.tf.parameters()) + list(m.act_shared.parameters()):
            pm.requires_grad_(False)                                       # freeze backbone
        r_idx = idx[idx < Nr]                                             # robot-only for精度
        _run_stage(m, tracks, vis, eef, r_idx, Nr, 2, SSm.WM_EPOCHS, seed)
    else:
        _run_stage(m, tracks, vis, eef, idx, Nr, 1, SSm.WM_EPOCHS, seed)
    return m.eval()


OUT = os.environ.get("OUT", "outputs/cross_embodiment_wm/combine_action"); os.makedirs(OUT, exist_ok=True)
DS = os.environ.get("DS", X.DS)
N_ROB_LIST = [int(x) for x in os.environ.get("N_ROB_LIST", "20,50,100,400").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
ANCHORS = ["world", "skel"]                                   # 上下界锚,走 V.train_feat
COMBINES = os.environ.get("COMBINES", "a1,align,align_wm,warm").split(",")
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
            teachers = {}                                     # align_wm 教师=本轮 skel 锚(同 idx 同 seed,Δ 诚实)
            for meth in METHODS:
                if meth in ANCHORS:
                    wm_ro = V.train_feat(mtr, mvs, mef, ri, meth, seed=seed)
                    wm_rh = V.train_feat(mtr, mvs, mef, rih, meth, seed=seed)
                    if meth == "skel":
                        teachers = {"ro": wm_ro, "rh": wm_rh}
                else:
                    t_ro = teachers.get("ro") if meth == "align_wm" else None
                    t_rh = teachers.get("rh") if meth == "align_wm" else None
                    wm_ro = train_comb(mtr, mvs, mef, ri, meth, Nr, seed=seed, teacher=t_ro)
                    wm_rh = train_comb(mtr, mvs, mef, rih, meth, Nr, seed=seed, teacher=t_rh)
                a_ro = X.ade_world(wm_ro, r_tr, r_ef, ho, False)
                a_rh = X.ade_world(wm_rh, r_tr, r_ef, ho, False)
                res[meth][N].append((a_ro, a_rh))
                print(f"  N={N:>4} seed={seed} {meth:>8} | ro {a_ro:6.2f}  rh {a_rh:6.2f}  Δ {a_ro-a_rh:+6.2f}", flush=True)
        _save(res, header)
    _save(res, header); _plot(res)
    print(f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
