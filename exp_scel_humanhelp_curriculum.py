"""HUMAN-HELP × 课程 × 架构 sweep (bridging-action 2606.28133 启发)。
本 session = 架构轴。三个来自 bridging paper 的机制,作用在 ② 动力学:
  (1) 课程:robot-only vs joint-mix vs human-pretrain→robot-finetune (paper 12→22→38 三段).
  (2) 绑定:pretrain-finetune 本身即"human 先验绑到 robot"(paper 表4 去掉崩38→12).
  (3) 砍旋转:dummy5 全 vs dummy5_trans(只 wrist+center 平移,扔朝向轴)——paper 6DoF 12→平移 22.
架构轴:ARCH=lwc(单视角FlowWM) | dualview(双视角联合) | dit(CDiT式AdaLN注入).
判据 = robot held-out ADE px + human-helps Δ = ADE(robot-only) - ADE(+human) (>0 帮).

Env: DS(单视角), DS_DUAL(双视角npz目录), ARCH, ACT(dummy5|dummy5_trans), N_ROB_LIST, SEEDS, OUT.
复用 exp_scel_velocity_action 的 featurize/ade/数据. iws, GPU.
"""
import os
import numpy as np, torch, torch.nn as nn
import exp_scel_velocity_action as X          # VelLWC/ACT_DIM/Lw/VEL_HALF
import exp_scel_agentframe as AF             # ade_world/H/HELDOUT/tracks_world(protocol)
import eval_scheduled_sampling as SSm
from amplify_wm import FlowWM_LWC, K, vel_to_class
device = X.device

SMOKE = os.environ.get("SMOKE", "0") == "1"
DS = os.environ.get("DS", "outputs/flow_render_dataset_can")
DS_DUAL = os.environ.get("DS_DUAL", "outputs/flow_render_dataset_can_dual")
ARCH = os.environ.get("ARCH", "lwc")           # lwc | dualview | dit
ACT = os.environ.get("ACT", "dummy5")          # dummy5 | dummy5_trans
CURR = os.environ.get("CURR", "all")           # all | robot_only | joint_mix | pretrain_ft
N_ROB_LIST = [int(x) for x in os.environ.get("N_ROB_LIST", "20,100").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
OUT = os.environ.get("OUT", f"outputs/cross_embodiment_wm/humanhelp_{ARCH}_{ACT}"); os.makedirs(OUT, exist_ok=True)
VEL_HALF = X.VEL_HALF


def act_dim(feat, P_view=1):
    if feat == "dummy5": return (K + SSm.F if False else X.Lw) * 10 * P_view
    if feat == "dummy5_trans": return X.Lw * 4 * P_view          # wrist(2)+center(2), 平移only
    return X.ACT_DIM.get(feat, X.Lw * 10) * P_view


def featurize(feat, eef3, objc):
    """eef3 (B,Lw,3,2), objc (B,1,2) -> action feats. dummy5_trans = 只平移(wrist+center)."""
    B = eef3.shape[0]
    if feat == "dummy5_trans":                                    # bridging: translation-only, drop orientation
        wrist = eef3[:, :, 0]; c = (eef3[:, :, 1] + eef3[:, :, 2]) / 2   # wrist + pinch center
        pts = torch.stack([wrist, c], 2)                          # (B,Lw,2,2) 无朝向轴
        return (pts - objc[:, :, None]).reshape(B, -1)
    # dummy5 full
    wrist, t1, t2 = eef3[:, :, 0], eef3[:, :, 1], eef3[:, :, 2]
    c = (t1 + t2) / 2; ax = (t1 - t2) / 2
    perp = torch.stack([-ax[..., 1], ax[..., 0]], -1)
    pts = torch.stack([wrist, c, c + ax, c - ax, c + perp], 2)
    return (pts - objc[:, :, None]).reshape(B, -1)


class CurrLWC(FlowWM_LWC):
    """单视角 ②,可配 dummy5/dummy5_trans + DiT(AdaLN)注入(action 同时作 token 与 AdaLN 调制,不会全崩)."""
    def __init__(s, P, feat="dummy5", arch="lwc", **kw):
        super().__init__(P, **kw); s.feat = feat; s.arch = arch
        s.act = nn.Linear(act_dim(feat), s.Dm)
        if arch == "dit":                                         # CDiT式: action -> per-block scale/shift
            s.ada = nn.Sequential(nn.SiLU(), nn.Linear(s.Dm, 2 * s.Dm))
            nn.init.zeros_(s.ada[1].weight); nn.init.zeros_(s.ada[1].bias)

    def trunk(s, hist, eef3):
        B, P = hist.shape[:2]
        anchor = hist[:, :, -1, :]; objc = anchor.mean(1, keepdim=True)
        obj = s.inp(torch.cat([(hist - anchor[:, :, None]).reshape(B, P, 2 * K), anchor], -1))
        a = s.act(featurize(s.feat, eef3, objc))                  # (B,Dm)
        x = s.tf(torch.cat([obj, a[:, None]], 1))[:, :P]
        if s.arch == "dit":                                       # AdaLN-zero: 起点=identity,不会崩
            scale, shift = s.ada(a).chunk(2, -1)
            x = x * (1 + scale[:, None]) + shift[:, None]
        return x, anchor


class CurrDual(FlowWM_LWC):
    """双视角 ②: 2P token + view emb + per-view dummy5/trans action."""
    def __init__(s, P, feat="dummy5", **kw):
        super().__init__(P, **kw); s.feat = feat; s.P1 = P
        s.view_emb = nn.Parameter(torch.zeros(2, s.Dm))
        s.act = nn.Linear(act_dim(feat, 2), s.Dm)

    def fwd_dual(s, hist, eef_a, eef_b, dom="r"):    # dom 吞掉: 共享 rollout_dual 现传 dom= (域头改动), 本子类单头忽略
        B, P2 = hist.shape[:2]; anchor = hist[:, :, -1, :]
        obj = s.inp(torch.cat([(hist - anchor[:, :, None]).reshape(B, P2, 2 * K), anchor], -1))
        obj = obj + torch.cat([s.view_emb[0].expand(B, s.P1, s.Dm), s.view_emb[1].expand(B, s.P1, s.Dm)], 1)
        oc_a = anchor[:, :s.P1].mean(1, keepdim=True); oc_b = anchor[:, s.P1:].mean(1, keepdim=True)
        fa = featurize(s.feat, eef_a, oc_a); fb = featurize(s.feat, eef_b, oc_b)
        a = s.act(torch.cat([fa, fb], -1))[:, None]
        x = s.tf(torch.cat([obj, a], 1))[:, :P2]
        return s.head(x).reshape(B, P2, SSm.F, s.W * s.W), anchor


def _train_steps(m, tr, ef, vs, idx, seed, ef_b=None, tr_dual=None, epochs=None):
    """SS 训练(单/双视角). 返回训好的 m. epochs=None 用 SSm.WM_EPOCHS(course 分段可短)."""
    opt = torch.optim.AdamW(m.parameters(), lr=SSm.WM_LR); g = torch.Generator().manual_seed(seed)
    EP = epochs or SSm.WM_EPOCHS
    dual = ef_b is not None
    G0 = torch.from_numpy(tr_dual if dual else tr).float()
    Vv0 = torch.from_numpy(vs).float(); Ef0 = torch.from_numpy(ef).float()
    Efb0 = torch.from_numpy(ef_b).float() if dual else None
    for ep in range(EP):
        p = 1.0 + (0.3 - 1.0) * ep / max(EP - 1, 1); m.train()
        pe = idx[torch.randperm(len(idx), generator=g)]
        for i in range(0, len(pe), SSm.WM_BS):
            b = pe[i:i + SSm.WM_BS]
            G = G0[b].to(device); Vv = Vv0[b].to(device); Ef = Ef0[b].to(device)
            Efb = Efb0[b].to(device) if dual else None
            buf = G[:, :K].clone(); losses = []
            for h in range(SSm.R_SS):
                if dual: logits, _ = m.fwd_dual(buf[:, -K:].permute(0, 2, 1, 3), Ef, Efb)
                else: logits, _ = m(buf[:, -K:].permute(0, 2, 1, 3), Ef)
                lg0 = logits[:, :, 0, :]
                gt_vel = G[:, K + h] - buf[:, -1]; cls = vel_to_class(gt_vel, m.W, m.vel_half)
                w = (Vv[:, K + h] * Vv[:, K - 1])
                ce = nn.functional.cross_entropy(lg0.reshape(-1, m.W * m.W), cls.reshape(-1), reduction="none")
                losses.append((ce * w.reshape(-1)).sum() / (w.sum() + 1e-6))
                nxt = buf[:, -1] + m.expected_vel(lg0)
                use_gt = (torch.rand(len(b), 1, 1, device=device) < p)
                buf = torch.cat([buf, torch.where(use_gt, G[:, K + h], nxt.detach())[:, None]], 1)
            loss = torch.stack(losses).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def mk_model(P):
    if ARCH == "dualview": return CurrDual(P, feat=ACT, Dm=384, layers=3, W=15, vel_half=VEL_HALF).to(device)
    return CurrLWC(P, feat=ACT, arch=ARCH, Dm=384, layers=3, W=15, vel_half=VEL_HALF).to(device)


def main():
    if SMOKE:
        SSm.WM_EPOCHS = 2; SSm.R_SS = 4; N_ROB_LIST[:] = [20]; SEEDS[:] = [0]
    dual = ARCH == "dualview"
    if dual:
        zr = np.load(f"{DS_DUAL}/clips_robot_L24.npz"); zh = np.load(f"{DS_DUAL}/clips_human_L24.npz")
        okr, okh = zr["low_valid"], zh["low_valid"]
        def dv(z, ok):
            trA = z["tracks"].astype(np.float32); trB = np.nan_to_num(z["tracks_low"].astype(np.float32), nan=0.5)
            vis2 = np.concatenate([z["vis"].astype(np.float32), z["vis_low"].astype(np.float32)], 2)  # 2P vis
            return (np.concatenate([trA, trB], 2)[ok], z["eef"].astype(np.float32)[ok],
                    np.nan_to_num(z["eef_low"].astype(np.float32), nan=0.5)[ok],
                    vis2[ok], trA[ok])
        r_trD, r_ef, r_efb, r_vs, r_trA = dv(zr, okr); h_trD, h_ef, h_efb, h_vs, _ = dv(zh, okh)
        Nr = len(r_trD); mtrD = np.concatenate([r_trD, h_trD]); mef = np.concatenate([r_ef, h_ef])
        mefb = np.concatenate([r_efb, h_efb]); mvs = np.concatenate([r_vs, h_vs])
    else:
        zr = np.load(f"{DS}/clips_robot.npz"); zh = np.load(f"{DS}/clips_human.npz")
        r_tr, r_ef, r_vs = zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32), zr["vis"].astype(np.float32)
        h_tr, h_ef, h_vs = zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32), zh["vis"].astype(np.float32)
        Nr = len(r_tr); mtr = np.concatenate([r_tr, h_tr]); mef = np.concatenate([r_ef, h_ef]); mvs = np.concatenate([r_vs, h_vs])
    perm = np.random.default_rng(0).permutation(Nr); ho, pool = perm[:AF.HELDOUT], perm[AF.HELDOUT:]
    hi = np.arange(Nr, Nr + (len(h_trD) if dual else len(h_tr)))
    AF.tracks_world, AF.vis_all = (r_trA if dual else r_tr), r_vs

    curricula = ["robot_only", "joint_mix", "pretrain_ft"] if CURR == "all" else [CURR]
    res = {c: {N: [] for N in N_ROB_LIST} for c in curricula}
    hdr = (f"HUMAN-HELP 课程sweep | ARCH={ARCH} ACT={ACT} | robot held-out ADE px | seeds={SEEDS}\n"
           f"曲线: robot_only(N) / joint_mix(N robot+全human) / pretrain_ft(全human预训→N robot finetune)\n"
           f"Δ(human-help)= robot_only ADE - 该曲线 ADE (>0=该策略比纯robot好)\n")
    print(hdr, flush=True)

    def train_c(curr, ri, seed):
        P = (r_trA if dual else r_tr).shape[2]      # 每视角P(48);CurrDual内部拼2P
        m = mk_model(P)
        if dual:
            if curr == "pretrain_ft":
                m = _train_steps(m, None, mef, mvs, torch.from_numpy(hi), seed, ef_b=mefb, tr_dual=mtrD)
                m = _train_steps(m, None, mef, mvs, ri, seed, ef_b=mefb, tr_dual=mtrD)
            else:
                idx = ri if curr == "robot_only" else torch.cat([ri, torch.from_numpy(hi)])
                m = _train_steps(m, None, mef, mvs, idx, seed, ef_b=mefb, tr_dual=mtrD)
        else:
            if curr == "pretrain_ft":
                m = _train_steps(m, mtr, mef, mvs, torch.from_numpy(hi), seed)
                m = _train_steps(m, mtr, mef, mvs, ri, seed)
            else:
                idx = ri if curr == "robot_only" else torch.cat([ri, torch.from_numpy(hi)])
                m = _train_steps(m, mtr, mef, mvs, idx, seed)
        return m

    def ade_eval(m):
        if dual:
            from exp_scel_dualview_wm import rollout_dual
            import exp_scel_flow3d_wm as F3
            P = r_trA.shape[2]
            gt = r_trA[ho][:, K:K + AF.H]
            trD = torch.from_numpy(r_trD[ho]).float().to(device)
            pr = rollout_dual(m, trD, torch.from_numpy(r_ef[ho]).float().to(device),
                              torch.from_numpy(r_efb[ho]).float().to(device), AF.H).cpu().numpy()[:, :, :P]
            return float(np.linalg.norm(pr - gt, axis=-1).mean() * 224)
        return AF.ade_world(m, r_tr, r_ef, ho, False)

    for N in N_ROB_LIST:
        for seed in SEEDS:
            sub = pool[np.random.default_rng(100 + seed).choice(len(pool), min(N, len(pool)), replace=False)]
            ri = torch.from_numpy(sub)
            ade = {}
            for curr in curricula:
                m = train_c(curr, ri, seed); ade[curr] = ade_eval(m)
            base = ade.get("robot_only", list(ade.values())[0])
            for curr in curricula:
                res[curr][N].append(ade[curr])
                print(f"  N={N:>4} seed={seed} {curr:>12} ADE {ade[curr]:6.2f}  Δvs_robot {base-ade[curr]:+6.2f}", flush=True)
        lines = [hdr]
        for curr in curricula:
            for NN in N_ROB_LIST:
                a = res[curr][NN]
                if a: lines.append(f"{curr:>12} N={NN:>4}: ADE {np.mean(a):6.2f} ±{np.std(a):.2f}")
        open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    # verdict
    lines = [hdr, "=== 判决(mean over seeds) ==="]
    for NN in N_ROB_LIST:
        b = np.mean(res["robot_only"][NN]) if "robot_only" in res and res["robot_only"][NN] else None
        for curr in curricula:
            if res[curr][NN]:
                a = np.mean(res[curr][NN]); dv_ = (b - a) if b is not None else 0
                lines.append(f"N={NN:>4} {curr:>12}: ADE {a:6.2f}  Δvs_robot {dv_:+.2f}")
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
