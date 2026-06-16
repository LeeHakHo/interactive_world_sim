"""M2 route B: transition-delta CONSISTENCY regularizer (WORLD frame) to AMPLIFY human-helps.

Hypothesis: explicitly pulling robot & human PREDICTED next-step centroid velocity together for
SIMILAR (relative-state, action) transitions transfers human's coverage to robot -> bigger
human-helps than plain mixed training. Centroid-level pairing avoids the robot/human 48-point
non-correspondence (CoTracker seeds points independently per clip).

Compares at a scarce N_rob: robot-only / mixed (=plain human-helps) / mixed+consistency.
Reuses amplify_wm.FlowWM_LWC + exp_scel_agentframe.ade_world (world frame). Output:
outputs/cross_embodiment_wm/scel_m2_consistency/."""
import os, numpy as np, torch, torch.nn as nn
import amplify_wm as A
import eval_scheduled_sampling as SSm
import exp_scel_agentframe as X
from amplify_wm import K, device, vel_to_class

SMOKE = os.environ.get("SMOKE", "0") == "1"
DS = "outputs/flow_render_dataset_v3"
OUT = "outputs/cross_embodiment_wm/scel_m2_consistency"; os.makedirs(OUT, exist_ok=True)
HELDOUT = 150; H = 16; P = 48
N_ROB = int(os.environ.get("N_ROB", "50"))
LAM = float(os.environ.get("LAM", "1.0"))
NORM_VEL = os.environ.get("NORM_VEL", "0") == "1"     # consistency on per-domain speed-normalized centroid vel (remove 0.987 magnitude gap)
BS = 64


def train_consist(tr, vs, ef, ridx, hidx, lam, seed=0, scale=(1.0, 1.0)):
    """SS-trained LWC (world frame). If lam>0 and human present, each batch = half robot + half
    human and a centroid-velocity consistency term is added on cross-domain nearest-neighbor pairs
    (by [anchor centroid - grasp pos, next grasp action]). lam=0 -> plain mixed/robot-only SS."""
    torch.manual_seed(seed)
    m = A.FlowWM_LWC(P, Dm=384, layers=3, W=15, vel_half=0.12).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=SSm.WM_LR)
    G = torch.from_numpy(tr).float(); Vv = torch.from_numpy(vs).float(); Ef = torch.from_numpy(ef).float()
    gen = torch.Generator().manual_seed(seed)
    ri = np.asarray(ridx); hi = np.asarray(hidx)
    use_c = lam > 0 and len(hi) > 0
    use_half = len(hi) > 0                                              # half robot+half human batch when human present
    half = BS // 2
    for ep in range(SSm.WM_EPOCHS):
        p = 1.0 + (0.3 - 1.0) * ep / max(SSm.WM_EPOCHS - 1, 1)
        m.train()
        if use_half:                                                   # half robot (cycled, pool may be small) + half human
            hp = hi[torch.randperm(len(hi), generator=gen).numpy()]
            nb = max(1, len(hp) // half)
            rp = ri[torch.randint(len(ri), (nb * half,), generator=gen).numpy()]
        else:                                                          # robot-only: plain full batches
            allidx = ri[torch.randperm(len(ri), generator=gen).numpy()]
            nb = max(1, len(allidx) // BS)
        for bi in range(nb):
            if use_half:
                rb = rp[bi * half:(bi + 1) * half]; hb = hp[bi * half:(bi + 1) * half]
                b = np.concatenate([rb, hb]); n_r = len(rb)
            else:
                b = allidx[bi * BS:(bi + 1) * BS]; n_r = 0
            Gb = G[b].to(device); Vb = Vv[b].to(device); Eb = Ef[b].to(device)
            buf = Gb[:, :K].clone(); losses = []; closs = torch.zeros((), device=device)
            for h in range(SSm.R_SS):
                logits, _ = m(buf[:, -K:].permute(0, 2, 1, 3), Eb)      # (B,P,F,WW)
                lg0 = logits[:, :, 0, :]
                gt_vel = Gb[:, K + h] - buf[:, -1]
                cls = vel_to_class(gt_vel, m.W, m.vel_half)
                w = (Vb[:, K + h] * Vb[:, K - 1])
                ce = nn.functional.cross_entropy(lg0.reshape(-1, m.W * m.W), cls.reshape(-1), reduction="none")
                losses.append((ce * w.reshape(-1)).sum() / (w.sum() + 1e-6))
                vel_pred = m.expected_vel(lg0)                          # (B,P,2)
                if use_c and h == 0:
                    cvel = vel_pred.mean(1)                             # (B,2) centroid vel
                    anc = buf[:, -1].mean(1)                            # (B,2) anchor centroid
                    rel = anc - Eb[:, K - 1, 0, :]                      # rel to grasp base
                    act = Eb[:, K, 0, :] - Eb[:, K - 1, 0, :]          # next grasp action
                    key = torch.cat([rel, act], 1)                      # (B,4)
                    kr, kh = key[:n_r], key[n_r:]; cr, ch = cvel[:n_r], cvel[n_r:]
                    if NORM_VEL:                                        # drop per-domain speed magnitude (probe 0.987) -> align motion pattern only
                        cr = cr / scale[0]; ch = ch / scale[1]
                    d = torch.cdist(kr, kh)                             # (n_r, n_h)
                    nn_i = d.argmin(1); dmin = d.min(1).values
                    sim = torch.exp(-dmin / (dmin.mean() + 1e-6))
                    closs = (sim * (cr - ch[nn_i]).pow(2).sum(1)).mean()
                nxt = buf[:, -1] + vel_pred
                use_gt = (torch.rand(len(b), 1, 1, device=device) < p)
                buf = torch.cat([buf, torch.where(use_gt, Gb[:, K + h], nxt.detach())[:, None]], 1)
            loss = torch.stack(losses).mean() + lam * closs
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def main():
    if SMOKE:
        SSm.WM_EPOCHS = 2; SSm.R_SS = 4
    zr = np.load(f"{DS}/clips_robot.npz"); zh = np.load(f"{DS}/clips_human.npz")
    r_tr, r_ef, r_vs = zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32), zr["vis"].astype(np.float32)
    h_tr, h_ef, h_vs = zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32), zh["vis"].astype(np.float32)
    Nr = len(r_tr); perm = np.random.default_rng(0).permutation(Nr); ho, pool = perm[:HELDOUT], perm[HELDOUT:]
    X.tracks_world, X.vis_all = r_tr, r_vs                              # ade_world reads these
    mtr = np.concatenate([r_tr, h_tr]); mef = np.concatenate([r_ef, h_ef]); mvs = np.concatenate([r_vs, h_vs])
    hi = np.arange(Nr, Nr + len(h_tr))
    sub = pool[np.random.default_rng(100).choice(len(pool), min(N_ROB, len(pool)), replace=False)]

    LAM_LIST = [float(x) for x in os.environ.get("LAM_LIST", "100,1000,10000").split(",")]
    print(f"=== robot-only / mixed / mixed+consist sweep (N_rob={N_ROB}, lams={LAM_LIST}) ===", flush=True)
    a_ro = X.ade_world(train_consist(mtr, mvs, mef, sub, np.array([], int), 0.0), r_tr, r_ef, ho, False)
    a_mix = X.ade_world(train_consist(mtr, mvs, mef, sub, hi, 0.0), r_tr, r_ef, ho, False)
    rcv = (r_tr[:, 1:].mean(2) - r_tr[:, :-1].mean(2)).reshape(-1, 2)       # robot centroid velocity
    hcv = (h_tr[:, 1:].mean(2) - h_tr[:, :-1].mean(2)).reshape(-1, 2)
    scale = (float(np.sqrt((rcv ** 2).sum(1).mean())) + 1e-6, float(np.sqrt((hcv ** 2).sum(1).mean())) + 1e-6)
    lines = [f"SCEL M2 route-B transition-delta consistency | world frame | N_rob={N_ROB} NORM_VEL={NORM_VEL} | held-out robot ADE px@224",
             f"centroid-vel scale robot={scale[0]:.4f} human={scale[1]:.4f}",
             f"{'model':>28} | {'ADE':>7} | {'gain vs mixed':>13}",
             f"{'robot-only':>28} | {a_ro:7.2f} | {'':>13}",
             f"{'mixed (plain human-helps)':>28} | {a_mix:7.2f} | {'(baseline)':>13}"]
    print("\n".join(lines), flush=True)
    for lam in LAM_LIST:
        a_con = X.ade_world(train_consist(mtr, mvs, mef, sub, hi, lam, scale=scale), r_tr, r_ef, ho, False)
        lines.append(f"{'mixed+consist lam=' + str(lam):>28} | {a_con:7.2f} | {a_mix - a_con:+13.2f}")
        print(lines[-1], flush=True)
    lines += ["", f"plain human-helps (robot-only - mixed) = {a_ro - a_mix:+.2f}",
              ">0 gain vs mixed = consistency amplifies human-helps beyond plain mixing"]
    open(f"{OUT}/{'summary_normvel' if NORM_VEL else 'summary'}.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
