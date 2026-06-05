"""Scheduled sampling on ② to fix the compounding (object flow drift in free-run rollout).
Original ② is teacher-forced one-shot (gain 1.07 ro / 1.21 rh -> free-run explodes). Here:
train ② AUTOREGRESSIVELY — predict next object point, feed back GT-or-own-prediction with
teacher-prob p annealed 1.0->0.3, so the model learns to stay stable on its OWN prediction
distribution. eef is the known action plan, held FIXED over the rollout (only object points
recurse), which sidesteps the clip's 24-frame eef-window limit. Compare teacher vs SS, ro vs
rh, on (a) per-step gain and (b) free-run ADE-vs-horizon. Goal: SS pushes gain toward 1 and
flattens the free-run curve; check whether human-helps stops hurting closed-loop.
Output: outputs/flow_wm/scheduled_sampling/
"""
import os, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import e2e_flow_wm_render as E
from e2e_flow_wm_render import FlowWM, train_wm

OUT = "outputs/flow_wm/scheduled_sampling"; os.makedirs(OUT, exist_ok=True)
K, F, device = E.K, E.F, E.device
R_SS = 16                                   # autoregressive rollout steps in SS training
WM_EPOCHS, WM_BS, WM_LR = E.WM_EPOCHS, E.WM_BS, E.WM_LR


def train_wm_ss(tracks, vis, eef, idx, seed=0, p_start=1.0, p_end=0.3):
    torch.manual_seed(seed); P = tracks.shape[2]
    m = FlowWM(P).to(device); opt = torch.optim.AdamW(m.parameters(), lr=WM_LR)
    g = torch.Generator().manual_seed(seed)
    tr = torch.from_numpy(tracks).float(); vs = torch.from_numpy(vis).float(); ef = torch.from_numpy(eef).float()
    for ep in range(WM_EPOCHS):
        p = p_start + (p_end - p_start) * ep / max(WM_EPOCHS - 1, 1)
        m.train(); pe = idx[torch.randperm(len(idx), generator=g)]
        for i in range(0, len(pe), WM_BS):
            b = pe[i:i + WM_BS]
            G = tr[b].to(device); Vv = vs[b].to(device); Ef = ef[b].to(device)  # G(B,L,P,2) Ef(B,L,3,2)
            buf = G[:, :K].clone(); losses = []
            for h in range(R_SS):
                pred = m(buf[:, -K:].permute(0, 2, 1, 3), Ef)        # eef FIXED (whole clip)
                nxt = pred[:, :, 0, :]; gt_h = G[:, K + h]
                w = (Vv[:, K + h] * Vv[:, K - 1])[..., None]
                losses.append(((nxt - gt_h) ** 2 * w).sum() / (w.sum() + 1e-6))
                use_gt = (torch.rand(len(b), 1, 1, device=device) < p)
                buf = torch.cat([buf, torch.where(use_gt, gt_h, nxt.detach())[:, None]], 1)
            loss = torch.stack(losses).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def rollout_free(model, tracks, eef, H):
    buf = tracks[:, :K].clone(); preds = []
    with torch.no_grad():
        for h in range(H):
            p = model(buf[:, -K:].permute(0, 2, 1, 3), eef[:, h:h + K + F])
            preds.append(p[:, :, 0, :]); buf = torch.cat([buf, p[:, :, 0, :][:, None]], 1)
    return torch.stack(preds, 1)


def ade_curve(pred, gt, vis):
    err = torch.linalg.norm(pred - gt, dim=-1) * 224.0
    return ((err * vis).sum((0, 2)) / (vis.sum((0, 2)) + 1e-6)).cpu().numpy()


@torch.no_grad()
def gain(model, hist, eef, sigma=0.01, ndraw=8):
    clean = model(hist, eef)[:, :, 0, :]; gs = []
    gen = torch.Generator(device=device).manual_seed(1)
    for _ in range(ndraw):
        d = torch.randn(hist[:, :, -1].shape, generator=gen, device=device) * sigma
        hp = hist.clone(); hp[:, :, -1] = hp[:, :, -1] + d
        pert = model(hp, eef)[:, :, 0, :]
        gs.append((torch.linalg.norm(pert - clean, dim=-1).mean() / (torch.linalg.norm(d, dim=-1).mean() + 1e-9)).item())
    return float(np.mean(gs))


def main():
    zr = np.load(f"{E.DS}/clips_robot.npz"); zh = np.load(f"{E.DS}/clips_human.npz")
    r_tr, r_ef, r_vs = zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32), zr["vis"].astype(np.float32)
    h_tr, h_ef, h_vs = zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32), zh["vis"].astype(np.float32)
    Nr = len(r_tr); perm = np.random.default_rng(0).permutation(Nr)
    ho = perm[:E.HELDOUT_ROB]; pool = perm[E.HELDOUT_ROB:]
    sub = pool[np.random.default_rng(100).choice(len(pool), min(E.N_ROB, len(pool)), replace=False)]
    mtr = np.concatenate([r_tr, h_tr]); mef = np.concatenate([r_ef, h_ef]); mvs = np.concatenate([r_vs, h_vs])
    ri = torch.from_numpy(sub); hi = torch.arange(Nr, Nr + len(h_tr))

    print("training 4 models (teacher ro/rh, SS ro/rh)...", flush=True)
    models = {
        "ro teacher": train_wm(mtr, mvs, mef, ri, seed=0),
        "rh teacher": train_wm(mtr, mvs, mef, torch.cat([ri, hi]), seed=0),
        "ro SS": train_wm_ss(mtr, mvs, mef, ri, seed=0),
        "rh SS": train_wm_ss(mtr, mvs, mef, torch.cat([ri, hi]), seed=0),
    }

    # gain on held-out clips hist
    hist = torch.from_numpy(r_tr[ho, :K]).float().to(device).permute(0, 2, 1, 3)
    eefh = torch.from_numpy(r_ef[ho]).float().to(device)
    gains = {k: gain(m, hist, eefh) for k, m in models.items()}

    # free-run rollout on long seqs
    z = np.load("outputs/flow_wm/rollout_eval/seqs.npz"); H = int(z["H"])
    tr = torch.from_numpy(z["tracks"]).float().to(device); ef = torch.from_numpy(z["eef"]).float().to(device)
    vs = torch.from_numpy(z["vis"]).float().to(device); gt = tr[:, K:K + H]; gvis = vs[:, K:K + H]
    curves = {k: ade_curve(rollout_free(m, tr, ef, H), gt, gvis) for k, m in models.items()}

    hs = np.arange(1, H + 1)
    L = ["=== Scheduled sampling vs teacher-forced ② (compounding fix) ===",
         f"per-step gain (>1 explodes): " + "  ".join(f"{k}={gains[k]:.3f}" for k in models), "",
         f"{'model':<14}" + "".join(f"h={h:<7}" for h in (1, 5, 10, 20, 40))]
    for k in models: L.append(f"{k:<14}" + "".join(f"{curves[k][min(h,H)-1]:<9.2f}" for h in (1, 5, 10, 20, 40)))
    L += ["",
          f"gain: teacher ro {gains['ro teacher']:.3f} rh {gains['rh teacher']:.3f}  ->  SS ro {gains['ro SS']:.3f} rh {gains['rh SS']:.3f}",
          f"free-run ADE@40: teacher ro {curves['ro teacher'][-1]:.1f} rh {curves['rh teacher'][-1]:.1f}  ->  "
          f"SS ro {curves['ro SS'][-1]:.1f} rh {curves['rh SS'][-1]:.1f}",
          "win if SS gain->~1 and SS free-run ADE@40 << teacher (esp. rh stops exploding); "
          "human-helps-closed-loop if rh SS <= ro SS."]
    print("\n".join(L), flush=True); open(f"{OUT}/summary.txt", "w").write("\n".join(L) + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    sty = {"ro teacher": ("C0", "--"), "rh teacher": ("C1", "--"), "ro SS": ("C0", "-"), "rh SS": ("C1", "-")}
    for k, (c, ls) in sty.items():
        ax[0].plot(hs, curves[k], ls, color=c, label=k, lw=2)
    ax[0].set_xlabel("rollout step"); ax[0].set_ylabel("free-run ADE (px@224)")
    ax[0].set_title("free-run rollout: teacher(dashed) vs scheduled-sampling(solid)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ks = list(models); ax[1].bar(ks, [gains[k] for k in ks], color=["C0", "C1", "C0", "C1"])
    ax[1].axhline(1.0, color="k", ls=":"); ax[1].set_ylabel("per-step gain"); ax[1].set_title("gain (>1 = compounds)")
    ax[1].tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(f"{OUT}/scheduled_sampling.png", dpi=130)
    print(f"saved {OUT}/scheduled_sampling.png + summary.txt\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
