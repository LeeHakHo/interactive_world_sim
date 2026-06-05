"""COMPOUNDING-ERROR test (the open question in FLOW_WM_REPORT §5). ② is trained one-shot
(K=4 -> F=20) and teacher-forced; we never fed its own predictions back. Here: true
AUTOREGRESSIVE rollout to horizon H=40, three regimes per model:
  - teacher-forced: hist = GT each step (open-loop multi-step, NO compounding) = upper bound
  - free-run:       hist = own predictions (TRUE autoregressive) = compounding
  - constant-vel:   floor
for robot-only ② and robot+human ②. ADE-vs-horizon curves. The free vs teacher gap = the
compounding error; rh-free vs ro-free = does human data reduce it. eef (= known action) is GT
each step (actions are inputs, not predicted). Reuses e2e FlowWM/train_wm (no architecture
change). Output: outputs/flow_wm/rollout_eval/
"""
import os, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import e2e_flow_wm_render as E
from e2e_flow_wm_render import FlowWM, train_wm  # noqa

OUT = "outputs/flow_wm/rollout_eval"; os.makedirs(OUT, exist_ok=True)
K, F = E.K, E.F                      # 4, 20
device = E.device


def rollout(model, tracks, eef, H, mode):
    # tracks (B,SEQ,P,2), eef (B,SEQ,3,2) on device
    buffer = tracks[:, :K].clone()
    preds = []
    with torch.no_grad():
        for h in range(H):
            hist = buffer[:, -K:]                              # (B,K,P,2)
            eef_win = eef[:, h:h + K + F]                      # (B,K+F,3,2)
            pred = model(hist.permute(0, 2, 1, 3), eef_win)    # (B,P,F,2)
            nxt = pred[:, :, 0, :]                             # next point (t=K+h)
            preds.append(nxt)
            nxt_in = nxt if mode == "free" else tracks[:, K + h]
            buffer = torch.cat([buffer, nxt_in[:, None]], 1)
    return torch.stack(preds, 1)                               # (B,H,P,2)


def cv_rollout(tracks, H):
    buffer = tracks[:, :K].clone(); preds = []
    for h in range(H):
        nxt = 2 * buffer[:, -1] - buffer[:, -2]
        preds.append(nxt); buffer = torch.cat([buffer, nxt[:, None]], 1)
    return torch.stack(preds, 1)


def ade_curve(pred, gt, vis):   # pred/gt (B,H,P,2), vis (B,H,P) -> (H,) px@224
    err = torch.linalg.norm(pred - gt, dim=-1) * 224.0
    return ((err * vis).sum((0, 2)) / (vis.sum((0, 2)) + 1e-6)).cpu().numpy()


def main():
    # ---- train ② robot-only & robot+human (same protocol as e2e) ----
    zr = np.load(f"{E.DS}/clips_robot.npz"); zh = np.load(f"{E.DS}/clips_human.npz")
    r_tr, r_ef, r_vs = zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32), zr["vis"].astype(np.float32)
    h_tr, h_ef, h_vs = zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32), zh["vis"].astype(np.float32)
    Nr = len(r_tr)
    rng = np.random.default_rng(0); perm = rng.permutation(Nr)
    pool = perm[E.HELDOUT_ROB:]
    sub = pool[np.random.default_rng(100).choice(len(pool), min(E.N_ROB, len(pool)), replace=False)]
    rob_idx = torch.from_numpy(sub)
    merged_tr = np.concatenate([r_tr, h_tr]); merged_ef = np.concatenate([r_ef, h_ef]); merged_vs = np.concatenate([r_vs, h_vs])
    hum_idx = torch.arange(Nr, Nr + len(h_tr))
    print(f"② train robot-only N={len(rob_idx)} ; robot+human N={len(rob_idx)+len(hum_idx)}", flush=True)
    wm_ro = train_wm(merged_tr, merged_vs, merged_ef, rob_idx, seed=0)
    wm_rh = train_wm(merged_tr, merged_vs, merged_ef, torch.cat([rob_idx, hum_idx]), seed=0)

    # ---- long rollout sequences ----
    z = np.load(f"{OUT}/seqs.npz"); H = int(z["H"])
    tr = torch.from_numpy(z["tracks"]).float().to(device)      # (B,SEQ,P,2)
    ef = torch.from_numpy(z["eef"]).float().to(device)
    vs = torch.from_numpy(z["vis"]).float().to(device)
    gt = tr[:, K:K + H]; gvis = vs[:, K:K + H]
    print(f"rollout seqs B={tr.shape[0]} SEQ={tr.shape[1]} H={H}", flush=True)

    curves = {}
    for name, m in [("robot-only", wm_ro), ("robot+human", wm_rh)]:
        for mode in ("teacher", "free"):
            curves[f"{name} {mode}"] = ade_curve(rollout(m, tr, ef, H, mode), gt, gvis)
    curves["constant-vel"] = ade_curve(cv_rollout(tr, H), gt, gvis)

    hs = np.arange(1, H + 1)
    L = ["=== COMPOUNDING-ERROR rollout (ADE px@224 vs horizon step) ===",
         f"B={tr.shape[0]} seqs, H={H}, ② trained one-shot K={K}->F={F}", "",
         f"{'regime':<26}" + "".join(f"h={h:<7}" for h in (1, 5, 10, 20, 40))]
    for k in ["robot-only teacher", "robot-only free", "robot+human teacher", "robot+human free", "constant-vel"]:
        c = curves[k]
        L.append(f"{k:<26}" + "".join(f"{c[min(h,H)-1]:<9.2f}" for h in (1, 5, 10, 20, 40)))
    # compounding gap (free - teacher) at h=40, and human effect on free-run
    def at(k, h): return curves[k][min(h, H) - 1]
    L += ["",
          f"compounding gap @h=40 (free-teacher): robot-only={at('robot-only free',40)-at('robot-only teacher',40):+.2f}  "
          f"robot+human={at('robot+human free',40)-at('robot+human teacher',40):+.2f}",
          f"human effect on FREE-RUN @h=40: robot-only={at('robot-only free',40):.2f} -> robot+human={at('robot+human free',40):.2f} "
          f"(Δ={at('robot-only free',40)-at('robot+human free',40):+.2f})",
          f"free-run vs constant-vel @h=40: ro={at('robot-only free',40):.2f} rh={at('robot+human free',40):.2f} cv={at('constant-vel',40):.2f}"]
    print("\n".join(L), flush=True); open(f"{OUT}/summary.txt", "w").write("\n".join(L) + "\n")

    plt.figure(figsize=(9, 6))
    sty = {"robot-only teacher": ("C0", "--"), "robot-only free": ("C0", "-"),
           "robot+human teacher": ("C1", "--"), "robot+human free": ("C1", "-"),
           "constant-vel": ("gray", ":")}
    for k, (c, ls) in sty.items():
        plt.plot(hs, curves[k], ls, color=c, label=k, lw=2)
    plt.xlabel("rollout step (horizon)"); plt.ylabel("ADE (px@224)")
    plt.title("Compounding error: free-run (solid) vs teacher-forced (dashed).\n"
              "gap = compounding; does robot+human (orange) drift less than robot-only (blue)?")
    plt.legend(); plt.grid(alpha=.3)
    plt.savefig(f"{OUT}/compounding_curve.png", dpi=130, bbox_inches="tight")
    print(f"saved {OUT}/compounding_curve.png + summary.txt\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
