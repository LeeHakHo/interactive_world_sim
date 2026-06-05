"""WHY does robot+human ② diverge under free-run while robot-only saturates? Mechanism test:
the per-step error amplification gain. In autoregressive rollout, the predicted next point
becomes the next anchor; if a perturbation δ on the last history frame (= the just-fed
prediction error) produces |Δpred_next| = gain·|δ|, then gain>1 -> error explodes step over
step, gain<1 -> it decays/saturates. Finite-difference estimate of that gain for robot-only
vs robot+human ②, swept over perturbation σ. Hypothesis: rh gain > ro gain (and >1) -> human
makes ② a sharper, higher-gain map -> positive feedback. Reuses e2e ②. Output:
outputs/flow_wm/rollout_eval/wm_sensitivity.{png,txt}
"""
import os, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import e2e_flow_wm_render as E
from e2e_flow_wm_render import train_wm

OUT = "outputs/flow_wm/rollout_eval"; os.makedirs(OUT, exist_ok=True)
K, F, device = E.K, E.F, E.device
SIGMAS = [0.002, 0.005, 0.01, 0.02, 0.04]   # normalized (×224 ≈ 0.45..9 px)
NDRAW = 8


def main():
    zr = np.load(f"{E.DS}/clips_robot.npz"); zh = np.load(f"{E.DS}/clips_human.npz")
    r_tr, r_ef, r_vs = zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32), zr["vis"].astype(np.float32)
    h_tr, h_ef, h_vs = zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32), zh["vis"].astype(np.float32)
    Nr = len(r_tr); perm = np.random.default_rng(0).permutation(Nr)
    ho = perm[:E.HELDOUT_ROB]; pool = perm[E.HELDOUT_ROB:]
    sub = pool[np.random.default_rng(100).choice(len(pool), min(E.N_ROB, len(pool)), replace=False)]
    mtr = np.concatenate([r_tr, h_tr]); mef = np.concatenate([r_ef, h_ef]); mvs = np.concatenate([r_vs, h_vs])
    wm_ro = train_wm(mtr, mvs, mef, torch.from_numpy(sub), seed=0)
    wm_rh = train_wm(mtr, mvs, mef, torch.cat([torch.from_numpy(sub), torch.arange(Nr, Nr + len(h_tr))]), seed=0)
    print("② trained", flush=True)

    # held-out hist (B,P,K,2) + eef
    hist = torch.from_numpy(r_tr[ho, :K]).float().to(device).permute(0, 2, 1, 3)  # (B,P,K,2)
    eef = torch.from_numpy(r_ef[ho]).float().to(device)
    gen = torch.Generator(device=device).manual_seed(0)

    @torch.no_grad()
    def gain_at(model, sigma):
        clean = model(hist, eef)[:, :, 0, :]           # (B,P,2) next-step pred
        gs = []
        for _ in range(NDRAW):
            d = torch.randn(hist[:, :, -1].shape, generator=gen, device=device) * sigma  # perturb last frame (=fed prediction)
            hp = hist.clone(); hp[:, :, -1] = hp[:, :, -1] + d
            pert = model(hp, eef)[:, :, 0, :]
            num = torch.linalg.norm(pert - clean, dim=-1).mean()
            den = torch.linalg.norm(d, dim=-1).mean()
            gs.append((num / (den + 1e-9)).item())
        return float(np.mean(gs))

    res = {"robot-only": [], "robot+human": []}
    for s in SIGMAS:
        res["robot-only"].append(gain_at(wm_ro, s)); res["robot+human"].append(gain_at(wm_rh, s))
        print(f"σ={s:.3f} (≈{s*224:.1f}px): gain ro={res['robot-only'][-1]:.3f}  rh={res['robot+human'][-1]:.3f}", flush=True)

    L = ["=== ② per-step error-amplification GAIN (|Δpred_next| / |δ_hist|) ===",
         "gain>1 => error grows each free-run step (explodes); gain<1 => decays (saturates)", "",
         f"{'σ (px@224)':<14}" + "".join(f"{s*224:<9.1f}" for s in SIGMAS)]
    L.append(f"{'robot-only':<14}" + "".join(f"{v:<9.3f}" for v in res["robot-only"]))
    L.append(f"{'robot+human':<14}" + "".join(f"{v:<9.3f}" for v in res["robot+human"]))
    ro_m, rh_m = np.mean(res["robot-only"]), np.mean(res["robot+human"])
    L += ["", f"mean gain: robot-only={ro_m:.3f}  robot+human={rh_m:.3f}  (ratio rh/ro={rh_m/ro_m:.2f})",
          "verdict: if rh>ro and rh>1 -> human makes ② a higher-gain map -> positive feedback in free-run,",
          "matching compounding_curve (rh explodes to 101, ro plateaus ~26). gain≈1 boundary = stable/unstable."]
    print("\n".join(L), flush=True); open(f"{OUT}/wm_sensitivity.txt", "w").write("\n".join(L) + "\n")

    plt.figure(figsize=(7.5, 5))
    px = [s * 224 for s in SIGMAS]
    plt.plot(px, res["robot-only"], "o-", label=f"robot-only (mean {ro_m:.2f})", lw=2)
    plt.plot(px, res["robot+human"], "o-", label=f"robot+human (mean {rh_m:.2f})", lw=2)
    plt.axhline(1.0, color="k", ls=":", label="gain=1 (stable/unstable boundary)")
    plt.xlabel("perturbation σ on fed prediction (px@224)"); plt.ylabel("error-amplification gain")
    plt.title("Why human hurts free-run: per-step gain.\ngain>1 ⇒ compounding explodes; rh higher ⇒ human amplifies")
    plt.legend(); plt.grid(alpha=.3); plt.savefig(f"{OUT}/wm_sensitivity.png", dpi=130, bbox_inches="tight")
    print(f"saved {OUT}/wm_sensitivity.png + .txt\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
