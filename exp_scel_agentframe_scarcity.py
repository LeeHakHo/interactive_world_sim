"""M1 follow-up: scarcity sweep of agent-frame vs world human-helps. Reuses the M1 core
(transform_tracks / train_one / ade_world) from exp_scel_agentframe — does NOT duplicate logic.
Question: does agent-frame keep human-helps at LOW N_rob (where human is supposed to help),
or did it just replace human by making robot-only saturate? Output summary_scarcity.txt."""
import os, numpy as np, torch
import exp_scel_agentframe as X

OUT = X.OUT
HELDOUT = X.HELDOUT
N_ROB_LIST = [int(x) for x in os.environ.get("N_ROB_LIST", "20,50,100,400").split(",")]


def main():
    zr = np.load(f"{X.DS}/clips_robot.npz"); zh = np.load(f"{X.DS}/clips_human.npz")
    r_tr, r_ef, r_vs = (zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32), zr["vis"].astype(np.float32))
    h_tr, h_ef, h_vs = (zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32), zh["vis"].astype(np.float32))
    Nr = len(r_tr); perm = np.random.default_rng(0).permutation(Nr)
    ho, pool = perm[:HELDOUT], perm[HELDOUT:]
    X.tracks_world, X.vis_all = r_tr, r_vs                    # ade_world reads these module globals
    mtr = np.concatenate([r_tr, h_tr]); mef = np.concatenate([r_ef, h_ef]); mvs = np.concatenate([r_vs, h_vs])
    hi = torch.arange(Nr, Nr + len(h_tr))

    lines = ["SCEL M1 scarcity sweep | agent-frame vs world | LWC | held-out robot ADE px@224",
             "(wΔ/afΔ = robot-only minus robot+human = human-helps; >0 means human helps)",
             f"{'N_rob':>6} | {'w_ro':>7} {'w_rh':>7} {'wΔ':>6} | {'af_ro':>7} {'af_rh':>7} {'afΔ':>6}"]
    print("\n".join(lines), flush=True)
    for N in N_ROB_LIST:
        sub = pool[np.random.default_rng(100).choice(len(pool), min(N, len(pool)), replace=False)]
        ri = torch.from_numpy(sub); r = {}
        for af in (False, True):
            wm_ro = X.train_one(mtr, mvs, mef, ri, af)
            wm_rh = X.train_one(mtr, mvs, mef, torch.cat([ri, hi]), af)
            r[af] = (X.ade_world(wm_ro, r_tr, r_ef, ho, af), X.ade_world(wm_rh, r_tr, r_ef, ho, af))
        wro, wrh = r[False]; aro, arh = r[True]
        lines.append(f"{N:>6} | {wro:7.2f} {wrh:7.2f} {wro-wrh:+6.2f} | {aro:7.2f} {arh:7.2f} {aro-arh:+6.2f}")
        print(lines[-1], flush=True)
    open(f"{OUT}/summary_scarcity.txt", "w").write("\n".join(lines) + "\n")
    print(f"\nsaved {OUT}/summary_scarcity.txt\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
