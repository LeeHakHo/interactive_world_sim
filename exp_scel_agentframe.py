# exp_scel_agentframe.py
"""M1: agent-frame relative coordinate as an invertible wrapper around LWC ②.
Compares world-frame LWC (baseline) vs agent-frame LWC on robot held-out ADE (px@224)
and downstream human-helps (robot-only vs robot+human). Output:
outputs/cross_embodiment_wm/scel_m1_agentframe/."""
import os, numpy as np, torch
import scel_repr as S
import amplify_wm as A
import eval_scheduled_sampling as SSm
from amplify_wm import K, F, device

SMOKE = os.environ.get("SMOKE", "0") == "1"
DS = "outputs/flow_render_dataset_v3"
OUT = "outputs/cross_embodiment_wm/scel_m1_agentframe"; os.makedirs(OUT, exist_ok=True)
HELDOUT = 150; H = 16
N_ROB = 100


def transform_tracks(tracks, eef, fwd=True):
    """tracks (N,L,48,2) world<->agent-frame using per-FRAME grasp frame from eef (N,L,3,2)."""
    gf = S.grasp_frame(eef)
    fn = S.to_agent_frame if fwd else S.from_agent_frame
    return fn(tracks, gf).astype(np.float32)


def ade_world(model, tracks, eef, idx, agentframe):
    """rollout in (world|agent) frame; always score in WORLD px@224."""
    src = transform_tracks(tracks, eef) if agentframe else tracks    # rollout INPUT must match train frame
    buf = torch.from_numpy(src[idx, :K]).float().to(device)
    ef = torch.from_numpy(eef[idx]).float().to(device)
    pred = A.rollout_lwc(model, buf, ef, H).cpu().numpy()            # (B,H,48,2) in train frame
    if agentframe:                                                   # back to world with KNOWN grasp frames
        gf = S.grasp_frame(eef[idx])[:, K:K + H]                     # (B,H,5)
        pred = S.from_agent_frame(pred, gf)
    gt = tracks_world[idx][:, K:K + H]
    vis = vis_all[idx][:, K:K + H]
    err = np.linalg.norm(pred - gt, axis=-1) * 224.0
    return float((err * vis).sum() / (vis.sum() + 1e-6))


def train_one(tracks, vis, eef, idx, agentframe, seed=0):
    tr = transform_tracks(tracks, eef) if agentframe else tracks
    return A.train_lwc_ss(tr, vis, eef, idx, seed=seed)


def main():
    global tracks_world, vis_all
    if SMOKE:
        SSm.WM_EPOCHS = 2; SSm.R_SS = 4
    zr = np.load(f"{DS}/clips_robot.npz"); zh = np.load(f"{DS}/clips_human.npz")
    r_tr, r_ef, r_vs = (zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32), zr["vis"].astype(np.float32))
    h_tr, h_ef, h_vs = (zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32), zh["vis"].astype(np.float32))
    Nr = len(r_tr); perm = np.random.default_rng(0).permutation(Nr)
    ho, pool = perm[:HELDOUT], perm[HELDOUT:]
    tracks_world, vis_all = r_tr, r_vs

    mtr = np.concatenate([r_tr, h_tr]); mef = np.concatenate([r_ef, h_ef]); mvs = np.concatenate([r_vs, h_vs])
    hi = torch.arange(Nr, Nr + len(h_tr))
    sub = pool[np.random.default_rng(100).choice(len(pool), min(N_ROB, len(pool)), replace=False)]
    ri = torch.from_numpy(sub)

    lines = [f"SCEL M1 agent-frame vs world | N_rob={N_ROB} | LWC(amplify_wm) | held-out robot ADE px@224",
             f"{'model':>22} | {'ADE':>7}"]
    res = {}
    for agentframe in (False, True):
        tag = "agentframe" if agentframe else "world"
        wm_ro = train_one(mtr, mvs, mef, ri, agentframe)
        wm_rh = train_one(mtr, mvs, mef, torch.cat([ri, hi]), agentframe)
        ade_ro = ade_world(wm_ro, r_tr, r_ef, ho, agentframe)
        ade_rh = ade_world(wm_rh, r_tr, r_ef, ho, agentframe)
        res[tag] = (ade_ro, ade_rh)
        lines += [f"{tag+' robot-only':>22} | {ade_ro:7.2f}",
                  f"{tag+' robot+human':>22} | {ade_rh:7.2f}",
                  f"{tag+' human-helps Δ':>22} | {ade_ro-ade_rh:+7.2f}"]
    w_help = res['world'][0] - res['world'][1]
    a_help = res['agentframe'][0] - res['agentframe'][1]
    lines += ["",
              f"GATE1 agent-frame ADE not worse: agentframe rh {res['agentframe'][1]:.2f} vs world rh {res['world'][1]:.2f}",
              f"GATE2 human-helps preserved/stronger: agentframe Δ {a_help:+.2f} vs world Δ {w_help:+.2f}"]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
