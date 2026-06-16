"""Visualize ② rollout predicted object-flow vs GT, world-frame vs agent-frame, on held-out robot.
Reuses M1 core (train_one / transform_tracks) from exp_scel_agentframe. Overlays predicted cube
centroid + trajectory (GT green / world red / agent-frame blue) on real frames; per-seq gifs +
static grid + per-seq ADE. Output: outputs/cross_embodiment_wm/scel_m1_flow_viz/."""
import os, numpy as np, torch, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
import exp_scel_agentframe as X
import amplify_wm as A
from amplify_wm import K

OUT = "outputs/cross_embodiment_wm/scel_m1_flow_viz"; os.makedirs(OUT, exist_ok=True)
H = X.H; HELDOUT = X.HELDOUT; IMG = 128; NSEQ = 6; N_ROB = 100
GT_C, W_C, A_C = (0, 255, 0), (255, 0, 0), (0, 0, 255)   # RGB: GT green, world red, agentframe blue


def rollout_world(model, r_tr, r_ef, idx, agentframe):
    """predicted cube points (len(idx),H,48,2) in WORLD frame."""
    src = X.transform_tracks(r_tr, r_ef) if agentframe else r_tr
    buf = torch.from_numpy(src[idx, :K]).float().to(X.device)
    ef = torch.from_numpy(r_ef[idx]).float().to(X.device)
    pred = A.rollout_lwc(model, buf, ef, H).cpu().numpy()
    if agentframe:
        gf = X.S.grasp_frame(r_ef[idx])[:, K:K + H]
        pred = X.S.from_agent_frame(pred, gf)
    return pred


def px(pt): return (int(np.clip(pt[0], 0, 1) * IMG), int(np.clip(pt[1], 0, 1) * IMG))


def draw_frame(frame, gtc, wc, ac, h):
    """frame (128,128,3) RGB uint8; *c (H,2) centroid trajectories; h = current step (draw up to h)."""
    im = np.ascontiguousarray(frame)
    for traj, col in [(gtc, GT_C), (wc, W_C), (ac, A_C)]:
        for t in range(1, h + 1):
            cv2.line(im, px(traj[t - 1]), px(traj[t]), col, 1)
        cv2.circle(im, px(traj[h]), 3, col, -1)
    return im


def draw_fp(frame, gtp, wp, ap):
    """convex-hull footprint outlines (what ③ fills as the cube): GT green / world red / agentframe blue.
    *p are (48,2) normalized predicted/GT cube points at one frame."""
    im = np.ascontiguousarray(frame)
    for pts, col in [(gtp, GT_C), (wp, W_C), (ap, A_C)]:
        p = np.clip(pts * IMG, 0, IMG - 1).astype(np.int32).reshape(-1, 1, 2)
        if len(p) >= 3:
            cv2.polylines(im, [cv2.convexHull(p)], True, col, 1)
    return im


def main():
    zr = np.load(f"{X.DS}/clips_robot.npz"); zh = np.load(f"{X.DS}/clips_human.npz")
    r_tr, r_ef, r_vs = zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32), zr["vis"].astype(np.float32)
    r_fr = zr["frames"]
    h_tr, h_ef, h_vs = zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32), zh["vis"].astype(np.float32)
    Nr = len(r_tr); perm = np.random.default_rng(0).permutation(Nr); ho, pool = perm[:HELDOUT], perm[HELDOUT:]
    X.tracks_world, X.vis_all = r_tr, r_vs
    mtr = np.concatenate([r_tr, h_tr]); mef = np.concatenate([r_ef, h_ef]); mvs = np.concatenate([r_vs, h_vs])
    hi = torch.arange(Nr, Nr + len(h_tr))
    sub = pool[np.random.default_rng(100).choice(len(pool), min(N_ROB, len(pool)), replace=False)]
    ri = torch.cat([torch.from_numpy(sub), hi])               # robot+human (best models for viz)

    print("training world rh ...", flush=True); wm_w = X.train_one(mtr, mvs, mef, ri, False)
    print("training agent-frame rh ...", flush=True); wm_a = X.train_one(mtr, mvs, mef, ri, True)

    gt = r_tr[ho][:, K:K + H]                                 # (HO,H,48,2)
    mot = np.linalg.norm(np.diff(gt.mean(2), axis=1), axis=-1).sum(1)
    chosen = list(np.argsort(-mot)[:NSEQ])                    # high-motion seqs
    idx = ho[chosen]
    pw = rollout_world(wm_w, r_tr, r_ef, idx, False)
    pa = rollout_world(wm_a, r_tr, r_ef, idx, True)
    gtc = gt[chosen].mean(2); pwc = pw.mean(2); pac = pa.mean(2)   # centroids (NSEQ,H,2)
    vis = r_vs[idx][:, K:K + H]
    ade_w = (np.linalg.norm(pw - gt[chosen], axis=-1) * 224 * vis).sum((1, 2)) / (vis.sum((1, 2)) + 1e-6)
    ade_a = (np.linalg.norm(pa - gt[chosen], axis=-1) * 224 * vis).sum((1, 2)) / (vis.sum((1, 2)) + 1e-6)

    cols = list(range(0, H, max(1, H // 6)))
    fig, ax = plt.subplots(NSEQ, len(cols), figsize=(2 * len(cols), 2 * NSEQ), squeeze=False)
    lines = ["SCEL M1 flow viz | GT(green) world-pred(red) agentframe-pred(blue) | held-out robot, high-motion",
             f"{'seq':>4} | {'world ADE':>9} | {'agentframe ADE':>14}"]
    for si, s in enumerate(chosen):
        for ci, h in enumerate(cols):
            im = draw_frame(r_fr[idx[si], K + h], gtc[si], pwc[si], pac[si], h)
            ax[si, ci].imshow(im); ax[si, ci].axis("off")
            if ci == 0: ax[si, ci].set_title(f"seq{s} w{ade_w[si]:.1f} a{ade_a[si]:.1f}", fontsize=7, loc="left")
        gif = [Image.fromarray(draw_frame(r_fr[idx[si], K + h], gtc[si], pwc[si], pac[si], h)) for h in range(H)]
        gif[0].save(f"{OUT}/seq{s}.gif", save_all=True, append_images=gif[1:], duration=180, loop=0)
        lines.append(f"{s:>4} | {ade_w[si]:9.2f} | {ade_a[si]:14.2f}")
    fig.suptitle("GT(green) / world-pred(red) / agent-frame-pred(blue) cube centroid+trajectory", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/flow_grid.png", dpi=115); plt.close(fig)

    # footprint (convex-hull) viz — closer to what ③ actually renders (cube = filled hull of 48 pred pts)
    fig2, ax2 = plt.subplots(NSEQ, len(cols), figsize=(2 * len(cols), 2 * NSEQ), squeeze=False)
    for si, s in enumerate(chosen):
        for ci, h in enumerate(cols):
            im = draw_fp(r_fr[idx[si], K + h], gt[chosen][si, h], pw[si, h], pa[si, h])
            ax2[si, ci].imshow(im); ax2[si, ci].axis("off")
            if ci == 0: ax2[si, ci].set_title(f"seq{s}", fontsize=7, loc="left")
        gif2 = [Image.fromarray(draw_fp(r_fr[idx[si], K + h], gt[chosen][si, h], pw[si, h], pa[si, h])) for h in range(H)]
        gif2[0].save(f"{OUT}/fp_seq{s}.gif", save_all=True, append_images=gif2[1:], duration=180, loop=0)
    fig2.suptitle("footprint convex-hull: GT(green) / world-pred(red) / agent-frame-pred(blue) — what ③ fills", fontsize=11)
    fig2.tight_layout(); fig2.savefig(f"{OUT}/footprint_grid.png", dpi=115); plt.close(fig2)

    lines.append(f"\nMEAN | world {ade_w.mean():.2f} | agentframe {ade_a.mean():.2f}")
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
