"""Long sliding-window action-replay rollout viz for the v4 flow world model.

Trains two models inline on robot data (thin = v3 baseline; thick = contact-gated +
anti-drift), then on ONE long held-out clip (vid=12 = play_robot_3_eef, built fresh in the
flow_ds_v3 convention via gen_flow_dataset_v3 machinery) does a continuous H=36 sliding-window
rollout (3 blocks of F=12) by action-replay, advancing the history with the model's own
predictions. Renders the GT cube flow, thick prediction, and thin prediction TOGETHER on the
single real RGB frame so divergence over the long horizon is obvious.

Outputs (outputs/flow_wm_v4/viz_rollout/):
  rollout_overlay.gif        all H frames, combined overlay (green=GT, red=thick, blue=thin, yellow=EEF) + centroid trails
  rollout_overlay_strip.png  one row of combined-overlay snapshots across the horizon
  rollout_3row_strip.png     3 rows (GT / thick / thin) for separated viewing
  summary.txt                centroid paths + overall & per-block ADE (thin vs thick)

Run: /scr/yusenluo/anaconda3/envs/iws/bin/python viz_rollout_replay.py
"""
import os, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import imageio.v2 as imageio

import train_flow_wm_scarcity_v4 as v4
import gen_flow_dataset_v3 as g3

OUT = "outputs/flow_wm_v4/viz_rollout"; os.makedirs(OUT, exist_ok=True)
K, F, L, P = v4.K, v4.F, v4.L, v4.P          # 4, 12, 16, 48
S, RES = g3.S, g3.RES                          # stride 3, 224px
H = 36                                          # rollout horizon = 3 blocks of F
Lseq = K + H                                     # 40 tracked frames
device = v4.device


# ---------------------------------------------------------------- (a) train two models
def train_models():
    tr, vis, eef3, dom, vid = v4.load()
    gstats = v4.fit_grasp_stats(v4.grasp_openness(eef3), dom)
    rob = torch.where((dom == 1) & (vid != v4.TEST_ROBOT_VID))[0]
    print(f"[train] robot-pool n={len(rob)}", flush=True)
    _, m_thin = v4.train_eval(tr, vis, eef3, dom, gstats, rob, rob[:2],
                              thin=True, seed=0, antidrift=False, return_model=True)
    _, m_thick = v4.train_eval(tr, vis, eef3, dom, gstats, rob, rob[:2],
                               thin=False, seed=0, antidrift=True, return_model=True)
    m_thin.eval(); m_thick.eval()
    print("[train] thin + thick models ready", flush=True)
    return m_thin, m_thick, gstats


# ------------------------------------------------ (b) build ONE long held-out clip (vid=12)
def build_clip(gstats):
    vpath = (f"{g3.ROBOT_DIRS[2]}/videos/chunk-000/observation.images.cam_high/"
             f"episode_000000.mp4")
    crops = g3.decode("robot", vpath)
    print(f"[clip] decoded {len(crops)} crops from {vpath}", flush=True)

    from interactive_world_sim.algorithms.latent_decompose.ot_align import robot_world_to_cam
    T_cw = np.asarray(robot_world_to_cam(), np.float64); Kmat = g3.load_K()
    eef_fn, _ = g3.robot_eef3_loader(g3.ROBOT_DIRS[2], T_cw, Kmat)

    ct = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device).eval()
    rng = np.random.default_rng(0)

    best = None  # (path_len, s0, trk, vsk, eseq_n, idxs)
    last = len(crops) - (Lseq * S) - F * S
    for s0 in range(0, max(1, last), 30):
        idxs = [s0 + k * S for k in range(Lseq)]
        if idxs[-1] >= len(crops):
            continue
        fr = [crops[i] for i in idxs]
        m0 = g3.cube_mask(fr[0]); ys, xs = np.where(m0 > 0)
        if len(xs) < P:
            continue
        sel = rng.choice(len(xs), P, replace=False)
        q = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
        # EEF must be available across the whole rollout (windows extend F beyond Lseq)
        eef_idx = [s0 + k * S for k in range(Lseq + F)]
        eseq = [eef_fn(i) for i in eef_idx]
        if any(e is None for e in eseq):
            continue
        trk, vsk = g3.track(ct, fr, q, device)                      # (Lseq,P,2) pixels
        c = trk.mean(1)
        plen = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
        if best is None or plen > best[0]:
            best = (plen, s0, trk, vsk, np.stack(eseq), idxs)
    if best is None:
        raise RuntimeError("no valid clip found (cube/eef availability)")
    plen, s0, trk, vsk, eseq, idxs = best
    print(f"[clip] chosen s0={s0}  GT cube centroid path={plen:.2f}px", flush=True)
    trk_n = (trk / RES).astype(np.float32)                          # (Lseq,P,2)
    eseq_n = (eseq / RES).astype(np.float32)                        # (Lseq+F,3,2)
    return dict(s0=s0, crops=crops, idxs=idxs, trk=trk, vsk=vsk,
                trk_n=trk_n, eseq_n=eseq_n, gt_path=plen, gstats=gstats)


# --------------------------------------------- (c) sliding-window action-replay rollout
def rollout(m, trk_n, eseq_n, gstats):
    hist = torch.from_numpy(trk_n[:K]).float().permute(1, 0, 2)[None].to(device)  # (1,P,K,2)
    out = []
    for bk in range(H // F):
        w0 = bk * F
        ew = torch.from_numpy(eseq_n[w0:w0 + L]).float()[None].to(device)         # (1,L,3,2)
        g = v4.normalize_grasp(v4.grasp_openness(ew),
                               torch.ones(1, dtype=torch.long), gstats).to(device)
        dm = torch.ones(1, dtype=torch.long, device=device)
        with torch.no_grad():
            pred, _ = m(hist, ew, g, dm)                                          # (1,P,F,2)
        out.append(pred[0])                                                       # (P,F,2)
        hist = pred[:, :, -K:, :]
    return torch.cat(out, 1).cpu().numpy()                                        # (P,H,2) normalized


# --------------------------------------------------------- helpers for metrics / drawing
def centroid_path_px(seq_phk):  # seq (P,T,2) normalized -> total centroid path in px
    c = seq_phk.mean(0)                                                           # (T,2)
    return float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1))) * RES


def ade_px(pred_ph, gt_ph, vis_ph):  # (P,T,2) norm, vis (P,T) -> ADE px on visible pts
    err = np.linalg.norm(pred_ph - gt_ph, axis=-1) * RES                          # (P,T)
    return float((err * vis_ph).sum() / (vis_ph.sum() + 1e-6))


# ------------------------------------------------------------------------------ main
def main():
    m_thin, m_thick, gstats = train_models()
    clip = build_clip(gstats)
    trk_n, eseq_n = clip["trk_n"], clip["eseq_n"]
    crops, idxs, s0 = clip["crops"], clip["idxs"], clip["s0"]

    pred_thin = rollout(m_thin, trk_n, eseq_n, gstats)            # (P,H,2) norm
    pred_thick = rollout(m_thick, trk_n, eseq_n, gstats)          # (P,H,2) norm
    gt = trk_n[K:K + H].transpose(1, 0, 2)                        # (P,H,2) norm
    gt_vis = clip["vsk"][K:K + H].transpose(1, 0)                 # (P,H)

    # ---- (e) metrics
    thin_path = centroid_path_px(pred_thin)
    thick_path = centroid_path_px(pred_thick)
    gt_path = centroid_path_px(gt)
    ade_thin = ade_px(pred_thin, gt, gt_vis)
    ade_thick = ade_px(pred_thick, gt, gt_vis)
    blocks = H // F
    block_lines = []
    for bk in range(blocks):
        a, b = bk * F, (bk + 1) * F
        bt = ade_px(pred_thin[:, a:b], gt[:, a:b], gt_vis[:, a:b])
        bk_ = ade_px(pred_thick[:, a:b], gt[:, a:b], gt_vis[:, a:b])
        block_lines.append((bk, K + a, K + b - 1, bt, bk_))

    lines = [
        "long sliding-window action-replay rollout (v4 flow WM) | vid=12 play_robot_3_eef",
        f"convention=flow_ds_v3 (224px, P=48 red-cube pts, eef3); K={K} F={F} L={L} H={H} Lseq={Lseq} S={S}",
        f"chosen s0 = {s0}",
        f"GT cube centroid path   = {gt_path:8.2f} px",
        f"thin  rollout cen. path = {thin_path:8.2f} px",
        f"thick rollout cen. path = {thick_path:8.2f} px",
        f"overall rollout ADE (H={H} frames, visible pts):",
        f"  thin  ADE = {ade_thin:7.2f} px",
        f"  thick ADE = {ade_thick:7.2f} px",
        "per-block ADE (block grows with rollout depth; expect block0 small/similar, later diverge):",
        f"  {'block':>5} | {'frames':>9} | {'thin_ADE':>9} | {'thick_ADE':>10}",
    ]
    for bk, f0, f1, bt, bk_ in block_lines:
        lines.append(f"  {bk:>5} | {f'{f0}..{f1}':>9} | {bt:9.2f} | {bk_:10.2f}")
    summary = "\n".join(lines) + "\n"
    open(os.path.join(OUT, "summary.txt"), "w").write(summary)
    print("\n" + summary, flush=True)

    # ---- (d) rendering
    gt_trk_px = clip["trk"][K:K + H]                               # (H,P,2) pixels (exact GT)
    thick_px = pred_thick.transpose(1, 0, 2) * RES                # (H,P,2)
    thin_px = pred_thin.transpose(1, 0, 2) * RES                  # (H,P,2)
    tips_px = clip["eseq_n"][K:K + H, 1:3] * RES                  # (H,2,2)

    # centroid trails (pixels): centroid per frame, frames K..K+h
    cen_gt = gt_trk_px.mean(1)                                     # (H,2)
    cen_thick = thick_px.mean(1)
    cen_thin = thin_px.mean(1)

    def running_ade(pred_px, h):
        e = np.linalg.norm(pred_px[:h + 1] - gt_trk_px[:h + 1], axis=-1)  # (h+1,P)
        w = gt_vis[:, :h + 1].T                                            # (h+1,P)
        return float((e * w).sum() / (w.sum() + 1e-6))

    def draw_overlay(ax, h, show_trail=True, sets=("gt", "thick", "thin", "eef")):
        ax.imshow(crops[idxs[K + h]]); ax.axis("off")
        if "gt" in sets:
            ax.scatter(gt_trk_px[h, :, 0], gt_trk_px[h, :, 1], s=6, c="lime",
                       edgecolors="none", alpha=0.9)
        if "thick" in sets:
            ax.scatter(thick_px[h, :, 0], thick_px[h, :, 1], s=6, c="red",
                       edgecolors="none", alpha=0.9)
        if "thin" in sets:
            ax.scatter(thin_px[h, :, 0], thin_px[h, :, 1], s=6, c="dodgerblue",
                       edgecolors="none", alpha=0.9)
        if "eef" in sets:
            ax.scatter(tips_px[h, :, 0], tips_px[h, :, 1], s=40, c="yellow",
                       marker="*", edgecolors="k", linewidths=0.4)
        if show_trail and h > 0:
            if "gt" in sets:
                ax.plot(cen_gt[:h + 1, 0], cen_gt[:h + 1, 1], c="lime", lw=1.4)
            if "thick" in sets:
                ax.plot(cen_thick[:h + 1, 0], cen_thick[:h + 1, 1], c="red", lw=1.4)
            if "thin" in sets:
                ax.plot(cen_thin[:h + 1, 0], cen_thin[:h + 1, 1], c="dodgerblue", lw=1.4)
        ax.set_xlim(0, RES); ax.set_ylim(RES, 0)

    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor="lime", label="GT", markersize=7),
              Line2D([0], [0], marker="o", color="w", markerfacecolor="red", label="thick", markersize=7),
              Line2D([0], [0], marker="o", color="w", markerfacecolor="dodgerblue", label="thin", markersize=7),
              Line2D([0], [0], marker="*", color="w", markerfacecolor="yellow", markeredgecolor="k", label="EEF", markersize=11)]

    # ---- GIF: combined overlay every frame
    gif_frames = []
    for h in range(H):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=90)
        draw_overlay(ax, h)
        ax.set_title(f"t={K + h} | thick ADE {running_ade(thick_px, h):.1f}px / "
                     f"thin ADE {running_ade(thin_px, h):.1f}px", fontsize=9)
        ax.legend(handles=legend, loc="upper right", fontsize=6, framealpha=0.7)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, hh = fig.canvas.get_width_height()
        gif_frames.append(buf.reshape(hh, w, 4)[..., :3].copy())
        plt.close(fig)
    gif_path = os.path.join(OUT, "rollout_overlay.gif")
    try:
        imageio.mimsave(gif_path, gif_frames, duration=0.15, loop=0)
    except Exception as e:
        gif_path = os.path.join(OUT, "rollout_overlay.mp4")
        imageio.mimsave(gif_path, gif_frames, fps=7)
        print(f"[warn] gif failed ({e}); wrote mp4 instead", flush=True)
    print(f"[render] {gif_path} ({H} frames)", flush=True)

    # ---- combined-overlay strip
    cols = [c for c in (0, 5, 11, 17, 23, 29, 35) if c < H]
    fig, ax = plt.subplots(1, len(cols), figsize=(2.4 * len(cols), 2.7))
    for ci, h in enumerate(cols):
        draw_overlay(ax[ci], h)
        ax[ci].set_title(f"t={K + h}", fontsize=9)
    fig.legend(handles=legend, loc="upper center", ncol=4, fontsize=8)
    fig.suptitle("combined overlay rollout: blue(thin) drifts off cube, red(thick) tracks green(GT)",
                 y=0.02, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    strip_path = os.path.join(OUT, "rollout_overlay_strip.png")
    fig.savefig(strip_path, dpi=115); plt.close(fig)
    print(f"[render] {strip_path}", flush=True)

    # ---- 3-row strip (GT / thick / thin separated)
    rows = [("GT-FLOW", ("gt", "eef")), ("Pred-thick", ("thick", "eef")), ("Pred-thin", ("thin", "eef"))]
    fig, ax = plt.subplots(3, len(cols), figsize=(2.4 * len(cols), 7.2))
    for ri, (rname, sets) in enumerate(rows):
        for ci, h in enumerate(cols):
            draw_overlay(ax[ri, ci], h, sets=sets)
            if ri == 0:
                ax[ri, ci].set_title(f"t={K + h}", fontsize=9)
            if ci == 0:
                ax[ri, ci].axis("on")
                ax[ri, ci].set_xticks([]); ax[ri, ci].set_yticks([])
                ax[ri, ci].set_ylabel(rname, fontsize=10)
    fig.suptitle("3-row rollout (GT=green / thick=red / thin=blue, yellow=EEF)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    strip3_path = os.path.join(OUT, "rollout_3row_strip.png")
    fig.savefig(strip3_path, dpi=115); plt.close(fig)
    print(f"[render] {strip3_path}", flush=True)

    print("\n=== DONE ===", flush=True)
    print(f"GIF:   {os.path.abspath(gif_path)}", flush=True)
    print(f"strip: {os.path.abspath(strip_path)}", flush=True)
    print(f"3row:  {os.path.abspath(strip3_path)}", flush=True)
    print(f"summary: {os.path.abspath(os.path.join(OUT, 'summary.txt'))}", flush=True)


if __name__ == "__main__":
    main()
