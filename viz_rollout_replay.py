"""viz_rollout_replay.py — Action-replay rollout visualization: GT | GT-FLOW | Pred-thick | Pred-thin.

Shows, on real robot video frames (vid=12 / play_robot_3_eef), the autoregressive object-flow
rollout for 4 conditions side by side so the user can judge whether the thick model helps.

Outputs:
  outputs/flow_wm_v4/viz_rollout/rollout_replay.mp4  (or .gif fallback)
  outputs/flow_wm_v4/viz_rollout/rollout_strip.png
  outputs/flow_wm_v4/viz_rollout/summary.txt
"""
import os, sys, warnings
os.environ["MPLBACKEND"] = "Agg"  # must be before matplotlib import
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, "/scr2/yusenluo/interactive_world_sim")

import train_flow_wm_scarcity_v4 as v4
import gen_flow_dataset_v3 as g3

# ─── constants (mirrored from v4 / g3) ───────────────────────────────────────
K, F, L = v4.K, v4.F, v4.L          # 4, 12, 16
P = v4.P                              # 48
RES = g3.RES                          # 224
S = g3.S                              # 3
device = v4.device
OUT_DIR = "outputs/flow_wm_v4/viz_rollout"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── (a) Train the two models ─────────────────────────────────────────────────
print("=== Loading dataset ===", flush=True)
tr, vis, eef3, dom, vid = v4.load()
gstats = v4.fit_grasp_stats(v4.grasp_openness(eef3), dom)

rob = torch.where((dom == 1) & (vid != v4.TEST_ROBOT_VID))[0]
dummy_test = rob[:1]

print(f"train pool: {len(rob)} clips  (robot, vid != {v4.TEST_ROBOT_VID})", flush=True)

print("=== Training thin baseline (seed=0) ===", flush=True)
_, m_thin = v4.train_eval(tr, vis, eef3, dom, gstats,
                          rob, dummy_test,
                          thin=True, seed=0, antidrift=False,
                          return_model=True)
m_thin.eval()
print("thin model trained.", flush=True)

print("=== Training thick full (seed=0) ===", flush=True)
_, m_thick = v4.train_eval(tr, vis, eef3, dom, gstats,
                           rob, dummy_test,
                           thin=False, seed=0, antidrift=True,
                           return_model=True)
m_thick.eval()
print("thick model trained.", flush=True)

# ─── (b) Get real frames + GT-FLOW for vid=12 (play_robot_3_eef) ─────────────
print("=== Loading CoTracker3 ===", flush=True)
ct = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline",
                    trust_repo=True).to(device).eval()

from interactive_world_sim.algorithms.latent_decompose.ot_align import robot_world_to_cam
T_cw = np.asarray(robot_world_to_cam(), np.float64)
Kmat = g3.load_K()

robot_dir = g3.ROBOT_DIRS[2]   # "play_robot_eef/play_robot_3_eef"
vpath = f"{robot_dir}/videos/chunk-000/observation.images.cam_high/episode_000000.mp4"

print(f"=== Decoding frames from {vpath} ===", flush=True)
crops = g3.decode("robot", vpath)
print(f"decoded {len(crops)} frames", flush=True)

eef_fn, n_eef = g3.robot_eef3_loader(robot_dir, T_cw, Kmat)

# Scan clips: find the one with largest cube centroid motion
print("=== Scanning for most-dynamic clip ===", flush=True)
best_s0 = None
best_motion = -1.0
best_trk = None
best_vs = None
best_eef3_clip = None
best_q = None

rng = np.random.default_rng(42)

scan_starts = list(range(0, len(crops) - L * S - 1, 30))
for s0 in scan_starts:
    idxs = [s0 + k * S for k in range(L)]
    if idxs[-1] >= len(crops):
        continue
    fr = [crops[i] for i in idxs]
    m0 = g3.cube_mask(fr[0])
    ys, xs = np.where(m0 > 0)
    if len(xs) < P:
        continue
    # check EEF early
    eef3_clip = [eef_fn(idxs[t]) for t in range(L)]
    if any(e is None for e in eef3_clip):
        continue
    sel = rng.choice(len(xs), P, replace=False)
    q = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)  # (P,2) x,y
    trk, vs = g3.track(ct, fr, q, device)   # trk (L,P,2) pixels
    c = trk.mean(1)                           # (L,2)
    motion = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
    if motion > best_motion:
        best_motion = motion
        best_s0 = s0
        best_trk = trk
        best_vs = vs
        best_eef3_clip = eef3_clip
        best_q = q

if best_s0 is None:
    raise RuntimeError("No valid clip found with P>=48 cube points and valid EEF in vid=12")

print(f"Best clip start={best_s0}, centroid motion={best_motion:.2f}px", flush=True)

# Reassemble clip data
idxs_chosen = [best_s0 + k * S for k in range(L)]
frames_chosen = [crops[i] for i in idxs_chosen]
trk_px = best_trk          # (L,P,2) pixels, float32
eef3_chosen = np.stack(best_eef3_clip)  # (L,3,2) pixels

# Normalize to [0,1] coords (RES=224)
tracks_n = torch.from_numpy(trk_px / RES).float()       # (L,P,2)
eef3_n   = torch.from_numpy(eef3_chosen / RES).float()  # (L,3,2)

# ─── (c) Autoregressive chained rollout ──────────────────────────────────────
def chained_traj(m, tracks_n, eef3_n):
    """Build chained (action-replay) trajectory for frames K..L-1.

    Returns pred_traj normalized (P, F, 2) where F=L-K=12.
    """
    g_raw = v4.grasp_openness(eef3_n[None])               # (1,L)
    g = v4.normalize_grasp(g_raw, torch.ones(1, dtype=torch.long), gstats)  # robot=1
    ef = eef3_n[None].to(device)                           # (1,L,3,2)
    dm = torch.ones(1, dtype=torch.long, device=device)   # robot=1
    hist = tracks_n[:K].permute(1, 0, 2)[None].to(device) # (1,P,K,2)

    blocks = []
    n_hops = F // K   # = 3
    for hop in range(n_hops):
        with torch.no_grad():
            pred, _ = m(hist, ef, g.to(device), dm)       # (1,P,F,2)
        blk = pred[:, :, :K, :]                            # (1,P,K,2): next K frames
        blocks.append(blk)
        hist = blk                                         # feed own prediction back

    traj = torch.cat(blocks, 2)[0]                         # (P, F, 2) normalized (F=12)
    return traj


print("=== Computing chained rollout: thin ===", flush=True)
traj_thin_n  = chained_traj(m_thin,  tracks_n, eef3_n)   # (P,12,2) normalized
print("=== Computing chained rollout: thick ===", flush=True)
traj_thick_n = chained_traj(m_thick, tracks_n, eef3_n)   # (P,12,2) normalized

# GT future: frames K..L-1
gt_future_n = tracks_n[K:]   # (12,P,2) normalized

# Convert to pixels for rendering and metrics
traj_thin_px  = (traj_thin_n  * RES).cpu().numpy()   # (P,12,2)
traj_thick_px = (traj_thick_n * RES).cpu().numpy()   # (P,12,2)
gt_future_px  = (gt_future_n  * RES).cpu().numpy()   # (12,P,2)
trk_history_px = trk_px[:K]                           # (K,P,2)

# ─── Metrics (ADE vs GT, 12 future frames) ────────────────────────────────────
def ade_px(pred_pf2, gt_fp2):
    """pred_pf2: (P,F,2), gt_fp2: (F,P,2) in pixels -> ADE scalar."""
    gt_pf2 = gt_fp2.transpose(1, 0, 2)   # (P,F,2)
    err = np.linalg.norm(pred_pf2 - gt_pf2, axis=-1)  # (P,F)
    return float(err.mean())

ade_thin  = ade_px(traj_thin_px,  gt_future_px)
ade_thick = ade_px(traj_thick_px, gt_future_px)

# Centroid paths (px)
def centroid_path(pts_pf2):
    """pts_pf2: (P,F,2) pixels -> (F,2) centroid array."""
    return pts_pf2.mean(0)   # (F,2)

gt_centroid    = gt_future_px.mean(1)                    # (12,2)
thin_centroid  = centroid_path(traj_thin_px)             # (12,2)
thick_centroid = centroid_path(traj_thick_px)            # (12,2)

# ─── Summary text ─────────────────────────────────────────────────────────────
def fmt_path(arr):
    return ", ".join(f"({r[0]:.1f},{r[1]:.1f})" for r in arr)

summary_lines = [
    f"Clip start frame: {best_s0}  (vid=12 / play_robot_3_eef, stride={S})",
    f"GT cube centroid motion over clip: {best_motion:.2f} px",
    "",
    "GT  future centroid path (px) [frames K..L-1]:",
    f"  {fmt_path(gt_centroid)}",
    "",
    "Thin rollout centroid path (px):",
    f"  {fmt_path(thin_centroid)}",
    "",
    "Thick rollout centroid path (px):",
    f"  {fmt_path(thick_centroid)}",
    "",
    f"ADE vs GT (12 future frames, all 48 points, px):",
    f"  thin  (no anti-drift): {ade_thin:.2f} px",
    f"  thick (full):          {ade_thick:.2f} px",
    f"  improvement (thin-thick): {ade_thin - ade_thick:.2f} px",
]
summary_str = "\n".join(summary_lines) + "\n"
print(summary_str, flush=True)
with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write(summary_str)


# ─── (d) Render ───────────────────────────────────────────────────────────────
def draw_panel(ax, frame_rgb, title,
               gt_pts=None,        # (P,2) px green
               pred_pts=None,      # (P,2) px colored
               hist_pts=None,      # (P,2) px gray (history)
               eef_pts=None,       # (3,2) px yellow
               centroid_trail=None,# (T,2) px trail
               trail_color="gray",
               pred_color="red"):
    ax.imshow(frame_rgb)
    ax.set_title(title, fontsize=8, pad=2)
    ax.axis("off")
    if gt_pts is not None:
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1], s=3, c="lime", alpha=0.7,
                   linewidths=0, zorder=3)
    if hist_pts is not None:
        ax.scatter(hist_pts[:, 0], hist_pts[:, 1], s=3, c="silver", alpha=0.5,
                   linewidths=0, zorder=3)
    if pred_pts is not None:
        ax.scatter(pred_pts[:, 0], pred_pts[:, 1], s=3, c=pred_color, alpha=0.7,
                   linewidths=0, zorder=4)
    if eef_pts is not None:
        ax.scatter(eef_pts[1:, 0], eef_pts[1:, 1], s=18, c="yellow", marker="*",
                   edgecolors="k", linewidths=0.3, zorder=5)
    if centroid_trail is not None and len(centroid_trail) >= 2:
        ax.plot(centroid_trail[:, 0], centroid_trail[:, 1],
                "-", color=trail_color, linewidth=1.0, alpha=0.55, zorder=2)


def make_frame(t):
    """Produce a 4-panel figure for frame index t (0..L-1). Returns RGBA array."""
    frame = frames_chosen[t]
    eef_px_t = eef3_chosen[t]   # (3,2) pixels

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.88, bottom=0.01)

    # ── Panel 0: GT ──
    draw_panel(axes[0], frame, f"GT  [t={t}]")

    # ── Panel 1: GT-FLOW ──
    gt_pts_t = trk_px[t]   # (P,2)
    # centroid trail up to t
    gt_trail = trk_px[:t+1].mean(1)   # (t+1,2)
    draw_panel(axes[1], frame, "GT-FLOW",
               gt_pts=gt_pts_t,
               eef_pts=eef_px_t,
               centroid_trail=gt_trail, trail_color="lime")

    # ── Panel 2: Pred-THICK ──
    if t < K:
        # history: show GT gray
        draw_panel(axes[2], frame, "Pred-FLOW-thick",
                   hist_pts=trk_px[t],
                   eef_pts=eef_px_t,
                   centroid_trail=trk_px[:t+1].mean(1),
                   trail_color="gray")
    else:
        tf = t - K   # future frame index (0..11)
        pred_pts_t = traj_thick_px[:, tf, :]   # (P,2)
        # trail: GT centroid for frames 0..K-1, then predicted centroid for K..t
        gt_c_hist = trk_px[:K].mean(1)                # (K,2)
        pred_c_fut = traj_thick_px[:, :tf+1, :].mean(0)  # (tf+1,2)
        trail = np.concatenate([gt_c_hist, pred_c_fut], axis=0)  # (K+tf+1,2)
        draw_panel(axes[2], frame, "Pred-FLOW-thick",
                   pred_pts=pred_pts_t,
                   eef_pts=eef_px_t,
                   centroid_trail=trail, trail_color="orangered",
                   pred_color="red")

    # ── Panel 3: Pred-THIN ──
    if t < K:
        draw_panel(axes[3], frame, "Pred-FLOW-thin",
                   hist_pts=trk_px[t],
                   eef_pts=eef_px_t,
                   centroid_trail=trk_px[:t+1].mean(1),
                   trail_color="gray")
    else:
        tf = t - K
        pred_pts_t = traj_thin_px[:, tf, :]
        gt_c_hist = trk_px[:K].mean(1)
        pred_c_fut = traj_thin_px[:, :tf+1, :].mean(0)
        trail = np.concatenate([gt_c_hist, pred_c_fut], axis=0)
        draw_panel(axes[3], frame, "Pred-FLOW-thin",
                   pred_pts=pred_pts_t,
                   eef_pts=eef_px_t,
                   centroid_trail=trail, trail_color="cornflowerblue",
                   pred_color="royalblue")

    # Render to array
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]   # RGBA->RGB
    plt.close(fig)
    return buf


# ── Generate all L frames ──
print("=== Rendering frames ===", flush=True)
rendered_frames = []
for t in range(L):
    frame_arr = make_frame(t)
    rendered_frames.append(frame_arr)
    if t % 4 == 0:
        print(f"  rendered {t+1}/{L} frames", flush=True)

# ── Save MP4 ──
mp4_path = os.path.join(OUT_DIR, "rollout_replay.mp4")
gif_path = os.path.join(OUT_DIR, "rollout_replay.gif")
saved_video = None

try:
    import imageio.v3 as iio3
    # try MP4 via imageio
    iio3.imwrite(mp4_path, rendered_frames, fps=4, codec="libx264",
                 quality=None, output_params=["-pix_fmt", "yuv420p"])
    saved_video = mp4_path
    print(f"Saved MP4: {mp4_path}", flush=True)
except Exception as e:
    print(f"MP4 failed ({e}), falling back to GIF...", flush=True)
    try:
        import imageio as iio
        iio.mimsave(gif_path, rendered_frames, fps=4)
        saved_video = gif_path
        print(f"Saved GIF: {gif_path}", flush=True)
    except Exception as e2:
        print(f"GIF also failed: {e2}. Only PNG strip will be saved.", flush=True)

# ── Save static key-frame strip PNG ──
# Key frames: K-1, K+3, K+7, L-1  (= 3, 7, 11, 15)
key_t = [K - 1, K + 3, K + 7, L - 1]
key_t = [t for t in key_t if t < L]   # safety

strip_path = os.path.join(OUT_DIR, "rollout_strip.png")
n_rows = len(key_t)
n_cols = 4   # GT | GT-FLOW | Pred-thick | Pred-thin

fig_strip, axes_strip = plt.subplots(n_rows, n_cols, figsize=(10, 2.8 * n_rows))
fig_strip.subplots_adjust(wspace=0.02, hspace=0.12, left=0.01, right=0.99,
                           top=0.96, bottom=0.01)

col_titles = ["GT", "GT-FLOW", "Pred-FLOW-thick", "Pred-FLOW-thin"]

for ri, t in enumerate(key_t):
    frame = frames_chosen[t]
    eef_px_t = eef3_chosen[t]

    # Col 0: GT
    ax = axes_strip[ri, 0]
    ax.imshow(frame); ax.axis("off")
    if ri == 0: ax.set_title(col_titles[0], fontsize=8)
    ax.set_ylabel(f"t={t}", fontsize=7, rotation=0, labelpad=22, va="center")

    # Col 1: GT-FLOW
    ax = axes_strip[ri, 1]
    ax.imshow(frame)
    ax.scatter(trk_px[t, :, 0], trk_px[t, :, 1], s=3, c="lime", alpha=0.7, linewidths=0)
    ax.scatter(eef_px_t[1:, 0], eef_px_t[1:, 1], s=18, c="yellow", marker="*",
               edgecolors="k", linewidths=0.3, zorder=5)
    trail = trk_px[:t+1].mean(1)
    if len(trail) >= 2:
        ax.plot(trail[:, 0], trail[:, 1], "-", color="lime", lw=1.0, alpha=0.55)
    ax.axis("off")
    if ri == 0: ax.set_title(col_titles[1], fontsize=8)

    # Col 2: Pred-thick
    ax = axes_strip[ri, 2]
    ax.imshow(frame)
    if t < K:
        ax.scatter(trk_px[t, :, 0], trk_px[t, :, 1], s=3, c="silver", alpha=0.5, linewidths=0)
    else:
        tf = t - K
        ax.scatter(traj_thick_px[:, tf, 0], traj_thick_px[:, tf, 1], s=3, c="red",
                   alpha=0.7, linewidths=0, zorder=4)
        gt_c_hist = trk_px[:K].mean(1)
        pred_c_fut = traj_thick_px[:, :tf+1, :].mean(0)
        trail = np.concatenate([gt_c_hist, pred_c_fut], axis=0)
        if len(trail) >= 2:
            ax.plot(trail[:, 0], trail[:, 1], "-", color="orangered", lw=1.0, alpha=0.55)
    ax.scatter(eef_px_t[1:, 0], eef_px_t[1:, 1], s=18, c="yellow", marker="*",
               edgecolors="k", linewidths=0.3, zorder=5)
    ax.axis("off")
    if ri == 0: ax.set_title(col_titles[2], fontsize=8)

    # Col 3: Pred-thin
    ax = axes_strip[ri, 3]
    ax.imshow(frame)
    if t < K:
        ax.scatter(trk_px[t, :, 0], trk_px[t, :, 1], s=3, c="silver", alpha=0.5, linewidths=0)
    else:
        tf = t - K
        ax.scatter(traj_thin_px[:, tf, 0], traj_thin_px[:, tf, 1], s=3, c="royalblue",
                   alpha=0.7, linewidths=0, zorder=4)
        gt_c_hist = trk_px[:K].mean(1)
        pred_c_fut = traj_thin_px[:, :tf+1, :].mean(0)
        trail = np.concatenate([gt_c_hist, pred_c_fut], axis=0)
        if len(trail) >= 2:
            ax.plot(trail[:, 0], trail[:, 1], "-", color="cornflowerblue", lw=1.0, alpha=0.55)
    ax.scatter(eef_px_t[1:, 0], eef_px_t[1:, 1], s=18, c="yellow", marker="*",
               edgecolors="k", linewidths=0.3, zorder=5)
    ax.axis("off")
    if ri == 0: ax.set_title(col_titles[3], fontsize=8)

fig_strip.suptitle(
    f"Action-replay rollout viz  |  clip start={best_s0}, vid=12 (play_robot_3_eef)\n"
    f"thin ADE={ade_thin:.1f}px   thick ADE={ade_thick:.1f}px   (12 future frames, 48 pts)",
    fontsize=8)
fig_strip.savefig(strip_path, dpi=150, bbox_inches="tight")
plt.close(fig_strip)
print(f"Saved strip PNG: {strip_path}", flush=True)

# ── Final report ──
print("\n=== DONE ===", flush=True)
print(f"Clip start frame: {best_s0}", flush=True)
print(f"Video: {saved_video or 'SKIPPED'}", flush=True)
print(f"Strip: {strip_path}", flush=True)
print(f"Summary: {os.path.join(OUT_DIR, 'summary.txt')}", flush=True)
print(f"\nFull summary:\n{summary_str}", flush=True)
