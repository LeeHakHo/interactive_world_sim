"""TRACK-POINT RESEEDING prototype (2026-07-15). Same bug family footprint_warp.py fixed for the
FOOTPRINT, now for the SEEDING itself: HSV-red seeding at gen time (gen_flow_render_dataset_caneef.
can_mask -> red rim/label only) clusters the 48 object tracks in outputs/flow_render_dataset_can_dual
/clips_robot.npz on the can's red band instead of covering the can body (cam_high convex hull as
small as 61-74px^2, can top ~600+px^2; QC: outputs/cross_embodiment_wm/track_coverage_qc/
track_points_frame0.png).

PROTOTYPE ONLY on a fixed rng(0) 30-clip sample, both views (cam_high + cam_low where low_valid).
Fix = SAM2-segment the can at frame 0 (positive prompts = the EXISTING track points -- they ARE on
the can, just clustered; negative = eef points) -> uniformly reseed 48 points inside the mask ->
CoTracker3-offline over the clip's stored 48 frames (same tracker call as gen_flow_render_dataset.
track(), run directly on the clip's stored 128x128 frames rather than re-decoding the source 224px
video -- a deliberate simplification for this prototype: those are the exact frames the downstream
WM/renderer consume, and tracks/eef in the npz are already crop-normalized so old and new points are
directly comparable in this space. NOT re-validated against 224-res tracking; flag for the full-regen
script if this prototype is approved).

SAM2 MASK SELECTION DIVERGES FROM footprint_warp.object_mask_sam2: that function's selection key
penalizes masks that are large relative to the OLD point hull ("blowup" term) -- correct for its use
case (hull usually already ~= object), wrong here (hull is 5-10x SMALLER than the true can body by
construction, that's the bug). Empirically (scratchpad sam2_mask_check.png, clips 50/900/2000) SAM2's
own predicted IoU score alone tracks true full-can coverage well and correctly downranks masks that
balloon into the tray/table (score 0.61 vs 0.90 for a correct can-only mask). So this script selects
by argmax(SAM2 score) subject to a >=70% old-point-inside gate (keeps the same object) -- see
object_mask_for_reseed(). Clips where nothing passes the gate are skipped and counted as sam2_fail
(nan-trap rule: report the failure/skip rate, do not silently drop into the mean).

Writes NOTHING into the existing npz. Outputs -> outputs/cross_embodiment_wm/track_reseed_proto/
(summary.txt with old-vs-new coverage/quality stats + 4-clip x 2-view PNG viz, old=green new=cyan,
frame0 + frame24).

Env: DS (source npz), OUT, N_CLIP (default 30), SEED (default 0). GPU (SAM2 + CoTracker3-offline).
"""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "phantom/submodules/sam2")

from gen_flow_render_dataset import track

DS = os.environ.get("DS", "outputs/flow_render_dataset_can_dual/clips_robot.npz")
OUT = os.environ.get("OUT", "outputs/cross_embodiment_wm/track_reseed_proto")
N_CLIP = int(os.environ.get("N_CLIP", "30"))
SEED = int(os.environ.get("SEED", "0"))
P = 48
IMG = 128
SAM2_CKPT = "phantom/submodules/sam2/checkpoints/sam2_hiera_large.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)

VIEWS = {
    "high": dict(frames="frames", tracks="tracks", eef="eef", vis="vis", valid=None),
    "low": dict(frames="frames_low", tracks="tracks_low", eef="eef_low", vis="vis_low", valid="low_valid"),
}

_P = {}


def load_sam2():
    if "p" not in _P:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _P["p"] = SAM2ImagePredictor(build_sam2("sam2_hiera_l.yaml", SAM2_CKPT, device=device))
    return _P["p"]


def object_mask_for_reseed(frame_u8, pts01, neg01, up=4, inside_gate=0.7):
    """SAM2 mask selected by argmax(native score) among candidates with >=inside_gate fraction of
    the (on-can) old track points inside -- see module docstring for why this diverges from
    footprint_warp.object_mask_sam2's blowup-penalized selection. Returns (mask_bool_HW, score) or
    (None, None) if no candidate passes the gate."""
    H, W = frame_u8.shape[:2]
    big = cv2.resize(frame_u8, (W * up, H * up), interpolation=cv2.INTER_LINEAR)
    p = load_sam2()
    p.set_image(big)
    sub = pts01[np.linspace(0, len(pts01) - 1, 8).astype(int)]
    pos = np.concatenate([sub, pts01.mean(0, keepdims=True)], 0)
    prompts, labels = pos, np.ones(len(pos), np.int32)
    if neg01 is not None and len(neg01):
        neg = np.asarray(neg01, np.float32)
        if len(neg) >= 3:                                  # eef=[wrist,tip1,tip2] (footprint_warp convention)
            wrist, t1, t2 = neg[0], neg[1], neg[2]
            d = wrist - (t1 + t2) / 2
            d = d / (np.linalg.norm(d) + 1e-6)
            extra = np.stack([(wrist + t1) / 2, (wrist + t2) / 2, wrist + d * 0.05])
            neg = np.concatenate([neg, np.clip(extra, 0, 1)], 0)
        prompts = np.concatenate([pos, neg], 0)
        labels = np.concatenate([labels, np.zeros(len(neg), np.int32)])
    masks, scores, _ = p.predict(point_coords=(prompts * np.array([W * up, H * up])).astype(np.float32),
                                 point_labels=labels, multimask_output=True)
    pts_px = (pts01 * np.array([W * up, H * up])).astype(int).clip(0, [W * up - 1, H * up - 1])
    best_i, best_sc = -1, -1.0
    for i, (m, sc) in enumerate(zip(masks, scores)):
        inside = m[pts_px[:, 1], pts_px[:, 0]].mean()
        if inside < inside_gate:
            continue
        if sc > best_sc:
            best_sc, best_i = float(sc), i
    if best_i < 0:
        return None, None
    mask = cv2.resize(masks[best_i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask, best_sc


def hull_area(pts01, hw=IMG):
    pts = np.clip((pts01 * hw).astype(np.int32), 0, hw - 1)
    if len(pts) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(pts.reshape(-1, 1, 2))))


def sample_in_mask(mask, n, rng):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    sel = rng.choice(len(xs), n, replace=len(xs) < n)
    pts_px = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
    return pts_px / mask.shape[0]


def step_disp(tr01, vs, hw=IMG):
    """mean/max per-step displacement (px@128) on visible->visible transitions (nan if no such step)."""
    d = np.linalg.norm(np.diff(tr01, axis=0), axis=-1) * hw
    w = (vs[:-1] > 0.5) & (vs[1:] > 0.5)
    if w.sum() == 0:
        return float("nan"), float("nan")
    return float(d[w].mean()), float(d[w].max())


def viz(view, ci, f0, f24, old0, new0, old24, new24, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.8))
    for a, img, op, npt, ttl in [(ax[0], f0, old0, new0, "t=0"), (ax[1], f24, old24, new24, "t=24")]:
        a.imshow(img)
        op_px = np.clip(op * IMG, 0, IMG - 1)
        npt_px = np.clip(npt * IMG, 0, IMG - 1)
        a.scatter(op_px[:, 0], op_px[:, 1], c="lime", s=10, label="old (HSV-red seed)", edgecolors="k", linewidths=0.2)
        a.scatter(npt_px[:, 0], npt_px[:, 1], c="cyan", s=10, label="new (SAM2 reseed)", edgecolors="k", linewidths=0.2)
        a.set_title(ttl, fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
    ax[0].legend(loc="upper right", fontsize=6, framealpha=0.6)
    fig.suptitle(f"{view} view | clip {ci} | old=green (HSV-red seed) vs new=cyan (SAM2 reseed)", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{view}_clip{ci}.png", dpi=130)
    plt.close(fig)


def main():
    ct = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device).eval()
    z = np.load(DS)
    N = z["frames"].shape[0]
    L = z["frames"].shape[1]
    t_mid = min(24, L - 1)
    rng_sample = np.random.default_rng(SEED)
    clip_ids = rng_sample.choice(N, N_CLIP, replace=False)
    viz_clip_ids = set(clip_ids[:4].tolist())
    rng_seed = np.random.default_rng(SEED)   # separate stream for mask-point sampling (reproducible)

    lines = [f"TRACK-POINT RESEEDING prototype | DS={DS} | N={N} | N_CLIP={N_CLIP} sample "
             f"(rng seed={SEED}) | clip_ids={sorted(clip_ids.tolist())}"]
    for view, keys in VIEWS.items():
        frames_all, tracks_all = z[keys["frames"]], z[keys["tracks"]].astype(np.float32)
        eef_all, vis_all = z[keys["eef"]].astype(np.float32), z[keys["vis"]].astype(np.float32)
        valid_all = z[keys["valid"]] if keys["valid"] else np.ones(N, bool)

        rows = dict(old_hull=[], new_hull=[], mask_area=[], sam2_score=[], old_cov=[], new_cov=[],
                    old_vis=[], new_vis=[], old_step_mean=[], new_step_mean=[], old_step_max=[], new_step_max=[])
        n_valid_view, n_used, sam2_fail, exploded = 0, 0, 0, 0

        for ci in clip_ids:
            if not valid_all[ci]:
                continue
            n_valid_view += 1
            fr = frames_all[ci]
            tr_old = tracks_all[ci]
            ef = eef_all[ci]
            vs_old = vis_all[ci]
            frame0 = fr[0]
            old_pts0 = np.clip(tr_old[0], 0, 1)
            eef0 = np.clip(ef[0], 0, 1)

            mask, sc = object_mask_for_reseed(frame0, old_pts0, eef0)
            if mask is None:
                sam2_fail += 1
                print(f"[{view}] clip {ci}: SAM2 gate failed (no candidate >=70% old-pts inside) - skip", flush=True)
                continue
            mask_area = float(mask.sum())
            oh = hull_area(old_pts0)
            new_pts0 = sample_in_mask(mask, P, rng_seed)
            nh = hull_area(new_pts0)

            q_px = new_pts0 * IMG
            tr_new, vs_new = track(ct, list(fr), q_px, device)
            tr_new01 = tr_new / IMG

            os_, om_ = step_disp(tr_old, vs_old)
            ns_, nm_ = step_disp(tr_new01, vs_new)
            if nm_ == nm_ and nm_ > 30.0:      # >30px@128 in one step across a 128px frame = exploded
                exploded += 1
                print(f"[{view}] clip {ci}: new-track max step {nm_:.1f}px@128 (old {om_:.1f}) - possible drift", flush=True)

            rows["old_hull"].append(oh); rows["new_hull"].append(nh); rows["mask_area"].append(mask_area)
            rows["sam2_score"].append(sc)
            rows["old_cov"].append(oh / mask_area); rows["new_cov"].append(nh / mask_area)
            rows["old_vis"].append(float(vs_old.mean())); rows["new_vis"].append(float(vs_new.mean()))
            rows["old_step_mean"].append(os_); rows["new_step_mean"].append(ns_)
            rows["old_step_max"].append(om_); rows["new_step_max"].append(nm_)
            n_used += 1

            if ci in viz_clip_ids:
                old24 = tr_old[t_mid]
                new24 = tr_new01[t_mid]
                viz(view, ci, frame0, fr[t_mid], old_pts0, new_pts0, old24, new24, OUT)

        def m(k):
            v = rows[k]
            return float(np.nanmean(v)) if len(v) else float("nan")

        lines.append(
            f"[{view}] valid={n_valid_view}/{N_CLIP} used={n_used}/{n_valid_view} sam2_fail={sam2_fail} "
            f"exploded(>30px@128 step)={exploded} | sam2_score={m('sam2_score'):.3f} | "
            f"hull(px^2 @128): OLD {m('old_hull'):.1f} -> NEW {m('new_hull'):.1f} | mask_area {m('mask_area'):.1f} | "
            f"coverage(hull/mask): OLD {m('old_cov'):.3f} -> NEW {m('new_cov'):.3f} | "
            f"vis-rate: OLD {m('old_vis'):.3f} -> NEW {m('new_vis'):.3f} | "
            f"step-disp px@128 mean: OLD {m('old_step_mean'):.2f} -> NEW {m('new_step_mean'):.2f} | "
            f"step-disp px@128 max(of-max): OLD {m('old_step_max'):.2f} -> NEW {m('new_step_max'):.2f}"
        )

    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"saved {OUT}/ (summary.txt + viz pngs)\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
