"""TRACK-POINT RESEEDING v2 -- FULL REGEN (2026-07-15). Adapted from the validated 30-clip
prototype (reseed_tracks_proto.py, commit 9466753; report .superpowers/sdd/task-track-reseed-
report.md) to the full can-dual dataset: robot (clips_robot.npz, N=2700, L=48) + human
(clips_human_L24.npz, N=1800, L=24; the can_dual dir has no L48 human clips), both cam_high +
cam_low. USER-APPROVED full run, in-place, branch phantom_dynamo.

USER HARDENING (they eyeballed the prototype): cam_high sometimes still fails -- the can TOP is
SILVER while the BODY is RED, and cam_high sees mostly the top, so a single-anchor SAM2 prompt
(frame 0 only, prototype behavior) can lock onto just the visible sub-part instead of the full
can. Relative to the prototype this script adds:

  1. MULTI-ANCHOR VOTING (footprint_warp.anchor_mask_voted precedent): SAM2 at the top-3
     least-occluded frames per clip/view (eef-to-object-centroid distance, footprint_warp.
     best_anchor_frame heuristic), not just frame 0. Each anchor's gated-best mask is warped to
     frame 0 via Umeyama similarity fit on the OLD tracks (motion-sensing use -- tracks are
     reliable for this even though bad at *coverage*, per footprint_warp's docstring) and
     pixel-voted (>=2/3 agree, or AND for 2 anchors -- footprint_warp's `acc >= min(2, len(order))`
     rule, reused verbatim). Mean pairwise IoU across the warped anchor masks is logged as
     `consistency`; low consistency (<CONSIST_THRESH) flags the clip ("anchor_disagreement").
  2. CANDIDATE GATING, extended from the prototype's inside-only gate: among SAM2's 3 proposals
     per anchor, a candidate must have >=70% of that anchor's OLD points inside AND area in
     [1.5x, 20x] that anchor's OLD hull. The upper bound is the silver-top-specific addition: a
     mask that is *only* the shiny top can still pass a naive inside-gate (the old, HSV-red-biased
     points already sit in a small sub-region) while being far too small relative to the true
     full-can hull -- 20x blocks that same failure mode from the other direction (SAM2 grabbing
     tray/table). Argmax(SAM2 score) among gated candidates, per anchor.
  3. If ALL anchors fail the gate, or the vote result itself fails the frame-0 gate, this falls
     back (in order) to: the single best-scoring passed anchor's warped mask, then the OLD point
     hull interior (prototype-era coverage, never worse than what existed before this regen).
     Every failure mode is recorded in `fail_reason` and folds into the boolean `flags` output --
     rows are NEVER dropped (fixed-shape arrays), only flagged for QC.

Writes SIDECAR ONLY: outputs/flow_render_dataset_can_dual/tracks_v2_{robot,human}.npz. NEVER
touches the source clips_*.npz -- `merge` mode md5-verifies the source file is byte-identical to
the md5 recorded by `run` mode before any GPU work started.

Modes (env MODE):
  run    -- process one (DOMAIN, VIEW) shard -> partial npz in outputs/flow_render_dataset_can_dual/
            tracksv2_partial/{DOMAIN}_{VIEW}[_{START}-{END}].npz + a per-clip .jsonl debug log.
            GPU (SAM2 + CoTracker3-offline). Env: DOMAIN (robot|human, required), VIEW (high|low,
            required), CLIP_START/CLIP_END (shard within domain, default full range).
  merge  -- concatenate shards, combine high+low -> outputs/flow_render_dataset_can_dual/
            tracks_v2_{robot,human}.npz (tracks2_high/low, vis2_high/low, seed_mask_coverage (N,2),
            flags (N,2) -- axis 1 = [high, low]). md5-checks the source clips_*.npz vs the manifest
            written by `run`. CPU-only.
  qc     -- coverage distribution (mean/p10/p2) per domain x view, flags/fail-reason counts, 10
            worst-case + 4 good-case overlay PNGs (old=green vs new=cyan, frame0 + mid-frame) ->
            outputs/cross_embodiment_wm/tracks_v2_qc/. Prints a BLOCKED banner if cam_high coverage
            or flag-rate looks systemically bad (silver-top mitigation did not work) -- does NOT
            refuse to write files; the STOP decision is made by whoever reads the QC output, per
            task instructions ("report BLOCKED with examples rather than shipping bad tracks").
            CPU-only (re-uses the merged sidecar + a re-open of the source frames for the overlay
            images only).
"""
import glob
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "phantom/submodules/sam2")

from gen_flow_render_dataset import track

MODE = os.environ.get("MODE", "run")
DOMAIN = os.environ.get("DOMAIN", "robot")
VIEW = os.environ.get("VIEW", "high")

DS_ROOT = "outputs/flow_render_dataset_can_dual"
DS_PATHS = {"robot": f"{DS_ROOT}/clips_robot.npz", "human": f"{DS_ROOT}/clips_human_L24.npz"}
PARTIAL_DIR = os.environ.get("PARTIAL_DIR", f"{DS_ROOT}/tracksv2_partial")
SIDECAR = {"robot": f"{DS_ROOT}/tracks_v2_robot.npz", "human": f"{DS_ROOT}/tracks_v2_human.npz"}
QC_DIR = os.environ.get("QC_DIR", "outputs/cross_embodiment_wm/tracks_v2_qc")

P = 48
IMG = 128
TOPK_ANCHOR = 3
INSIDE_GATE = 0.7
AREA_LO, AREA_HI = 1.5, 20.0
CONSIST_THRESH = 0.30
STEP_EXPLODE_PX = 30.0
SAM2_CKPT = "phantom/submodules/sam2/checkpoints/sam2_hiera_large.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

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


def md5sum(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------- geometry helpers (footprint_warp.py precedent) -----------------------------
def umeyama2d(p0, p1, w=None, allow_scale=True):
    """weighted similarity fit p1 ~= s*R@p0 + t. p0,p1 (P,2), w (P,). Returns 2x3 affine matrix.
    Copied from footprint_warp.py (same scale-clamp rationale: tracks collapse under grasp/
    occlusion, clamp stops spurious shrink)."""
    if w is None:
        w = np.ones(len(p0), np.float32)
    sw = w.sum() + 1e-6
    c0 = (p0 * w[:, None]).sum(0) / sw
    c1 = (p1 * w[:, None]).sum(0) / sw
    q0, q1 = p0 - c0, p1 - c1
    H = (w[:, None] * q0).T @ q1 / sw
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, d]) @ U.T
    s = 1.0
    if allow_scale:
        var0 = (w[:, None] * q0 ** 2).sum() / sw
        s = float((S * np.array([1, d])).sum() / (var0 + 1e-9))
        lo = float(os.environ.get("FP_SCALE_LO", "0.6"))
        hi = float(os.environ.get("FP_SCALE_HI", "1.7"))
        s = float(np.clip(s, lo, hi))
    A = np.zeros((2, 3), np.float32)
    A[:, :2] = s * R
    A[:, 2] = c1 - s * R @ c0
    return A


def warp_footprint(mask0_u8, tr0_01, trt_01, w=None, allow_scale=True):
    H, W = mask0_u8.shape
    A = umeyama2d(tr0_01 * np.array([W, H]), trt_01 * np.array([W, H]), w, allow_scale)
    return cv2.warpAffine(mask0_u8.astype(np.uint8), A, (W, H), flags=cv2.INTER_NEAREST)


def hull_area(pts01, hw=IMG):
    pts = np.clip((pts01 * hw).astype(np.int32), 0, hw - 1)
    if len(pts) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(pts.reshape(-1, 1, 2))))


def old_hull_mask(pts01, hw=IMG):
    """last-resort fallback: filled convex hull of the OLD points -- the prototype-era behavior
    (bad coverage but never crashes, never worse than before this regen)."""
    fp = np.zeros((hw, hw), np.uint8)
    pts = np.clip((pts01 * hw).astype(np.int32), 0, hw - 1)
    if len(pts) >= 3:
        cv2.fillConvexPoly(fp, cv2.convexHull(pts.reshape(-1, 1, 2)), 1)
    return fp.astype(bool)


def constrain_to_old_region(mask, old_pts0, expand_px=int(os.environ.get("CONSTRAIN_PX", "30"))):
    """约束撒点掩码到 '旧点凸包膨胀区' 内: 旧点虽挤但可靠在物体上, 以它膨胀 expand_px 覆盖整个物体(含银顶),
    同时挡住 SAM2 误纳的托盘/蓝条/桌面(那些离旧簇 >expand_px). expand_px=0 关闭(退回纯 SAM2 掩码)."""
    if expand_px <= 0:
        return mask
    H_ = mask.shape[0]
    reg = np.zeros((H_, H_), np.uint8)
    vp = np.clip((np.asarray(old_pts0) * H_).astype(np.int32), 0, H_ - 1)
    if len(vp) >= 3:
        cv2.fillConvexPoly(reg, cv2.convexHull(vp.reshape(-1, 1, 2)), 1)
    else:
        for x, y in vp:
            cv2.circle(reg, (int(x), int(y)), 3, 1, -1)
    k = 2 * expand_px + 1
    reg = cv2.dilate(reg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    out = (mask.astype(np.uint8) & reg).astype(bool)
    return out if out.sum() >= 8 else mask   # 交集太小(旧点异常)则退回原掩码, 不制造空掩码


def sample_in_mask(mask, n, rng):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    sel = rng.choice(len(xs), n, replace=len(xs) < n)
    pts_px = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
    return pts_px / mask.shape[0]


def step_disp(tr01, vs, hw=IMG):
    d = np.linalg.norm(np.diff(tr01, axis=0), axis=-1) * hw
    w = (vs[:-1] > 0.5) & (vs[1:] > 0.5)
    if w.sum() == 0:
        return float("nan"), float("nan")
    return float(d[w].mean()), float(d[w].max())


# ----------------------------- SAM2 candidate gen + gating -----------------------------
def sam2_candidates(frame_u8, pts01, neg01, up=4):
    """All 3 raw SAM2 proposals at ORIGINAL resolution, no selection (selection is gated_best's
    job -- kept separate so multi-anchor voting can inspect every anchor's full candidate set)."""
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
    masks_full = [cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                  for m in masks]
    return masks_full, [float(s) for s in scores]


def gated_best(masks, scores, pts01, old_hull_area, inside_gate=INSIDE_GATE, lo=AREA_LO, hi=AREA_HI):
    """argmax(SAM2 score) among candidates with >=inside_gate old-pts-inside AND area in
    [lo,hi]*old_hull_area. Returns (mask, score) or (None, None) if nothing passes both gates."""
    H, W = masks[0].shape
    pts_px = (pts01 * np.array([W, H])).astype(int).clip(0, [W - 1, H - 1])
    best_i, best_sc = -1, -1.0
    for i, (m, sc) in enumerate(zip(masks, scores)):
        inside = m[pts_px[:, 1], pts_px[:, 0]].mean()
        if inside < inside_gate:
            continue
        area = float(m.sum())
        if not (lo * old_hull_area <= area <= hi * old_hull_area):
            continue
        if sc > best_sc:
            best_sc, best_i = float(sc), i
    if best_i < 0:
        return None, None
    return masks[best_i], best_sc


def pick_anchors(tr_old, ef_old, vs_old, topk=TOPK_ANCHOR):
    """top-k least-occluded frames (eef-to-object-centroid distance, footprint_warp.
    best_anchor_frame precedent). Purely occlusion-driven -- NOT forced to include frame 0, since
    frame-0 occlusion is exactly the failure mode multi-anchor voting exists to escape."""
    c = tr_old.mean(1)
    d = np.linalg.norm(ef_old - c[:, None], axis=-1).min(1) * (vs_old.mean(1) > 0.5)
    order = np.argsort(-d)[:topk]
    return [int(x) for x in order]


def voted_mask_at_frame0(frame_all_u8, tr_old, ef_old, vs_old):
    """Multi-anchor SAM2 mask, gated per-anchor, majority-voted, warped to frame 0.
    Returns dict(mask=bool(H,W)|None, fail_reason=str|None, n_anchor_pass=int,
    consistency=float, mask_area=float, sam2_score=float)."""
    anchors = pick_anchors(tr_old, ef_old, vs_old)
    passed = []  # (ta, mask_at_ta, score)
    for ta in anchors:
        pts_ta = np.clip(tr_old[ta], 0, 1)
        neg_ta = np.clip(ef_old[ta], 0, 1)
        oh_ta = hull_area(pts_ta)
        if oh_ta <= 0:
            continue
        masks, scores = sam2_candidates(frame_all_u8[ta], pts_ta, neg_ta)
        m, sc = gated_best(masks, scores, pts_ta, oh_ta)
        if m is not None:
            passed.append((ta, m, sc))

    if not passed:
        return dict(mask=None, fail_reason="sam2_total_fail", n_anchor_pass=0,
                     consistency=float("nan"), mask_area=float("nan"), sam2_score=float("nan"))

    old_pts0 = np.clip(tr_old[0], 0, 1)
    warped = []
    for ta, m, sc in passed:
        w = vs_old[ta] * vs_old[0]
        mw = warp_footprint(m.astype(np.uint8), np.clip(tr_old[ta], 0, 1), old_pts0, w=w)
        warped.append(mw.astype(bool))

    consistency = 1.0
    if len(warped) >= 2:
        ious = []
        for i in range(len(warped)):
            for j in range(i + 1, len(warped)):
                inter = float((warped[i] & warped[j]).sum())
                union = float((warped[i] | warped[j]).sum()) + 1e-6
                ious.append(inter / union)
        consistency = float(np.mean(ious))

    acc = np.zeros_like(warped[0], np.int32)
    for mw in warped:
        acc += mw.astype(np.int32)
    voted = acc >= min(2, len(warped))  # footprint_warp.anchor_mask_voted rule

    oh0 = hull_area(old_pts0)
    pts0_px = np.clip((old_pts0 * IMG).astype(int), 0, IMG - 1)

    def gate_ok(mask):
        if mask.sum() == 0:
            return False
        inside = mask[pts0_px[:, 1], pts0_px[:, 0]].mean()
        area = float(mask.sum())
        return inside >= INSIDE_GATE and AREA_LO * oh0 <= area <= AREA_HI * oh0

    fail_reason = None
    if gate_ok(voted):
        final = voted
    else:
        best_idx = int(np.argmax([sc for _, _, sc in passed]))
        if gate_ok(warped[best_idx]):
            final = warped[best_idx]
            fail_reason = "voted_gate_fail_used_best_anchor"
        else:
            return dict(mask=None, fail_reason="voted_and_fallback_fail", n_anchor_pass=len(passed),
                         consistency=consistency, mask_area=float("nan"), sam2_score=float("nan"))

    if consistency < CONSIST_THRESH and len(warped) >= 2:
        fail_reason = (fail_reason + "+anchor_disagreement") if fail_reason else "anchor_disagreement"

    return dict(mask=final, fail_reason=fail_reason, n_anchor_pass=len(passed), consistency=consistency,
                mask_area=float(final.sum()), sam2_score=float(np.mean([sc for _, _, sc in passed])))


# ----------------------------- run mode -----------------------------
def run():
    domain, view = DOMAIN, VIEW
    assert domain in DS_PATHS, f"bad DOMAIN={domain}"
    assert view in VIEWS, f"bad VIEW={view}"
    src = DS_PATHS[domain]
    os.makedirs(PARTIAL_DIR, exist_ok=True)

    manifest = f"{PARTIAL_DIR}/{domain}_source.md5"
    if not os.path.exists(manifest):
        print(f"[{domain}/{view}] hashing source {src} for the never-touch manifest...", flush=True)
        open(manifest, "w").write(md5sum(src) + "\n")
    print(f"[{domain}/{view}] source md5 manifest: {open(manifest).read().strip()}", flush=True)

    z = np.load(src)
    N = z["frames"].shape[0]
    L = z["frames"].shape[1]
    start = int(os.environ.get("CLIP_START", "0"))
    end = int(os.environ.get("CLIP_END", str(N)))

    keys = VIEWS[view]
    frames_all = z[keys["frames"]]
    tracks_all = z[keys["tracks"]].astype(np.float32)
    eef_all = z[keys["eef"]].astype(np.float32)
    vis_all = z[keys["vis"]].astype(np.float32)
    valid_all = z[keys["valid"]] if keys["valid"] else np.ones(N, bool)

    ct = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device).eval()
    rng_seed = np.random.default_rng(0)  # deterministic in-mask point sampling (prototype convention)

    tag = f"{domain}_{view}" + (f"_{start}-{end}" if (start != 0 or end != N) else "")
    out_npz = f"{PARTIAL_DIR}/{tag}.npz"
    log_path = f"{PARTIAL_DIR}/{tag}.jsonl"

    n_rows = end - start
    tracks2 = np.zeros((n_rows, L, P, 2), np.float16)
    vis2 = np.zeros((n_rows, L, P), np.float16)
    coverage = np.full(n_rows, np.nan, np.float32)
    flags = np.zeros(n_rows, bool)

    logf = open(log_path, "w")
    n_ok = n_invalid = n_total_fail = n_flagged = 0
    for row, ci in enumerate(range(start, end)):
        if not valid_all[ci]:
            flags[row] = True
            n_invalid += 1
            logf.write(json.dumps(dict(ci=int(ci), reason="view_invalid")) + "\n")
            continue

        fr = frames_all[ci]
        tr_old = tracks_all[ci]
        ef = eef_all[ci]
        vs_old = vis_all[ci]
        old_pts0 = np.clip(tr_old[0], 0, 1)

        res = voted_mask_at_frame0(fr, tr_old, ef, vs_old)
        fail_reason = res["fail_reason"]
        if res["mask"] is None:
            mask = old_hull_mask(old_pts0)
            n_total_fail += 1
        else:
            mask = res["mask"]

        mask = constrain_to_old_region(mask, old_pts0)          # 挡托盘/蓝条溢出(旧点簇膨胀区约束)
        new_pts0 = sample_in_mask(mask, P, rng_seed)
        if new_pts0 is None:
            new_pts0 = old_pts0.copy()
            fail_reason = (fail_reason + "+empty_mask_fallback") if fail_reason else "empty_mask_fallback"

        q_px = (new_pts0 * IMG).astype(np.float32)
        tr_new, vs_new = track(ct, list(fr), q_px, device)
        tr_new01 = tr_new / IMG

        os_, om_ = step_disp(tr_old, vs_old)
        ns_, nm_ = step_disp(tr_new01, vs_new)
        if nm_ == nm_ and nm_ > STEP_EXPLODE_PX:
            fail_reason = (fail_reason + "+exploded") if fail_reason else "exploded"

        mask_area = float(mask.sum())
        oh = hull_area(old_pts0)
        nh = hull_area(new_pts0)
        cov = nh / mask_area if mask_area > 0 else float("nan")

        tracks2[row] = tr_new01.astype(np.float16)
        vis2[row] = vs_new.astype(np.float16)
        coverage[row] = cov
        is_flag = fail_reason is not None
        flags[row] = is_flag
        n_flagged += int(is_flag)
        n_ok += 1

        logf.write(json.dumps(dict(
            ci=int(ci), fail_reason=fail_reason, coverage=float(cov), old_hull=float(oh), new_hull=float(nh),
            mask_area=mask_area, sam2_score=res["sam2_score"], consistency=res["consistency"],
            n_anchor_pass=res["n_anchor_pass"], old_vis=float(vs_old.mean()), new_vis=float(vs_new.mean()),
            old_step_mean=os_, new_step_mean=ns_, old_step_max=om_, new_step_max=nm_,
        )) + "\n")
        logf.flush()

        if (row + 1) % 50 == 0:
            print(f"[{domain}/{view}] {row + 1}/{n_rows} (ok={n_ok} invalid={n_invalid} "
                  f"total_fail={n_total_fail} flagged={n_flagged})", flush=True)

    logf.close()
    np.savez(out_npz, tracks2=tracks2, vis2=vis2, coverage=coverage, flags=flags,
             clip_start=start, clip_end=end, N=N)
    print(f"[{domain}/{view}] DONE shard[{start}:{end}] -> {out_npz} | ok={n_ok} invalid={n_invalid} "
          f"total_fail={n_total_fail} flagged={n_flagged}\n=== DONE ===", flush=True)


# ----------------------------- merge mode -----------------------------
def _load_shards(domain, view):
    files = sorted(glob.glob(f"{PARTIAL_DIR}/{domain}_{view}*.npz"),
                    key=lambda p: int(np.load(p)["clip_start"]))
    if not files:
        raise FileNotFoundError(f"no partial shards found for {domain}/{view} in {PARTIAL_DIR}")
    parts = [np.load(f) for f in files]
    N = int(parts[0]["N"])
    L = parts[0]["tracks2"].shape[1]
    tracks2 = np.zeros((N, L, P, 2), np.float16)
    vis2 = np.zeros((N, L, P), np.float16)
    coverage = np.full(N, np.nan, np.float32)
    flags = np.zeros(N, bool)
    covered = np.zeros(N, bool)
    for f, part in zip(files, parts):
        s, e = int(part["clip_start"]), int(part["clip_end"])
        tracks2[s:e] = part["tracks2"]
        vis2[s:e] = part["vis2"]
        coverage[s:e] = part["coverage"]
        flags[s:e] = part["flags"]
        covered[s:e] = True
        print(f"  merged shard {f} [{s}:{e}]", flush=True)
    if not covered.all():
        missing = np.where(~covered)[0]
        raise RuntimeError(f"{domain}/{view}: shards don't cover full range, missing rows "
                            f"{missing[0]}..{missing[-1]} ({len(missing)} total) -- resume those shards")
    return tracks2, vis2, coverage, flags


def merge():
    for domain in ("robot", "human"):
        src = DS_PATHS[domain]
        manifest = f"{PARTIAL_DIR}/{domain}_source.md5"
        if not os.path.exists(manifest):
            print(f"[merge] WARNING: no md5 manifest for {domain} at {manifest} (run mode never ran?) - skipping", flush=True)
            continue
        before = open(manifest).read().strip()
        after = md5sum(src)
        assert before == after, (f"[merge] REFUSING TO WRITE: source {src} md5 changed "
                                  f"({before} -> {after}) -- something touched the original dataset!")
        print(f"[merge] {domain}: source md5 verified untouched ({after})", flush=True)

        th, vh, ch, fh = _load_shards(domain, "high")
        tl, vl, cl, fl = _load_shards(domain, "low")
        assert th.shape[0] == tl.shape[0], f"{domain}: high/low N mismatch {th.shape[0]} vs {tl.shape[0]}"

        seed_mask_coverage = np.stack([ch, cl], axis=1)  # (N,2) [high, low]
        flags = np.stack([fh, fl], axis=1)               # (N,2) [high, low]

        out_path = SIDECAR[domain]
        np.savez(out_path, tracks2_high=th, tracks2_low=tl, vis2_high=vh, vis2_low=vl,
                 seed_mask_coverage=seed_mask_coverage, flags=flags)
        print(f"[merge] wrote {out_path} | N={th.shape[0]} L={th.shape[1]} | "
              f"flagged: high={fh.sum()} low={fl.sum()} | "
              f"coverage mean: high={np.nanmean(ch):.3f} low={np.nanmean(cl):.3f}", flush=True)

        # re-verify untouched AFTER writing the sidecar too (paranoia -- sidecar path must differ from src)
        assert md5sum(src) == before, f"[merge] source {src} changed DURING merge -- investigate immediately"
    print("=== DONE (merge) ===", flush=True)


# ----------------------------- qc mode -----------------------------
def _viz_overlay(view, ci, f0, fmid, old0, new0, oldmid, newmid, mid_idx, out_path, title_extra=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.8))
    for a, img, op, npt, ttl in [(ax[0], f0, old0, new0, "t=0"), (ax[1], fmid, oldmid, newmid, f"t={mid_idx}")]:
        a.imshow(img)
        op_px = np.clip(op * IMG, 0, IMG - 1)
        npt_px = np.clip(npt * IMG, 0, IMG - 1)
        a.scatter(op_px[:, 0], op_px[:, 1], c="lime", s=10, label="old", edgecolors="k", linewidths=0.2)
        a.scatter(npt_px[:, 0], npt_px[:, 1], c="cyan", s=10, label="new (v2)", edgecolors="k", linewidths=0.2)
        a.set_title(ttl, fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
    ax[0].legend(loc="upper right", fontsize=6, framealpha=0.6)
    fig.suptitle(f"{view} | clip {ci} | old=green vs new(v2)=cyan {title_extra}", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def qc():
    os.makedirs(QC_DIR, exist_ok=True)
    lines = ["TRACKS-V2 FULL REGEN QC | augment_clips_tracksv2.py"]
    worst_candidates = []  # (coverage, domain, view, ci)
    good_candidates = []
    all_flag_reason_counts = {}

    for domain in ("robot", "human"):
        sc_path = SIDECAR[domain]
        if not os.path.exists(sc_path):
            lines.append(f"[{domain}] MISSING sidecar {sc_path} -- run merge first")
            continue
        z = np.load(sc_path)
        cov = z["seed_mask_coverage"]  # (N,2)
        flags = z["flags"]             # (N,2)
        for vi, view in enumerate(("high", "low")):
            c = cov[:, vi]
            f = flags[:, vi]
            valid_c = c[~np.isnan(c)]
            p2 = float(np.nanpercentile(c, 2)) if len(valid_c) else float("nan")
            p10 = float(np.nanpercentile(c, 10)) if len(valid_c) else float("nan")
            mean = float(np.nanmean(c)) if len(valid_c) else float("nan")
            lines.append(f"[{domain}/{view}] N={len(c)} coverage mean={mean:.3f} p10={p10:.3f} p2={p2:.3f} "
                         f"| flagged={int(f.sum())}/{len(f)} ({100 * f.mean():.1f}%)")
            # aggregate fail_reason counts from jsonl shard logs
            for jf in glob.glob(f"{PARTIAL_DIR}/{domain}_{view}*.jsonl"):
                for line in open(jf):
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    r = d.get("fail_reason") or d.get("reason")
                    if r:
                        for part in str(r).split("+"):
                            all_flag_reason_counts[f"{domain}/{view}/{part}"] = \
                                all_flag_reason_counts.get(f"{domain}/{view}/{part}", 0) + 1
            order = np.argsort(valid_c) if len(valid_c) else np.array([])
            n_worst = max(1, int(0.02 * len(valid_c))) if len(valid_c) else 0
            idx_map = np.where(~np.isnan(c))[0]
            if len(idx_map):
                worst_idx = idx_map[np.argsort(c[idx_map])[:n_worst]]
                best_idx = idx_map[np.argsort(-c[idx_map])[:2]]
                for ci in worst_idx:
                    worst_candidates.append((float(c[ci]), domain, view, int(ci)))
                for ci in best_idx:
                    good_candidates.append((float(c[ci]), domain, view, int(ci)))

    lines.append("")
    lines.append("fail_reason breakdown (from per-shard .jsonl logs, may double count multi-reason rows):")
    for k, v in sorted(all_flag_reason_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k}: {v}")

    # ---- render overlays: 10 worst (global, sorted ascending coverage) + 4 good ----
    worst_candidates.sort(key=lambda x: x[0])
    good_candidates.sort(key=lambda x: -x[0])
    worst_sel = worst_candidates[:10]
    good_sel = good_candidates[:4]

    src_cache = {}
    sc_cache = {}
    for cov_val, domain, view, ci in worst_sel + good_sel:
        if domain not in src_cache:
            src_cache[domain] = np.load(DS_PATHS[domain])
            sc_cache[domain] = np.load(SIDECAR[domain])
        zsrc, zsc = src_cache[domain], sc_cache[domain]
        keys = VIEWS[view]
        fr = zsrc[keys["frames"]][ci]
        tr_old = zsrc[keys["tracks"]][ci].astype(np.float32)
        L = fr.shape[0]
        mid = min(L // 2, L - 1)
        tr_new = zsc[f"tracks2_{view}"][ci].astype(np.float32)
        kind = "WORST" if (cov_val, domain, view, ci) in worst_sel else "GOOD"
        out_path = f"{QC_DIR}/{kind.lower()}_{domain}_{view}_clip{ci}_cov{cov_val:.2f}.png"
        _viz_overlay(view, ci, fr[0], fr[mid], np.clip(tr_old[0], 0, 1), np.clip(tr_new[0], 0, 1),
                     np.clip(tr_old[mid], 0, 1), np.clip(tr_new[mid], 0, 1), mid, out_path,
                     title_extra=f"[{kind} coverage={cov_val:.3f}]")
        lines.append(f"overlay: {out_path}")

    # ---- systemic silver-top failure check (cam_high specifically) ----
    lines.append("")
    blocked = False
    for domain in ("robot", "human"):
        sc_path = SIDECAR[domain]
        if not os.path.exists(sc_path):
            continue
        z = np.load(sc_path)
        cov_high = z["seed_mask_coverage"][:, 0]
        flag_high = z["flags"][:, 0]
        mean_h = float(np.nanmean(cov_high))
        flagrate_h = float(np.mean(flag_high))
        if mean_h < 0.5 or flagrate_h > 0.20:
            blocked = True
            lines.append(f"*** POSSIBLE SYSTEMIC cam_high FAILURE [{domain}]: mean coverage={mean_h:.3f} "
                         f"(<0.5) or flag-rate={flagrate_h:.1%} (>20%) -- EYEBALL the worst_* overlays above "
                         f"before trusting tracks2_high for {domain} ***")
    if blocked:
        lines.insert(1, "STATUS: BLOCKED-CANDIDATE -- see systemic cam_high warning below, eyeball overlays "
                        "before shipping")
    else:
        lines.insert(1, "STATUS: no systemic cam_high coverage/flag-rate red flag by the automated threshold "
                        "(mean>=0.5, flagrate<=20%) -- still eyeball the worst_* overlays")

    open(f"{QC_DIR}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"saved {QC_DIR}/ (summary.txt + overlay pngs)\n=== DONE (qc) ===", flush=True)


if __name__ == "__main__":
    if MODE == "run":
        run()
    elif MODE == "merge":
        merge()
    elif MODE == "qc":
        qc()
    else:
        raise ValueError(f"bad MODE={MODE!r}, expected run|merge|qc")
