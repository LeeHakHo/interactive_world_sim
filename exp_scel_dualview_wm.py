"""② DUAL-VIEW joint flow prediction (the missing middle of the interface axis).

2026-07-05 verdict: per-view 2D beats 3D (whose weakness = depth-lift input
noise). Dual-view 2D predicts BOTH views' 48-pt flow jointly: two projections
determine 3D implicitly, no depth consumed. Eval adds TRIANGULATED 3D error
(predicted view pair -> ray intersection -> mm vs depth-lifted GT): if it
approaches the 6.9mm depth-3D quality at 2D-native px accuracy, dual-view is
the interface sweet spot.

Model: FlowWM_LWC trunk, 2P point tokens (learned view embedding), per-token
2D velocity classes (unchanged head); action = per-view dummy5 constellation
(the adopted ② action rep, project_detmem_dit) concatenated.
Data: aligned dual-view clips (gen_dualview_aligned.py).
Env: DS(npz), SMOKE, OUT_DIR, SPLIT(legacy|okfirst; okfirst=③-style filter-then-permute,
audit item5 leakage fix). iws env, GPU.
-> outputs/cross_embodiment_wm/dualview_wm/ (wm_dual.pt + summary.txt)

--- SKELETON ACTION REPRESENTATION experiment (2026-07-15) ---
Attacks the diagnosed human-helps bottleneck (project_action_dflow_separability): dummy5's action
INPUT is the absolute eef constellation, which is domain-disjoint between human/robot (probe 1.0);
an ISOMORPHIC skeleton action rep (same 4-joint geometric chain, isomorphic v2 design from
FLOW_WARP_REPR_LOG.md §1 "同构骨架"/exp_scel_dualview_dit_formal.py's flowskelv2 render-condition
precedent -- SAME sidecar files, but consumed here as the WM's action INPUT, not a render condition)
should let human data help the robot more, especially in the robot-scarce regime.
Env additions (all default to byte-identical old behavior):
  ACTION(dummy5|skel): skel = 4 tokens/view, robot [link_6,fingertipL',fingertipR',link_5] from
    skel_sidecar_robot_v2.npz (idx [5,6,7,4]; link_5->link_6 = approach dir, mirrors human
    forearm_stub->wrist), human [wrist,fin1,fin2,forearm_stub] from skel_sidecar_human.npz verbatim.
    Fed the same way dummy5 feeds its 5 points (object-centroid-relative, concat both views into
    s.act); no-leakage: both sidecars are built from joints/eef/hand-kpts only, never future frames.
  MIX(r|rh): rh = co-train on clips_human_L24.npz (L=24 vs robot's L=48) alongside the robot pool,
    same vis-weighted CE loss, SS rollout steps capped to the human clip length (R_SS_h<=L_h-K).
  NROB(""|int): robot-scarce protocol -- subsample the (okfirst) robot TRAINING pool to NROB clips
    (rng(0), deterministic/shared across arms); held-out eval is always the full robot okfirst
    heldout set (never subsampled).
  OUT_DIR default routes to outputs/cross_embodiment_wm/dualview_wm_skelact/{ACTION}_{MIX}_n{NROB}/
    whenever any of ACTION/MIX/NROB is non-default; the all-default run keeps the original path.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

os.chdir("/scr2/yusenluo/interactive_world_sim")
import eval_scheduled_sampling as SSm
from amplify_wm import FlowWM_LWC, K, F, vel_to_class
from scipy.spatial.transform import Rotation

SMOKE = os.environ.get("SMOKE", "0") == "1"
DS = os.environ.get("DS", "outputs/flow_render_dataset_can_dual/clips_robot.npz")
SPLIT = os.environ.get("SPLIT", "legacy")                     # legacy | okfirst (audit item5 leakage fix)
ACTION = os.environ.get("ACTION", "dummy5")                   # dummy5 (default, unchanged) | skel (isomorphic v2)
MIX = os.environ.get("MIX", "r")                               # r (default, robot-only) | rh (human co-train)
NROB = os.environ.get("NROB", "")                               # "" = full robot pool | int = robot-scarce N
_NROB_TAG = NROB if NROB else "all"
_DEFAULT_OUT = ("outputs/cross_embodiment_wm/dualview_wm" if (ACTION == "dummy5" and MIX == "r" and not NROB)
                else f"outputs/cross_embodiment_wm/dualview_wm_skelact/{ACTION}_{MIX}_n{_NROB_TAG}")
OUT = os.environ.get("OUT_DIR", _DEFAULT_OUT)
os.makedirs(OUT, exist_ok=True)
H = 40; HELDOUT = 150; IMG = 128; VEL_HALF = 0.06
device = "cuda" if torch.cuda.is_available() else "cpu"

CROPS = {0: (60, 60, 390, 390), 1: (0, 0, 640, 480)}          # view0=cam_high, view1=cam_low
_CAL = {0: json.load(open("calib/rgb_cam_calib_can_REAL.json")),
        1: json.load(open("calib/rgb_cam_calib_can_low_REAL.json"))}
_RT = {v: (Rotation.from_quat(c["quat_xyzw"]).as_matrix(), np.asarray(c["t_world"]))
       for v, c in _CAL.items()}


def dummy5(eef3):
    """(B,Lw,3,2) -> (B,Lw,5,2) DexWM virtual constellation (per view)."""
    wrist, t1, t2 = eef3[:, :, 0], eef3[:, :, 1], eef3[:, :, 2]
    c = (t1 + t2) / 2; ax = (t1 - t2) / 2
    perp = torch.stack([-ax[..., 1], ax[..., 0]], -1)
    return torch.stack([wrist, c, c + ax, c - ax, c + perp], 2)


# --- ACTION=skel: isomorphic v2 skeleton action rep (module docstring §skeleton experiment) ---
# skel_sidecar_robot_v2.npz joint order: [link_1,link_2,link_3,link_4,link_5,link_6,fingertipL',fingertipR']
# (idx 0..7); we take [link_6, fingertipL', fingertipR', link_5] = idx [5,6,7,4], isomorphic to
# skel_sidecar_human.npz's verbatim [wrist, fingertip1, fingertip2, forearm_stub] (idx 0..3, used as-is).
SKEL_ROBOT_IDX = [5, 6, 7, 4]


def load_action_tokens(mode, dom, z, sk=None):
    """Per-view raw action-point arrays (N,L,n_raw,2), windowed downstream exactly like the original
    dummy5 eef arrays (train_dual/rollout_dual index [:, h:h+K+F] and pad).
    mode='dummy5' (default): n_raw=3 eef triplet (wrist,fin1,fin2) -- IDENTICAL to the pre-existing
      main() computation (efA has no nan_to_num, matching the original: z['eef'] is 0%-NaN in this
      dataset for both domains, verified 2026-07-15).
    mode='skel': n_raw=4, the isomorphic v2 skeleton tokens (dom='r' selects SKEL_ROBOT_IDX from the
      8-pt robot_v2 sidecar; dom='h' uses the human sidecar's 4 points verbatim); nan_to_num(0.5) on
      both views (matches wm convention for track/action arrays with occasional NaN, e.g. undetected
      human hand frames or a view-1 fall-off)."""
    if mode == "dummy5":
        efA = z["eef"].astype(np.float32)
        efB = np.nan_to_num(z["eef_low"].astype(np.float32), nan=0.5)
        return efA, efB
    assert sk is not None, "ACTION=skel requires the skeleton sidecar npz"
    a, b = sk["skel2d_high"], sk["skel2d_low"]
    if dom == "r":
        a, b = a[:, :, SKEL_ROBOT_IDX], b[:, :, SKEL_ROBOT_IDX]
    efA = np.nan_to_num(a.astype(np.float32), nan=0.5)
    efB = np.nan_to_num(b.astype(np.float32), nan=0.5)
    return efA, efB


class DualLWC(FlowWM_LWC):
    def __init__(s, P, action="dummy5", **kw):
        super().__init__(P, **kw)
        s.P1 = P
        s.action = action
        s.n_tok = 5 if action == "dummy5" else 4                # dummy5: 3->5 virtual pts | skel: 4 tokens as-is
        s.view_emb = nn.Parameter(torch.zeros(2, s.Dm))
        s.act = nn.Linear((K + F) * s.n_tok * 2 * 2, s.Dm)      # action tokens x 2 views

    def _act_pts(s, eef):
        """eef (B,Lw,n_raw,2) -> (B,Lw,n_tok,2) action tokens. dummy5: DexWM virtual constellation.
        skel: tokens already ARE the isomorphic geometric skeleton points -> identity (no expansion).
        getattr default covers unpickling checkpoints saved before this attribute existed."""
        return dummy5(eef) if getattr(s, "action", "dummy5") == "dummy5" else eef

    def fwd_dual(s, hist, eef_a, eef_b):
        """hist (B,2P,K,2) [view0 pts | view1 pts]; eef_a/b (B,K+F,n_raw,2) per view."""
        B, P2 = hist.shape[:2]
        anchor = hist[:, :, -1, :]
        obj = s.inp(torch.cat([(hist - anchor[:, :, None]).reshape(B, P2, 2 * K), anchor], -1))
        obj = obj + torch.cat([s.view_emb[0].expand(B, s.P1, s.Dm),
                               s.view_emb[1].expand(B, s.P1, s.Dm)], 1)
        oc_a = anchor[:, :s.P1].mean(1, keepdim=True)          # per-view object centroid
        oc_b = anchor[:, s.P1:].mean(1, keepdim=True)
        d5a = s._act_pts(eef_a) - oc_a[:, :, None]
        d5b = s._act_pts(eef_b) - oc_b[:, :, None]
        act = s.act(torch.cat([d5a.reshape(B, -1), d5b.reshape(B, -1)], -1))[:, None]
        x = s.tf(torch.cat([obj, act], 1))[:, :P2]
        logits = s.head(x).reshape(B, P2, F, s.W * s.W)
        return logits, anchor


def _ss_batch_step(m, opt, G, Vv, Ea, Eb, R_SS, pteach):
    """One optimizer step of scheduled-sampling rollout over R_SS steps for one batch. Domain-agnostic
    (robot or human -- caller picks the matching R_SS/tensors); verbatim body of the original
    train_dual inner loop, factored out so MIX=rh can reuse it unchanged for the human pass."""
    buf = G[:, :K].clone(); losses = []
    for h in range(R_SS):
        wa = Ea[:, h:h + K + F]; wb = Eb[:, h:h + K + F]
        if wa.shape[1] < K + F:
            pad = K + F - wa.shape[1]
            wa = torch.cat([wa, wa[:, -1:].repeat(1, pad, 1, 1)], 1)
            wb = torch.cat([wb, wb[:, -1:].repeat(1, pad, 1, 1)], 1)
        logits, _ = m.fwd_dual(buf[:, -K:].permute(0, 2, 1, 3), wa, wb)
        lg0 = logits[:, :, 0, :]
        gt_vel = G[:, K + h] - buf[:, -1]
        cls = vel_to_class(gt_vel, m.W, m.vel_half)
        w = (Vv[:, K + h] * Vv[:, K - 1])
        ce = nn.functional.cross_entropy(lg0.reshape(-1, m.W * m.W), cls.reshape(-1),
                                         reduction="none")
        ce = (ce * w.reshape(-1)).sum() / (w.sum() + 1e-6)
        losses.append(ce)
        nxt = buf[:, -1] + m.expected_vel(lg0)
        use_gt = (torch.rand(len(G), 1, 1, device=device) < pteach)
        buf = torch.cat([buf, torch.where(use_gt, G[:, K + h], nxt.detach())[:, None]], 1)
    loss = torch.stack(losses).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss)


def train_dual(trD, vsD, efA, efB, idx, seed=0, R_SS=32, action="dummy5",
               trDh=None, vsDh=None, efAh=None, efBh=None, idxh=None):
    """SS training, robot pool (trD/.../idx). MIX=rh (trDh et al not None): after each robot epoch's
    batches, also SS-train on the human co-train pool with the SAME vis-weighted CE loss (no extra
    reweighting) -- rollout steps capped to the human clip length (R_SS_h = min(R_SS, L_human-K),
    since clips_human_L24.npz is L=24 vs the robot's L=48). Default args (action='dummy5', no *h)
    reproduce the original robot-only loop exactly (same rng call order/shapes -> byte-identical)."""
    torch.manual_seed(seed); P2 = trD.shape[2]
    m = DualLWC(P2 // 2, action=action, Dm=384, layers=3, W=15, vel_half=VEL_HALF).to(device)
    # base __init__ sized inp for P; token count doesn't affect Linear dims -> fine
    opt = torch.optim.AdamW(m.parameters(), lr=SSm.WM_LR)
    g = torch.Generator().manual_seed(seed)
    trT = torch.from_numpy(trD).float(); vsT = torch.from_numpy(vsD).float()
    eAT = torch.from_numpy(efA).float(); eBT = torch.from_numpy(efB).float()
    mix_human = trDh is not None
    if mix_human:
        trTh = torch.from_numpy(trDh).float(); vsTh = torch.from_numpy(vsDh).float()
        eATh = torch.from_numpy(efAh).float(); eBTh = torch.from_numpy(efBh).float()
        R_SS_h = min(R_SS, trDh.shape[1] - K)
        gh = torch.Generator().manual_seed(seed + 1000)
    epochs = 2 if SMOKE else SSm.WM_EPOCHS
    for ep in range(epochs):
        pteach = 1.0 + (0.3 - 1.0) * ep / max(epochs - 1, 1)
        m.train(); pe = idx[torch.randperm(len(idx), generator=g)]
        loss = 0.0
        for i in range(0, len(pe), SSm.WM_BS):
            b = pe[i:i + SSm.WM_BS]
            G = trT[b].to(device); Vv = vsT[b].to(device)
            Ea = eAT[b].to(device); Eb = eBT[b].to(device)
            loss = _ss_batch_step(m, opt, G, Vv, Ea, Eb, R_SS, pteach)
        if mix_human:
            peh = idxh[torch.randperm(len(idxh), generator=gh)]
            for i in range(0, len(peh), SSm.WM_BS):
                b = peh[i:i + SSm.WM_BS]
                G = trTh[b].to(device); Vv = vsTh[b].to(device)
                Ea = eATh[b].to(device); Eb = eBTh[b].to(device)
                loss = _ss_batch_step(m, opt, G, Vv, Ea, Eb, R_SS_h, pteach)
        print(f"ep{ep} loss {loss:.4f}", flush=True)
    return m.eval()


@torch.no_grad()
def rollout_dual(m, trD, efA, efB, Hn):
    buf = trD[:, :K].clone(); Lw = K + F; preds = []
    for h in range(Hn):
        wa = efA[:, h:h + Lw]; wb = efB[:, h:h + Lw]
        if wa.shape[1] < Lw:
            pad = Lw - wa.shape[1]
            wa = torch.cat([wa, wa[:, -1:].repeat(1, pad, 1, 1)], 1)
            wb = torch.cat([wb, wb[:, -1:].repeat(1, pad, 1, 1)], 1)
        logits, _ = m.fwd_dual(buf[:, -K:].permute(0, 2, 1, 3), wa, wb)
        nxt = buf[:, -1] + m.expected_vel(logits[:, :, 0, :])
        preds.append(nxt); buf = torch.cat([buf, nxt[:, None]], 1)
    return torch.stack(preds, 1)                               # (B,H,2P,2)


def rays_from_norm(uv_norm, view):
    """crop-norm (…,2) -> (origin (3,), dirs (…,3)) in world."""
    x, y, w, h = CROPS[view]
    c = _CAL[view]; R, t = _RT[view]
    u = uv_norm[..., 0] * w + x; v = uv_norm[..., 1] * h + y
    d_cam = np.stack([(u - c["cx"]) / c["f"], (v - c["cy"]) / c["f"], np.ones_like(u)], -1)
    return t, d_cam @ R.T


def triangulate(uv0, uv1):
    """two crop-norm point sets (…,2) -> world (…,3) least-squares midpoint."""
    o0, d0 = rays_from_norm(uv0, 0); o1, d1 = rays_from_norm(uv1, 1)
    d0 = d0 / np.linalg.norm(d0, axis=-1, keepdims=True)
    d1 = d1 / np.linalg.norm(d1, axis=-1, keepdims=True)
    b = o1 - o0
    d0d1 = (d0 * d1).sum(-1)
    denom = 1 - d0d1 ** 2
    t0 = ((b * d0).sum(-1) - (b * d1).sum(-1) * d0d1) / (denom + 1e-9)
    t1 = ((b * d0).sum(-1) * d0d1 - (b * d1).sum(-1)) / (denom + 1e-9)
    p0 = o0 + t0[..., None] * d0
    p1 = o1 + t1[..., None] * d1
    return (p0 + p1) / 2


def subsample_robot_pool(pool, nrob):
    """Robot-scarce protocol: deterministic rng(0).choice subset of the (okfirst) robot TRAINING pool,
    SAME subset for a given (pool, nrob) pair regardless of ACTION/MIX (all 2x2 arms at a given N train
    on the identical robot clips) -- rng(0) depends only on pool/nrob, not on ACTION/MIX/SPLIT choices
    upstream. No-op (returns pool unchanged) if nrob is falsy or >= len(pool)."""
    if not nrob:
        return pool
    n = int(nrob)
    if n >= len(pool):
        return pool
    return np.random.default_rng(0).choice(pool, size=n, replace=False)


def split_okfirst(ok, heldout=HELDOUT):
    """③-style split (filter-then-permute), verbatim: exp_scel_dualview_dit_formal.py
    `okr = np.where(R["ok"])[0]; perm = rng(0).permutation(okr); ho, pool = perm[:150], perm[150:]`.
    Fixes audit item5: legacy ②-split permutes ALL indices then filters `ok` post-hoc, which
    draws a *different* Fisher-Yates sequence than ③'s filter-first split whenever `ok` has
    holes -> ②'s held-out set is not the same as ③'s, so ② ends up training on most of ③'s
    published eval clips. `ho`/`pool` are returned already ok-filtered, same as legacy."""
    okr = np.where(ok)[0]
    perm = np.random.default_rng(0).permutation(okr)
    return perm[:heldout], perm[heldout:]


def main():
    z = np.load(DS)
    ok = z["low_valid"]
    trA = z["tracks"].astype(np.float32); trB = z["tracks_low"].astype(np.float32)
    vsA = z["vis"].astype(np.float32); vsB = z["vis_low"].astype(np.float32)
    t3 = z["tracks3d"].astype(np.float64)
    trD = np.concatenate([trA, np.nan_to_num(trB, nan=0.5)], 2)   # (N,L,2P,2)
    vsD = np.concatenate([vsA, vsB], 2)

    ds_dir = os.path.dirname(DS)
    sk_r = np.load(f"{ds_dir}/skel_sidecar_robot_v2.npz") if ACTION == "skel" else None
    efA, efB = load_action_tokens(ACTION, "r", z, sk_r)

    print(f"split={SPLIT} action={ACTION} mix={MIX} nrob={NROB or 'all'}", flush=True)
    if SPLIT == "okfirst":
        ho, pool = split_okfirst(ok, HELDOUT)
    else:
        perm = np.random.default_rng(0).permutation(len(trD))
        ho = np.array([i for i in perm[:HELDOUT] if ok[i]])
        pool = np.array([i for i in perm[HELDOUT:] if ok[i]])
    if SMOKE: pool = pool[:200]; ho = ho[:24]
    pool = subsample_robot_pool(pool, NROB)
    print(f"aligned clips: {int(ok.sum())}/{len(ok)}; train {len(pool)} ho {len(ho)}", flush=True)

    trDh = vsDh = efAh = efBh = idxh = None
    n_human = 0
    if MIX == "rh":
        zh = np.load(f"{ds_dir}/clips_human_L24.npz")
        okh = zh["low_valid"]
        trAh = zh["tracks"].astype(np.float32); trBh = zh["tracks_low"].astype(np.float32)
        vsAh = zh["vis"].astype(np.float32); vsBh = zh["vis_low"].astype(np.float32)
        trDh = np.concatenate([trAh, np.nan_to_num(trBh, nan=0.5)], 2)
        vsDh = np.concatenate([vsAh, vsBh], 2)
        sk_h = np.load(f"{ds_dir}/skel_sidecar_human.npz") if ACTION == "skel" else None
        efAh, efBh = load_action_tokens(ACTION, "h", zh, sk_h)
        idxh_np = np.where(okh)[0]
        if SMOKE: idxh_np = idxh_np[:200]
        idxh = torch.from_numpy(idxh_np)
        n_human = len(idxh)
        print(f"human co-train clips: {n_human}/{len(okh)} (L={trDh.shape[1]})", flush=True)

    m = train_dual(trD, vsD, efA, efB, torch.from_numpy(pool), action=ACTION,
                   trDh=trDh, vsDh=vsDh, efAh=efAh, efBh=efBh, idxh=idxh)
    torch.save(m, f"{OUT}/wm_dual.pt")

    pr = rollout_dual(m, torch.from_numpy(trD[ho]).float().to(device),
                      torch.from_numpy(efA[ho]).float().to(device),
                      torch.from_numpy(efB[ho]).float().to(device), H).cpu().numpy()
    P = trA.shape[2]
    prA, prB = pr[:, :, :P], pr[:, :, P:]
    gtA = trA[ho][:, K:K + H]; gtB = trB[ho][:, K:K + H]
    eA = np.linalg.norm(prA - gtA, axis=-1) * IMG
    eB = np.linalg.norm(prB - np.nan_to_num(gtB, nan=0.5), axis=-1) * IMG
    # triangulated 3D error vs depth-lifted GT
    tri = triangulate(prA, prB)
    gt3 = t3[ho][:, K:K + H]
    vmask = np.isfinite(gt3[..., 0])
    e3 = np.linalg.norm(tri - np.nan_to_num(gt3), axis=-1) * 1000
    e3 = float(e3[vmask].mean())
    ez = np.abs(tri[..., 2] - np.nan_to_num(gt3[..., 2])) * 1000
    ez = float(ez[vmask].mean())
    zr = np.nanmax(t3[ho][..., 2], axis=(1, 2)) - np.nanmin(t3[ho][..., 2], axis=(1, 2))
    lift = zr > 0.08
    metrics = {
        "action": ACTION, "mix": MIX, "nrob": int(NROB) if NROB else None,
        "n_train_robot": int(len(pool)), "n_human_cotrain": int(n_human), "n_heldout": int(len(ho)),
        "drift_px_cam_high": float(eA.mean()), "drift_px_cam_low": float(eB.mean()),
        "drift_px_cam_high_lift": float(eA[lift].mean()), "drift_px_cam_low_lift": float(eB[lift].mean()),
        "n_lift": int(lift.sum()), "tri3d_err_mm": e3, "tri3d_z_err_mm": ez,
    }
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)
    lines = [
        f"DUAL-VIEW ② (joint 2-view flow, action={ACTION}, mix={MIX}, nrob={NROB or 'all'}) "
        f"H={H} held-out n={len(ho)} train_robot={len(pool)} human_cotrain={n_human}",
        f"drift px cam_high: dual {metrics['drift_px_cam_high']:.2f}   (single-view 2D baseline 2.96, 3D 3.11)",
        f"drift px cam_low : dual {metrics['drift_px_cam_low']:.2f}   (single-view 2D baseline 3.10, 3D 7.78)",
        f"lift subset (n={metrics['n_lift']}): high {metrics['drift_px_cam_high_lift']:.2f} "
        f"| low {metrics['drift_px_cam_low_lift']:.2f}",
        f"TRIANGULATED 3D err: {e3:.1f} mm (z {ez:.1f} mm)  [3D-② z-err was 6.9mm clean / 12.5mm camlow-lifted]",
    ]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + "\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
