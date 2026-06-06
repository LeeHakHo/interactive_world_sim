"""IN-DISTRIBUTION S=3 pixel-rendered multi-trajectory action-replay rollout.

The prior render script (eval_rollout_replay_v4_3way.py) rolled out on seqs.npz, which
were built with S=4 (OOD for our v4 ② model, whose training convention is S=3). That OOD
sampling exaggerated hallucination. Here we build our OWN held-out sequences with the
EXACT v4 clip-extraction convention (crop 195,195,256,256; RES=224; S=3; P=48; cube_mask;
CoTracker; robot_eef3_loader) from play_robot_3 (vid=12, the held-out video), and roll out
the v4-RETRAINED models (trained on the new static-inclusive dataset flow_ds_v4.npz).

We render GT | render(thick) | render(thin) side-by-side for SEVERAL held-out sequences,
a MIX of static (cube should stay still) and dynamic ones, so the user can SEE whether the
cube stays put when it should, and how thick-vs-thin behave.

Renderer = densev4e ③ (flow + g-mask), identical to eval_full_rollout_replay.py.

SMOKE=1 env var -> NSEQ=1 (one dynamic seq), Hroll=12 (1 block), render only a couple
frames, just to verify the full pipeline runs end-to-end.

Output -> outputs/flow_wm_v4/viz_indist/
  replay_indist_seq{i}_{tag}.gif : per frame  GT | render(thick) | render(thin)
  replay_indist_strips.png       : per seq 3 rows (GT/thick/thin) at stride-3 horizons
  summary.txt                    : per seq tag, GT cube motion px, thick/thin ADE + drift
"""
import os, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pandas as pd

import train_flow_cond_decoder_densev4e as D
import train_flow_wm_scarcity_v4 as v4
import gen_flow_dataset_v4 as g4

SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT = "outputs/flow_wm_v4/REPORT/4_rollout_videos"; os.makedirs(OUT, exist_ok=True)
DEC = "outputs/flow_wm/flow_cond_decoder__densev4e_robot__armmask_gmask/decoder_flowcond_gmask.pt"
LAT, IMG = D.LAT_RES, D.IMG_RES
device = v4.device

# convention constants (from g4 / v4)
K, F, L, P = v4.K, v4.F, v4.L, v4.P           # 4, 12, 16, 48
S, RES = g4.S, g4.RES                          # 3, 224
CX, CY, CW, CH = g4.CX, g4.CY, g4.CW, g4.CH    # 195,195,256,256

NSEQ = 1 if SMOKE else 6
Hroll = 12 if SMOKE else 36                     # 1 / 3 blocks of F=12
Lseq = K + Hroll                                # tracked frames per seq
NEED_EEF = Lseq + F                             # eef/joint indices needed

gstats = None  # filled in main()


# ----------------------------------------------------------------------------- (1)
def train_v4_models():
    z = np.load("outputs/flow_dataset/flow_ds_v4.npz")
    tr = torch.from_numpy(z["tracks"]).float(); vis = torch.from_numpy(z["vis"]).float()
    eef3 = torch.from_numpy(z["eef3"]).float()
    dom = torch.from_numpy(z["domain"]).long(); vid = torch.from_numpy(z["vid"]).long()
    gs = v4.fit_grasp_stats(v4.grasp_openness(eef3), dom)
    rob = torch.where((dom == 1) & (vid != v4.TEST_ROBOT_VID))[0]
    # WINNER per 5-seed long-rollout: thick-antidrift (contact-gate ON, anti-drift OFF) =
    # least cube hallucination. Contrast with plain thin baseline.
    _, m_thick = v4.train_eval(tr, vis, eef3, dom, gs, rob, rob[:2],
                               thin=False, seed=0, antidrift=False, return_model=True)
    _, m_thin = v4.train_eval(tr, vis, eef3, dom, gs, rob, rob[:2],
                              thin=True, seed=0, antidrift=False, return_model=True)
    m_thick.eval(); m_thin.eval()
    return m_thick, m_thin, gs


# ----------------------------------------------------------------------------- (2)
def decode_dual(path):
    """Decode robot mp4 -> (crop224 list, crop128 list), mirroring gen_flow_render_dataset.decode."""
    import av
    c224, c128 = [], []
    cont = av.open(path); st = cont.streams.video[0]
    for frame in cont.decode(st):
        if len(c224) >= g4.MAXF: break
        rgb = frame.to_ndarray(format="rgb24")
        roi = rgb[CY:CY + CH, CX:CX + CW]
        c224.append(cv2.resize(roi, (RES, RES)))
        c128.append(cv2.resize(roi, (IMG, IMG)))
    cont.close()
    return c224, c128


def build_indist_seqs(m_dummy=None):
    """Build NSEQ in-distribution S=3 held-out sequences from play_robot_3 (vid=12).
    A mix of dynamic (high cube motion) + static (low cube motion) seqs.
    Returns list of dicts with: idxs, frames128, trk_n (Lseq,P,2), vis (Lseq,P),
    eef3_n (NEED_EEF,3,2), joint (NEED_EEF,7), motion (px), tag.
    """
    robot_dir = g4.ROBOT_DIRS[2]
    vpath = f"{robot_dir}/videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
    from interactive_world_sim.algorithms.latent_decompose.ot_align import robot_world_to_cam
    T_cw = np.asarray(robot_world_to_cam(), np.float64); Kmat = g4.load_K()
    eef_fn, neef = g4.robot_eef3_loader(robot_dir, T_cw, Kmat)
    df = pd.read_parquet(f"{robot_dir}/data/chunk-000/episode_000000.parquet")
    joint_all = np.stack(df["observation.state"].values).astype(np.float32)  # (T,7)
    njoint = len(joint_all)

    print(f"decoding {vpath} ...", flush=True)
    c224, c128 = decode_dual(vpath)
    nframes = len(c224)
    print(f"  decoded {nframes} frames; eef={neef} joint={njoint}", flush=True)

    ct = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device).eval()
    rng = np.random.default_rng(0)

    # candidate starts spanning the episode; need s0 + (Lseq-1)*S + F*S indices for eef/joint
    span_frames = (Lseq - 1) * S + 1                 # last tracked source frame offset
    max_src = (NEED_EEF - 1) * S                      # last eef/joint source index offset
    max_s0 = min(nframes - span_frames, njoint - 1 - max_src, neef - 1 - max_src)
    if max_s0 <= 0:
        raise RuntimeError(f"video too short: nframes={nframes} njoint={njoint} neef={neef}")
    n_scan = 6 if SMOKE else 60
    starts = np.linspace(0, max_s0, n_scan).astype(int).tolist()

    cands = []  # (motion, tag, dict)
    for s0 in starts:
        idxs = [s0 + k * S for k in range(Lseq)]          # tracked source frame indices
        eef_idxs = [s0 + k * S for k in range(NEED_EEF)]  # eef/joint source indices
        # eef + joint must be valid for all needed indices
        eef3 = [eef_fn(i) for i in eef_idxs]
        if any(e is None for e in eef3):
            continue
        if any(i >= njoint for i in eef_idxs):
            continue
        fr224 = [c224[i] for i in idxs]
        m0 = g4.cube_mask(fr224[0]); ys, xs = np.where(m0 > 0)
        if len(xs) < P:
            continue
        sel = rng.choice(len(xs), P, replace=False)
        q = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
        trk, vs = g4.track(ct, fr224, q, device)          # (Lseq,P,2),(Lseq,P)
        c = trk.mean(1)
        motion = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())  # px over full Lseq (display)
        # classify by the ROLLOUT window (frames K..Lseq-1) so "static" = cube barely moves
        # during the part we actually roll out (Lseq*6px-over-16 doesn't scale to 40 frames).
        fut_motion = float(np.linalg.norm(np.diff(trk[K:].mean(1), axis=0), axis=1).sum())
        tag = "static" if fut_motion < 18.0 else "dynamic"
        d = dict(
            idxs=idxs,
            frames128=np.stack([c128[i] for i in idxs]).astype(np.uint8),   # (Lseq,128,128,3)
            trk_n=(trk / RES).astype(np.float32),                            # (Lseq,P,2)
            vis=vs.astype(np.float32),                                       # (Lseq,P)
            eef3_n=(np.stack(eef3) / RES).astype(np.float32),                # (NEED_EEF,3,2)
            joint=np.stack([joint_all[i] for i in eef_idxs]).astype(np.float32),  # (NEED_EEF,7)
            motion=motion, tag=tag,
        )
        cands.append((motion, tag, d))
        print(f"  s0={s0:5d} motion={motion:6.2f}px -> {tag}", flush=True)

    if not cands:
        raise RuntimeError("no valid candidate sequences found")

    if SMOKE:
        dyn = [c for c in cands if c[1] == "dynamic"]
        chosen = (dyn or cands)[:1]
    else:
        dyn = sorted([c for c in cands if c[1] == "dynamic"], key=lambda x: -x[0])
        sta = sorted([c for c in cands if c[1] == "static"], key=lambda x: x[0])
        n_dyn = (NSEQ + 1) // 2
        n_sta = NSEQ - n_dyn
        chosen = dyn[:n_dyn] + sta[:n_sta]
        # backfill if one bucket is short
        if len(chosen) < NSEQ:
            rest = [c for c in (dyn + sta) if c not in chosen]
            chosen += rest[:NSEQ - len(chosen)]
    seqs = [c[2] for c in chosen]
    print(f"selected {len(seqs)} seqs: " +
          ", ".join(f"{s['tag']}({s['motion']:.1f}px)" for s in seqs), flush=True)
    return seqs


# ----------------------------------------------------------------------------- (3)
def v4_rollout(m, seq, gs):
    """Sliding-window rollout: trk_n[:K] history + eef3_n action windows -> pr (Hroll,P,2)."""
    trk = seq["trk_n"]; eff = seq["eef3_n"]
    hist = torch.from_numpy(trk[:K]).float().permute(1, 0, 2)[None].to(device)  # (1,P,K,2)
    pr = np.zeros((Hroll, P, 2), np.float32)
    filled, bk = 0, 0
    while filled < Hroll:
        w0 = bk * F
        ew_np = eff[w0:w0 + L]
        if len(ew_np) < L:
            ew_np = np.concatenate([ew_np, np.repeat(ew_np[-1:], L - len(ew_np), 0)], 0)
        ew = torch.from_numpy(ew_np).float()[None].to(device)                   # (1,L,3,2)
        g = v4.normalize_grasp(v4.grasp_openness(ew), torch.ones(1, dtype=torch.long), gs).to(device)
        dm = torch.ones(1, dtype=torch.long, device=device)
        with torch.no_grad():
            pred, _ = m(hist, ew, g, dm)                                        # (1,P,F,2)
        take = min(F, Hroll - filled)
        pr[filled:filled + take] = pred[0, :, :take].cpu().numpy().transpose(1, 0, 2)
        if take >= K:
            hist = pred[:, :, take - K:take, :]
        else:
            hist = torch.cat([hist[:, :, take:, :], pred[:, :, :take, :]], 2)
        filled += take; bk += 1
    return pr


# ----------------------------------------------------------------------------- (4)
def build_renderer():
    model = D.load_model(); D.widen_control_net(model.decoder.control_net, D.C_EXTRA)
    model.decoder.load_state_dict(torch.load(DEC, map_location="cpu")["decoder"])
    model.eval(); DT = model.dtype; nm = model.normalizer[model.obs_keys[0]]
    gck = torch.load(D.GCK, map_location="cpu")
    g = D.MaskGen(gck["indim"]).to(device); g.load_state_dict(gck["g"]); g.eval()
    xm = torch.from_numpy(np.asarray(gck["xm"], np.float32)).to(device)
    xs = torch.from_numpy(np.asarray(gck["xs"], np.float32)).to(device)
    from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

    @torch.no_grad()
    def enc0(fr):  # fr (1,128,128,3) uint8
        x = torch.from_numpy(fr).float().permute(0, 3, 1, 2).to(device) / 255.0
        return model.encoder_forward(nm.normalize(x).to(DT))

    @torch.no_grad()
    def render_seq(seq, pr, hmax):
        z0 = enc0(seq["frames128"][0:1])
        trk = seq["trk_n"]; eff = seq["eef3_n"]; vis_ = seq["vis"]; joint = seq["joint"]
        outs = []
        for h in range(hmax):
            t = K + h
            feat = np.concatenate([joint[t], eff[t].reshape(6)])[None].astype(np.float32)
            fn = (torch.from_numpy(feat).to(device) - xm) / xs
            pm = (torch.sigmoid(g(fn)[:, 0]) > 0.5).float().cpu().numpy()[0]
            gm = cv2.resize(pm, (LAT, LAT))[None]
            pos0 = np.concatenate([trk[0], eff[0]], 0)
            post = np.concatenate([pr[h], eff[t]], 0)
            v = np.concatenate([vis_[t], np.ones(3, np.float32)], 0)
            flow = D.splat_flow(pos0, post, v)[None]
            cond = torch.cat([z0, torch.from_numpy(np.concatenate([flow, gm[None]], 1)).to(DT).to(device)], 1)
            torch.manual_seed(0)
            r = render_img_cm(model, cond, IMG, model.normalizer, num_views=1, batch_size=1)[0].cpu().numpy()
            outs.append(np.clip(r.transpose(1, 2, 0), 0, 1))
        return outs

    return render_seq


# ----------------------------------------------------------------------------- metrics
def centroid_path(arr):  # arr (T,P,2) normalized -> total centroid path px
    c = arr.mean(1)
    return float(np.linalg.norm(np.diff(c, axis=0), axis=-1).sum() * RES)


def rollout_ade(pr, seq):  # pr (Hroll,P,2) vs GT tracks over Hroll, visible pts
    gt = np.stack([seq["trk_n"][K + h] for h in range(Hroll)], 0)   # (Hroll,P,2)
    w = np.stack([seq["vis"][K + h] for h in range(Hroll)], 0)      # (Hroll,P)
    err = np.linalg.norm(pr - gt, axis=-1) * RES
    m = w > 0.5
    return float(err[m].mean()) if m.any() else float("nan")


# ----------------------------------------------------------------------------- main
def main():
    global gstats
    print(f"=== v4 in-distribution rollout-replay (SMOKE={SMOKE}) NSEQ={NSEQ} Hroll={Hroll} ===", flush=True)
    m_thick, m_thin, gstats = train_v4_models()
    print("v4 models retrained (thick + thin) on flow_ds_v4.npz", flush=True)

    seqs = build_indist_seqs()
    render_seq = build_renderer()
    print("densev4e renderer + g-mask loaded", flush=True)

    # how many horizons to render (SMOKE: just a couple)
    hmax = min(2, Hroll) if SMOKE else Hroll

    lines = [f"v4 IN-DISTRIBUTION (S=3) pixel rollout replay | NSEQ={NSEQ} Hroll={Hroll} hmax={hmax}",
             "data=flow_ds_v4.npz (static-inclusive, RETRAINED) | renderer=densev4e ③ (flow+g-mask)",
             "② thick=contact-gate (no anti-drift, WINNER: least hallucination) | thin=baseline | held-out vid=12",
             "GT cube motion = GT centroid path over Lseq (px) | ADE = rollout ADE px (visible pts)",
             f"{'seq':>3} | {'tag':>7} | {'GTmotion':>8} | {'ADE_thick':>9} | {'ADE_thin':>9} | {'drift_thick':>11} | {'drift_thin':>10}"]
    static_notes = []

    cols = list(range(0, hmax, 3)) or [0]
    nrows = 3 * NSEQ
    fig, ax = plt.subplots(nrows, len(cols), figsize=(1.6 * len(cols), 1.6 * nrows), squeeze=False)

    for si, seq in enumerate(seqs):
        pr_thick = v4_rollout(m_thick, seq, gstats)
        pr_thin = v4_rollout(m_thin, seq, gstats)
        rt = render_seq(seq, pr_thick, hmax)
        rn = render_seq(seq, pr_thin, hmax)
        print(f"seq{si} ({seq['tag']}): rendered {hmax} frames", flush=True)

        at = rollout_ade(pr_thick, seq); an = rollout_ade(pr_thin, seq)
        dt = centroid_path(pr_thick); dn = centroid_path(pr_thin)
        # GT cube motion over the rolled-out window (K..K+Hroll-1) for a fair drift comparison
        gt_roll = np.stack([seq["trk_n"][K + h] for h in range(Hroll)], 0)
        gt_motion_roll = centroid_path(gt_roll)
        lines.append(f"{si:>3} | {seq['tag']:>7} | {seq['motion']:8.2f} | {at:9.2f} | {an:9.2f} | {dt:11.2f} | {dn:10.2f}")
        print(lines[-1], flush=True)
        if seq["tag"] == "static":
            static_notes.append(f"  seq{si}: GT static {gt_motion_roll:.1f}px -> thick pred {dt:.1f}px / thin pred {dn:.1f}px")

        # strip rows: GT / thick / thin at stride-3 horizons
        for ci, h in enumerate(cols):
            t = K + h
            r0 = 3 * si
            ax[r0, ci].imshow(seq["frames128"][t] / 255.); ax[r0, ci].axis("off")
            ax[r0 + 1, ci].imshow(rt[h]); ax[r0 + 1, ci].axis("off")
            ax[r0 + 2, ci].imshow(rn[h]); ax[r0 + 2, ci].axis("off")
            if si == 0:
                ax[r0, ci].set_title(f"t={t}", fontsize=7)
            if ci == 0:
                ax[r0, ci].set_title(f"seq{si} GT [{seq['tag']}]\nmot={seq['motion']:.0f}px", fontsize=7, loc="left")
                ax[r0 + 1, ci].set_title(f"thick ADE={at:.0f}", fontsize=7, loc="left")
                ax[r0 + 2, ci].set_title(f"thin ADE={an:.0f}", fontsize=7, loc="left")

        # GIF: GT | thick | thin (all rendered horizons)
        gif = []
        pad = np.full((IMG, 4, 3), 255, np.uint8)
        for h in range(hmax):
            t = K + h
            gtf = seq["frames128"][t].astype(np.uint8)
            tk = (rt[h] * 255).astype(np.uint8)
            tn = (rn[h] * 255).astype(np.uint8)
            panel = np.concatenate([gtf, pad, tk, pad, tn], 1)
            im = Image.fromarray(panel); dr = ImageDraw.Draw(im)
            dr.text((2, 2), "GT", fill=(255, 255, 0))
            dr.text((IMG + 6, 2), "thick", fill=(255, 255, 0))
            dr.text((2 * IMG + 10, 2), "thin", fill=(255, 255, 0))
            gif.append(im)
        gpath = f"{OUT}/replay_indist_seq{si}_{seq['tag']}.gif"
        gif[0].save(gpath, save_all=True, append_images=gif[1:], duration=150, loop=0)
        print(f"seq{si}: saved {gpath} ({hmax} frames)", flush=True)

    fig.suptitle("v4 IN-DISTRIBUTION (S=3) rollout replay: per seq GT / render(thick) / render(thin) (stride-3). "
                 "Static seqs: cube should stay still.", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/replay_indist_strips.png", dpi=115)
    plt.close(fig)

    if static_notes:
        lines.append("\nStatic-cube behaviour (GT motion -> predicted drift over rolled-out window):")
        lines += static_notes
    msg = "\n".join(lines) + "\n"
    open(f"{OUT}/summary.txt", "w").write(msg)
    print("\n" + msg, flush=True)
    print(f"saved {OUT}/replay_indist_strips.png + replay_indist_seq*.gif + summary.txt\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
