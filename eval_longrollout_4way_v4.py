"""4-way (arch x anti-drift) LONG sliding-window rollout ADE on IN-DISTRIBUTION S=3
held-out sequences (vid=12 play_robot_3), split by GT cube motion (static vs dynamic).
Settles: is thin+antidrift the winner, and is the contact-gate harmful on long rollout
(under-moves the cube)? Trains 4 variants on flow_ds_v4 (static-inclusive), builds many
long held-out sequences, rolls each variant out for Hroll frames, reports rollout ADE +
the cube-centroid motion the model predicts on STATIC clips (= hallucination).
Output: outputs/flow_wm_v4/longroll_4way/summary.txt
"""
import os, numpy as np, torch
import train_flow_wm_scarcity_v4 as v4
import gen_flow_dataset_v4 as g4

OUT = "outputs/flow_wm_v4/longroll_4way"; os.makedirs(OUT, exist_ok=True)
dev = v4.device
K, F, L, P, S, RES = v4.K, v4.F, v4.L, v4.P, g4.S, 224
HROLL = 36                      # 3 sliding blocks of F=12
LSEQ = K + HROLL               # tracked frames per sequence
STATIC_PX = 6.0                # GT cube full-seq centroid path < this -> static
NSEQ = 40                      # held-out sequences to build


def v4_rollout(m, trk_n, eseq_n, gstats, Hroll):
    # trk_n (LSEQ,P,2), eseq_n (>=LSEQ+F,3,2) normalized -> pr (Hroll,P,2)
    hist = torch.from_numpy(trk_n[:K]).float().permute(1, 0, 2)[None].to(dev)
    out, filled, bk = [], 0, 0
    while filled < Hroll:
        w0 = bk * F
        ew_np = eseq_n[w0:w0 + L]
        if len(ew_np) < L:
            ew_np = np.concatenate([ew_np, np.repeat(ew_np[-1:], L - len(ew_np), 0)], 0)
        ew = torch.from_numpy(ew_np).float()[None].to(dev)
        g = v4.normalize_grasp(v4.grasp_openness(ew), torch.ones(1, dtype=torch.long), gstats).to(dev)
        dm = torch.ones(1, dtype=torch.long, device=dev)
        with torch.no_grad():
            pred, _ = m(hist, ew, g, dm)
        take = min(F, Hroll - filled)
        out.append(pred[0, :, :take].cpu())
        hist = pred[:, :, take - K:take, :] if take >= K else torch.cat([hist[:, :, take:, :], pred[:, :, :take, :]], 2)
        filled += take; bk += 1
    return torch.cat(out, 1).numpy().transpose(1, 0, 2)   # (Hroll,P,2)


def cpath(pts):   # (T,P,2) normalized -> centroid path px
    c = pts.mean(1); return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum() * RES)


def build_seqs():
    from interactive_world_sim.algorithms.latent_decompose.ot_align import robot_world_to_cam
    T_cw = np.asarray(robot_world_to_cam(), np.float64); Kmat = g4.load_K()
    rdir = g4.ROBOT_DIRS[2]
    eef_fn, _ = g4.robot_eef3_loader(rdir, T_cw, Kmat)
    ct = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(dev).eval()
    vpath = f"{rdir}/videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
    crops = g4.decode("robot", vpath)
    rng = np.random.default_rng(0)
    starts = list(range(0, len(crops) - (LSEQ + F) * S, 40)); rng.shuffle(starts)
    seqs = []
    for s0 in starts:
        if len(seqs) >= NSEQ: break
        idxs = [s0 + k * S for k in range(LSEQ)]
        fr = [crops[i] for i in idxs]
        m0 = g4.cube_mask(fr[0]); ys, xs = np.where(m0 > 0)
        if len(xs) < P: continue
        sel = rng.choice(len(xs), P, replace=False)
        q = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
        trk, vsk = g4.track(ct, fr, q, dev)
        eidx = [s0 + k * S for k in range(LSEQ + F)]
        eseq = [eef_fn(i) for i in eidx]
        if any(e is None for e in eseq): continue
        trk_n = (trk / RES).astype(np.float32)
        gt_motion = cpath(trk_n)
        seqs.append(dict(trk_n=trk_n, vis=vsk.astype(np.float32),
                         eseq_n=(np.stack(eseq) / RES).astype(np.float32), gt_motion=gt_motion))
    return seqs


def main():
    z = np.load("outputs/flow_dataset/flow_ds_v4.npz")
    tr = torch.from_numpy(z["tracks"]).float(); vis = torch.from_numpy(z["vis"]).float()
    eef3 = torch.from_numpy(z["eef3"]).float(); dom = torch.from_numpy(z["domain"]).long()
    vid = torch.from_numpy(z["vid"]).long()
    gstats = v4.fit_grasp_stats(v4.grasp_openness(eef3), dom)
    rob = torch.where((dom == 1) & (vid != v4.TEST_ROBOT_VID))[0]
    variants = [("thin", True, False), ("thin+antidrift", True, True),
                ("thick-antidrift", False, False), ("thick+antidrift", False, True)]
    models = {}
    for name, thin, ad in variants:
        _, m = v4.train_eval(tr, vis, eef3, dom, gstats, rob, rob[:2],
                             thin=thin, seed=0, antidrift=ad, return_model=True)
        m.eval(); models[name] = m
        print(f"trained {name}", flush=True)

    seqs = build_seqs()
    statc = [s for s in seqs if s["gt_motion"] < STATIC_PX]
    dynm = [s for s in seqs if s["gt_motion"] >= STATIC_PX]
    print(f"built {len(seqs)} seqs: {len(statc)} static + {len(dynm)} dynamic", flush=True)

    lines = [f"4-way LONG rollout (S=3 in-dist, Hroll={HROLL}) | vid=12 | seqs={len(seqs)} "
             f"({len(statc)} static / {len(dynm)} dynamic) | data=flow_ds_v4",
             "ADE = rollout ADE px vs GT (visible pts) | hallu = predicted cube cpath on STATIC clips "
             f"(GT static cpath mean={np.mean([s['gt_motion'] for s in statc]) if statc else 0:.1f}px, should match)",
             f"{'variant':>16} | {'dyn_ADE':>8} | {'stat_ADE':>8} | {'stat_hallu_cpath':>16} | {'dyn_hallu_cpath':>15}"]
    for name, _, _ in variants:
        m = models[name]
        def rolladE(group):
            if not group: return float('nan'), float('nan')
            ades, halls = [], []
            for s in group:
                pr = v4_rollout(m, s["trk_n"], s["eseq_n"], gstats, HROLL)   # (Hroll,P,2)
                gt = s["trk_n"][K:K + HROLL]                                  # (Hroll,P,2)
                w = s["vis"][K:K + HROLL]                                     # (Hroll,P)
                err = np.linalg.norm(pr - gt, axis=-1) * RES                  # (Hroll,P)
                ades.append((err * w).sum() / (w.sum() + 1e-6))
                halls.append(cpath(pr))
            return float(np.mean(ades)), float(np.mean(halls))
        d_ade, d_hall = rolladE(dynm)
        s_ade, s_hall = rolladE(statc)
        lines.append(f"{name:>16} | {d_ade:8.2f} | {s_ade:8.2f} | {s_hall:16.2f} | {d_hall:15.2f}")
        print(lines[-1], flush=True)
    gtdyn = np.mean([cpath(s['trk_n'][K:K+HROLL]) for s in dynm]) if dynm else 0
    lines.append(f"\nREF GT cube cpath: static={np.mean([cpath(s['trk_n'][K:K+HROLL]) for s in statc]) if statc else 0:.2f}px  dynamic={gtdyn:.2f}px")
    msg = "\n".join(lines) + "\n"
    open(os.path.join(OUT, "summary.txt"), "w").write(msg)
    print("\n" + msg, flush=True)


if __name__ == "__main__":
    main()
