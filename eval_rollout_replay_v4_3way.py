"""PIXEL-RENDERED 3-way rollout-replay video: GT | render(thick-②) | render(thin-②).

Mirrors eval_full_rollout_replay.py (densev4e ③ renderer + g-mask action), but the ②
flow is produced by our v4 FlowWMThick instead of the SS-②. Two v4 models are trained on
flow_ds_v3 (thick = full contact-gate+grasp+anti-drift; thin = baseline) and rolled out
autoregressively over the held-out robot seqs. Their predicted object flow (48 pts) is fed
into the densev4e renderer (IDENTICAL coordinate convention: crop 195,195,256,256; /224
norm; eef3 loader; cube_mask). Only difference is clip length/stride (v4 L=16/S=3 vs
E.DS L=24/S=4) — comparison thick-vs-thin is fair regardless.

Outputs -> outputs/flow_wm_v4/viz_rollout_rendered/
  replay_v4_seq{s}.gif : per frame  GT | render(thick) | render(thin)  side by side
  replay_v4_strips.png : per seq 3 rows (GT/thick/thin) + seq0 GT-flow sanity row
  summary.txt          : per-seq rollout ADE (px) thick/thin + cube-centroid paths
"""
import os, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
import train_flow_cond_decoder_densev4e as D
import train_flow_wm_scarcity_v4 as v4

OUT = "outputs/flow_wm_v4/viz_rollout_rendered"; os.makedirs(OUT, exist_ok=True)
DEC = "outputs/flow_wm/flow_cond_decoder__densev4e_robot__armmask_gmask/decoder_flowcond_gmask.pt"
LAT, IMG = D.LAT_RES, D.IMG_RES
NSEQ = 4

# globals filled by main() so v4_rollout / render_seq can read them
gstats = None


def v4_rollout(m, trk, eff, Hroll):
    K, F, L, P = v4.K, v4.F, v4.L, v4.P; dev = v4.device
    N = trk.shape[0]; pr = np.zeros((N, Hroll, P, 2), np.float32)
    for s in range(N):
        hist = torch.from_numpy(trk[s, :K]).float().permute(1, 0, 2)[None].to(dev)  # (1,P,K,2)
        filled, bk = 0, 0
        while filled < Hroll:
            w0 = bk * F
            ew_np = eff[s, w0:w0 + L]
            if len(ew_np) < L:
                ew_np = np.concatenate([ew_np, np.repeat(ew_np[-1:], L - len(ew_np), 0)], 0)
            ew = torch.from_numpy(ew_np).float()[None].to(dev)                       # (1,L,3,2)
            g = v4.normalize_grasp(v4.grasp_openness(ew), torch.ones(1, dtype=torch.long), gstats).to(dev)
            dm = torch.ones(1, dtype=torch.long, device=dev)
            with torch.no_grad():
                pred, _ = m(hist, ew, g, dm)                                          # (1,P,F,2)
            take = min(F, Hroll - filled)
            pr[s, filled:filled + take] = pred[0, :, :take].cpu().numpy().transpose(1, 0, 2)
            if take >= K:
                hist = pred[:, :, take - K:take, :]
            else:
                hist = torch.cat([hist[:, :, take:, :], pred[:, :, :take, :]], 2)
            filled += take; bk += 1
    return pr


def main():
    global gstats
    # --- (2) train the two v4 ② models on flow_ds_v3 ---
    tr, vis, eef3, dom, vid = v4.load()
    gstats = v4.fit_grasp_stats(v4.grasp_openness(eef3), dom)
    rob = torch.where((dom == 1) & (vid != v4.TEST_ROBOT_VID))[0]
    _, m_thin = v4.train_eval(tr, vis, eef3, dom, gstats, rob, rob[:2],
                              thin=True, seed=0, antidrift=False, return_model=True)
    _, m_thick = v4.train_eval(tr, vis, eef3, dom, gstats, rob, rob[:2],
                               thin=False, seed=0, antidrift=True, return_model=True)
    m_thin.eval(); m_thick.eval()
    print("v4 ② trained (thin + thick)", flush=True)

    # --- (3) load held-out long sequences ---
    z = np.load("outputs/flow_wm/rollout_eval/seqs.npz")
    trk = z["tracks"]; eff = z["eef"]; vis_ = z["vis"]
    frames = z["frames"]; joint = z["joint"]
    H = int(z["H"]); Kr = int(z["K"]); T = trk.shape[1]
    K = v4.K  # OUR rollout history length (=4)
    Hroll = min(H, T - K)
    # clamp so the last F-block's eef window fits (w0 + L <= T); pad otherwise
    while Hroll > 0:
        last_w0 = ((Hroll - 1) // v4.F) * v4.F
        if last_w0 + v4.L <= T:
            break
        Hroll -= 1
    print(f"H={H} T={T} K={K} -> Hroll={Hroll}", flush=True)

    pr_thick = v4_rollout(m_thick, trk, eff, Hroll)
    pr_thin = v4_rollout(m_thin, trk, eff, Hroll)
    # (6) GT-flow sanity: feed GT object tracks (shifted by K) through the renderer
    pr_gt = np.stack([trk[:, K + h] for h in range(Hroll)], axis=1)  # (N,Hroll,48,2)
    print("v4 rollouts done (thick/thin/gt-sanity)", flush=True)

    # --- (5) densev4e renderer setup (copied from reference) ---
    model = D.load_model(); D.widen_control_net(model.decoder.control_net, D.C_EXTRA)
    model.decoder.load_state_dict(torch.load(DEC, map_location="cpu")["decoder"])
    model.eval(); DT = model.dtype; nm = model.normalizer[model.obs_keys[0]]
    gck = torch.load(D.GCK, map_location="cpu")
    g = D.MaskGen(gck["indim"]).to(v4.device); g.load_state_dict(gck["g"]); g.eval()
    xm = torch.from_numpy(np.asarray(gck["xm"], np.float32)).to(v4.device)
    xs = torch.from_numpy(np.asarray(gck["xs"], np.float32)).to(v4.device)
    from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

    @torch.no_grad()
    def enc0(fr):
        x = torch.from_numpy(fr).float().permute(0, 3, 1, 2).to(v4.device) / 255.0
        return model.encoder_forward(nm.normalize(x).to(DT))

    @torch.no_grad()
    def render_seq(s, pr):
        """render every horizon h (t=K+h) for seq s through densev4e using rollout pr."""
        z0 = enc0(frames[s:s + 1, 0])
        outs = []
        for h in range(Hroll):
            t = K + h
            feat = np.concatenate([joint[s, t], eff[s, t].reshape(6)])[None].astype(np.float32)
            fn = (torch.from_numpy(feat).to(v4.device) - xm) / xs
            pm = (torch.sigmoid(g(fn)[:, 0]) > 0.5).float().cpu().numpy()[0]
            gm = cv2.resize(pm, (LAT, LAT))[None]
            pos0 = np.concatenate([trk[s, 0], eff[s, 0]], 0)
            post = np.concatenate([pr[s, h], eff[s, t]], 0)
            v = np.concatenate([vis_[s, t], np.ones(3, np.float32)], 0)
            flow = D.splat_flow(pos0, post, v)[None]
            cond = torch.cat([z0, torch.from_numpy(np.concatenate([flow, gm[None]], 1)).to(DT).to(v4.device)], 1)
            torch.manual_seed(0)
            r = render_img_cm(model, cond, IMG, model.normalizer, num_views=1, batch_size=1)[0].cpu().numpy()
            outs.append(np.clip(r.transpose(1, 2, 0), 0, 1))
        return outs

    # --- (7) outputs ---
    sel = np.arange(NSEQ)
    cols = list(range(0, Hroll, 3))
    nrows = 3 * NSEQ + 1  # +1 sanity row for seq0
    fig, ax = plt.subplots(nrows, len(cols), figsize=(1.5 * len(cols), 1.5 * nrows))

    def ade_px(pr):
        # rollout ADE (px) vs GT tracks over Hroll, visible pts
        gt = np.stack([trk[:, K + h] for h in range(Hroll)], axis=1)  # (N,Hroll,48,2)
        w = np.stack([vis_[:, K + h] for h in range(Hroll)], axis=1)  # (N,Hroll,48)
        err = np.linalg.norm(pr - gt, axis=-1) * 224.0
        return err, w

    err_thick, wmask = ade_px(pr_thick)
    err_thin, _ = ade_px(pr_thin)

    def centroid_path(arr):  # arr (Hroll,48,2) normalized -> path length px
        c = arr.mean(1)
        return float(np.linalg.norm(np.diff(c, axis=0), axis=-1).sum() * 224.0)

    lines = [f"v4 pixel-rendered 3-way rollout replay | NSEQ={NSEQ} Hroll={Hroll} (T={T},H={H},K={K})",
             "renderer=densev4e ③ (flow+g-mask) | ② thick=full(contact-gate+grasp+antidrift) thin=baseline",
             "ADE = rollout ADE px vs GT tracks (visible pts) over Hroll | centroid path = cube-centroid total path px",
             f"{'seq':>3} | {'ADE_thick':>9} | {'ADE_thin':>9} | {'cpath_GT':>9} | {'cpath_thick':>11} | {'cpath_thin':>10}"]

    sanity_rend = None
    for si, s in enumerate(sel):
        rt = render_seq(s, pr_thick)
        rn = render_seq(s, pr_thin)
        if s == 0:
            sanity_rend = render_seq(0, pr_gt)
        # strip rows: GT / thick / thin
        for ci, h in enumerate(cols):
            t = K + h
            r0 = 3 * si
            ax[r0, ci].imshow(frames[s, t] / 255.); ax[r0, ci].axis("off")
            ax[r0 + 1, ci].imshow(rt[h]); ax[r0 + 1, ci].axis("off")
            ax[r0 + 2, ci].imshow(rn[h]); ax[r0 + 2, ci].axis("off")
            if si == 0:
                ax[r0, ci].set_title(f"t={t}", fontsize=7)
            if ci == 0:
                ax[r0, ci].set_title(f"seq{s} GT", fontsize=8, loc="left")
                ax[r0 + 1, ci].set_title("render(thick-②)", fontsize=8, loc="left")
                ax[r0 + 2, ci].set_title("render(thin-②)", fontsize=8, loc="left")
        # GIF: GT | thick | thin
        gif = []
        for h in range(Hroll):
            t = K + h
            gtf = frames[s, t].astype(np.uint8)
            tk = (rt[h] * 255).astype(np.uint8)
            tn = (rn[h] * 255).astype(np.uint8)
            pad = np.full((IMG, 4, 3), 255, np.uint8)
            gif.append(Image.fromarray(np.concatenate([gtf, pad, tk, pad, tn], 1)))
        gif[0].save(f"{OUT}/replay_v4_seq{s}.gif", save_all=True,
                    append_images=gif[1:], duration=150, loop=0)
        # metrics
        m = wmask[s] > 0.5
        at = float((err_thick[s][m]).mean()) if m.any() else float("nan")
        an = float((err_thin[s][m]).mean()) if m.any() else float("nan")
        cg = centroid_path(np.stack([trk[s, K + h] for h in range(Hroll)], 0))
        ct = centroid_path(pr_thick[s]); cn = centroid_path(pr_thin[s])
        lines.append(f"{s:>3} | {at:9.2f} | {an:9.2f} | {cg:9.2f} | {ct:11.2f} | {cn:10.2f}")
        print(lines[-1], flush=True)
        print(f"seq{s}: saved replay_v4_seq{s}.gif ({Hroll} frames)", flush=True)

    # seq0 sanity row (bottom): GT-flow render
    rs = nrows - 1
    for ci, h in enumerate(cols):
        ax[rs, ci].imshow(sanity_rend[h]); ax[rs, ci].axis("off")
        if ci == 0:
            ax[rs, ci].set_title("seq0 render(GT-flow) sanity", fontsize=8, loc="left")

    fig.suptitle("v4 pixel rollout replay: per seq  GT / render(thick-②) / render(thin-②)  (stride-3). "
                 "Bottom: seq0 render(GT-flow) sanity (should match GT).", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/replay_v4_strips.png", dpi=115)
    plt.close(fig)

    msg = "\n".join(lines) + "\n"
    open(f"{OUT}/summary.txt", "w").write(msg)
    print("\n" + msg, flush=True)
    print(f"saved {OUT}/replay_v4_strips.png + replay_v4_seq*.gif + summary.txt\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
