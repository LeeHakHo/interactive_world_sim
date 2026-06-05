"""Full continuous rollout = ACTION REPLAY. Given I0 + the whole known action (EEF) sequence,
autoregressively roll ② out for every frame and render each with densev4e ③ (+ g mask). Uses
the scheduled-sampling ② (best, non-exploding). Outputs per-seq continuous strips (GT row vs
render row across all horizons) + GIFs you can scrub. Agent mask = g(joint,eef) (known action,
no drift); only the cube rides ②'s flow (slight drift = the position gap you saw).
Output: outputs/flow_wm/scheduled_sampling/replay/
"""
import os, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
import train_flow_cond_decoder_densev4e as D
import e2e_flow_wm_render as E
from eval_scheduled_sampling import train_wm_ss, rollout_free

OUT = "outputs/flow_wm/scheduled_sampling/replay"; os.makedirs(OUT, exist_ok=True)
DEC = "outputs/flow_wm/flow_cond_decoder__densev4e_robot__armmask_gmask/decoder_flowcond_gmask.pt"
K, F, device, LAT, IMG = E.K, E.F, D.device, D.LAT_RES, D.IMG_RES
NSEQ = 4


def main():
    zr = np.load(f"{E.DS}/clips_robot.npz"); zh = np.load(f"{E.DS}/clips_human.npz")
    r_tr, r_ef, r_vs = zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32), zr["vis"].astype(np.float32)
    h_tr, h_ef, h_vs = zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32), zh["vis"].astype(np.float32)
    Nr = len(r_tr); pool = np.random.default_rng(0).permutation(Nr)[E.HELDOUT_ROB:]
    sub = pool[np.random.default_rng(100).choice(len(pool), min(E.N_ROB, len(pool)), replace=False)]
    mtr = np.concatenate([r_tr, h_tr]); mef = np.concatenate([r_ef, h_ef]); mvs = np.concatenate([r_vs, h_vs])
    rhidx = torch.cat([torch.from_numpy(sub), torch.arange(Nr, Nr + len(h_tr))])
    wm = train_wm_ss(mtr, mvs, mef, rhidx, seed=0)
    print("SS-② trained", flush=True)

    z = np.load("outputs/flow_wm/rollout_eval/seqs.npz"); H = int(z["H"])
    trk, eff, vis = z["tracks"], z["eef"], z["vis"]; frames = z["frames"]; joint = z["joint"]
    pr = rollout_free(wm, torch.from_numpy(trk).float().to(device),
                      torch.from_numpy(eff).float().to(device), H).cpu().numpy()

    model = D.load_model(); D.widen_control_net(model.decoder.control_net, D.C_EXTRA)
    model.decoder.load_state_dict(torch.load(DEC, map_location="cpu")["decoder"])
    model.eval(); DT = model.dtype; nm = model.normalizer[model.obs_keys[0]]
    gck = torch.load(D.GCK, map_location="cpu")
    g = D.MaskGen(gck["indim"]).to(device); g.load_state_dict(gck["g"]); g.eval()
    xm = torch.from_numpy(np.asarray(gck["xm"], np.float32)).to(device)
    xs = torch.from_numpy(np.asarray(gck["xs"], np.float32)).to(device)
    from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

    @torch.no_grad()
    def enc0(fr):
        x = torch.from_numpy(fr).float().permute(0, 3, 1, 2).to(device) / 255.0
        return model.encoder_forward(nm.normalize(x).to(DT))

    @torch.no_grad()
    def render_seq(s):
        """render every horizon frame t=K..K+H-1 for seq s (N=1, sharp)."""
        z0 = enc0(frames[s:s + 1, 0])
        outs = []
        for h in range(H):
            t = K + h
            feat = np.concatenate([joint[s, t], eff[s, t].reshape(6)])[None].astype(np.float32)
            fn = (torch.from_numpy(feat).to(device) - xm) / xs
            pm = (torch.sigmoid(g(fn)[:, 0]) > 0.5).float().cpu().numpy()[0]
            gm = cv2.resize(pm, (LAT, LAT))[None]
            pos0 = np.concatenate([trk[s, 0], eff[s, 0]], 0); post = np.concatenate([pr[s, h], eff[s, t]], 0)
            v = np.concatenate([vis[s, t], np.ones(3, np.float32)], 0)
            flow = D.splat_flow(pos0, post, v)[None]
            cond = torch.cat([z0, torch.from_numpy(np.concatenate([flow, gm[None]], 1)).to(DT).to(device)], 1)
            torch.manual_seed(0)
            r = render_img_cm(model, cond, IMG, model.normalizer, num_views=1, batch_size=1)[0].cpu().numpy()
            outs.append(np.clip(r.transpose(1, 2, 0), 0, 1))
        return outs

    sel = np.arange(NSEQ)
    # static strip: per seq GT row + render row, stride-3 horizons
    cols = list(range(0, H, 3))
    fig, ax = plt.subplots(2 * NSEQ, len(cols), figsize=(1.5 * len(cols), 3.0 * NSEQ))
    for si, s in enumerate(sel):
        rend = render_seq(s)
        for ci, h in enumerate(cols):
            t = K + h
            ax[2 * si, ci].imshow(frames[s, t] / 255.); ax[2 * si, ci].axis("off")
            ax[2 * si + 1, ci].imshow(rend[h]); ax[2 * si + 1, ci].axis("off")
            if ci == 0:
                ax[2 * si, ci].set_title(f"seq{s} GT", fontsize=8, loc="left")
                ax[2 * si + 1, ci].set_title("SS-② render", fontsize=8, loc="left")
            ax[2 * si, ci].set_title(f"t={t}", fontsize=7) if si == 0 else None
        # GIF: GT | render side by side, all frames
        gif = []
        for h in range(H):
            t = K + h
            gtf = (frames[s, t]).astype(np.uint8)
            rd = (rend[h] * 255).astype(np.uint8)
            pad = np.full((IMG, 4, 3), 255, np.uint8)
            gif.append(Image.fromarray(np.concatenate([gtf, pad, rd], 1)))
        gif[0].save(f"{OUT}/replay_seq{s}.gif", save_all=True, append_images=gif[1:], duration=150, loop=0)
        print(f"seq{s}: saved gif ({H} frames)", flush=True)
    fig.suptitle("ACTION REPLAY: full continuous rollout (SS-②). per seq: GT (top) vs render (bottom). "
                 "agent=g(action) stays; cube rides ②-flow (slight drift).", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/replay_strips.png", dpi=115)
    open(f"{OUT}/README.txt", "w").write(
        f"Action-replay full rollout, SS-② + densev4e + g-mask, {NSEQ} held-out robot seqs, H={H} frames.\n"
        "replay_strips.png = GT-vs-render strips (stride 3). replay_seq*.gif = GT|render side-by-side, all frames.\n"
        "cube position drifts slightly (② flow); agent (g from known action) stays correct; frame never collapses (I0-anchor).\n")
    print(f"saved {OUT}/replay_strips.png + replay_seq*.gif + README\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
