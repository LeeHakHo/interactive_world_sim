"""Diagnose the ghosting the user sees in the rollout GIFs. Hypothesis: ③ re-injects z0=enc(I0)
every frame, so the object/agent appearance AT THEIR INITIAL (I0) POSITION leaks in as a ghost
while flow draws them at the new position -> double image. Test per horizon t:
  (a) GT frame
  (b) zero-flow render = decode(z0 only) -> shows where I0 puts object/agent (the ghost source)
  (c) full render (SS-② flow + g-mask) -> if it shows (b)'s initial-position object/agent AS WELL
      AS the moved one, ghosting = I0 residual confirmed.
Mark the I0 cube position (cyan) on (c). Output: outputs/flow_wm/scheduled_sampling/ghost_diag.png
"""
import os, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import train_flow_cond_decoder_densev4e as D
import e2e_flow_wm_render as E
from eval_scheduled_sampling import train_wm_ss, rollout_free

OUT = "outputs/flow_wm/scheduled_sampling"; os.makedirs(OUT, exist_ok=True)
DEC = "outputs/flow_wm/flow_cond_decoder__densev4e_robot__armmask_gmask/decoder_flowcond_gmask.pt"
K, F, device, LAT, IMG = E.K, E.F, D.device, D.LAT_RES, D.IMG_RES
TS = [12, 20, 28, 36]; SHOW = [0, 1]


def main():
    zr = np.load(f"{E.DS}/clips_robot.npz"); zh = np.load(f"{E.DS}/clips_human.npz")
    r_tr, r_ef, r_vs = zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32), zr["vis"].astype(np.float32)
    h_tr, h_ef, h_vs = zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32), zh["vis"].astype(np.float32)
    Nr = len(r_tr); pool = np.random.default_rng(0).permutation(Nr)[E.HELDOUT_ROB:]
    sub = pool[np.random.default_rng(100).choice(len(pool), min(E.N_ROB, len(pool)), replace=False)]
    mtr = np.concatenate([r_tr, h_tr]); mef = np.concatenate([r_ef, h_ef]); mvs = np.concatenate([r_vs, h_vs])
    wm = train_wm_ss(mtr, mvs, mef, torch.cat([torch.from_numpy(sub), torch.arange(Nr, Nr + len(h_tr))]), seed=0)

    z = np.load("outputs/flow_wm/rollout_eval/seqs.npz"); H = int(z["H"])
    trk, eff, vis, frames, joint = z["tracks"], z["eef"], z["vis"], z["frames"], z["joint"]
    pr = rollout_free(wm, torch.from_numpy(trk).float().to(device), torch.from_numpy(eff).float().to(device), H).cpu().numpy()

    model = D.load_model(); D.widen_control_net(model.decoder.control_net, D.C_EXTRA)
    model.decoder.load_state_dict(torch.load(DEC, map_location="cpu")["decoder"])
    model.eval(); DT = model.dtype; nm = model.normalizer[model.obs_keys[0]]
    gck = torch.load(D.GCK, map_location="cpu")
    g = D.MaskGen(gck["indim"]).to(device); g.load_state_dict(gck["g"]); g.eval()
    xm = torch.from_numpy(np.asarray(gck["xm"], np.float32)).to(device); xs = torch.from_numpy(np.asarray(gck["xs"], np.float32)).to(device)
    from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

    @torch.no_grad()
    def enc0(fr):
        x = torch.from_numpy(fr).float().permute(0, 3, 1, 2).to(device) / 255.0
        return model.encoder_forward(nm.normalize(x).to(DT))

    @torch.no_grad()
    def rend(z0, flow, gm):
        cond = torch.cat([z0, torch.from_numpy(np.concatenate([flow, gm], 0)[None]).to(DT).to(device)], 1)
        torch.manual_seed(0)
        return render_img_cm(model, cond, IMG, model.normalizer, num_views=1, batch_size=1)[0].cpu().numpy().transpose(1, 2, 0)

    def gm_at(s, t):
        feat = np.concatenate([joint[s, t], eff[s, t].reshape(6)])[None].astype(np.float32)
        fn = (torch.from_numpy(feat).to(device) - xm) / xs
        pm = (torch.sigmoid(g(fn)[:, 0]) > 0.5).float().cpu().numpy()[0]
        return cv2.resize(pm, (LAT, LAT))[None]

    zfl = np.zeros((4, LAT, LAT), np.float32); zmk = np.zeros((1, LAT, LAT), np.float32)
    rows = ["GT frame", "zero-flow = decode(z0)\n(I0 ghost source)", "full render (SS)\ncyan=I0 cube pos"]
    fig, ax = plt.subplots(len(rows) * len(SHOW), len(TS), figsize=(2.3 * len(TS), 2.4 * len(rows) * len(SHOW)))
    for si, s in enumerate(SHOW):
        z0 = enc0(frames[s:s + 1, 0])
        cube0 = trk[s, 0].mean(0)  # I0 cube centroid (normalized)
        for ti, t in enumerate(TS):
            pos0 = np.concatenate([trk[s, 0], eff[s, 0]], 0); post = np.concatenate([pr[s, t - K], eff[s, t]], 0)
            v = np.concatenate([vis[s, t], np.ones(3, np.float32)], 0)
            flow = D.splat_flow(pos0, post, v)
            full = rend(z0, flow, gm_at(s, t))
            zero = rend(z0, zfl, zmk)
            full_m = (full * 255).astype(np.uint8).copy()
            cx, cy = int(cube0[0] * IMG), int(cube0[1] * IMG); cv2.circle(full_m, (cx, cy), 8, (0, 255, 255), 2)
            imgs = [frames[s, t] / 255., zero, full_m / 255.]
            for r, im in enumerate(imgs):
                a = ax[si * len(rows) + r, ti]; a.imshow(np.clip(im, 0, 1)); a.axis("off")
                if ti == 0: a.set_title(rows[r], fontsize=8, loc="left")
            ax[si * len(rows), ti].set_title(f"t={t}", fontsize=9)
    fig.suptitle("Ghosting diagnosis: does the full render (row3) still show the cube/agent at the I0 position "
                 "(cyan ring = I0 cube) on top of the moved one? -> I0 residual.", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/ghost_diag.png", dpi=120)
    print(f"saved {OUT}/ghost_diag.png\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
