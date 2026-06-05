"""Diagnose WHERE the blur comes from (user: "agent 和 cube 有时候有些糊").
Two candidate causes, separated by a CEILING row:
  (A) latent bottleneck (32x32x4) loses high-freq -> small cube / agent edges blur no matter
      how good the conditioning is. Test: decode the TARGET frame's OWN latent z0=enc(I_t)
      (best-case anchor). If even THIS blurs the cube, it's the decode wall, not conditioning.
  (B) conditioning imprecise (flow splat for the small cube, soft g-mask for agent). Shows up
      as: ceiling sharp, but current pred (z0=enc(I0)+flow+gmask) blurry.
Per-region MSE: full / cube(red HSV) / agent(robot_v3), for pred vs ceiling vs (I0 naive copy).
Reuses the EXACT trained densev4d decoder + g. Output: outputs/diag_blur_source/
"""
import os, sys, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from apply_robot_mask_v3 import robot_v3
from gen_flow_render_dataset import cube_mask
import train_flow_cond_decoder_densev4d as D

OUT = "outputs/flow_wm/diag_blur_source"; os.makedirs(OUT, exist_ok=True)
TRAINED = "outputs/flow_cond_decoder_train__densev4d_robot__learned_gmask/decoder_flowcond_gmask.pt"
device = D.device; IMG_RES, LAT_RES = D.IMG_RES, D.LAT_RES
T_HI = D.T_HI; N_SAMPLE = 6


def main():
    model = D.load_model(); D.widen_control_net(model.decoder.control_net, D.C_EXTRA)
    sd = torch.load(TRAINED, map_location="cpu")["decoder"]
    model.decoder.load_state_dict(sd); model.eval(); DT = model.dtype
    nm = model.normalizer[model.obs_keys[0]]

    z = np.load(D.DS)
    frames = z["frames"]; tracks = z["tracks"].astype(np.float32); eef = z["eef"].astype(np.float32)
    vis = z["vis"].astype(np.float32); joint = z["joint"].astype(np.float32)
    N, L = len(frames), frames.shape[1]

    gck = torch.load(D.GCK, map_location="cpu")
    g = D.MaskGen(gck["indim"]).to(device); g.load_state_dict(gck["g"]); g.eval()
    xm = torch.from_numpy(np.asarray(gck["xm"], np.float32)).to(device)
    xs = torch.from_numpy(np.asarray(gck["xs"], np.float32)).to(device)
    feat = np.concatenate([joint, eef.reshape(N, L, 6)], -1).astype(np.float32)
    g_mask = np.zeros((N, L, LAT_RES, LAT_RES), np.float32)
    with torch.no_grad():
        for i in range(N):
            fn = (torch.from_numpy(feat[i]).to(device) - xm) / xs
            pm = (torch.sigmoid(g(fn)[:, 0]) > 0.5).float().cpu().numpy()
            for t in range(L): g_mask[i, t] = cv2.resize(pm[t], (LAT_RES, LAT_RES))

    rng = np.random.default_rng(0); perm = rng.permutation(N)
    n_ho = int(N * D.HELDOUT_FRAC); ho_idx = perm[:n_ho]
    EV = ho_idx[:min(8, len(ho_idx))]; ev_t = np.full(len(EV), T_HI - 1)

    @torch.no_grad()
    def enc(fr):
        x = torch.from_numpy(fr).float().permute(0, 3, 1, 2).to(device) / 255.0
        return model.encoder_forward(nm.normalize(x).to(DT))

    z0_I0 = enc(frames[EV, 0]); z0_It = enc(frames[EV, ev_t])

    def cond(z0, use_flow):
        fms = []
        for k, (j, t) in enumerate(zip(EV, ev_t)):
            f = D.make_flow(tracks[j], eef[j], vis[j], t) if use_flow else np.zeros((D.C_FLOW, LAT_RES, LAT_RES), np.float32)
            fms.append(np.concatenate([f, g_mask[j, t][None]], 0))
        return torch.cat([z0, torch.from_numpy(np.stack(fms)).to(DT).to(device)], 1)

    from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

    @torch.no_grad()
    def rmean(c):
        acc = None
        for _ in range(N_SAMPLE):
            r = render_img_cm(model, c, IMG_RES, model.normalizer, num_views=1, batch_size=8)
            acc = r if acc is None else acc + r
        return (acc / N_SAMPLE).cpu().numpy()

    with torch.no_grad():
        pred = rmean(cond(z0_I0, True))          # current deployable pipeline
        ceil = rmean(cond(z0_It, False))         # CEILING: decode target's own latent
    gt = (frames[EV, ev_t].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    I0 = (frames[EV, 0].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)

    cubeR = np.stack([cube_mask(np.ascontiguousarray(frames[j, t])) > 0 for j, t in zip(EV, ev_t)])
    agentR = np.stack([robot_v3(np.ascontiguousarray(frames[j, t])) > 0 for j, t in zip(EV, ev_t)])

    def mse(a, region):
        s, n = 0.0, 0
        for i in range(len(EV)):
            m = region[i]
            if m.sum() < 1: continue
            s += (((a[i] - gt[i]) ** 2).mean(0))[m].mean(); n += 1
        return s / max(n, 1)

    lines = ["diag_blur_source: where does the blur come from?",
             f"N_eval={len(EV)} N_SAMPLE={N_SAMPLE}", "",
             f"{'variant':<22}{'FULL':>9}{'CUBE':>9}{'AGENT':>9}"]
    full = np.ones_like(cubeR)
    for tag, im in [("I0 naive-copy", I0), ("pred(enc I0+flow+gmask)", pred), ("CEILING(enc I_t)", ceil)]:
        lines.append(f"{tag:<22}{mse(im, full):>9.4f}{mse(im, cubeR):>9.4f}{mse(im, agentR):>9.4f}")
    lines += ["",
              "read: if CEILING cube ~ pred cube (both >> small) -> latent-bottleneck wall (need",
              "higher latent res / perceptual loss). if CEILING cube << pred cube -> conditioning",
              "(flow splat for cube) is the fixable culprit."]
    print("\n".join(lines), flush=True)
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")

    ns = len(EV)
    fig, ax = plt.subplots(5, ns, figsize=(2.1 * ns, 11))
    rows = [("GT I_t", gt), ("CEILING decode(enc I_t)", ceil), ("pred(enc I0+flow+gmask)", pred)]
    for j in range(ns):
        for r, (lab, im) in enumerate(rows):
            ax[r, j].imshow(np.clip(im[j].transpose(1, 2, 0), 0, 1)); ax[r, j].axis("off")
            if j == 0: ax[r, j].set_title(lab, fontsize=9, loc="left")
        ov = (gt[j].transpose(1, 2, 0)).copy(); ov[cubeR[j]] = [1, 1, 0]
        ax[3, j].imshow(ov); ax[3, j].axis("off")
        if j == 0: ax[3, j].set_title("cube region (yellow)", fontsize=9, loc="left")
        ov2 = (gt[j].transpose(1, 2, 0)).copy(); ov2[agentR[j]] = [0, 1, 0]
        ax[4, j].imshow(ov2); ax[4, j].axis("off")
        if j == 0: ax[4, j].set_title("agent region (green)", fontsize=9, loc="left")
    fig.suptitle("BLUR SOURCE: row2 CEILING = decode target's own latent (best case). "
                 "if cube still blurry there -> 32x32x4 latent wall, not conditioning.", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/diag.png", dpi=115)
    print(f"saved {OUT}/diag.png + summary.txt\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
