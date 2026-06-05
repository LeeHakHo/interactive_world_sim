"""Final demo for the agent line: densev4e (g supervised by robot_arm_mask = fuller mask, no
broken rods) rendered at N=1 (no self-inflicted averaging blur). Shows both fixes together:
- agent middle-break gone (complete g-mask)
- agent sharper (single sample vs averaged)
Side-by-side rows: I0 / GT / pred N=1 / pred N=4(averaged, blurry) / g-mask. Fixed seed.
Reuses densev4e decoder + g v3. Output: outputs/flow_wm/densev4e_demo_n1/
"""
import os, sys, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ".")
import train_flow_cond_decoder_densev4e as D

OUT = "outputs/flow_wm/densev4e_demo_n1"; os.makedirs(OUT, exist_ok=True)
TRAINED = "outputs/flow_wm/flow_cond_decoder__densev4e_robot__armmask_gmask/decoder_flowcond_gmask.pt"
device = D.device; IMG_RES, LAT_RES = D.IMG_RES, D.LAT_RES; T_HI = D.T_HI


def main():
    model = D.load_model(); D.widen_control_net(model.decoder.control_net, D.C_EXTRA)
    model.decoder.load_state_dict(torch.load(TRAINED, map_location="cpu")["decoder"])
    model.eval(); DT = model.dtype; nm = model.normalizer[model.obs_keys[0]]

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
    ho = perm[:int(N * D.HELDOUT_FRAC)]; EV = ho[:8]; ev_t = np.full(len(EV), T_HI - 1)

    @torch.no_grad()
    def enc(fr):
        x = torch.from_numpy(fr).float().permute(0, 3, 1, 2).to(device) / 255.0
        return model.encoder_forward(nm.normalize(x).to(DT))
    z0 = enc(frames[EV, 0])
    fms = [np.concatenate([D.make_flow(tracks[j], eef[j], vis[j], t), g_mask[j, t][None]], 0)
           for j, t in zip(EV, ev_t)]
    cond = torch.cat([z0, torch.from_numpy(np.stack(fms)).to(DT).to(device)], 1)

    from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

    @torch.no_grad()
    def render_n(n, fixed_seed=None):
        acc = None
        for _ in range(n):
            if fixed_seed is not None: torch.manual_seed(fixed_seed)
            r = render_img_cm(model, cond, IMG_RES, model.normalizer, num_views=1, batch_size=8)
            acc = r if acc is None else acc + r
        return (acc / n).cpu().numpy()

    p1 = render_n(1, fixed_seed=0)   # single sample, reproducible
    p4 = render_n(4)                 # 4 varying samples averaged (the blur source)
    gt = (frames[EV, ev_t].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    I0 = (frames[EV, 0].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)

    ns = len(EV)
    fig, ax = plt.subplots(5, ns, figsize=(2.1 * ns, 11))
    rows = [("I0", I0), ("GT I_t", gt), ("pred N=1 (sharp)", p1), ("pred N=4 (averaged, blurry)", p4)]
    for j in range(ns):
        for r, (lab, im) in enumerate(rows):
            ax[r, j].imshow(np.clip(im[j].transpose(1, 2, 0), 0, 1)); ax[r, j].axis("off")
            if j == 0: ax[r, j].set_title(lab, fontsize=9, loc="left")
        ax[4, j].imshow(cv2.resize(g_mask[EV[j], ev_t[j]], (IMG_RES, IMG_RES)), cmap="gray"); ax[4, j].axis("off")
        if j == 0: ax[4, j].set_title("g-mask (robot_arm_mask sup, complete)", fontsize=9, loc="left")
    fig.suptitle("densev4e FINAL: g supervised by robot_arm_mask (complete -> no agent break) @ N=1 (sharp). "
                 "row3 vs row4 = sharpness cost of averaging.", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/demo.png", dpi=120)
    open(f"{OUT}/summary.txt", "w").write("densev4e N=1 demo: complete g-mask (robot_arm_mask sup) + single-sample sharp.\n"
                                           "row3 N=1 sharp, row4 N=4 averaged-blurry, row5 g-mask complete (no break).\n")
    print(f"saved {OUT}/demo.png + summary.txt\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
