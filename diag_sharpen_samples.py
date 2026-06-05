"""Is part of the blur SELF-INFLICTED by multi-sample averaging? eval averages N_SAMPLE=6
stochastic renders to kill sampling noise -- but averaging random samples also smears
high-freq edges (each sample places the sharp edge slightly differently; mean = blur).
A/B render-sample count {1,2,4,8} on the SAME clips. Metric: Laplacian variance (sharpness,
higher=sharper) in cube/agent/full, plus MSE-to-GT (does fewer samples hurt accuracy?).
Reuses trained densev4d decoder + g. Output: outputs/diag_sharpen_samples/
"""
import os, sys, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from apply_robot_mask_v3 import robot_v3
from gen_flow_render_dataset import cube_mask
import train_flow_cond_decoder_densev4d as D

OUT = "outputs/flow_wm/diag_sharpen_samples"; os.makedirs(OUT, exist_ok=True)
TRAINED = "outputs/flow_cond_decoder_train__densev4d_robot__learned_gmask/decoder_flowcond_gmask.pt"
device = D.device; IMG_RES, LAT_RES = D.IMG_RES, D.LAT_RES; T_HI = D.T_HI
SAMPLES = [1, 2, 4, 8]


def lapvar(img, region):
    g = cv2.cvtColor((img.transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lv = cv2.Laplacian(g.astype(np.float32), cv2.CV_32F)
    return float((lv[region] ** 2).mean()) if region.sum() > 0 else 0.0


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
    ho_idx = perm[:int(N * D.HELDOUT_FRAC)]
    EV = ho_idx[:min(6, len(ho_idx))]; ev_t = np.full(len(EV), T_HI - 1)

    @torch.no_grad()
    def enc(fr):
        x = torch.from_numpy(fr).float().permute(0, 3, 1, 2).to(device) / 255.0
        return model.encoder_forward(nm.normalize(x).to(DT))
    z0 = enc(frames[EV, 0])
    fms = []
    for j, t in zip(EV, ev_t):
        f = D.make_flow(tracks[j], eef[j], vis[j], t)
        fms.append(np.concatenate([f, g_mask[j, t][None]], 0))
    cond = torch.cat([z0, torch.from_numpy(np.stack(fms)).to(DT).to(device)], 1)

    from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

    @torch.no_grad()
    def render_n(n):  # mean of n stochastic renders
        acc = None
        for _ in range(n):
            r = render_img_cm(model, cond, IMG_RES, model.normalizer, num_views=1, batch_size=8)
            acc = r if acc is None else acc + r
        return (acc / n).cpu().numpy()

    gt = (frames[EV, ev_t].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    cubeR = np.stack([cube_mask(np.ascontiguousarray(frames[j, t])) > 0 for j, t in zip(EV, ev_t)])
    agentR = np.stack([robot_v3(np.ascontiguousarray(frames[j, t])) > 0 for j, t in zip(EV, ev_t)])
    full = np.ones_like(cubeR)

    def mse(a, region):
        s, n = 0.0, 0
        for i in range(len(EV)):
            if region[i].sum() < 1: continue
            s += (((a[i] - gt[i]) ** 2).mean(0))[region[i]].mean(); n += 1
        return s / max(n, 1)

    def sharp(imgs, region):
        return float(np.mean([lapvar(imgs[i], region[i]) for i in range(len(EV)) if region[i].sum() > 0]))

    preds = {n: render_n(n) for n in SAMPLES}
    lines = ["diag_sharpen_samples: does multi-sample averaging cause blur?",
             f"N_eval={len(EV)}", "",
             f"{'render':<10}{'MSE_full':>10}{'MSE_cube':>10}{'MSE_agent':>10}"
             f"{'shrp_cube':>11}{'shrp_agent':>12}{'shrp_full':>11}"]
    lines.append(f"{'GT':<10}{'-':>10}{'-':>10}{'-':>10}"
                 f"{sharp(gt, cubeR):>11.1f}{sharp(gt, agentR):>12.1f}{sharp(gt, full):>11.1f}")
    for n in SAMPLES:
        p = preds[n]
        lines.append(f"{'N='+str(n):<10}{mse(p, full):>10.4f}{mse(p, cubeR):>10.4f}{mse(p, agentR):>10.4f}"
                     f"{sharp(p, cubeR):>11.1f}{sharp(p, agentR):>12.1f}{sharp(p, full):>11.1f}")
    lines += ["", "read: if N=1 sharpness >> N=8 with similar MSE -> averaging was self-inflicted blur,",
              "use fewer samples. if MSE jumps at N=1 -> sampling noise real, keep averaging."]
    print("\n".join(lines), flush=True)
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")

    ns = len(EV); R = len(SAMPLES) + 1
    fig, ax = plt.subplots(R, ns, figsize=(2.1 * ns, 2.2 * R))
    rows = [("GT", gt)] + [(f"N={n}", preds[n]) for n in SAMPLES]
    for r, (lab, im) in enumerate(rows):
        for j in range(ns):
            ax[r, j].imshow(np.clip(im[j].transpose(1, 2, 0), 0, 1)); ax[r, j].axis("off")
            if j == 0: ax[r, j].set_title(lab, fontsize=9, loc="left")
    fig.suptitle("render-sample count A/B. fewer samples = noisier but sharper? eyeball cube + gripper.", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/diag.png", dpi=115)
    print(f"saved {OUT}/diag.png + summary.txt\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
