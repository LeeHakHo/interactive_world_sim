"""A: how sensitive is ③'s agent rendering to mask QUALITY? Decides whether it's worth
improving g's supervision. Fix flow + fix the render noise seed (so variants differ ONLY by
the conditioning mask), sweep mask IoU from 0 (zero) -> g(0.65) -> degraded-oracle -> oracle(1.0),
plot agent-region MSE & sharpness vs the mask's actual agent-IoU.
  steep slope -> mask quality is the bottleneck, improving supervision pays off.
  flat slope  -> mask is good enough, the wall is decoder capacity / sampling.
CAVEAT: decoder was TRAINED with g-mask, so oracle/shifted masks are mildly OOD; trend (not
absolute) is the signal. N=1 single sample (no averaging blur). Output: outputs/diag_mask_sensitivity/
"""
import os, sys, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from apply_robot_mask_v3 import robot_v3
import train_flow_cond_decoder_densev4d as D

OUT = "outputs/flow_wm/diag_mask_sensitivity"; os.makedirs(OUT, exist_ok=True)
TRAINED = "outputs/flow_cond_decoder_train__densev4d_robot__learned_gmask/decoder_flowcond_gmask.pt"
device = D.device; IMG_RES, LAT_RES = D.IMG_RES, D.LAT_RES; T_HI = D.T_HI
SEED = 0


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

    rng = np.random.default_rng(0); perm = rng.permutation(N)
    ho_idx = perm[:int(N * D.HELDOUT_FRAC)]
    EV = ho_idx[:min(8, len(ho_idx))]; ev_t = np.full(len(EV), T_HI - 1)

    @torch.no_grad()
    def enc(fr):
        x = torch.from_numpy(fr).float().permute(0, 3, 1, 2).to(device) / 255.0
        return model.encoder_forward(nm.normalize(x).to(DT))
    z0 = enc(frames[EV, 0])

    # real target-frame robot_v3 mask @128 = ground-truth agent region & oracle source
    v3_128 = np.stack([robot_v3(np.ascontiguousarray(frames[j, t])) > 0 for j, t in zip(EV, ev_t)])

    # g-mask @128 (for the 'g' variant and to report its IoU)
    g128 = np.zeros((len(EV), IMG_RES, IMG_RES), bool)
    with torch.no_grad():
        for k, j in enumerate(EV):
            fn = (torch.from_numpy(feat[j]).to(device) - xm) / xs
            pm = (torch.sigmoid(g(fn)[:, 0]) > 0.5).float().cpu().numpy()
            g128[k] = cv2.resize(pm[ev_t[k]], (IMG_RES, IMG_RES)) > 0.5

    def to_lat(m128):  # bool@128 -> float latent32
        return cv2.resize(m128.astype(np.float32), (LAT_RES, LAT_RES))

    def shift(m, d):  # diagonal shift by d px
        return np.roll(np.roll(m, d, 0), d, 1)

    # build mask variants per clip; record per-clip latent mask + its @128 version (for IoU)
    def make_variant(name, k):
        if name == "zero": return np.zeros((LAT_RES, LAT_RES), np.float32), np.zeros((IMG_RES, IMG_RES), bool)
        if name == "g": return to_lat(g128[k]), g128[k]
        base = v3_128[k]
        if name == "oracle": m = base
        else: m = shift(base, int(name.split("sh")[1]))
        return to_lat(m), m

    VARIANTS = ["zero", "oracle_sh8", "oracle_sh6", "oracle_sh4", "g", "oracle_sh3", "oracle_sh2", "oracle"]
    flows = [D.make_flow(tracks[j], eef[j], vis[j], t) for j, t in zip(EV, ev_t)]
    gt = (frames[EV, ev_t].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)

    from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

    @torch.no_grad()
    def render_seeded(masks_lat):
        cond = torch.cat([z0, torch.from_numpy(np.stack(
            [np.concatenate([flows[k], masks_lat[k][None]], 0) for k in range(len(EV))])).to(DT).to(device)], 1)
        torch.manual_seed(SEED)
        return render_img_cm(model, cond, IMG_RES, model.normalizer, num_views=1, batch_size=8).cpu().numpy()

    def agent_iou(m128_list):
        return float(np.mean([(m128_list[k] & v3_128[k]).sum() / max((m128_list[k] | v3_128[k]).sum(), 1)
                              for k in range(len(EV)) if v3_128[k].sum() > 0]))

    def agent_mse(pred):
        s, n = 0.0, 0
        for k in range(len(EV)):
            if v3_128[k].sum() < 1: continue
            s += (((pred[k] - gt[k]) ** 2).mean(0))[v3_128[k]].mean(); n += 1
        return s / max(n, 1)

    def agent_sharp(pred):
        return float(np.mean([lapvar(pred[k], v3_128[k]) for k in range(len(EV)) if v3_128[k].sum() > 0]))

    results = {}; preds = {}
    for name in VARIANTS:
        mlat = [make_variant(name, k)[0] for k in range(len(EV))]
        m128 = [make_variant(name, k)[1] for k in range(len(EV))]
        pred = render_seeded(mlat); preds[name] = pred
        results[name] = dict(iou=agent_iou(m128), mse=agent_mse(pred), sharp=agent_sharp(pred))
        print(f"{name:12s} mask-IoU {results[name]['iou']:.3f}  agentMSE {results[name]['mse']:.4f}  "
              f"sharp {results[name]['sharp']:.1f}", flush=True)

    gt_sharp = agent_sharp(gt)
    order = sorted(VARIANTS, key=lambda n: results[n]["iou"])
    ious = [results[n]["iou"] for n in order]; mses = [results[n]["mse"] for n in order]
    shrp = [results[n]["sharp"] for n in order]
    L = ["=== A: agent-render sensitivity to mask quality ===",
         "(N=1 single sample, fixed seed; decoder trained on g-mask -> oracle is mildly OOD)", "",
         f"{'variant':<12}{'mask-IoU':>10}{'agentMSE':>10}{'sharp':>9}"]
    for n in order: L.append(f"{n:<12}{results[n]['iou']:>10.3f}{results[n]['mse']:>10.4f}{results[n]['sharp']:>9.1f}")
    g_iou = results["g"]["iou"]
    # slope around g: how much does agentMSE improve per +0.1 IoU near g?
    L += ["", f"GT agent sharpness (ceiling) = {gt_sharp:.1f}",
          f"g sits at IoU={g_iou:.2f}, agentMSE={results['g']['mse']:.4f}, sharp={results['g']['sharp']:.1f}",
          f"oracle(IoU~1) agentMSE={results['oracle']['mse']:.4f}, sharp={results['oracle']['sharp']:.1f}",
          "read: if oracle MSE << g MSE -> steep, improving mask pays. if ~equal -> mask already enough."]
    print("\n".join(L), flush=True); open(f"{OUT}/summary.txt", "w").write("\n".join(L) + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(ious, mses, "o-"); ax[0].set_xlabel("mask agent-IoU"); ax[0].set_ylabel("agent-region MSE (lower better)")
    for n in order: ax[0].annotate(n, (results[n]["iou"], results[n]["mse"]), fontsize=7)
    ax[0].axvline(g_iou, c="g", ls="--", label=f"g (IoU {g_iou:.2f})"); ax[0].legend(); ax[0].set_title("MSE vs mask quality")
    ax[1].plot(ious, shrp, "o-"); ax[1].axhline(gt_sharp, c="k", ls=":", label=f"GT sharp {gt_sharp:.0f}")
    ax[1].set_xlabel("mask agent-IoU"); ax[1].set_ylabel("agent sharpness (higher better)")
    ax[1].axvline(g_iou, c="g", ls="--"); ax[1].legend(); ax[1].set_title("sharpness vs mask quality")
    fig.suptitle("A: does better mask -> better agent render? slope tells if improving g's supervision is worth it.", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/curve.png", dpi=120)

    # overlay strip: GT + a few variants
    show = ["zero", "g", "oracle_sh4", "oracle"]; ns = min(6, len(EV))
    fig2, ax2 = plt.subplots(len(show) + 1, ns, figsize=(2.1 * ns, 2.2 * (len(show) + 1)))
    rows = [("GT", gt)] + [(f"{n} (IoU{results[n]['iou']:.2f})", preds[n]) for n in show]
    for r, (lab, im) in enumerate(rows):
        for j in range(ns):
            ax2[r, j].imshow(np.clip(im[j].transpose(1, 2, 0), 0, 1)); ax2[r, j].axis("off")
            if j == 0: ax2[r, j].set_title(lab, fontsize=8, loc="left")
    fig2.suptitle("agent render under different mask quality (fixed seed, N=1). eyeball the gripper region.", fontsize=10)
    fig2.tight_layout(); fig2.savefig(f"{OUT}/overlay.png", dpi=115)
    print(f"saved {OUT}/curve.png + overlay.png + summary.txt\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
