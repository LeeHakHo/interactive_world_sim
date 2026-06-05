"""PROPER training of the flow-conditioned decoder (not overfit).

Motivation: the overfit derisk (job45964) showed the decoder走捷径 -- with 30 samples
it memorizes z0->target and ignores flow (spread 0.25 after de-noising). User's call:
train the flow-cond structure properly on FULL data so it CANNOT memorize and must
learn to USE flow, and "尽量不要让 decoder 就复原原图".

Setup:
- reuse job45895 AE (encoder frozen, PSNR42) + widen the ControlNet bypass for flow.
- data = outputs/flow_render_dataset/clips_{human,robot}.npz (full, big time step).
- each step: I0=frame0, I_t=frame_t (t large), cond=cat([enc(I0), splat_flow_{0->t}]),
  CTM-reconstruct I_t. z0=enc(frame0) is PRECOMPUTED (I0 fixed per clip).
- anti-shortcut: large time step (I0 != I_t), full data + many epochs, 10% HELD-OUT clips.
- eval on held-out (multi-sample avg): GT-flow vs SHUFFLE-flow vs ZERO-flow -> proves
  (generalizing) whether flow is actually used + quantifies its net contribution.

Output: outputs/flow_cond_decoder_train__job45895AE__fulldata_bigstep/
"""
import os, json, numpy as np, cv2, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from omegaconf import OmegaConf

_otl = torch.load
torch.load = lambda *a, **k: _otl(*a, **{**k, "weights_only": False})
OmegaConf.register_new_resolver("eval", lambda e: eval(e, {"np": np}), replace=True)
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x), replace=True)

CKPT = "outputs/2026-05-29/job45895_phase0_stage1_mask_subtract_2gpu/checkpoints/epoch=1-step=25000.ckpt"
CFG = "outputs/2026-05-29/job45895_phase0_stage1_mask_subtract_2gpu/.hydra/config.yaml"
DS = "outputs/flow_render_dataset"
OUTDIR = "outputs/flow_cond_decoder_train__densev3_80ep__ckpt"
IMG_RES, LAT_RES, C_FLOW = 128, 32, 4   # +1 channel = OCCUPANCY at t-position
T_LO, T_HI = 12, 24          # target frame range (big time step; clip has L=24 frames)
FLOW_SCALE = 10.0
EPOCHS, BS, LR = 80, 8, 1e-4   # densev3: more epochs (exp6 wnorm not saturated)
HELDOUT_FRAC = 0.12
SPLAT_RADIUS, SPLAT_SIGMA = 4, 5.0      # pixel-level disk + big Gaussian (DragNUWA-style)
device = "cuda" if torch.cuda.is_available() else "cpu"


def splat_flow(pos0, post, vis):
    """DENSE v2 (fixes exp1-5's too-weak splat): build at PIXEL res 128 with a disk per
    point + BIG Gaussian, then downsample to latent 32 (DragNUWA/Image Conductor recipe,
    not the old 32x32 sparse single-pixel splat). 4 channels:
      0,1 = (dx,dy) displacement 0->t at START position
      2   = visibility at start
      3   = OCCUPANCY at t-position (where the point ENDS UP) -> renderer knows the
            target location, not just "move from here by this much"."""
    H = IMG_RES
    fm = np.zeros((4, H, H), np.float32); cnt = np.zeros((H, H), np.float32)
    yy, xx = np.ogrid[:H, :H]
    for (x0, y0), (xt, yt), v in zip(pos0, post, vis):
        gx, gy = int(np.clip(x0 * H, 0, H - 1)), int(np.clip(y0 * H, 0, H - 1))
        disk = (xx - gx) ** 2 + (yy - gy) ** 2 <= SPLAT_RADIUS ** 2
        fm[0][disk] += (xt - x0) * FLOW_SCALE; fm[1][disk] += (yt - y0) * FLOW_SCALE
        fm[2][disk] += float(v); cnt += disk
        tx, ty = int(np.clip(xt * H, 0, H - 1)), int(np.clip(yt * H, 0, H - 1))
        tdisk = (xx - tx) ** 2 + (yy - ty) ** 2 <= SPLAT_RADIUS ** 2
        fm[3][tdisk] += float(v)
    nz = cnt > 0
    for ch in range(3): fm[ch][nz] /= cnt[nz]
    for ch in range(4): fm[ch] = cv2.GaussianBlur(fm[ch], (15, 15), SPLAT_SIGMA)
    return np.stack([cv2.resize(fm[ch], (LAT_RES, LAT_RES)) for ch in range(4)])


def make_flow(tr, ef, vs, t):
    pts0 = np.concatenate([tr[0], ef[0]], 0); ptst = np.concatenate([tr[t], ef[t]], 0)
    vis = np.concatenate([vs[t], np.ones(3, np.float32)], 0)
    return splat_flow(pts0, ptst, vis)


def region_mask(pts_list, res=IMG_RES, radius=10):
    m = np.zeros((res, res), np.uint8)
    for pts in pts_list:
        for x, y in pts:
            cv2.circle(m, (int(np.clip(x, 0, 1) * res), int(np.clip(y, 0, 1) * res)), radius, 1, -1)
    return m.astype(bool)


def widen_control_net(cn, c_flow):
    old = cn.num_cond_channel; new = old + c_flow
    dt = next(cn.parameters()).dtype; dev = next(cn.parameters()).device
    for up in cn.upsample_conds:
        oc = up.conv; nc = nn.Conv2d(new, new, 3, padding=1).to(dev, dt)
        with torch.no_grad():
            nc.weight.zero_(); nc.weight[:old, :old] = oc.weight; nc.weight[old:, old:] = oc.weight[:c_flow, :c_flow]
            nc.bias.zero_(); nc.bias[:old] = oc.bias
        up.conv = nc; up.channels = new; up.out_channels = new
    oc0 = cn.input_blocks_cond[0]; nc0 = nn.Conv2d(new, oc0.out_channels, 3, padding=1).to(dev, dt)
    with torch.no_grad():
        nc0.weight.zero_(); nc0.weight[:, :old] = oc0.weight
        nc0.weight[:, old:] = torch.randn_like(nc0.weight[:, old:]) * 0.05   # non-zero so grad flows
        nc0.bias.copy_(oc0.bias)
    cn.input_blocks_cond[0] = nc0; cn.num_cond_channel = new
    print(f"widened control_net cond {old}->{new}", flush=True)


def load_model():
    cfg = OmegaConf.load(CFG)
    from interactive_world_sim.algorithms.latent_dynamics import LatentWorldModel
    m = LatentWorldModel.load_from_checkpoint(CKPT, cfg=cfg.algorithm, map_location="cpu")
    m.eval().to(device); torch.cuda.empty_cache()
    print(f"model loaded dtype={m.dtype} dec_infer_steps={m.dec_infer_steps} timesteps={m.timesteps}", flush=True)
    return m


def recon_loss(model, xs, cond):
    t, s = model._generate_noise_levels(xs[None], model.dec_infer_steps)
    wt = model.noise_scheduler.get_weights(t)[0]; ws = model.noise_scheduler.get_weights(s)[0]
    nt, ns = model.noise_scheduler.add_noise_to_t_s(xs[None], t, s)
    nt, ns = nt.squeeze(0), ns.squeeze(0); t, s = t.squeeze(0), s.squeeze(0); u = torch.zeros_like(t)
    ps = model._forward(model.decoder, nt, t, s, external_cond=cond)
    loss = nn.functional.mse_loss(ps, ns.detach(), reduction="none") * wt.view(*wt.shape, *((1,) * (ps.ndim - 1)))
    if model.dec_infer_steps > 1:
        pu = model._forward(model.decoder, ns, s, u, external_cond=cond)
        loss = loss + nn.functional.mse_loss(pu, xs.detach(), reduction="none") * ws.view(*ws.shape, *((1,) * (ps.ndim - 1)))
    return loss.mean()


def main():
    os.makedirs(OUTDIR, exist_ok=True); summary = []
    model = load_model(); widen_control_net(model.decoder.control_net, C_FLOW); DT = model.dtype
    nm = model.normalizer[model.obs_keys[0]]

    # load full data
    parts = []
    for tag in ("human", "robot"):
        z = np.load(f"{DS}/clips_{tag}.npz")
        parts.append((z["frames"], z["tracks"].astype(np.float32), z["eef"].astype(np.float32),
                      z["vis"].astype(np.float32), z["vid"]))
    frames = np.concatenate([p[0] for p in parts]); tracks = np.concatenate([p[1] for p in parts])
    eef = np.concatenate([p[2] for p in parts]); vis = np.concatenate([p[3] for p in parts])
    N = len(frames); print(f"data N={N} frames={frames.shape}", flush=True)

    rng = np.random.default_rng(0); perm = rng.permutation(N)
    n_ho = int(N * HELDOUT_FRAC); ho_idx, tr_idx = perm[:n_ho], perm[n_ho:]
    print(f"train={len(tr_idx)} held-out={len(ho_idx)}", flush=True)

    # PRECOMPUTE z0 = enc(frame0) for every clip (I0 is fixed per clip -> huge speedup)
    @torch.no_grad()
    def enc0(fr_u8):
        x = torch.from_numpy(fr_u8).float().permute(0, 3, 1, 2).to(device) / 255.0
        return model.encoder_forward(nm.normalize(x).to(DT))
    z0_all = torch.zeros(N, 4, LAT_RES, LAT_RES, device=device, dtype=DT)
    for i in range(0, N, 64):
        z0_all[i:i + 64] = enc0(frames[i:i + 64, 0])
    print(f"precomputed z0_all {tuple(z0_all.shape)}", flush=True)

    for p in model.decoder.parameters(): p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in model.decoder.parameters()], lr=LR)
    g = np.random.default_rng(1)

    def batch_cond_target(idx, ts, flow_mode="gt"):
        z0 = z0_all[idx]
        fms = []
        for j, t in zip(idx, ts):
            if flow_mode == "zero":
                fms.append(np.zeros((C_FLOW, LAT_RES, LAT_RES), np.float32))
            else:
                fms.append(make_flow(tracks[j], eef[j], vis[j], t))
        fm = torch.from_numpy(np.stack(fms)).to(DT).to(device)
        cond = torch.cat([z0, fm], 1)
        It = torch.from_numpy(frames[idx, ts]).float().permute(0, 3, 1, 2).to(device) / 255.0
        xs = nm.normalize(It).to(DT)
        return cond, xs, It

    print("=== train ===", flush=True)
    model.decoder.train()
    for ep in range(EPOCHS):
        pe = tr_idx[g.permutation(len(tr_idx))]; tot = 0.0; nb = 0
        for bi in range(0, len(pe), BS):
            idx = pe[bi:bi + BS]; ts = g.integers(T_LO, T_HI, len(idx))
            cond, xs, _ = batch_cond_target(idx, ts, "gt")
            loss = recon_loss(model, xs, cond)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        if ep % 2 == 0 or ep == EPOCHS - 1:
            wn = model.decoder.control_net.input_blocks_cond[0].weight[:, 4:].norm().item()
            print(f"ep {ep:3d}  loss {tot/nb:.5f}  flow_bypass_wnorm {wn:.3f}", flush=True)
    summary.append(f"final train loss {tot/nb:.5f}")

    # ===== held-out eval: GT vs SHUFFLE vs ZERO flow, multi-sample averaged =====
    from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
    model.decoder.eval()
    EV = ho_idx[:min(40, len(ho_idx))]; ev_t = np.full(len(EV), T_HI - 1)
    N_SAMPLE = 4

    @torch.no_grad()
    def rmean(cond):
        acc = None
        for _ in range(N_SAMPLE):
            r = render_img_cm(model, cond, IMG_RES, model.normalizer, num_views=1, batch_size=8)
            acc = r if acc is None else acc + r
        return (acc / N_SAMPLE).cpu().numpy()

    with torch.no_grad():
        cond_gt, _, It = batch_cond_target(EV, ev_t, "gt")
        # shuffle flow within eval batch (mismatch); zero flow = z0-only baseline
        z0e = z0_all[EV]
        fm_gt = cond_gt[:, 4:]
        cond_sh = torch.cat([z0e, fm_gt[torch.randperm(len(EV))]], 1)
        cond_zero = torch.cat([z0e, torch.zeros_like(fm_gt)], 1)
        pred_gt = rmean(cond_gt); pred_sh = rmean(cond_sh); pred_zero = rmean(cond_zero)
        # I0 frames for reference
        I0 = (frames[EV, 0].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
        gt = It.cpu().numpy()

    def mse(a, b): return float(np.mean((a - b) ** 2))
    e_gt = np.mean([mse(pred_gt[i], gt[i]) for i in range(len(EV))])
    e_sh = np.mean([mse(pred_sh[i], gt[i]) for i in range(len(EV))])
    e_zero = np.mean([mse(pred_zero[i], gt[i]) for i in range(len(EV))])
    for ln in [f"[HELD-OUT] pred(GT flow)   vs GT MSE = {e_gt:.5f}",
               f"[HELD-OUT] pred(SHUFFLE)    vs GT MSE = {e_sh:.5f}  ratio {e_sh/(e_gt+1e-8):.2f}x",
               f"[HELD-OUT] pred(ZERO flow)  vs GT MSE = {e_zero:.5f}  ratio {e_zero/(e_gt+1e-8):.2f}x (flow net gain)"]:
        print(ln, flush=True); summary.append(ln)

    # figure: I0 / GT / pred(GT) / pred(shuffle) / pred(zero)
    ns = min(8, len(EV))
    fig, ax = plt.subplots(5, ns, figsize=(2.1 * ns, 11))
    rows = ["I0", "GT I_t", "pred (GT flow)", "pred (shuffle)", "pred (zero flow)"]
    imgs_rows = [I0, gt, pred_gt, pred_sh, pred_zero]
    for j in range(ns):
        for r in range(5):
            ax[r, j].imshow(np.clip(imgs_rows[r][j].transpose(1, 2, 0), 0, 1)); ax[r, j].axis("off")
            if j == 0: ax[r, j].set_title(rows[r], fontsize=9, loc="left")
    fig.suptitle("HELD-OUT flow-cond decoder (full-data trained). pred(GT) should beat pred(shuffle)&pred(zero) "
                 "AND show moved object/agent. If GT~=zero, decoder still ignores flow.", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/heldout_eval.png", dpi=110)
    print(f"saved {OUTDIR}/heldout_eval.png", flush=True)

    # save the trained flow-conditioned decoder (for downstream R1 template / ② dynamics)
    torch.save({"decoder": model.decoder.state_dict(), "C_FLOW": C_FLOW,
                "ckpt_AE": CKPT, "epochs": EPOCHS}, f"{OUTDIR}/decoder_flowcond.pt")
    print(f"saved {OUTDIR}/decoder_flowcond.pt", flush=True)

    with open(f"{OUTDIR}/summary.txt", "w") as f:
        f.write(f"flow-cond decoder proper training (densev3, 80ep, ckpt)\nckpt={CKPT}\nN={N} epochs={EPOCHS} "
                f"T_range=[{T_LO},{T_HI}) FLOW_SCALE={FLOW_SCALE}\n\n" + "\n".join(summary) + "\n")
    print(f"saved {OUTDIR}/summary.txt\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
