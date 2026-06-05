"""densev4d FULL: feed the LEARNED, DEPLOYABLE silhouette g(joint,eef) into ③ (robot-only).
Unlike densev4a (oracle real mask = leaks the GT frame), the conditioning mask here comes
from g, which only eats the joint action + EEF points (rollout inputs, no future frame) ->
deployable + non-leaking. train AND test use g's mask. eval AGENT region uses the real
robot_v3 mask (NOT g's, so the eval region isn't self-defined).

Compare agent-region MSE to: densev4a oracle (0.00131, leaks), densev3-like flow+NOmask
(~0.07-0.09, no shape). If flow+gmask lands near oracle -> g's mask is good enough for ③.
Needs outputs/flow_render_dataset/clips2_robot.npz (joint+eef) and the g ckpt.
Output: outputs/flow_cond_decoder_train__densev4d_robot__learned_gmask/
"""
import os, sys, numpy as np, cv2, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from omegaconf import OmegaConf
sys.path.insert(0, ".")
from apply_robot_mask_v3 import robot_v3

_otl = torch.load
torch.load = lambda *a, **k: _otl(*a, **{**k, "weights_only": False})
OmegaConf.register_new_resolver("eval", lambda e: eval(e, {"np": np}), replace=True)
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x), replace=True)

CKPT = "outputs/2026-05-29/job45895_phase0_stage1_mask_subtract_2gpu/checkpoints/epoch=1-step=25000.ckpt"
CFG = "outputs/2026-05-29/job45895_phase0_stage1_mask_subtract_2gpu/.hydra/config.yaml"
DS = "outputs/flow_render_dataset/clips2_robot.npz"
GCK = "outputs/densev4d_maskgen_joint2mask_v2/maskgen.pt"
OUTDIR = "outputs/flow_cond_decoder_train__densev4d_robot__learned_gmask"
IMG_RES, LAT_RES = 128, 32
C_FLOW, C_MASK, C_EXTRA = 4, 1, 5
T_LO, T_HI = 12, 24
FLOW_SCALE = 10.0
EPOCHS, BS, LR = 80, 8, 1e-4
HELDOUT_FRAC = 0.12
SPLAT_RADIUS, SPLAT_SIGMA = 4, 5.0
device = "cuda" if torch.cuda.is_available() else "cpu"


class MaskGen(nn.Module):
    def __init__(s, indim=13):
        super().__init__()
        s.fc = nn.Sequential(nn.Linear(indim, 512), nn.ReLU(), nn.Linear(512, 512 * 4 * 4), nn.ReLU())
        s.dc = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1))
    def forward(s, x): return s.dc(s.fc(x).view(-1, 512, 4, 4))


def splat_flow(pos0, post, vis):
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


def widen_control_net(cn, c_extra):
    old = cn.num_cond_channel; new = old + c_extra
    dt = next(cn.parameters()).dtype; dev = next(cn.parameters()).device
    for up in cn.upsample_conds:
        oc = up.conv; nc = nn.Conv2d(new, new, 3, padding=1).to(dev, dt)
        with torch.no_grad():
            nc.weight.zero_(); nc.bias.zero_()
            nc.weight[:old, :old] = oc.weight; nc.bias[:old] = oc.bias
            for k in range(c_extra): nc.weight[old + k, old + k, 1, 1] = 1.0
        up.conv = nc; up.channels = new; up.out_channels = new
    oc0 = cn.input_blocks_cond[0]; nc0 = nn.Conv2d(new, oc0.out_channels, 3, padding=1).to(dev, dt)
    with torch.no_grad():
        nc0.weight.zero_(); nc0.weight[:, :old] = oc0.weight
        nc0.weight[:, old:] = torch.randn_like(nc0.weight[:, old:]) * 0.05; nc0.bias.copy_(oc0.bias)
    cn.input_blocks_cond[0] = nc0; cn.num_cond_channel = new


def load_model():
    cfg = OmegaConf.load(CFG)
    from interactive_world_sim.algorithms.latent_dynamics import LatentWorldModel
    m = LatentWorldModel.load_from_checkpoint(CKPT, cfg=cfg.algorithm, map_location="cpu")
    m.eval().to(device); torch.cuda.empty_cache(); return m


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
    model = load_model(); widen_control_net(model.decoder.control_net, C_EXTRA); DT = model.dtype
    nm = model.normalizer[model.obs_keys[0]]

    z = np.load(DS)
    frames = z["frames"]; tracks = z["tracks"].astype(np.float32); eef = z["eef"].astype(np.float32)
    vis = z["vis"].astype(np.float32); joint = z["joint"].astype(np.float32)
    N, L = len(frames), frames.shape[1]; print(f"ROBOT clips2 N={N} L={L}", flush=True)

    # ---- g(joint,eef) -> mask, precompute g_mask at latent res for every (clip,frame) ----
    gck = torch.load(GCK, map_location="cpu")
    g = MaskGen(gck["indim"]).to(device); g.load_state_dict(gck["g"]); g.eval()
    xm = torch.from_numpy(np.asarray(gck["xm"], np.float32)).to(device)
    xs = torch.from_numpy(np.asarray(gck["xs"], np.float32)).to(device)
    feat = np.concatenate([joint, eef.reshape(N, L, 6)], -1).astype(np.float32)   # (N,L,13) == g's input order
    g_mask = np.zeros((N, L, LAT_RES, LAT_RES), np.float32)
    with torch.no_grad():
        for i in range(N):
            fn = (torch.from_numpy(feat[i]).to(device) - xm) / xs
            pm = (torch.sigmoid(g(fn)[:, 0]) > 0.5).float().cpu().numpy()
            for t in range(L): g_mask[i, t] = cv2.resize(pm[t], (LAT_RES, LAT_RES))
    print(f"precomputed g_mask {g_mask.shape} mean_cov={g_mask.mean():.4f}", flush=True)

    rng = np.random.default_rng(0); perm = rng.permutation(N)
    n_ho = int(N * HELDOUT_FRAC); ho_idx, tr_idx = perm[:n_ho], perm[n_ho:]
    print(f"train {len(tr_idx)} held-out {len(ho_idx)}", flush=True)

    @torch.no_grad()
    def enc0(fr):
        x = torch.from_numpy(fr).float().permute(0, 3, 1, 2).to(device) / 255.0
        return model.encoder_forward(nm.normalize(x).to(DT))
    z0_all = torch.zeros(N, 4, LAT_RES, LAT_RES, device=device, dtype=DT)
    for i in range(0, N, 64): z0_all[i:i + 64] = enc0(frames[i:i + 64, 0])

    for p in model.decoder.parameters(): p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in model.decoder.parameters()], lr=LR)
    gg = np.random.default_rng(1)

    def batch_cond(idx, ts, use_flow=True, use_mask=True):
        z0 = z0_all[idx]; fms = []
        for j, t in zip(idx, ts):
            f = make_flow(tracks[j], eef[j], vis[j], t) if use_flow else np.zeros((C_FLOW, LAT_RES, LAT_RES), np.float32)
            mk = g_mask[j, t][None] if use_mask else np.zeros((C_MASK, LAT_RES, LAT_RES), np.float32)
            fms.append(np.concatenate([f, mk], 0))
        return torch.cat([z0, torch.from_numpy(np.stack(fms)).to(DT).to(device)], 1)

    def batch_target(idx, ts):
        It = torch.from_numpy(frames[idx, ts]).float().permute(0, 3, 1, 2).to(device) / 255.0
        return nm.normalize(It).to(DT), It

    print("=== train ===", flush=True); model.decoder.train()
    for ep in range(EPOCHS):
        pe = tr_idx[gg.permutation(len(tr_idx))]; tot = 0.0; nb = 0
        for bi in range(0, len(pe), BS):
            idx = pe[bi:bi + BS]; ts = gg.integers(T_LO, T_HI, len(idx))
            loss = recon_loss(model, batch_target(idx, ts)[0], batch_cond(idx, ts, True, True))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        if ep % 8 == 0 or ep == EPOCHS - 1:
            w = model.decoder.control_net.input_blocks_cond[0].weight
            print(f"ep {ep:3d} loss {tot/nb:.5f} flow_wn {w[:,4:8].norm():.3f} mask_wn {w[:,8:9].norm():.3f}", flush=True)
    summary.append(f"final train loss {tot/nb:.5f}")

    from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
    model.decoder.eval()
    EV = ho_idx[:min(40, len(ho_idx))]; ev_t = np.full(len(EV), T_HI - 1); N_SAMPLE = 4

    @torch.no_grad()
    def rmean(cond):
        acc = None
        for _ in range(N_SAMPLE):
            r = render_img_cm(model, cond, IMG_RES, model.normalizer, num_views=1, batch_size=8)
            acc = r if acc is None else acc + r
        return (acc / N_SAMPLE).cpu().numpy()

    with torch.no_grad():
        pred_fm = rmean(batch_cond(EV, ev_t, True, True))
        pred_f0 = rmean(batch_cond(EV, ev_t, True, False))
        pred_0m = rmean(batch_cond(EV, ev_t, False, True))
        _, It = batch_target(EV, ev_t); gt = It.cpu().numpy()
        I0 = (frames[EV, 0].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    # eval region = REAL robot_v3 mask (not g's), upsampled
    reg = np.stack([cv2.resize((robot_v3(np.ascontiguousarray(frames[j, t])) > 0).astype(np.float32), (IMG_RES, IMG_RES)) > 0.5
                    for j, t in zip(EV, ev_t)])

    def mse_reg(a, b):
        s = 0.0; n = 0
        for i in range(len(EV)):
            m = reg[i]
            if m.sum() < 1: continue
            s += (((a[i] - b[i]) ** 2).mean(0))[m].mean(); n += 1
        return s / max(n, 1)
    rows = [("flow+gmask", pred_fm), ("flow+NOmask", pred_f0), ("NOflow+gmask", pred_0m)]
    summary.append("=== AGENT-REGION MSE vs GT (real robot_v3 region) ===")
    rs = {nm_: mse_reg(pr, gt) for nm_, pr in rows}
    for nm_ in rs: summary.append(f"  {nm_:14s} agent={rs[nm_]:.5f}")
    summary.append(f"** g-mask net gain (agent) = flow+NOmask / flow+gmask = {rs['flow+NOmask']/(rs['flow+gmask']+1e-8):.2f}x **")
    summary.append("(ref: densev4a ORACLE-leak 0.00131 ; densev3-like flow+NOmask ~0.07-0.09)")
    for ln in summary[-6:]: print(ln, flush=True)

    ns = min(8, len(EV))
    fig, ax = plt.subplots(5, ns, figsize=(2.1 * ns, 11))
    rr = ["I0", "GT I_t", "pred flow+gmask", "pred flow+NOmask", "g mask(cond)"]
    imgs = [I0, gt, pred_fm, pred_f0]
    for j in range(ns):
        for r in range(4):
            ax[r, j].imshow(np.clip(imgs[r][j].transpose(1, 2, 0), 0, 1)); ax[r, j].axis("off")
            if j == 0: ax[r, j].set_title(rr[r], fontsize=9, loc="left")
        ax[4, j].imshow(cv2.resize(g_mask[EV[j], ev_t[j]], (IMG_RES, IMG_RES)), cmap="gray"); ax[4, j].axis("off")
        if j == 0: ax[4, j].set_title(rr[4], fontsize=9, loc="left")
    fig.suptitle("densev4d DEPLOYABLE learned g-mask. row3(flow+gmask) agent vs row4(flow+NOmask=densev3). "
                 "non-leaking, g only eats joint+eef.", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/heldout_eval.png", dpi=110)
    torch.save({"decoder": model.decoder.state_dict()}, f"{OUTDIR}/decoder_flowcond_gmask.pt")
    open(f"{OUTDIR}/summary.txt", "w").write("densev4d learned g-mask (robot)\n" + "\n".join(summary) + "\n")
    print(f"saved {OUTDIR}/heldout_eval.png + summary.txt\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
