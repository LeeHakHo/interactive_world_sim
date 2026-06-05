"""END-TO-END demo: ② FlowWM predicts object flow -> ③ densev3 renderer draws pixels.
Shows the data-scarcity human-helps result IN PIXEL SPACE: render a held-out robot frame
from (a) GT flow (oracle), (b) robot-only ② prediction, (c) robot+human ② prediction.
If human helps, (c) should look closer to GT than (b).

Zero cross-dataset alignment: everything runs on outputs/flow_render_dataset/clips_*.npz
(same crop/res/normalization as ③ densev3). ② is retrained HERE on clips (K=4 hist ->
predict object points through t=23), ③ is the frozen densev3 decoder (decoder_flowcond.pt).
The ONLY variable is ②'s predicted object flow; ③ is identical across conditions.

Output: outputs/e2e_flow_wm_render__robotonly_vs_human/
"""
import os, numpy as np, cv2, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from omegaconf import OmegaConf

_otl = torch.load
torch.load = lambda *a, **k: _otl(*a, **{**k, "weights_only": False})
OmegaConf.register_new_resolver("eval", lambda e: eval(e, {"np": np}), replace=True)
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x), replace=True)

CKPT_AE = "outputs/2026-05-29/job45895_phase0_stage1_mask_subtract_2gpu/checkpoints/epoch=1-step=25000.ckpt"
CFG = "outputs/2026-05-29/job45895_phase0_stage1_mask_subtract_2gpu/.hydra/config.yaml"
DEC_CKPT = "outputs/flow_cond_decoder_train__densev3_80ep__ckpt/decoder_flowcond.pt"
DS = "outputs/flow_render_dataset"
OUTDIR = "outputs/e2e_flow_wm_render__robotonly_vs_human"
IMG_RES, LAT_RES, C_FLOW = 128, 32, 4
FLOW_SCALE = 10.0
SPLAT_RADIUS, SPLAT_SIGMA = 4, 5.0
K, L = 4, 24                       # hist 4 frames, clip length 24
F = L - K                          # predict t=K..L-1 (t=23 is render target)
RENDER_T = L - 1                   # 23
Dm = 128
WM_EPOCHS, WM_BS, WM_LR = 60, 128, 3e-4
N_ROB = 100                        # data-scarcity point (human helps most here)
HELDOUT_ROB = 100
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- ② FlowWM (clips version, F=20) ----------------
class FlowWM(nn.Module):
    def __init__(self, P):
        super().__init__()
        self.inp = nn.Linear(2 * K + 2, Dm)
        self.act = nn.Linear((K + F) * 3 * 2, Dm)
        enc = nn.TransformerEncoderLayer(Dm, 4, Dm * 2, batch_first=True, dropout=0.0)
        self.tf = nn.TransformerEncoder(enc, 3)
        self.head = nn.Linear(Dm, F * 2)

    def forward(self, hist, eef3):
        B, P = hist.shape[:2]
        anchor = hist[:, :, -1, :]
        obj = self.inp(torch.cat([(hist - anchor[:, :, None]).reshape(B, P, 2 * K), anchor], -1))
        objc = anchor.mean(1, keepdim=True)
        act = self.act((eef3 - objc[:, :, None]).reshape(B, -1))[:, None, :]
        x = self.tf(torch.cat([obj, act], 1))[:, :P]
        return anchor[:, :, None, :] + self.head(x).reshape(B, P, F, 2)


def train_wm(tracks, vis, eef, idx, seed=0):
    torch.manual_seed(seed)
    P = tracks.shape[2]; m = FlowWM(P).to(device); opt = torch.optim.AdamW(m.parameters(), lr=WM_LR)
    g = torch.Generator().manual_seed(seed)
    tr = torch.from_numpy(tracks).float(); vs = torch.from_numpy(vis).float(); ef = torch.from_numpy(eef).float()
    for ep in range(WM_EPOCHS):
        m.train(); pe = idx[torch.randperm(len(idx), generator=g)]
        for i in range(0, len(pe), WM_BS):
            b = pe[i:i + WM_BS]
            h, fut = tr[b, :K].to(device), tr[b, K:].to(device)
            e = ef[b].to(device); hv, fv = vs[b, :K].to(device), vs[b, K:].to(device)
            pred = m(h.permute(0, 2, 1, 3), e); fut = fut.permute(0, 2, 1, 3)
            w = (fv.permute(0, 2, 1) * hv[:, -1:].permute(0, 2, 1))[..., None]
            loss = ((pred - fut) ** 2 * w).sum() / (w.sum() + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def wm_predict_objt(m, tracks, eef, idx):
    """Return predicted object points at RENDER_T for clips idx. (len,P,2)."""
    tr = torch.from_numpy(tracks[idx, :K]).float().to(device)
    ef = torch.from_numpy(eef[idx]).float().to(device)
    pred = m(tr.permute(0, 2, 1, 3), ef)        # (B,P,F,2)
    return pred[:, :, RENDER_T - K, :].cpu().numpy()


# ---------------- ③ renderer pieces (densev3) ----------------
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


def make_flow_e2e(tr0_obj, obj_t, ef0, ef_t, vis_t):
    pts0 = np.concatenate([tr0_obj, ef0], 0); ptst = np.concatenate([obj_t, ef_t], 0)
    vis = np.concatenate([vis_t, np.ones(3, np.float32)], 0)
    return splat_flow(pts0, ptst, vis)


def widen_control_net(cn, c_flow):
    old = cn.num_cond_channel; new = old + c_flow
    dt = next(cn.parameters()).dtype; dev = next(cn.parameters()).device
    for up in cn.upsample_conds:
        oc = up.conv; nc = nn.Conv2d(new, new, 3, padding=1).to(dev, dt)
        with torch.no_grad():
            nc.weight.zero_(); nc.bias.zero_()
            nc.weight[:old, :old] = oc.weight; nc.bias[:old] = oc.bias
            for k in range(c_flow): nc.weight[old + k, old + k, 1, 1] = 1.0
        up.conv = nc; up.channels = new; up.out_channels = new
    oc0 = cn.input_blocks_cond[0]; nc0 = nn.Conv2d(new, oc0.out_channels, 3, padding=1).to(dev, dt)
    with torch.no_grad():
        nc0.weight.zero_(); nc0.weight[:, :old] = oc0.weight
        nc0.weight[:, old:] = torch.randn_like(nc0.weight[:, old:]) * 0.05; nc0.bias.copy_(oc0.bias)
    cn.input_blocks_cond[0] = nc0; cn.num_cond_channel = new


def load_renderer():
    cfg = OmegaConf.load(CFG)
    from interactive_world_sim.algorithms.latent_dynamics import LatentWorldModel
    m = LatentWorldModel.load_from_checkpoint(CKPT_AE, cfg=cfg.algorithm, map_location="cpu")
    m.eval().to(device); widen_control_net(m.decoder.control_net, C_FLOW)
    sd = torch.load(DEC_CKPT, map_location="cpu"); m.decoder.load_state_dict(sd["decoder"], strict=False)
    m.decoder.to(device).eval(); print("renderer loaded", flush=True)
    return m


def main():
    os.makedirs(OUTDIR, exist_ok=True); summary = []
    zr = np.load(f"{DS}/clips_robot.npz"); zh = np.load(f"{DS}/clips_human.npz")
    r_tr, r_ef, r_vs, r_fr = (zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32),
                              zr["vis"].astype(np.float32), zr["frames"])
    h_tr, h_ef, h_vs = zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32), zh["vis"].astype(np.float32)
    Nr = len(r_tr); print(f"robot N={Nr} human N={len(h_tr)}", flush=True)

    rng = np.random.default_rng(0); perm = rng.permutation(Nr)
    ho = perm[:HELDOUT_ROB]; pool = perm[HELDOUT_ROB:]
    sub = pool[np.random.default_rng(100).choice(len(pool), min(N_ROB, len(pool)), replace=False)]
    rob_idx = torch.from_numpy(sub)

    # human pool appended with index offset into a merged array
    merged_tr = np.concatenate([r_tr, h_tr]); merged_ef = np.concatenate([r_ef, h_ef]); merged_vs = np.concatenate([r_vs, h_vs])
    hum_idx = torch.arange(Nr, Nr + len(h_tr))
    print(f"② train: robot-only N={len(rob_idx)} ; robot+human N={len(rob_idx)+len(hum_idx)} ; held-out robot={len(ho)}", flush=True)

    print("=== train ② robot-only ===", flush=True)
    wm_ro = train_wm(merged_tr, merged_vs, merged_ef, rob_idx, seed=0)
    print("=== train ② robot+human ===", flush=True)
    wm_rh = train_wm(merged_tr, merged_vs, merged_ef, torch.cat([rob_idx, hum_idx]), seed=0)

    # ② held-out ADE (px@... normalized->*IMG? tracks normalized [0,1]; report *IMG_RES px-equiv on 128)
    @torch.no_grad()
    def ade(m):
        tr = torch.from_numpy(r_tr[ho, :K]).float().to(device); ef = torch.from_numpy(r_ef[ho]).float().to(device)
        pred = m(tr.permute(0, 2, 1, 3), ef)
        fut = torch.from_numpy(r_tr[ho, K:]).float().permute(0, 2, 1, 3).to(device)
        fv = torch.from_numpy(r_vs[ho, K:]).float().permute(0, 2, 1).to(device)
        err = torch.linalg.norm(pred - fut, dim=-1) * 224.0
        return ((err * fv).sum() / (fv.sum() + 1e-6)).item()
    ade_ro, ade_rh = ade(wm_ro), ade(wm_rh)
    summary += [f"② held-out robot ADE (px@224): robot-only={ade_ro:.2f}  robot+human={ade_rh:.2f}  Δ={ade_ro-ade_rh:+.2f}"]
    print(summary[-1], flush=True)

    # ---------------- end-to-end render on held-out robot ----------------
    model = load_renderer(); DT = model.dtype; nm = model.normalizer[model.obs_keys[0]]
    from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
    EV = ho[:8]
    pred_ro = wm_predict_objt(wm_ro, r_tr, r_ef, EV)
    pred_rh = wm_predict_objt(wm_rh, r_tr, r_ef, EV)

    @torch.no_grad()
    def enc0(fr):
        x = torch.from_numpy(fr).float().permute(0, 3, 1, 2).to(device) / 255.0
        return model.encoder_forward(nm.normalize(x).to(DT))
    z0 = enc0(r_fr[EV, 0])

    def flows_for(objt_arr):
        fl = [make_flow_e2e(r_tr[j, 0], objt_arr[i], r_ef[j, 0], r_ef[j, RENDER_T], r_vs[j, RENDER_T])
              for i, j in enumerate(EV)]
        return torch.from_numpy(np.stack(fl)).to(DT).to(device)

    @torch.no_grad()
    def rmean(flow):
        cond = torch.cat([z0, flow], 1); acc = None
        for _ in range(4):
            r = render_img_cm(model, cond, IMG_RES, model.normalizer, num_views=1, batch_size=8)
            acc = r if acc is None else acc + r
        return (acc / 4).cpu().numpy()

    gt_obj = r_tr[EV, RENDER_T]
    pr_gt = rmean(flows_for(gt_obj)); pr_ro = rmean(flows_for(pred_ro)); pr_rh = rmean(flows_for(pred_rh))
    gtf = (r_fr[EV, RENDER_T].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    I0 = (r_fr[EV, 0].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)

    # object (red cube) region on GT frame, dilated to cover the moved-object neighborhood,
    # UNION over GT/robot-only/robot+human-predicted object positions so a misplaced object
    # is penalized at BOTH where it should be and where it was wrongly drawn.
    def cube_reg(fr_u8):
        hsv = cv2.cvtColor(np.ascontiguousarray(fr_u8), cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        red = (((h < 12) | (h > 168)) & (s > 90) & (v > 35) & (v < 175)).astype(np.uint8)
        return cv2.dilate(cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)), np.ones((9, 9), np.uint8)) > 0
    def pts_reg(objt):                                   # disk around predicted object points
        m = np.zeros((IMG_RES, IMG_RES), np.uint8)
        for x, y in objt: cv2.circle(m, (int(np.clip(x, 0, 1) * IMG_RES), int(np.clip(y, 0, 1) * IMG_RES)), 8, 1, -1)
        return m > 0
    obj_reg = [cube_reg(r_fr[j, RENDER_T]) | pts_reg(pred_ro[i]) | pts_reg(pred_rh[i]) | pts_reg(gt_obj[i])
               for i, j in enumerate(EV)]

    def mse(a, b): return float(np.mean((a - b) ** 2))
    def mse_obj(a, b):
        s = 0.0; n = 0
        for i in range(len(EV)):
            m = obj_reg[i]
            if m.sum() < 1: continue
            s += (((a[i] - b[i]) ** 2).mean(0))[m].mean(); n += 1
        return s / max(n, 1)
    e_gt = np.mean([mse(pr_gt[i], gtf[i]) for i in range(len(EV))])
    e_ro = np.mean([mse(pr_ro[i], gtf[i]) for i in range(len(EV))])
    e_rh = np.mean([mse(pr_rh[i], gtf[i]) for i in range(len(EV))])
    o_gt, o_ro, o_rh = mse_obj(pr_gt, gtf), mse_obj(pr_ro, gtf), mse_obj(pr_rh, gtf)
    summary += ["=== END-TO-END pixel MSE vs GT frame (held-out robot) ===",
                f"  [FULL] oracle={e_gt:.5f}  robot-only={e_ro:.5f}  robot+human={e_rh:.5f}  ro/rh={e_ro/(e_rh+1e-8):.2f}x",
                f"  [OBJ ] oracle={o_gt:.5f}  robot-only={o_ro:.5f}  robot+human={o_rh:.5f}  ro/rh={o_ro/(o_rh+1e-8):.2f}x",
                f"  human helps (object region, de-diluted) = {o_ro/(o_rh+1e-8):.2f}x "
                f"({'human better' if o_rh < o_ro else 'no gain'})"]
    for ln in summary[-4:]: print(ln, flush=True)

    ns = len(EV)
    fig, ax = plt.subplots(5, ns, figsize=(2.1 * ns, 11))
    rows = ["I0", "GT frame(t=23)", "② GT-flow (oracle)", "② robot-only", "② robot+human"]
    imgs = [I0, gtf, pr_gt, pr_ro, pr_rh]
    for j in range(ns):
        for r in range(5):
            ax[r, j].imshow(np.clip(imgs[r][j].transpose(1, 2, 0), 0, 1)); ax[r, j].axis("off")
            if j == 0: ax[r, j].set_title(rows[r], fontsize=9, loc="left")
    fig.suptitle(f"END-TO-END ②→③ (held-out robot, N_rob={N_ROB}). row4(robot-only ②) vs row5(robot+human ②): "
                 f"human should put the object closer to GT(row2). pixel MSE ro/rh={e_ro/(e_rh+1e-8):.2f}x", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/e2e_render.png", dpi=110)
    print(f"saved {OUTDIR}/e2e_render.png", flush=True)

    with open(f"{OUTDIR}/summary.txt", "w") as f:
        f.write(f"end-to-end ②FlowWM→③densev3 render (N_rob={N_ROB})\n\n" + "\n".join(summary) + "\n")
    print(f"saved {OUTDIR}/summary.txt\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
