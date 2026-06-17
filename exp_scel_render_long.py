"""SCEL H=40 long-horizon eval: world vs agent-frame ② on LONG data (L=48), many seqs, rendered
with the pre-trained renderer_long (footprint+novis+gmask). Reuses eval_pixel_H40 machinery
(renderer_long ckpt, sliding-window rollout, nan-trap metric). LONG data is robot-only -> this
measures ② long-horizon flow accuracy + render quality, NOT human-helps.

env: RENDERER, NSEQ, HORIZON, VEL_HALF."""
import os
os.environ["USE_GMASK"] = "1"; os.environ["FOOTPRINT"] = "1"; os.environ["DROP_VIS"] = "1"; os.environ["USE_PREV"] = "0"
import numpy as np, torch
from PIL import Image, ImageDraw
from exp_v3_human_helps_pixels import render_seq, load_gmask, cube_pos_err, K, device
from amplify_wm import train_lwc_ss, rollout_lwc
import exp_scel_agentframe as X

DS_TRAIN = "outputs/flow_render_dataset_v3"; DS_LONG = "outputs/flow_render_dataset_v3_long"
H = int(os.environ.get("HORIZON", "40")); NSEQ = int(os.environ.get("NSEQ", "12")); HELDOUT = 150; IMG = 128
VEL_HALF = float(os.environ.get("VEL_HALF", "0.06"))     # match eval_pixel_H40
RENDERER = os.environ.get("RENDERER", "outputs/cross_embodiment_wm/renderer_long_fp_novis/renderer.pt")
OUT = "outputs/cross_embodiment_wm/scel_render_long_H40"; os.makedirs(f"{OUT}/gifs", exist_ok=True)


def metric(rseq, gt_s):
    """rendered cube pos err with nan-trap (undetected = IMG/2 penalty); returns (err, det rate)."""
    e = np.array([cube_pos_err(rseq[h], gt_s[h].mean(0)) for h in range(len(rseq))])
    d = ~np.isnan(e); return float(np.where(d, e, IMG / 2).mean()), float(d.mean())


def train(tr_w, vs, ef, idx, agentframe):
    src = X.transform_tracks(tr_w, ef) if agentframe else tr_w
    return train_lwc_ss(src, vs, ef, torch.from_numpy(idx), W=15, vel_half=VEL_HALF, seed=0)


def rollout_world(wm, r_tr, r_ef, ho, agentframe):
    """rollout H steps on long data (sliding eef), return predicted cube pts in WORLD frame."""
    src = X.transform_tracks(r_tr[ho], r_ef[ho]) if agentframe else r_tr[ho]
    pr = rollout_lwc(wm, torch.from_numpy(src).float().to(device),
                     torch.from_numpy(r_ef[ho]).float().to(device), H).cpu().numpy()
    if agentframe:
        pr = X.S.from_agent_frame(pr, X.S.grasp_frame(r_ef[ho])[:, K:K + H])
    return pr


def main():
    load_gmask()
    ren = torch.load(RENDERER, map_location=device, weights_only=False).to(device).eval()
    zt = np.load(f"{DS_TRAIN}/clips_robot.npz")
    t_tr, t_ef, t_vs = zt["tracks"].astype(np.float32), zt["eef"].astype(np.float32), zt["vis"].astype(np.float32)
    pool = np.random.default_rng(0).permutation(len(t_tr))[HELDOUT:]
    print("=== train world ② / agent-frame ② (normal 24-frame robot pool) ===", flush=True)
    wm_w = train(t_tr, t_vs, t_ef, pool, False)
    wm_a = train(t_tr, t_vs, t_ef, pool, True)

    zl = np.load(f"{DS_LONG}/clips_robot.npz")
    r_tr, r_ef, r_vs, r_fr = (zl["tracks"].astype(np.float32), zl["eef"].astype(np.float32),
                              zl["vis"].astype(np.float32), zl["frames"]); r_jt = zl["joint"].astype(np.float32)
    ho = np.random.default_rng(0).permutation(len(r_tr))[:HELDOUT]; gt = r_tr[ho][:, K:K + H]
    pr_w = rollout_world(wm_w, r_tr, r_ef, ho, False)
    pr_a = rollout_world(wm_a, r_tr, r_ef, ho, True)
    gc = gt.mean(2)
    ew = np.linalg.norm(pr_w.mean(2) - gc, axis=-1) * IMG; ea = np.linalg.norm(pr_a.mean(2) - gc, axis=-1) * IMG
    print(f"②flow centroid err (px, NO renderer)  world: s0 {ew[:,0].mean():.1f} s{H-1} {ew[:,-1].mean():.1f} overall {ew.mean():.1f}", flush=True)
    print(f"②flow centroid err (px, NO renderer)  agentfr: s0 {ea[:,0].mean():.1f} s{H-1} {ea[:,-1].mean():.1f} overall {ea.mean():.1f}", flush=True)

    mot = np.linalg.norm(np.diff(gt.mean(2), axis=1), axis=-1).sum(1)
    order = list(np.argsort(-mot)); pick = sorted(set(int(x) for x in np.linspace(0, len(order) - 1, NSEQ)))
    chosen = [order[i] for i in pick][:NSEQ]
    lines = [f"SCEL H={H} pred-flow render | world vs agent-frame ② | renderer_long fp+novis+gmask | robot held-out (long, robot-only)",
             f"②flow overall (NO renderer): world {ew.mean():.1f} | agentframe {ea.mean():.1f}  px@128",
             f"{'seq':>4} | {'GTflow':>11} | {'world②':>11} | {'agentfr②':>11}  (px/det)"]
    gp, gd, wp, wd, ap, ad = [], [], [], [], [], []
    u8 = lambda x: (np.clip(x, 0, 1) * 255).astype(np.uint8); pad = np.full((IMG, 4, 3), 255, np.uint8)
    for s in chosen:
        I0 = r_fr[ho[s], 0]; ef_seq = r_ef[ho[s], K:K + H]; vis_seq = r_vs[ho[s], K:K + H]; jt_seq = r_jt[ho[s], K:K + H]
        rg = render_seq(ren, I0, r_tr[ho[s], 0], r_ef[ho[s], 0], gt[s], ef_seq, vis_seq, jt_seq)
        rw = render_seq(ren, I0, r_tr[ho[s], 0], r_ef[ho[s], 0], pr_w[s], ef_seq, vis_seq, jt_seq)
        ra = render_seq(ren, I0, r_tr[ho[s], 0], r_ef[ho[s], 0], pr_a[s], ef_seq, vis_seq, jt_seq)
        pg, dg = metric(rg, gt[s]); pw_, dw_ = metric(rw, gt[s]); pa_, da_ = metric(ra, gt[s])
        gp.append(pg); gd.append(dg); wp.append(pw_); wd.append(dw_); ap.append(pa_); ad.append(da_)
        lines.append(f"{s:>4} | {pg:5.1f}/{dg:.2f} | {pw_:5.1f}/{dw_:.2f} | {pa_:5.1f}/{da_:.2f}"); print(lines[-1], flush=True)
        bg = r_fr[ho[s], K:K + H].astype(np.uint8); gif = []
        for h in range(H):
            panel = np.concatenate([bg[h], pad, u8(rg[h]), pad, u8(rw[h]), pad, u8(ra[h])], 1)
            im = Image.fromarray(panel); dr = ImageDraw.Draw(im)
            for x, t in [(2, "GT"), (IMG + 6, "GTflow"), (2 * IMG + 10, "world2"), (3 * IMG + 14, "agentfr2")]:
                dr.text((x, 2), t, fill=(255, 255, 0))
            gif.append(im)
        gif[0].save(f"{OUT}/gifs/seq{s}.gif", save_all=True, append_images=gif[1:], duration=140, loop=0)
    f = lambda a: float(np.mean(a))
    lines += ["", f"MEAN GTflow {f(gp):.1f}/det{f(gd):.2f} | world {f(wp):.1f}/det{f(wd):.2f} | agentframe {f(ap):.1f}/det{f(ad):.2f}  (n={len(chosen)} H={H})",
              f"  det<0.9: world {sum(d<0.9 for d in wd)}/{len(wd)} agentfr {sum(d<0.9 for d in ad)}/{len(ad)}  | px>20: world {sum(p>20 for p in wp)}/{len(wp)} agentfr {sum(p>20 for p in ap)}/{len(ap)}"]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines) + f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
