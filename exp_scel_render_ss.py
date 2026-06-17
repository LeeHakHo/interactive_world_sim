"""Visualize Route 1 IN PIXELS: short-clip-trained ② vs long-data-trained ② rendered at H=40.
Shows the 6.9->2px long-horizon drift improvement as actual rendered cube stability. Reuses
renderer_long + exp_scel_ss_long.train_long + flow-column gifs (② pred red vs GT green).
Output: outputs/cross_embodiment_wm/scel_render_ss_H40/"""
import os
os.environ["USE_GMASK"] = "1"; os.environ["FOOTPRINT"] = "1"; os.environ["DROP_VIS"] = "1"; os.environ["USE_PREV"] = "0"
import numpy as np, torch
from viz_combined import build_flow_cols, save_combined_gif
from exp_v3_human_helps_pixels import render_seq, load_gmask, cube_pos_err, K, F, device
from amplify_wm import train_lwc_ss, rollout_lwc
import exp_scel_ss_long as SL

SMOKE = os.environ.get("SMOKE", "0") == "1"
DS_TRAIN = "outputs/flow_render_dataset_v3"; DS_LONG = "outputs/flow_render_dataset_v3_long"
H = 40; NSEQ = 12; HELDOUT = 150; IMG = 128; VEL_HALF = 0.06
RENDERER = os.environ.get("RENDERER", "outputs/cross_embodiment_wm/renderer_long_fp_novis/renderer.pt")
ABLATE = os.environ.get("ABLATE", "0") == "1"     # True: teacher-forced vs SS (both long); False: short vs long data
OUT = "outputs/cross_embodiment_wm/scel_render_ss" + ("_ablation" if ABLATE else "") + "_H40"; os.makedirs(f"{OUT}/gifs", exist_ok=True)


def metric(rseq, gt_s):
    e = np.array([cube_pos_err(rseq[h], gt_s[h].mean(0)) for h in range(len(rseq))])
    d = ~np.isnan(e); return float(np.where(d, e, IMG / 2).mean()), float(d.mean())


def main():
    if SMOKE:
        import eval_scheduled_sampling as SSm; SSm.WM_EPOCHS = 2
    load_gmask()
    ren = torch.load(RENDERER, map_location=device, weights_only=False).to(device).eval()
    # short-clip ② (the old setup that drifts to 6.9px)
    zt = np.load(f"{DS_TRAIN}/clips_robot.npz")
    st, se, sv = zt["tracks"].astype(np.float32), zt["eef"].astype(np.float32), zt["vis"].astype(np.float32)
    spool = np.random.default_rng(0).permutation(len(st))[HELDOUT:]
    if SMOKE: spool = spool[:200]
    zl = np.load(f"{DS_LONG}/clips_robot.npz")
    lt, le, lv, lf = (zl["tracks"].astype(np.float32), zl["eef"].astype(np.float32),
                      zl["vis"].astype(np.float32), zl["frames"]); ljt = zl["joint"].astype(np.float32)
    perm = np.random.default_rng(0).permutation(len(lt)); ho, lpool = perm[:HELDOUT], perm[HELDOUT:]
    if SMOKE: lpool = lpool[:200]
    R = 8 if SMOKE else 32
    if ABLATE:                                                  # SS protocol on/off (both long data, isolate SS)
        print("=== train long teacher-forced ② (SS OFF) / long SS ② (SS ON) ===", flush=True)
        wm_s = SL.train_long(lt, lv, le, torch.from_numpy(lpool), R, p_fixed=1.0)
        wm_l = SL.train_long(lt, lv, le, torch.from_numpy(lpool), R)
        lab_s, lab_l = "teacher(SSoff)", "SS(on)"
    else:                                                       # training data length: short vs long
        print("=== train short-clip ② / long-data ② ===", flush=True)
        wm_s = train_lwc_ss(st, sv, se, torch.from_numpy(spool), W=15, vel_half=VEL_HALF, seed=0)
        wm_l = SL.train_long(lt, lv, le, torch.from_numpy(lpool), R)
        lab_s, lab_l = "short-clip", "long-data"

    gt = lt[ho][:, K:K + H]
    tt = torch.from_numpy(lt[ho]).float().to(device); ee = torch.from_numpy(le[ho]).float().to(device)
    pr_s = rollout_lwc(wm_s, tt, ee, H).cpu().numpy()
    pr_l = rollout_lwc(wm_l, tt, ee, H).cpu().numpy()
    mot = np.linalg.norm(np.diff(gt.mean(2), axis=1), axis=-1).sum(1)
    order = list(np.argsort(-mot)); pick = sorted(set(int(x) for x in np.linspace(0, len(order) - 1, NSEQ)))
    chosen = [order[i] for i in pick][:NSEQ]
    lines = [f"viz IN PIXELS: {lab_s}② vs {lab_l}② rendered H={H} | renderer_long | robot held-out",
             f"{'seq':>4} | {'GTflow':>11} | {lab_s:>13} | {lab_l:>13}  (px/det)"]
    gp, gd, sp, sd, lp, ld = [], [], [], [], [], []
    u8 = lambda x: (np.clip(x, 0, 1) * 255).astype(np.uint8)
    for s in chosen:
        I0 = lf[ho[s], 0]; ef_seq = le[ho[s], K:K + H]; vis_seq = lv[ho[s], K:K + H]; jt_seq = ljt[ho[s], K:K + H]
        rg = render_seq(ren, I0, lt[ho[s], 0], le[ho[s], 0], gt[s], ef_seq, vis_seq, jt_seq)
        rs = render_seq(ren, I0, lt[ho[s], 0], le[ho[s], 0], pr_s[s], ef_seq, vis_seq, jt_seq)
        rl = render_seq(ren, I0, lt[ho[s], 0], le[ho[s], 0], pr_l[s], ef_seq, vis_seq, jt_seq)
        pg, dg = metric(rg, gt[s]); ps, ds = metric(rs, gt[s]); pl, dl = metric(rl, gt[s])
        gp.append(pg); gd.append(dg); sp.append(ps); sd.append(ds); lp.append(pl); ld.append(dl)
        lines.append(f"{s:>4} | {pg:5.1f}/{dg:.2f} | {ps:5.1f}/{ds:.2f} | {pl:5.1f}/{dl:.2f}"); print(lines[-1], flush=True)
        bg = lf[ho[s], K:K + H].astype(np.uint8)
        render4 = np.stack([bg, u8(rg), u8(rs), u8(rl)])
        flow4 = build_flow_cols(bg, gt[s], [None, None, pr_s[s], pr_l[s]], ef_seq)
        save_combined_gif(f"{OUT}/gifs/seq{s}.gif", render4, flow4,
                          ["GT", "GTflow3", lab_s, lab_l], [None, pg, ps, pl], K)
    f = lambda a: float(np.mean(a))
    lines += ["", f"MEAN GTflow {f(gp):.1f}/det{f(gd):.2f} | {lab_s}② {f(sp):.1f}/det{f(sd):.2f} | {lab_l}② {f(lp):.1f}/det{f(ld):.2f}  (n={len(chosen)} H={H})"]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines) + f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
