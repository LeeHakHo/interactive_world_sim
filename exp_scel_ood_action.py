"""OOD-action eval: the REAL test of whether eef-3 action is 'too weak'. In-dist action replay is
all in-distribution (act layer already eats the eef window -> any thickening is ~0 marginal, proven:
pos+vel+acc 1.8->1.9). Here we drive ② with SYNTHETIC eef trajectories OUT of the training
distribution (= what keyboard input will do), roll out + render, and EYEBALL physical plausibility
(no GT exists for synthetic actions). Compare baseline-eef3 vs action-thick (pos+vel+acc).

Synthetic modes (real K-frame history kept; future eef replaced):
  freeze : eef held at last-hist pose -> cube SHOULD stay (gain<<1 worry: does ② still drag it?)
  drag   : eef centroid moves at constant velocity in a fixed (novel) direction
  circle : eef centroid orbits -> clearly OOD periodic motion

No GT -> report sanity proxies: cube_travel px (freeze should be ~0), follow_cos (cube net disp vs
eef net disp; drag/circle should be >0 if cube tracks), det rate. Plus gifs for eyeball.
Output: outputs/cross_embodiment_wm/scel_ood_action/"""
import os
os.environ["USE_GMASK"] = "1"; os.environ["FOOTPRINT"] = "1"; os.environ["DROP_VIS"] = "1"; os.environ["USE_PREV"] = "0"
import numpy as np, torch
from viz_combined import build_flow_cols, save_combined_gif
from exp_v3_human_helps_pixels import render_seq, load_gmask, cube_pos_err, K, F, device
from amplify_wm import rollout_lwc
import exp_scel_ss_long as SL

SMOKE = os.environ.get("SMOKE", "0") == "1"
DS_LONG = "outputs/flow_render_dataset_v3_long"
H = int(os.environ.get("HORIZON", "40")); NSEQ = int(os.environ.get("NSEQ", "5")); HELDOUT = 150; IMG = 128
RENDERER = os.environ.get("RENDERER", "outputs/cross_embodiment_wm/renderer_long_fp_novis/renderer.pt")
MODES = os.environ.get("MODES", "freeze,drag,circle").split(",")
OUT = "outputs/cross_embodiment_wm/scel_ood_action"; os.makedirs(f"{OUT}/gifs", exist_ok=True)


def synth_eef(ef_hist_last, mode, n, speed):
    """ef_hist_last (3,2) last-hist eef; -> (n,3,2) synthetic future eef, rigid 3-pt offset kept."""
    base_c = ef_hist_last.mean(0); off = ef_hist_last - base_c                      # rigid finger offsets
    t = np.arange(1, n + 1)[:, None]                                                # (n,1)
    if mode == "freeze":
        c = np.repeat(base_c[None], n, 0)
    elif mode == "drag":
        d = np.array([1.0, -0.6]); d = d / np.linalg.norm(d)                        # fixed novel dir (up-right)
        c = base_c[None] + t * speed * d[None]
    elif mode == "circle":
        r = max(speed * n / (2 * np.pi), 0.04); w = 2 * np.pi / n
        c = base_c[None] + r * np.concatenate([np.sin(w * t) , (1 - np.cos(w * t))], 1)
    else:
        raise ValueError(mode)
    return np.clip(c[:, None] + off[None], 0.02, 0.98).astype(np.float32)           # (n,3,2)


@torch.no_grad()
def rollout_synth(wm, tr0, ef_full):
    """tr0 (1,K,P,2) real history cube; ef_full (1,K+F+H,3,2) real-hist + synthetic future. -> (H,P,2)."""
    pr = rollout_lwc(wm, torch.from_numpy(tr0).float().to(device),
                     torch.from_numpy(ef_full).float().to(device), H)
    return pr.cpu().numpy()[0]


def main():
    if SMOKE:
        import eval_scheduled_sampling as SSm; SSm.WM_EPOCHS = 2
    load_gmask()
    ren = torch.load(RENDERER, map_location=device, weights_only=False).to(device).eval()
    zl = np.load(f"{DS_LONG}/clips_robot.npz")
    lt, le, lv, lf = (zl["tracks"].astype(np.float32), zl["eef"].astype(np.float32),
                      zl["vis"].astype(np.float32), zl["frames"]); ljt = zl["joint"].astype(np.float32)
    perm = np.random.default_rng(0).permutation(len(lt)); ho, lpool = perm[:HELDOUT], perm[HELDOUT:]
    if SMOKE: lpool = lpool[:200]; ho = ho[:8]
    R = 8 if SMOKE else 32
    print("=== train baseline-eef3 ② / action-thick ② (long data, SS) ===", flush=True)
    wm_b = SL.train_long(lt, lv, le, torch.from_numpy(lpool), R)
    wm_t = SL.train_long(lt, lv, le, torch.from_numpy(lpool), R, model_cls=SL.ActThickLWC)
    models = [("baseline-eef3", wm_b), ("action-thick", wm_t)]

    # pick high-motion seqs (real motion) so the cube is grabbable, then drive it OOD
    mot = np.linalg.norm(np.diff(lt[ho][:, K:K + H].mean(2), axis=1), axis=-1).sum(1)
    chosen = [int(x) for x in np.argsort(-mot)[:NSEQ]]
    # synthetic speed = median real eef centroid step (so motion magnitude is plausible, direction is OOD)
    speed = float(np.median(np.linalg.norm(np.diff(le[ho].mean(2), axis=1), axis=-1)))
    print(f"synthetic eef speed={speed:.4f}/frame  modes={MODES}  H={H}  n={len(chosen)}", flush=True)

    lines = [f"OOD-action eval | synthetic eef (out-of-train-dist) | H={H} | renderer_long | robot ②",
             f"synthetic speed={speed:.4f}/frame (median real)  modes={MODES}",
             "no GT: cube_travel px (freeze~0 good), follow_cos (drag/circle >0 good), det rate", ""]
    u8 = lambda x: (np.clip(x, 0, 1) * 255).astype(np.uint8)
    agg = {}
    for mode in MODES:
        lines.append(f"--- mode={mode} ---")
        lines.append(f"{'seq':>4} | " + " | ".join(f"{n:>13}" for n, _ in models) + "   (travel/follow/det)")
        for s in chosen:
            si = ho[s]
            ef_fut = synth_eef(le[si, K - 1], mode, F + H, speed)                  # (F+H,3,2)
            ef_full = np.concatenate([le[si, :K], ef_fut], 0)[None]                # (1,K+F+H,3,2)
            ef_render = ef_full[0, K:K + H]                                        # synthetic eef shown/rendered
            I0 = lf[si, 0]; jt_seq = ljt[si, K:K + H]; vis_seq = lv[si, K:K + H]
            cube0 = np.repeat(lt[si, K][None], H, 0)                               # green anchor = cube start
            row = [f"{s:>4} | "]; render_cols = [lf[si, K:K + H].astype(np.uint8)]; pred_cols = [None]; titles = ["bg"]
            for name, wm in models:
                pr = rollout_synth(wm, lt[si:si + 1, :K], ef_full)                 # (H,P,2)
                rseq = render_seq(ren, I0, lt[si, 0], le[si, 0], pr, ef_render, vis_seq, jt_seq)
                cc = pr.mean(1)                                                    # (H,2) cube centroid
                travel = float(np.linalg.norm(cc[-1] - cc[0]) * IMG)
                ec = ef_render.mean(1); cnet = cc[-1] - cc[0]; enet = ec[-1] - ec[0]
                fcos = float((cnet @ enet) / (np.linalg.norm(cnet) * np.linalg.norm(enet) + 1e-6))
                det = float(np.mean([~np.isnan(cube_pos_err(rseq[h], cc[h])) for h in range(H)]))
                row.append(f"{travel:5.1f}/{fcos:+.2f}/{det:.2f}")
                render_cols.append(u8(rseq)); pred_cols.append(pr); titles.append(name)
                agg.setdefault((mode, name), []).append((travel, fcos, det))
            lines.append(" | ".join(row)); print(lines[-1], flush=True)
            flow_cols = build_flow_cols(lf[si, K:K + H].astype(np.uint8), cube0, pred_cols, ef_render)
            save_combined_gif(f"{OUT}/gifs/{mode}_seq{s}.gif", np.stack(render_cols), flow_cols,
                              titles, [None] * len(titles), K)
        lines.append("")
    lines.append("=== MEAN per (mode,model): travel px / follow_cos / det ===")
    for mode in MODES:
        for name, _ in models:
            a = np.array(agg[(mode, name)])
            lines.append(f"  {mode:>7} {name:>13}: travel {a[:,0].mean():5.1f} | follow {a[:,1].mean():+.2f} | det {a[:,2].mean():.2f}")
    lines += ["", "EYEBALL the gifs: freeze=cube should sit still (drift=over-following bug);",
              "drag/circle=cube should track the yellow eef smoothly without teleport/vanish.",
              "If action-thick == baseline here too, action representation is NOT the lever (object",
              "state-dependence / contact is). If thick tracks better OOD, richer action matters."]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines) + f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
