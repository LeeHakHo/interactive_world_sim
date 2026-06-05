"""Test 1 (FLOW_WM_REPORT §5 follow-up): ②'s free-run flow diverges to ~101 px @ h=40. Does
③'s I₀-anchor design BOUND the PIXEL error, or do the rendered frames blow up too?
③ renders each frame independently as [z₀=enc(I₀), splat_flow(0→t)] → I_t (no pixel recursion;
only ②'s predicted object positions drift). So: render the held-out rollout frames from
(a) GT flow (upper bound), (b) robot+human free-run flow (drifts worst), (c) robot-only free-run.
Measure object-region pixel MSE vs horizon + eyeball whether the object degrades gracefully
(rendered off-position but frame still coherent) or the frame collapses.
Reuses e2e ② (FlowWM/train_wm) + ③ densev3 renderer. Output: outputs/flow_wm/rollout_eval/
"""
import os, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import e2e_flow_wm_render as E
from e2e_flow_wm_render import FlowWM, train_wm, make_flow_e2e, load_renderer

OUT = "outputs/flow_wm/rollout_eval"; os.makedirs(OUT, exist_ok=True)
K, F, device, IMG_RES = E.K, E.F, E.device, E.IMG_RES
TS = [4, 8, 12, 16, 20, 28, 36, 40]      # sample-frame horizons (t=K+h)
NSEQ_MSE, NSAMP = 16, 2


def rollout_free(model, tracks, eef, H):
    buffer = tracks[:, :K].clone(); preds = []
    with torch.no_grad():
        for h in range(H):
            pred = model(buffer[:, -K:].permute(0, 2, 1, 3), eef[:, h:h + K + F])
            nxt = pred[:, :, 0, :]; preds.append(nxt)
            buffer = torch.cat([buffer, nxt[:, None]], 1)
    return torch.stack(preds, 1)             # (B,H,P,2)


def main():
    # ---- ② robot-only & robot+human (same protocol as e2e/compounding) ----
    zr = np.load(f"{E.DS}/clips_robot.npz"); zh = np.load(f"{E.DS}/clips_human.npz")
    r_tr, r_ef, r_vs = zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32), zr["vis"].astype(np.float32)
    h_tr, h_ef, h_vs = zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32), zh["vis"].astype(np.float32)
    Nr = len(r_tr); perm = np.random.default_rng(0).permutation(Nr); pool = perm[E.HELDOUT_ROB:]
    sub = pool[np.random.default_rng(100).choice(len(pool), min(E.N_ROB, len(pool)), replace=False)]
    merged_tr = np.concatenate([r_tr, h_tr]); merged_ef = np.concatenate([r_ef, h_ef]); merged_vs = np.concatenate([r_vs, h_vs])
    rob_idx = torch.from_numpy(sub); hum_idx = torch.arange(Nr, Nr + len(h_tr))
    wm_ro = train_wm(merged_tr, merged_vs, merged_ef, rob_idx, seed=0)
    wm_rh = train_wm(merged_tr, merged_vs, merged_ef, torch.cat([rob_idx, hum_idx]), seed=0)
    print("② trained", flush=True)

    # ---- rollout seqs (with frames) ----
    z = np.load(f"{OUT}/seqs.npz"); H = int(z["H"])
    tr = torch.from_numpy(z["tracks"]).float().to(device)
    ef = torch.from_numpy(z["eef"]).float().to(device)
    vs = z["vis"].astype(np.float32); frames = z["frames"]               # (B,SEQ,128,128,3)
    B = tr.shape[0]
    pr_ro = rollout_free(wm_ro, tr, ef, H).cpu().numpy()                 # (B,H,P,2)
    pr_rh = rollout_free(wm_rh, tr, ef, H).cpu().numpy()
    trk = z["tracks"]; eff = z["eef"]                                    # numpy

    # ---- ③ renderer ----
    model = load_renderer(); DT = model.dtype; nm = model.normalizer[model.obs_keys[0]]
    from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

    @torch.no_grad()
    def enc0(fr):
        x = torch.from_numpy(fr).float().permute(0, 3, 1, 2).to(device) / 255.0
        return model.encoder_forward(nm.normalize(x).to(DT))

    @torch.no_grad()
    def render(z0, flows):
        cond = torch.cat([z0, torch.from_numpy(np.stack(flows)).to(DT).to(device)], 1)
        acc = None
        for _ in range(NSAMP):
            r = render_img_cm(model, cond, IMG_RES, model.normalizer, num_views=1, batch_size=8)
            acc = r if acc is None else acc + r
        return (acc / NSAMP).cpu().numpy()

    idxM = np.arange(min(NSEQ_MSE, B))
    z0M = enc0(frames[idxM, 0])

    def disk_region(pts, H_=IMG_RES, r=6):
        m = np.zeros((H_, H_), bool); yy, xx = np.ogrid[:H_, :H_]
        for (x, y) in pts:
            gx, gy = int(np.clip(x * H_, 0, H_ - 1)), int(np.clip(y * H_, 0, H_ - 1))
            m |= (xx - gx) ** 2 + (yy - gy) ** 2 <= r * r
        return m

    def obj_at(t, src):
        if src == "gt": return trk[:, t]
        return (pr_ro if src == "ro" else pr_rh)[:, t - K]

    def flows_for(idx, t, src):
        return [make_flow_e2e(trk[j, 0], obj_at(t, src)[j], eff[j, 0], eff[j, t], vs[j, t]) for j in idx]

    # ---- MSE vs horizon ----
    curves = {"GT-flow": [], "robot+human free": [], "robot-only free": []}
    srcs = {"GT-flow": "gt", "robot+human free": "rh", "robot-only free": "ro"}
    for t in TS:
        gt_fr = (frames[idxM, t].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
        # same region for all conditions: where object is (GT) or predicted to be
        region = [cube_union(frames[j, t], trk[j, t], pr_rh[i, t - K], pr_ro[i, t - K])
                  for i, j in enumerate(idxM)]
        for name, src in srcs.items():
            pred = render(z0M, flows_for(idxM, t, src))
            mse = np.mean([(((pred[i] - gt_fr[i]) ** 2).mean(0))[region[i]].mean()
                           for i in range(len(idxM)) if region[i].sum() > 0])
            curves[name].append(float(mse))
        print(f"t={t}: GT {curves['GT-flow'][-1]:.4f} rh {curves['robot+human free'][-1]:.4f} ro {curves['robot-only free'][-1]:.4f}", flush=True)

    L = ["=== ③ PIXEL-level rollout: object-region MSE vs horizon ===",
         f"NSEQ={len(idxM)} NSAMP={NSAMP}; ② flow free-run drifts (rh to ~101px@224)", "",
         f"{'t(sample-frame)':<18}" + "".join(f"{t:<8}" for t in TS)]
    for k in curves: L.append(f"{k:<18}" + "".join(f"{v:<8.4f}" for v in curves[k]))
    L += ["", f"GT-flow stays ~{np.mean(curves['GT-flow']):.4f} (③ upper bound, flow correct).",
          "If rh/ro free-run MSE saturates (not unbounded) -> ③'s I0-anchor keeps pixels coherent",
          "(object rendered off-position but frame doesn't collapse). If it grows unbounded -> anchor fails."]
    print("\n".join(L), flush=True); open(f"{OUT}/pixel_rollout_summary.txt", "w").write("\n".join(L) + "\n")

    plt.figure(figsize=(8, 5.5))
    for k, c in [("GT-flow", "gray"), ("robot-only free", "C0"), ("robot+human free", "C1")]:
        plt.plot(TS, curves[k], "o-", color=c, label=k, lw=2)
    plt.xlabel("rollout horizon (sample frame t)"); plt.ylabel("object-region pixel MSE")
    plt.title("③ pixel rollout: does I0-anchor bound pixel error when ②'s flow diverges?")
    plt.legend(); plt.grid(alpha=.3); plt.savefig(f"{OUT}/pixel_rollout_curve.png", dpi=130, bbox_inches="tight")

    # ---- visual grid: 2 seqs across horizons, rows GT / GT-flow / rh-free ----
    VS_ = [4, 12, 20, 40]; show = np.arange(min(2, B))
    z0S = enc0(frames[show, 0])
    fig, ax = plt.subplots(3 * len(show), len(VS_), figsize=(2.1 * len(VS_), 6.6 * len(show)))
    for si, s in enumerate(show):
        for ti, t in enumerate(VS_):
            gt = frames[s, t].astype(np.float32) / 255.0
            rg = render(z0S[si:si+1], [make_flow_e2e(trk[s, 0], trk[s, t], eff[s, 0], eff[s, t], vs[s, t])])[0]
            rh = render(z0S[si:si+1], [make_flow_e2e(trk[s, 0], pr_rh[s, t-K], eff[s, 0], eff[s, t], vs[s, t])])[0]
            for r, im in enumerate([gt, rg.transpose(1,2,0), rh.transpose(1,2,0)]):
                a = ax[si*3+r, ti]; a.imshow(np.clip(im, 0, 1)); a.axis("off")
                if ti == 0: a.set_title(["GT frame", "render GT-flow", "render rh-free(drift)"][r], fontsize=8, loc="left")
            ax[si*3, ti].set_title(f"t={t}", fontsize=9)
    fig.suptitle("③ pixel rollout: GT frame vs render(GT-flow) vs render(②-free-run drifted flow). "
                 "Does the object degrade gracefully or collapse?", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/pixel_rollout_grid.png", dpi=115)
    print(f"saved {OUT}/pixel_rollout_curve.png + grid + summary\n=== DONE ===", flush=True)


def cube_union(frame_t, gt_obj, rh_obj, ro_obj):
    from gen_flow_render_dataset import cube_mask
    H_ = IMG_RES
    base = cv2.resize(cube_mask(cv2.resize(frame_t, (224, 224))), (H_, H_)) > 0
    yy, xx = np.ogrid[:H_, :H_]
    for pts in (gt_obj, rh_obj, ro_obj):
        for (x, y) in pts:
            gx, gy = int(np.clip(x * H_, 0, H_ - 1)), int(np.clip(y * H_, 0, H_ - 1))
            base = base | ((xx - gx) ** 2 + (yy - gy) ** 2 <= 36)
    return base


if __name__ == "__main__":
    main()
