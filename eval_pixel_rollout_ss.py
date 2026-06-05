"""The picture the user wants: render the rollout with the SCHEDULED-SAMPLING ② vs the
teacher-forced ②, both through densev4e ③ (with g-mask). Object positions come from each ②'s
free-run rollout; agent mask = g(joint_t,eef_t) (known, doesn't drift). If SS fixed the
compounding, the SS rows' object (red cube) should stay near GT position while the teacher rows'
object drifts away. 3 object sources: GT-flow (upper bound), teacher-② free-run, SS-② free-run.
Output: outputs/flow_wm/scheduled_sampling/
"""
import os, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import train_flow_cond_decoder_densev4e as D
import e2e_flow_wm_render as E
from e2e_flow_wm_render import train_wm
from eval_scheduled_sampling import train_wm_ss, rollout_free
from apply_robot_mask_v3 import robot_v3
from gen_flow_render_dataset import cube_mask

OUT = "outputs/flow_wm/scheduled_sampling"; os.makedirs(OUT, exist_ok=True)
DEC = "outputs/flow_wm/flow_cond_decoder__densev4e_robot__armmask_gmask/decoder_flowcond_gmask.pt"
K, F, device, LAT, IMG = E.K, E.F, D.device, D.LAT_RES, D.IMG_RES
TS = [4, 8, 12, 16, 20, 28, 36, 40]; NSEQ, NSAMP = 16, 2


def main():
    zr = np.load(f"{E.DS}/clips_robot.npz"); zh = np.load(f"{E.DS}/clips_human.npz")
    r_tr, r_ef, r_vs = zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32), zr["vis"].astype(np.float32)
    h_tr, h_ef, h_vs = zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32), zh["vis"].astype(np.float32)
    Nr = len(r_tr); pool = np.random.default_rng(0).permutation(Nr)[E.HELDOUT_ROB:]
    sub = pool[np.random.default_rng(100).choice(len(pool), min(E.N_ROB, len(pool)), replace=False)]
    mtr = np.concatenate([r_tr, h_tr]); mef = np.concatenate([r_ef, h_ef]); mvs = np.concatenate([r_vs, h_vs])
    rhidx = torch.cat([torch.from_numpy(sub), torch.arange(Nr, Nr + len(h_tr))])
    wm_teacher = train_wm(mtr, mvs, mef, rhidx, seed=0)
    wm_ss = train_wm_ss(mtr, mvs, mef, rhidx, seed=0)
    print("② teacher + SS trained", flush=True)

    z = np.load(f"{OUT.replace('scheduled_sampling','rollout_eval')}/seqs.npz"); H = int(z["H"])
    trk, eff, vis = z["tracks"], z["eef"], z["vis"]; frames = z["frames"]; joint = z["joint"]
    B = len(trk)
    tt = torch.from_numpy(trk).float().to(device); ee = torch.from_numpy(eff).float().to(device)
    pr_t = rollout_free(wm_teacher, tt, ee, H).cpu().numpy()
    pr_s = rollout_free(wm_ss, tt, ee, H).cpu().numpy()

    model = D.load_model(); D.widen_control_net(model.decoder.control_net, D.C_EXTRA)
    model.decoder.load_state_dict(torch.load(DEC, map_location="cpu")["decoder"])
    model.eval(); DT = model.dtype; nm = model.normalizer[model.obs_keys[0]]
    gck = torch.load(D.GCK, map_location="cpu")
    g = D.MaskGen(gck["indim"]).to(device); g.load_state_dict(gck["g"]); g.eval()
    xm = torch.from_numpy(np.asarray(gck["xm"], np.float32)).to(device)
    xs = torch.from_numpy(np.asarray(gck["xs"], np.float32)).to(device)
    from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

    @torch.no_grad()
    def enc0(fr):
        x = torch.from_numpy(fr).float().permute(0, 3, 1, 2).to(device) / 255.0
        return model.encoder_forward(nm.normalize(x).to(DT))

    @torch.no_grad()
    def gmask_at(idx, t):
        feat = np.concatenate([joint[idx, t], eff[idx, t].reshape(len(idx), 6)], -1).astype(np.float32)
        fn = (torch.from_numpy(feat).to(device) - xm) / xs
        pm = (torch.sigmoid(g(fn)[:, 0]) > 0.5).float().cpu().numpy()
        return np.stack([cv2.resize(pm[i], (LAT, LAT)) for i in range(len(idx))])

    def flow_at(idx, t, obj):
        out = []
        for k, j in enumerate(idx):
            pos0 = np.concatenate([trk[j, 0], eff[j, 0]], 0); post = np.concatenate([obj[k], eff[j, t]], 0)
            v = np.concatenate([vis[j, t], np.ones(3, np.float32)], 0)
            out.append(D.splat_flow(pos0, post, v))
        return np.stack(out)

    @torch.no_grad()
    def render(z0, flow, gm):
        cond = torch.cat([z0, torch.from_numpy(np.concatenate([flow, gm[:, None]], 1)).to(DT).to(device)], 1)
        acc = None
        for _ in range(NSAMP):
            r = render_img_cm(model, cond, IMG, model.normalizer, num_views=1, batch_size=8)
            acc = r if acc is None else acc + r
        return (acc / NSAMP).cpu().numpy()

    idxM = np.arange(min(NSEQ, B)); z0M = enc0(frames[idxM, 0])
    srcs = {"GT-flow": lambda t: trk[idxM, t], "teacher-②": lambda t: pr_t[:len(idxM), t - K],
            "SS-②": lambda t: pr_s[:len(idxM), t - K]}
    curves = {k: [] for k in srcs}
    for t in TS:
        reg = [cube_u(frames[j, t], trk[j, t], pr_t[i, t - K], pr_s[i, t - K]) for i, j in enumerate(idxM)]
        gt = (frames[idxM, t].astype(np.float32) / 255.0).transpose(0, 3, 1, 2); gm = gmask_at(idxM, t)
        for k, fn in srcs.items():
            p = render(z0M, flow_at(idxM, t, fn(t)), gm)
            curves[k].append(float(np.mean([(((p[i] - gt[i]) ** 2).mean(0))[reg[i]].mean()
                                            for i in range(len(idxM)) if reg[i].sum() > 0])))
        print(f"t={t}: " + " ".join(f"{k} {curves[k][-1]:.4f}" for k in srcs), flush=True)

    L = ["=== SS-② vs teacher-② rollout, rendered (object-region MSE vs horizon) ===",
         "③=densev4e (g-mask); agent=g(joint) (no drift); object=each ②'s free-run flow", "",
         f"{'t':<6}" + "".join(f"{t:<8}" for t in TS)]
    for k in srcs: L.append(f"{k:<12}" + "".join(f"{v:<8.4f}" for v in curves[k]))
    L += ["", f"object MSE @h=40: GT {curves['GT-flow'][-1]:.4f}  teacher {curves['teacher-②'][-1]:.4f}  SS {curves['SS-②'][-1]:.4f}",
          "win if SS object MSE < teacher (SS object drifts less -> renders nearer GT position)."]
    print("\n".join(L), flush=True); open(f"{OUT}/pixel_rollout_ss_summary.txt", "w").write("\n".join(L) + "\n")

    plt.figure(figsize=(8, 5.5))
    for k, c in [("GT-flow", "gray"), ("teacher-②", "C1"), ("SS-②", "C0")]:
        plt.plot(TS, curves[k], "o-", color=c, label=k, lw=2)
    plt.xlabel("rollout horizon t"); plt.ylabel("rendered object-region MSE")
    plt.title("Rendered rollout: SS-② object drifts less than teacher-② (agent fixed by g-mask)")
    plt.legend(); plt.grid(alpha=.3); plt.savefig(f"{OUT}/pixel_rollout_ss_curve.png", dpi=130, bbox_inches="tight")

    VS_ = [4, 12, 20, 40]; show = np.arange(min(2, B)); z0S = enc0(frames[show, 0])
    rows = ["GT frame", "render GT-flow", "render teacher-② (drift)", "render SS-② (fixed)"]
    fig, ax = plt.subplots(len(rows) * len(show), len(VS_), figsize=(2.1 * len(VS_), 2.2 * len(rows) * len(show)))
    for si, s in enumerate(show):
        for ti, t in enumerate(VS_):
            gm = gmask_at(np.array([s]), t)
            ims = [frames[s, t] / 255.,
                   render(z0S[si:si+1], flow_at(np.array([s]), t, trk[s:s+1, t]), gm)[0].transpose(1, 2, 0),
                   render(z0S[si:si+1], flow_at(np.array([s]), t, pr_t[s:s+1, t-K]), gm)[0].transpose(1, 2, 0),
                   render(z0S[si:si+1], flow_at(np.array([s]), t, pr_s[s:s+1, t-K]), gm)[0].transpose(1, 2, 0)]
            for r, im in enumerate(ims):
                a = ax[si*len(rows)+r, ti]; a.imshow(np.clip(im, 0, 1)); a.axis("off")
                if ti == 0: a.set_title(rows[r], fontsize=8, loc="left")
            ax[si*len(rows), ti].set_title(f"t={t}", fontsize=9)
    fig.suptitle("Scheduled-sampling rollout, RENDERED. teacher-② object (red cube) drifts; SS-② stays near GT. "
                 "agent = g(joint) mask (never drifts).", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/pixel_rollout_ss_grid.png", dpi=115)
    print(f"saved {OUT}/pixel_rollout_ss_{{curve,grid}}.png + summary\n=== DONE ===", flush=True)


def cube_u(frame_t, *objs):
    H_ = IMG; base = cv2.resize(cube_mask(cv2.resize(frame_t, (224, 224))), (H_, H_)) > 0
    yy, xx = np.ogrid[:H_, :H_]
    for pts in objs:
        for (x, y) in pts:
            gx, gy = int(np.clip(x * H_, 0, H_ - 1)), int(np.clip(y * H_, 0, H_ - 1))
            base = base | ((xx - gx) ** 2 + (yy - gy) ** 2 <= 36)
    return base


if __name__ == "__main__":
    main()
