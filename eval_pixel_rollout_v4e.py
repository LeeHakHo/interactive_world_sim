"""Redo the ③ pixel rollout with the RIGHT renderer: densev4e (has the agent g-mask), not
densev3 (no mask -> agent blur/ghost, what the user spotted). Key point: the agent mask is
g(joint_t, eef_t) and joint is a KNOWN rollout input (agent's own action, never goes through ②,
never drifts) -> the agent mask stays CORRECT through the whole rollout; only the OBJECT flow
(②) drifts. So this separates: agent (mask, correct) vs object (② flow, drifts). Render
GT-flow vs ②-free-run-flow, both with the correct g-mask. Output: outputs/flow_wm/rollout_eval/
"""
import os, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import train_flow_cond_decoder_densev4e as D
import e2e_flow_wm_render as E
from e2e_flow_wm_render import FlowWM, train_wm
from apply_robot_mask_v3 import robot_v3
from gen_flow_render_dataset import cube_mask

OUT = "outputs/flow_wm/rollout_eval"; os.makedirs(OUT, exist_ok=True)
DEC = "outputs/flow_wm/flow_cond_decoder__densev4e_robot__armmask_gmask/decoder_flowcond_gmask.pt"
K, F, device = E.K, E.F, D.device
LAT, IMG = D.LAT_RES, D.IMG_RES
TS = [4, 8, 12, 16, 20, 28, 36, 40]; NSEQ, NSAMP = 16, 2


def rollout_free(model, tracks, eef, H):
    buf = tracks[:, :K].clone(); preds = []
    with torch.no_grad():
        for h in range(H):
            p = model(buf[:, -K:].permute(0, 2, 1, 3), eef[:, h:h + K + F])
            preds.append(p[:, :, 0, :]); buf = torch.cat([buf, p[:, :, 0, :][:, None]], 1)
    return torch.stack(preds, 1)


def main():
    # ② robot+human (worst-drift) free-run flow source
    zr = np.load(f"{E.DS}/clips_robot.npz"); zh = np.load(f"{E.DS}/clips_human.npz")
    r_tr, r_ef, r_vs = zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32), zr["vis"].astype(np.float32)
    h_tr, h_ef, h_vs = zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32), zh["vis"].astype(np.float32)
    Nr = len(r_tr); pool = np.random.default_rng(0).permutation(Nr)[E.HELDOUT_ROB:]
    sub = pool[np.random.default_rng(100).choice(len(pool), min(E.N_ROB, len(pool)), replace=False)]
    mtr = np.concatenate([r_tr, h_tr]); mef = np.concatenate([r_ef, h_ef]); mvs = np.concatenate([r_vs, h_vs])
    wm_rh = train_wm(mtr, mvs, mef, torch.cat([torch.from_numpy(sub), torch.arange(Nr, Nr + len(h_tr))]), seed=0)
    print("② rh trained", flush=True)

    z = np.load(f"{OUT}/seqs.npz"); H = int(z["H"])
    trk, eff, vis = z["tracks"], z["eef"], z["vis"]; frames = z["frames"]; joint = z["joint"]
    B = len(trk)
    pr_rh = rollout_free(wm_rh, torch.from_numpy(trk).float().to(device),
                         torch.from_numpy(eff).float().to(device), H).cpu().numpy()

    # ③ densev4e + g v3
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

    def flow_at(idx, t, obj):                       # obj (len,P,2) at horizon t
        out = []
        for k, j in enumerate(idx):
            pos0 = np.concatenate([trk[j, 0], eff[j, 0]], 0)
            post = np.concatenate([obj[k], eff[j, t]], 0)
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
    obj_reg = {}; agt_reg = {}
    for t in TS:
        obj_reg[t] = [cube_u(frames[j, t], trk[j, t], pr_rh[i, t - K]) for i, j in enumerate(idxM)]
        agt_reg[t] = [robot_v3(np.ascontiguousarray(frames[j, t])) > 0 for j in idxM]

    def mse_reg(pred, t, reg):
        gt = (frames[idxM, t].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
        return float(np.mean([(((pred[i] - gt[i]) ** 2).mean(0))[reg[t][i]].mean()
                              for i in range(len(idxM)) if reg[t][i].sum() > 0]))

    curves = {"GT-flow obj": [], "free-flow obj": [], "GT-flow agent": [], "free-flow agent": []}
    for t in TS:
        gm = gmask_at(idxM, t)
        p_gt = render(z0M, flow_at(idxM, t, trk[idxM, t]), gm)
        p_fr = render(z0M, flow_at(idxM, t, pr_rh[:len(idxM), t - K]), gm)
        curves["GT-flow obj"].append(mse_reg(p_gt, t, obj_reg)); curves["free-flow obj"].append(mse_reg(p_fr, t, obj_reg))
        curves["GT-flow agent"].append(mse_reg(p_gt, t, agt_reg)); curves["free-flow agent"].append(mse_reg(p_fr, t, agt_reg))
        print(f"t={t}: obj GT {curves['GT-flow obj'][-1]:.4f} free {curves['free-flow obj'][-1]:.4f} | "
              f"agent GT {curves['GT-flow agent'][-1]:.4f} free {curves['free-flow agent'][-1]:.4f}", flush=True)

    L = ["=== ③ densev4e (WITH g-mask) pixel rollout: object & AGENT region MSE vs horizon ===",
         "agent mask = g(joint_t) — joint is a known rollout input, does NOT drift", "",
         f"{'t':<6}" + "".join(f"{t:<8}" for t in TS)]
    for k in curves: L.append(f"{k:<16}" + "".join(f"{v:<8.4f}" for v in curves[k]))
    L += ["", "agent MSE should stay ~flat for BOTH (mask correct, joint known); object free-flow grows (② drift).",
          "If agent free-flow ~ agent GT-flow -> mask fixed the agent ghosting the user saw in densev3."]
    print("\n".join(L), flush=True); open(f"{OUT}/pixel_rollout_v4e_summary.txt", "w").write("\n".join(L) + "\n")

    plt.figure(figsize=(8, 5.5))
    plt.plot(TS, curves["GT-flow obj"], "o-", color="gray", label="object | GT-flow")
    plt.plot(TS, curves["free-flow obj"], "o-", color="C3", label="object | ②-free (drift)")
    plt.plot(TS, curves["GT-flow agent"], "s--", color="C2", label="agent | GT-flow")
    plt.plot(TS, curves["free-flow agent"], "s--", color="C0", label="agent | ②-free")
    plt.xlabel("horizon t"); plt.ylabel("region pixel MSE")
    plt.title("densev4e rollout: agent (g-mask, joint known) stays flat; object (② flow) drifts")
    plt.legend(); plt.grid(alpha=.3); plt.savefig(f"{OUT}/pixel_rollout_v4e_curve.png", dpi=130, bbox_inches="tight")

    VS_ = [4, 12, 20, 40]; show = np.arange(min(2, B)); z0S = enc0(frames[show, 0])
    fig, ax = plt.subplots(3 * len(show), len(VS_), figsize=(2.1 * len(VS_), 6.6 * len(show)))
    for si, s in enumerate(show):
        for ti, t in enumerate(VS_):
            gm = gmask_at(np.array([s]), t)
            rg = render(z0S[si:si+1], flow_at(np.array([s]), t, trk[s:s+1, t]), gm)[0]
            rf = render(z0S[si:si+1], flow_at(np.array([s]), t, pr_rh[s:s+1, t-K]), gm)[0]
            for r, im in enumerate([frames[s, t]/255., rg.transpose(1,2,0), rf.transpose(1,2,0)]):
                a = ax[si*3+r, ti]; a.imshow(np.clip(im, 0, 1)); a.axis("off")
                if ti == 0: a.set_title(["GT frame", "render GT-flow+gmask", "render ②free+gmask"][r], fontsize=8, loc="left")
            ax[si*3, ti].set_title(f"t={t}", fontsize=9)
    fig.suptitle("densev4e (g-mask) rollout. agent uses g(joint) (known->correct); object uses ② flow (GT vs drift).", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/pixel_rollout_v4e_grid.png", dpi=115)
    print(f"saved {OUT}/pixel_rollout_v4e_{{curve,grid}}.png + summary\n=== DONE ===", flush=True)


def cube_u(frame_t, gt_obj, fr_obj):
    H_ = IMG; base = cv2.resize(cube_mask(cv2.resize(frame_t, (224, 224))), (H_, H_)) > 0
    yy, xx = np.ogrid[:H_, :H_]
    for pts in (gt_obj, fr_obj):
        for (x, y) in pts:
            gx, gy = int(np.clip(x * H_, 0, H_ - 1)), int(np.clip(y * H_, 0, H_ - 1))
            base = base | ((xx - gx) ** 2 + (yy - gy) ** 2 <= 36)
    return base


if __name__ == "__main__":
    main()
