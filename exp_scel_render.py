"""SCEL ② -> ③ PIXELS: does the world-vs-agentframe ADE gap (a few px) actually show up in the
RENDERED cube? Reuses ③ renderer (exp_v3_human_helps_pixels: footprint+novis U-Net) + SCEL ②
(exp_scel_agentframe, world/agent-frame). Renders GT-flow | world-② | agentframe-② with the SAME
③; reports rendered cube position error + detection rate (nan-trap safe) + 4-panel gifs.

NOTE (declare component versions): ③ = exp_v3_human_helps_pixels footprint+novis (FOOTPRINT=1,
DROP_VIS=1, USE_GMASK=0 -> cube only, agent stays from I0); ② = amplify_wm LWC via exp_scel_agentframe."""
import os
os.environ.setdefault("FOOTPRINT", "1"); os.environ.setdefault("DROP_VIS", "1"); os.environ.setdefault("USE_GMASK", "0")
import numpy as np, torch
from PIL import Image, ImageDraw
import exp_v3_human_helps_pixels as PX
import exp_scel_agentframe as X
import amplify_wm as A
import eval_scheduled_sampling as SSm
from amplify_wm import K, device

SMOKE = os.environ.get("SMOKE", "0") == "1"
DS = X.DS
OUT = ("outputs/cross_embodiment_wm/scel_m1_render" + ("_gmask" if PX.USE_GMASK else "")
       + (f"_lpips{PX.LAMBDA_LPIPS:g}" if PX.LAMBDA_LPIPS > 0 else "")); os.makedirs(f"{OUT}/gifs", exist_ok=True)
H = 16; HELDOUT = 150; N_ROB = 100; NSEQ = 2 if SMOKE else 6; IMG = 128


def rollout_world(model, r_tr, r_ef, idx, agentframe):
    src = X.transform_tracks(r_tr, r_ef) if agentframe else r_tr
    buf = torch.from_numpy(src[idx, :K]).float().to(device)
    ef = torch.from_numpy(r_ef[idx]).float().to(device)
    pred = A.rollout_lwc(model, buf, ef, H).cpu().numpy()
    if agentframe:
        pred = X.S.from_agent_frame(pred, X.S.grasp_frame(r_ef[idx])[:, K:K + H])
    return pred


def agg(render, gc):
    """rendered cube pos err with NAN-TRAP guard: undetected cube = penalty IMG/2, report det rate."""
    errs, det = [], 0
    for h in range(H):
        e = PX.cube_pos_err(render[h], gc[h])
        if np.isnan(e):
            errs.append(IMG / 2.0)            # cube disappeared -> penalize, do NOT reward by skipping
        else:
            errs.append(e); det += 1
    return float(np.mean(errs)), det / H


def main():
    if SMOKE: SSm.WM_EPOCHS = 2; SSm.R_SS = 4
    zr = np.load(f"{DS}/clips_robot.npz"); zh = np.load(f"{DS}/clips_human.npz")
    r_tr, r_ef, r_vs, r_fr = (zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32),
                              zr["vis"].astype(np.float32), zr["frames"])
    r_jt = zr["joint"].astype(np.float32)
    h_tr, h_ef, h_vs = zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32), zh["vis"].astype(np.float32)
    Nr = len(r_tr); perm = np.random.default_rng(0).permutation(Nr); ho, pool = perm[:HELDOUT], perm[HELDOUT:]
    X.tracks_world, X.vis_all = r_tr, r_vs

    if PX.USE_GMASK: PX.load_gmask()
    print(f"=== train ③ renderer (footprint+novis, robot pool, USE_GMASK={PX.USE_GMASK}) ===", flush=True)
    ren = PX.train_renderer(r_fr, r_tr, r_ef, r_vs, r_jt, pool if not SMOKE else pool[:120])

    mtr = np.concatenate([r_tr, h_tr]); mef = np.concatenate([r_ef, h_ef]); mvs = np.concatenate([r_vs, h_vs])
    hi = np.arange(Nr, Nr + len(h_tr))
    sub = pool[np.random.default_rng(100).choice(len(pool), min(N_ROB, len(pool)), replace=False)]
    rit = torch.cat([torch.from_numpy(sub), torch.from_numpy(hi)])
    print("=== train ② world / agent-frame (mixed rh) ===", flush=True)
    wm_w = X.train_one(mtr, mvs, mef, rit, False)
    wm_a = X.train_one(mtr, mvs, mef, rit, True)

    gt = r_tr[ho][:, K:K + H]
    mot = np.linalg.norm(np.diff(gt.mean(2), axis=1), axis=-1).sum(1)
    chosen = list(np.argsort(-mot)[:(NSEQ + 1) // 2]) + list(np.argsort(mot)[:NSEQ // 2])
    chosen = list(dict.fromkeys(int(s) for s in chosen))[:NSEQ]
    idx = ho[chosen]
    pw = rollout_world(wm_w, r_tr, r_ef, idx, False)
    pa = rollout_world(wm_a, r_tr, r_ef, idx, True)
    gt_obj = r_tr[idx][:, K:K + H]

    lines = ["SCEL ②->③ rendered cube pos err px@128 (+det rate) | GT-flow / world-② / agentframe-② | same ③",
             f"{'seq':>4} | {'GTflow':>13} | {'world②':>13} | {'agentframe②':>13}   (err / det)"]
    G, W_, Aa = [], [], []
    for si, s in enumerate(chosen):
        I0 = r_fr[idx[si], 0]; ef_seq = r_ef[idx[si], K:K + H]; vis_seq = r_vs[idx[si], K:K + H]
        jt_seq = r_jt[idx[si], K:K + H]
        rn_g = PX.render_seq(ren, I0, r_tr[idx[si], 0], r_ef[idx[si], 0], gt_obj[si], ef_seq, vis_seq, jt_seq)
        rn_w = PX.render_seq(ren, I0, r_tr[idx[si], 0], r_ef[idx[si], 0], pw[si], ef_seq, vis_seq, jt_seq)
        rn_a = PX.render_seq(ren, I0, r_tr[idx[si], 0], r_ef[idx[si], 0], pa[si], ef_seq, vis_seq, jt_seq)
        gc = gt_obj[si].mean(1)
        eg, dg = agg(rn_g, gc); ew, dw = agg(rn_w, gc); ea, da = agg(rn_a, gc)
        G.append(eg); W_.append(ew); Aa.append(ea)
        lines.append(f"{s:>4} | {eg:6.1f}/{dg:4.2f} | {ew:6.1f}/{dw:4.2f} | {ea:6.1f}/{da:4.2f}")
        print(lines[-1], flush=True)
        gif = []; pad = np.full((IMG, 4, 3), 255, np.uint8)
        for h in range(H):
            panel = np.concatenate([r_fr[idx[si], K + h].astype(np.uint8), pad,
                                    (np.clip(rn_g[h], 0, 1) * 255).astype(np.uint8), pad,
                                    (np.clip(rn_w[h], 0, 1) * 255).astype(np.uint8), pad,
                                    (np.clip(rn_a[h], 0, 1) * 255).astype(np.uint8)], 1)
            im = Image.fromarray(panel); dr = ImageDraw.Draw(im)
            for x, t in [(2, "GT"), (IMG + 6, "GTflow3"), (2 * IMG + 10, "world2"), (3 * IMG + 14, "agentfr2")]:
                dr.text((x, 2), t, fill=(255, 255, 0))
            gif.append(im)
        gif[0].save(f"{OUT}/gifs/seq{s}.gif", save_all=True, append_images=gif[1:], duration=180, loop=0)
    G, W_, Aa = map(np.array, (G, W_, Aa))
    lines += ["", f"MEAN rendered cube pos err px@128: GTflow {G.mean():.1f} | world {W_.mean():.1f} | agentframe {Aa.mean():.1f}",
              "GTflow = ③ ceiling; world vs agentframe = whether ②'s few-px ADE gap is VISIBLE in pixels."]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
