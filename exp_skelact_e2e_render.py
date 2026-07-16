"""端到端: ② 骨架动作 human-helps -> flowskel ③ 好渲染器, 渲染 H 帧, 报 obj-LPIPS + agent-LPIPS.
回答用户: "② 的 human-helps 到底有没有让最终画面变好"(而非抽象 drift px / 糊的 latent-residual 解码器).
四臂 ②: {dummy5,skel} x {r-only, +human} @ NROB=100, 各 rollout H 步 pred flow -> 同一个 flowskel ③
渲染器(默认 ep120 最清). 输出 5 列 gif(GT|4臂) + metrics.json. HORIZON env (默认 44).
iws env + GPU. 输出 outputs/cross_embodiment_wm/skelact_e2e_render/.
"""
import os, json
os.environ.setdefault("SPLIT", "okfirst")
os.environ.setdefault("COND", "flowskel")            # 让 formal 以 flowskel 模式 import
import numpy as np, torch, cv2
import __main__ as _m
import exp_scel_dualview_wm as W
_m.DualLWC = W.DualLWC                                # skelact ckpt 由 wm 作 __main__ 保存
import exp_scel_dualview_dit_formal as F
from exp_scel_dualview_dit_formal import (render_formal, obj_lpips_audit, agent_lpips_audit,
                                          load_dual, _fp, u8, K, H, ROOT, device)
from viz_combined import save_combined_gif, build_flow_cols

HR = int(os.environ.get("HORIZON", "44"))
F.H = HR                                              # render_formal 用全局 H
NSEQ = int(os.environ.get("NSEQ", "8"))
REN = os.environ.get("REN", "flowskel_cv1_s0_ep120") # 最清渲染器
OUT = os.environ.get("OUT_DIR", "outputs/cross_embodiment_wm/skelact_e2e_render")
os.makedirs(f"{OUT}/gifs", exist_ok=True)
SKB = "outputs/cross_embodiment_wm/dualview_wm_skelact"
ARMS = [("dummy5_r_n100", "dummy5 r-only", "dummy5"), ("dummy5_rh_n100", "dummy5 +human", "dummy5"),
        ("skel_r_n100", "skel r-only", "skel"), ("skel_rh_n100", "skel +human", "skel")]


def main():
    R, _ = load_dual()
    z = np.load("outputs/flow_render_dataset_can_dual/clips_robot.npz")
    sk_r = np.load("outputs/flow_render_dataset_can_dual/skel_sidecar_robot_v2.npz")
    skel_hl = {0: sk_r["skel2d_high"], 1: sk_r["skel2d_low"]}   # agent-region crop centering
    ok = np.where(R["ok"])[0]; ho = np.random.default_rng(0).permutation(ok)[:150]
    P = R["tr"][0].shape[2]
    trA = z["tracks"].astype(np.float32); trB = np.nan_to_num(z["tracks_low"].astype(np.float32), nan=0.5)
    trD_all = np.concatenate([trA, trB], 2)
    mot = np.array([np.linalg.norm(np.diff(trA[si, K:K + HR].mean(1), axis=0), axis=-1).sum() for si in ho])
    chosen = [int(x) for x in ho[np.argsort(-mot)[:NSEQ]]]

    ren = torch.load(f"{ROOT}/{REN}/dvdit.pt", map_location=device, weights_only=False).eval()
    from interactive_world_sim.algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device).eval()

    wms = {}
    for d, _, act in ARMS:
        wms[d] = torch.load(f"{SKB}/{d}/wm_dual.pt", map_location=device, weights_only=False).eval()

    agg = {d: {f"{k}{v}": [] for v in range(2) for k in ("obj", "agt")} for d, _, _ in ARMS}
    vname = {0: "high", 1: "low"}
    for si in chosen:
        trD = torch.from_numpy(trD_all[si:si + 1]).float().to(device)
        rr = {}
        for d, lab, act in ARMS:
            efA, efB = W.load_action_tokens(act, "r", z, sk_r if act == "skel" else None)
            ea = torch.from_numpy(efA[si:si + 1]).float().to(device); eb = torch.from_numpy(efB[si:si + 1]).float().to(device)
            with torch.no_grad():
                pr = W.rollout_dual(wms[d], trD, ea, eb, HR).cpu().numpy()[0]     # (HR,2P,2)
            pred_tr = np.stack([pr[:, :P], pr[:, P:]])                            # (2,HR,P,2)
            rr[d] = render_formal(ren, R, si, "flowskel", pred_tr=pred_tr)        # (2,HR,128,128,3)
        for v in range(2):
            gtf = R["fr"][v][si, K:K + HR].astype(np.float32) / 255.0
            objm = np.stack([_fp(R["tr"][v][si, K + h]) for h in range(HR)])
            skpts = skel_hl[v][si, K:K + HR]                     # (HR,9,2) 骨架点用于 agent 裁剪中心
            for d, _, _ in ARMS:
                o, _, _ = obj_lpips_audit(lp, rr[d][v].astype(np.float32), gtf, objm)
                a, _, _ = agent_lpips_audit(lp, rr[d][v].astype(np.float32), gtf, skpts)
                agg[d][f"obj{v}"].append(o); agg[d][f"agt{v}"].append(a)
        # gif: GT | 4 臂
        for v in range(2):
            gt = R["fr"][v][si, K:K + HR].astype(np.uint8)
            cols = np.stack([gt] + [u8(rr[d][v]) for d, _, _ in ARMS])
            labels = [f"GT {vname[v]}"] + [lab for _, lab, _ in ARMS]
            fc = build_flow_cols(gt, R["tr"][v][si, K:K + HR], [None] * 5, R["ef"][v][si, K:K + HR])
            save_combined_gif(f"{OUT}/gifs/seq{si}_cam{vname[v]}.gif", cols, fc, labels, [None] * 5, K,
                              caption=f"② human-helps -> flowskel ③ ({REN}) | H={HR} | cam_{vname[v]}")
        print(f"seq{si} done", flush=True)

    res = {}
    for d, lab, _ in ARMS:
        for v in range(2):
            res[f"{d}_obj_v{v}"] = float(np.nanmean(agg[d][f"obj{v}"]))
            res[f"{d}_agt_v{v}"] = float(np.nanmean(agg[d][f"agt{v}"]))
    json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=2)
    lines = [f"② human-helps -> flowskel ③ 端到端渲染 | renderer={REN} | H={HR} | n={len(chosen)}",
             "arm                    obj v0/v1        agent v0/v1"]
    for d, lab, _ in ARMS:
        lines.append(f"{lab:20s}   {res[f'{d}_obj_v0']:.3f}/{res[f'{d}_obj_v1']:.3f}      "
                     f"{res[f'{d}_agt_v0']:.3f}/{res[f'{d}_agt_v1']:.3f}")
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + "\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
