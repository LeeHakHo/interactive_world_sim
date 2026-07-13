"""端到端:② pred-flow → dual-view DiT ③(vs GT-flow 天花板 vs naive eef-cond)。

补 dual-view 正式化 M2(spec 2026-07-12-dualview-dit-formalize):③ 此前只吃 replay GT
flow;本脚本用 ②(DualLWC/DualCombLWC,可换 combine 机制 ckpt)rollout 的预测 tracks
构 flow cond 驱动 ③,四列对照隔离 ②/③ 误差:
  GT | GT-flow→③(③天花板) | ②pred-flow→③(端到端) | eef-cond ③(naive)
数据 can_dual L24(=②正当 heldout,K=4+H=20 恰满 24 帧;窗口级重叠为项目既有协议)。
额外报 ② flow ADE px 作中间诊断。Env: WM2(② ckpt)/NSEQ/OUT。iws env,GPU。
Output: outputs/cross_embodiment_wm/dualview_e2e/"""
import os, sys
os.environ.setdefault("VAE_NAME", "ostris/vae-kl-f8-d16"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch

os.chdir("/scr2/yusenluo/interactive_world_sim")
import exp_scel_dualview_wm as W
import exp_scel_dualview_comb as DC
import exp_scel_dualview_dit as DIT
from exp_scel_dualview_dit import flow_cond, DualViewDiT, enc, dec, obj_lpips, psnr, _fp, u8
from viz_combined import save_combined_gif, build_flow_cols

device = "cuda" if torch.cuda.is_available() else "cpu"
K = 4; H = 20; IMG = 128; HELDOUT = 150
DS = "outputs/flow_render_dataset_can_dual"
WM2 = os.environ.get("WM2", "outputs/cross_embodiment_wm/dualview_wm/wm_dual.pt")
NSEQ = int(os.environ.get("NSEQ", "6"))
OUT = os.environ.get("OUT", "outputs/cross_embodiment_wm/dualview_e2e"); os.makedirs(f"{OUT}/gifs", exist_ok=True)

# 旧 ckpt 从各脚本 __main__ 整存 -> 注入类名供 unpickle
_m = sys.modules["__main__"]
for cls in [W.DualLWC, DC.DualCombLWC, DualViewDiT]:
    setattr(_m, cls.__name__, cls)


def pred_flow_cond(tr0, pred_t, ef0, eft, vis_t):
    """②预测 tracks 构 ③ flow cond。与 DIT.flow_cond 同一实现,trt=pred(footprint 也随 pred)。"""
    return flow_cond(tr0, pred_t, ef0, eft, vis_t)


def load_l24():
    z = np.load(f"{DS}/clips_robot_L24.npz")
    f32 = lambda k, nan=0.0: np.nan_to_num(z[k].astype(np.float32), nan=nan)
    return {"fr": [z["frames"], z["frames_low"]],
            "tr": [f32("tracks"), f32("tracks_low", 0.5)],
            "ef": [f32("eef"), f32("eef_low", 0.5)],
            "vs": [z["vis"].astype(np.float32), z["vis_low"].astype(np.float32)],
            "ok": z["low_valid"]}


@torch.no_grad()
def rollout2(wm2, R, seqs):
    """② rollout H 步 -> pred tracks (n,H,2P,2) 帧 K..K+H-1 + 每视角 ADE px。"""
    trD = np.concatenate([R["tr"][0][seqs], R["tr"][1][seqs]], 2)
    efA = torch.from_numpy(R["ef"][0][seqs]).float().to(device)
    efB = torch.from_numpy(R["ef"][1][seqs]).float().to(device)
    pr = W.rollout_dual(wm2, torch.from_numpy(trD).float().to(device), efA, efB, H).cpu().numpy()
    P = trD.shape[2] // 2
    gt = trD[:, K:K + H]
    adeH = float((np.linalg.norm(pr[:, :, :P] - gt[:, :, :P], axis=-1) * IMG).mean())
    adeL = float((np.linalg.norm(pr[:, :, P:] - gt[:, :, P:], axis=-1) * IMG).mean())
    return pr, adeH, adeL


@torch.no_grad()
def render3(mf, R, si, predtr=None):
    """③ 渲染 H 帧双视角。predtr None=replay GT flow;否则 (H,2P,2) 用②预测。"""
    DIT.COND = "flow"
    fr01 = lambda a: torch.from_numpy(a.astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to(device)
    z0 = torch.stack([enc(fr01(R["fr"][v][si, 0]))[0] for v in range(2)])[None]
    prev = z0.clone(); outs = [[], []]
    P = R["tr"][0].shape[2]
    for h in range(H):
        conds = []
        for v in range(2):
            trt = (R["tr"][v][si, K + h] if predtr is None else
                   predtr[h, v * P:(v + 1) * P])
            conds.append(pred_flow_cond(R["tr"][v][si, 0], trt, R["ef"][v][si, 0],
                                        R["ef"][v][si, K + h], R["vs"][v][si, K + h]))
        cond = torch.from_numpy(np.stack(conds)[None].astype(np.float32)).to(device)
        pred = mf(z0, prev, cond); prev = pred
        for v in range(2): outs[v].append(dec(pred[:, v])[0].cpu().numpy())
    return np.stack([np.stack(outs[v]).transpose(0, 2, 3, 1) for v in range(2)])


@torch.no_grad()
def render_eef(me, R, si):
    DIT.COND = "eef"
    fr01 = lambda a: torch.from_numpy(a.astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to(device)
    z0 = torch.stack([enc(fr01(R["fr"][v][si, 0]))[0] for v in range(2)])[None]
    prev = z0.clone(); outs = [[], []]
    for h in range(H):
        conds = [DIT.eef_point(R["ef"][v][si, 0], R["ef"][v][si, K + h]) for v in range(2)]
        cond = torch.from_numpy(np.stack(conds)[None].astype(np.float32)).to(device)
        pred = me(z0, prev, cond); prev = pred
        for v in range(2): outs[v].append(dec(pred[:, v])[0].cpu().numpy())
    return np.stack([np.stack(outs[v]).transpose(0, 2, 3, 1) for v in range(2)])


def main():
    R = load_l24()
    Nc = len(R["ok"])
    perm = np.random.default_rng(0).permutation(Nc)
    ho = np.array([i for i in perm[:HELDOUT] if R["ok"][i]])          # 与 ② sweep 同 split
    mf = torch.load("outputs/cross_embodiment_wm/dualview_dit/flow/dvdit.pt",
                    map_location=device, weights_only=False).eval()
    me = torch.load("outputs/cross_embodiment_wm/dualview_dit/eef/dvdit.pt",
                    map_location=device, weights_only=False).eval()
    wm2 = torch.load(WM2, map_location=device, weights_only=False).eval()
    print(f"WM2={WM2} ({type(wm2).__name__}"
          f"{',' + getattr(wm2, 'combine', '?') if hasattr(wm2, 'combine') else ''})", flush=True)
    mot = np.array([np.linalg.norm(np.diff(R["tr"][0][si, K:K + H].mean(1), axis=0), axis=-1).sum() for si in ho])
    chosen = ho[np.argsort(-mot)[:NSEQ]]
    pr, adeH, adeL = rollout2(wm2, R, chosen)
    print(f"② flow ADE px@128: cam_high {adeH:.2f} | cam_low {adeL:.2f}", flush=True)

    from interactive_world_sim.algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device).eval()
    cols_name = ["GT", "GT-flow→③", "②pred-flow→③", "eef-cond(naive)"]
    agg = {f"v{v}_{c}_{k}": [] for v in range(2) for c in ["gtf", "e2e", "eef"] for k in ["ps", "lp"]}
    vname = {0: "high", 1: "low"}
    os.makedirs(f"{OUT}/render_cache", exist_ok=True)
    np.save(f"{OUT}/render_cache/chosen.npy", chosen)
    np.save(f"{OUT}/render_cache/predtr.npy", pr)
    for n, si in enumerate(chosen):
        si = int(si)
        r_gtf = render3(mf, R, si)                              # ③ 天花板
        r_e2e = render3(mf, R, si, predtr=pr[n])                # 端到端
        r_eef = render_eef(me, R, si)                           # naive
        for cname, rr in [("gtf", r_gtf), ("e2e", r_e2e), ("eef", r_eef)]:
            np.save(f"{OUT}/render_cache/seq{si}_{cname}.npy", u8(rr))   # 跨 run 合成用
        for v in range(2):
            gtf = R["fr"][v][si, K:K + H].astype(np.float32) / 255.0
            objm = np.stack([_fp(R["tr"][v][si, K + h]) for h in range(H)])
            for cname, rr in [("gtf", r_gtf), ("e2e", r_e2e), ("eef", r_eef)]:
                agg[f"v{v}_{cname}_ps"].append(np.mean([psnr(rr[v, h], gtf[h]) for h in range(H)]))
                agg[f"v{v}_{cname}_lp"].append(obj_lpips(lp, rr[v].astype(np.float32), gtf, objm))
            gt = R["fr"][v][si, K:K + H].astype(np.uint8)
            cols = np.stack([gt, u8(r_gtf[v]), u8(r_e2e[v]), u8(r_eef[v])])
            gt_obj = R["tr"][v][si, K:K + H]
            pred_obj = pr[n][:, v * 48:(v + 1) * 48]
            fc = build_flow_cols(gt, gt_obj, [None, gt_obj, pred_obj, None], R["ef"][v][si, K:K + H])
            save_combined_gif(f"{OUT}/gifs/seq{si}_cam{vname[v]}.gif", cols, fc, cols_name,
                              [None] * 4, K,
                              caption=f"e2e: ② {os.path.basename(WM2)} → ③ flow-DiT | cam_{vname[v]}")
        print(f"  seq{si} done", flush=True)
    mn = lambda k: float(np.nanmean(agg[k]))
    lines = [f"DUAL-VIEW e2e | ② {WM2} | ③ flow-DiT(30ep 2026-07-07) vs eef-DiT | n={len(chosen)} L24 heldout",
             f"② flow ADE px@128: cam_high {adeH:.2f} | cam_low {adeL:.2f}"]
    for v in range(2):
        lines.append(f"cam_{vname[v]}: " + " | ".join(
            f"{c} PSNR {mn(f'v{v}_{c}_ps'):.2f} objLPIPS {mn(f'v{v}_{c}_lp'):.3f}" for c in ["gtf", "e2e", "eef"]))
    import json
    json.dump({k: float(np.nanmean(vv)) for k, vv in agg.items()} | {"adeH": adeH, "adeL": adeL},
              open(f"{OUT}/metrics.json", "w"), indent=2)
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + f"\ngifs -> {OUT}/gifs/\n=== DONE ===", flush=True)


def compare_runs(dirA, dirB, labelA="②align_wm→③", labelB="②dummy5→③"):
    """两次 e2e run 的渲染缓存 -> 5 列对比 gif:GT | GT-flow→③ | A-pred | B-pred | eef。
    用法:COMPARE=<dirA>,<dirB> python exp_scel_dualview_e2e.py"""
    R = load_l24()
    chosen = np.load(f"{dirA}/render_cache/chosen.npy")
    chB = np.load(f"{dirB}/render_cache/chosen.npy")
    assert (chosen == chB).all(), "两 run 的 seq 选择必须一致"
    prA = np.load(f"{dirA}/render_cache/predtr.npy"); prB = np.load(f"{dirB}/render_cache/predtr.npy")
    OUTC = os.environ.get("OUT", "outputs/cross_embodiment_wm/dualview_e2e/final_compare")
    os.makedirs(f"{OUTC}/gifs", exist_ok=True)
    vname = {0: "high", 1: "low"}
    for n, si in enumerate(chosen):
        si = int(si)
        ld = lambda d, c: np.load(f"{d}/render_cache/seq{si}_{c}.npy")
        gtf, eA, eB, eef = ld(dirA, "gtf"), ld(dirA, "e2e"), ld(dirB, "e2e"), ld(dirA, "eef")
        for v in range(2):
            gt = R["fr"][v][si, K:K + H].astype(np.uint8)
            cols = np.stack([gt, gtf[v], eA[v], eB[v], eef[v]])
            gt_obj = R["tr"][v][si, K:K + H]
            pA = prA[n][:, v * 48:(v + 1) * 48]; pB = prB[n][:, v * 48:(v + 1) * 48]
            fc = build_flow_cols(gt, gt_obj, [None, gt_obj, pA, pB, None], R["ef"][v][si, K:K + H])
            save_combined_gif(f"{OUTC}/gifs/seq{si}_cam{vname[v]}.gif", cols, fc,
                              ["GT", "GT-flow→③", labelA, labelB, "eef-cond(naive)"], [None] * 5, K,
                              caption=f"e2e 机制对比: {labelA} vs {labelB} | cam_{vname[v]}")
    print(f"saved 5-col compare gifs -> {OUTC}/gifs/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    if os.environ.get("COMPARE"):
        compare_runs(*os.environ["COMPARE"].split(","))
    else:
        main()
