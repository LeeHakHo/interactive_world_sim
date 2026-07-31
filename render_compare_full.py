"""综合对比 gif(用户要:H拉长+flow overlay+GT-flow列+双视角+多seq+metrics).
5列: GT | grip③-GTflow(天花板) | mp→ro_80k | mp→grip③ | dummy5→grip③. 双视角各出一gif.
Rendered行(③输出)+ Flow overlay行(绿GT/红②pred/黄eef). 每列sub-label=cube_px. metrics落summary."""
import os, sys, numpy as np, torch, cv2, imageio, lpips
os.environ.setdefault("HF_HUB_OFFLINE", "1"); sys.path.insert(0, ".")
import exp_scel_dualview_wm as W
import exp_scel_dualview_dit as DIT
from eval_mh_render import sample, gt256, detect_can
from video_multihead_wm import MultiHeadVideoWM, AuxHead
from wan_vae import WanVAE
from viz_combined import build_flow_cols, compose_frame
for _c in [W.DualLWC, MultiHeadVideoWM, AuxHead]:
    setattr(sys.modules["__main__"], _c.__name__, _c)
dev = "cuda"; TLCAP = int(os.environ.get("TLCAP", "12")); NS = 20; GRID = 16; POOL = 8
def pool16(x): return torch.nn.functional.avg_pool2d(torch.from_numpy(x[None]).float(), POOL)[0].numpy()
LAT = np.load("outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz")
lat, lat_low = LAT["lat"][:, :, :TLCAP], LAT["lat_low"][:, :, :TLCAP]
DS = np.load("outputs/flow_render_dataset_can_dual/clips_robot.npz")
CG = np.load("outputs/video_arch_wm/cond_can_dual/cond_skel_grip_retrack.npz")["cond"]      # 7ch flow+grip-skel+warp
CA = np.load("outputs/video_arch_wm/cond_can_dual/cond_skel_all_retrack.npz")["cond"]        # 7ch flow+skel+warp
vae = WanVAE(device=dev); lp = lpips.LPIPS(net="alex").to(dev).eval()
WMS = {"mp": torch.load("outputs/cross_embodiment_wm/epsplit_L48/mp_rh_all/wm_dual.pt", map_location=dev, weights_only=False).eval(),
       "dummy5": torch.load("outputs/cross_embodiment_wm/epsplit_L48/dummy5_rh_all/wm_dual.pt", map_location=dev, weights_only=False).eval()}
M_ro = torch.load("outputs/video_arch_wm/epsplit_L48_mh/ro_80k/mh_ema.pt", map_location=dev, weights_only=False).eval()
M_grip = torch.load("outputs/video_arch_wm/epsplit_L48_mh/grip_ro/mh_ema.pt", map_location=dev, weights_only=False).eval()
EF = {a: W.load_action_tokens(a, "r", DS) for a in WMS}
K, P, L = W.K, 48, 48
u8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
r128 = lambda im: cv2.resize(im, (128, 128))
OUT = os.environ.get("OUTDIR","outputs/video_arch_wm/COMPARE_full"); os.makedirs(f"{OUT}/gifs", exist_ok=True)
SEQS = [int(x) for x in os.environ.get("SEQS", "332,418,59,442").split(",")]

def rollout(action, si):
    trg = [np.nan_to_num(DS["tracks"][si].astype(np.float32)), np.nan_to_num(DS["tracks_low"][si].astype(np.float32))]
    trD = np.concatenate([trg[0][:K], trg[1][:K]], 1)[None]
    eA, eB = EF[action]
    with torch.no_grad():
        pr = W.rollout_dual(WMS[action], torch.from_numpy(trD).float().to(dev),
                            torch.from_numpy(eA[si][None]).float().to(dev),
                            torch.from_numpy(eB[si][None]).float().to(dev), L - K)[0].cpu().numpy()
    return [np.concatenate([trg[v][:K], pr[:, v*P:(v+1)*P]], 0) for v in range(2)]

def render(m3, condfull, si, v, pred=None, efg=None, vsg=None):
    cond = condfull[si].copy()[:, :, :TLCAP]  # (2,7,TLCAP,16,16)
    if pred is not None:                       # e2e: 换flow通道
        for k in range(TLCAP):
            rf = 0 if k == 0 else min(4*k, L-1)
            for vv in range(2):
                fc = DIT.flow_cond(pred[vv][0], pred[vv][rf], efg[vv][0], efg[vv][rf], vsg[vv][rf])
                cond[vv, :3, k] = pool16(fc)
    C = 4 if m3 is M_ro else 7
    c = torch.from_numpy(cond[None, :, :C].astype(np.float32)).to(dev)   # 切channel(dim2)非tL
    za = torch.from_numpy(np.stack([lat[si], lat_low[si]])[None, :, :, :1].astype(np.float32)).to(dev)
    with torch.no_grad():
        xs = sample(m3, za, c, NS)
        return vae.decode(xs[:, v])[0].permute(1, 2, 3, 0).cpu().numpy()

summ = ["综合对比 metrics (cube_px=检测帧render罐位vs GT罐位, LPIPS含agent) | 双视角 held-out\n"]
for si in SEQS:
    vid = int(DS["vid"][si]); fidx = DS["fidx"][si]
    trg = [np.nan_to_num(DS["tracks"][si].astype(np.float32)), np.nan_to_num(DS["tracks_low"][si].astype(np.float32))]
    efg = [np.nan_to_num(DS["eef"][si].astype(np.float32)), np.nan_to_num(DS["eef_low"][si].astype(np.float32))]
    vsg = [np.nan_to_num(DS["vis"][si].astype(np.float32)), np.nan_to_num(DS["vis_low"][si].astype(np.float32))]
    pr_mp = rollout("mp", si); pr_d5 = rollout("dummy5", si)
    for v in range(2):
        # 5列渲染
        cols = {
            "grip3-GTflow(ceil)": render(M_grip, CG, si, v),
            "mp->ro_80k":         render(M_ro, CA, si, v, pr_mp, efg, vsg),
            "mp->grip3":          render(M_grip, CG, si, v, pr_mp, efg, vsg),
            "dummy5->grip3":      render(M_grip, CG, si, v, pr_d5, efg, vsg),
        }
        Tp = min(min(r.shape[0] for r in cols.values()), 44)
        gt = gt256(vid, fidx[:Tp], v).astype(np.float32) / 255.
        gt128 = np.stack([r128(u8(gt[t])) for t in range(Tp)])
        # metrics: cube_px + LPIPS 每列
        errs = [None]  # GT列无
        titles = ["GT"] + list(cols.keys())
        render128 = [gt128]
        for lab, dec in cols.items():
            d128 = np.stack([r128(u8(dec[t])) for t in range(Tp)])
            render128.append(d128)
            # cube_px
            e = []
            for t in range(Tp):
                gc = detect_can(u8(gt[t])); pc = detect_can(u8(dec[t]))
                if gc and pc: e.append(np.hypot((pc[0]-gc[0])/2, (pc[1]-gc[1])/2))  # /2: 256→128尺度
            cpx = float(np.mean(e)) if e else float("nan")
            errs.append(cpx)
            a = torch.from_numpy(gt[:Tp]).permute(0,3,1,2).to(dev)*2-1
            b = torch.from_numpy(np.clip(dec[:Tp],0,1)).permute(0,3,1,2).to(dev)*2-1
            summ.append(f"seq{si} cam{'high' if v==0 else 'low'} {lab:20s}: cube_px {cpx:.1f}  LPIPS {lp(a,b).mean().item():.4f}")
        # flow overlay: GT列=绿only; ceiling=绿only(GT flow); mp列=红mp cube; dummy5列=红d5 cube
        gt_obj = trg[v][:Tp]; eef_v = efg[v][:Tp]
        preds = [None, None, pr_mp[v][:Tp], pr_mp[v][:Tp], pr_d5[v][:Tp]]
        flow_all = build_flow_cols(gt128, gt_obj, preds, eef_v)  # (5,Tp,128,128,3)
        frames = []
        for t in range(Tp):
            fr = compose_frame([render128[c][t] for c in range(5)], [flow_all[c][t] for c in range(5)],
                               titles, errs, t, caption=f"seq{si} cam{'high' if v==0 else 'low'} | 绿GT红②pred黄eef | sub=cube_px")
            frames.append(np.array(fr))
        out = f"{OUT}/gifs/compare_seq{si}_cam{'high' if v==0 else 'low'}.gif"
        imageio.mimsave(out, frames, fps=6)
        print(f"=== saved {out} ({Tp}帧) ===", flush=True)
open(f"{OUT}/summary.txt", "w").write("\n".join(summ) + "\n")
print("DONE\n" + "\n".join(summ[-10:]))
