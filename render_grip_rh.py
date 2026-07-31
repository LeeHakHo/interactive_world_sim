"""mp② → grip_ro③(robot-only) vs grip_rh③(robot+human) e2e, 测human帮不帮grip渲染器.
cube_px + LPIPS, 16 heldout seq × 双视角."""
import os, sys, numpy as np, torch, cv2, lpips, collections
os.environ.setdefault("HF_HUB_OFFLINE", "1"); sys.path.insert(0, ".")
import exp_scel_dualview_wm as W
import exp_scel_dualview_dit as DIT
from eval_mh_render import sample, gt256, detect_can
from video_multihead_wm import MultiHeadVideoWM, AuxHead
from wan_vae import WanVAE
for _c in [W.DualLWC, MultiHeadVideoWM, AuxHead]:
    setattr(sys.modules["__main__"], _c.__name__, _c)
dev = "cuda"; TLCAP = 6; NS = 20; POOL = 8
def pool16(x): return torch.nn.functional.avg_pool2d(torch.from_numpy(x[None]).float(), POOL)[0].numpy()
LAT = np.load("outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz")
lat, lat_low = LAT["lat"][:, :, :TLCAP], LAT["lat_low"][:, :, :TLCAP]
DS = np.load("outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz")   # ★retrack口径
CG = np.load("outputs/video_arch_wm/cond_can_dual/cond_skel_grip_retrack.npz")["cond"]
vae = WanVAE(device=dev); lp = lpips.LPIPS(net="alex").to(dev).eval()
wm = torch.load("outputs/cross_embodiment_wm/epsplit_L48/mp_rh_all/wm_dual.pt", map_location=dev, weights_only=False).eval()
M = {"grip_ro": torch.load("outputs/video_arch_wm/epsplit_L48_mh/grip_ro/mh_ema.pt", map_location=dev, weights_only=False).eval(),
     "grip_rh": torch.load("outputs/video_arch_wm/epsplit_L48_mh/grip_rh/mh_ema.pt", map_location=dev, weights_only=False).eval()}
eA, eB = W.load_action_tokens("mp", "r", DS)
K, P, L = W.K, 48, 48
u8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
SEQS = [int(x) for x in os.environ.get("SEQS", "340,412,50,426,54,376,423,330,41,14,64,371,10,48,369,312").split(",")]
agg = collections.defaultdict(lambda: [[], []])
for si in SEQS:
    vid = int(DS["vid"][si]); fidx = DS["fidx"][si]
    trg = [np.nan_to_num(DS["tracks"][si].astype(np.float32)), np.nan_to_num(DS["tracks_low"][si].astype(np.float32))]
    efg = [np.nan_to_num(DS["eef"][si].astype(np.float32)), np.nan_to_num(DS["eef_low"][si].astype(np.float32))]
    vsg = [np.nan_to_num(DS["vis"][si].astype(np.float32)), np.nan_to_num(DS["vis_low"][si].astype(np.float32))]
    trD = np.concatenate([trg[0][:K], trg[1][:K]], 1)[None]
    with torch.no_grad():
        pr = W.rollout_dual(wm, torch.from_numpy(trD).float().to(dev), torch.from_numpy(eA[si][None]).float().to(dev),
                            torch.from_numpy(eB[si][None]).float().to(dev), L - K)[0].cpu().numpy()
    predtr = [np.concatenate([trg[v][:K], pr[:, v*P:(v+1)*P]], 0) for v in range(2)]
    for v in range(2):
        cond = CG[si].copy()[:, :, :TLCAP]
        for k in range(TLCAP):
            rf = 0 if k == 0 else min(4*k, L-1)
            for vv in range(2):
                fc = DIT.flow_cond(predtr[vv][0], predtr[vv][rf], efg[vv][0], efg[vv][rf], vsg[vv][rf])
                cond[vv, :3, k] = pool16(fc)
        c = torch.from_numpy(cond[None, :, :7].astype(np.float32)).to(dev)
        za = torch.from_numpy(np.stack([lat[si], lat_low[si]])[None, :, :, :1].astype(np.float32)).to(dev)
        for name, m3 in M.items():
            with torch.no_grad():
                xs = sample(m3, za, c, NS)
                dec = vae.decode(xs[:, v])[0].permute(1, 2, 3, 0).cpu().numpy()
            Tp = min(dec.shape[0], 44)
            gt = gt256(vid, fidx[:Tp], v).astype(np.float32) / 255.
            e = []
            for t in range(Tp):
                gc = detect_can(u8(gt[t])); pc = detect_can(u8(dec[t]))
                if gc and pc: e.append(np.hypot((pc[0]-gc[0])/2, (pc[1]-gc[1])/2))
            if e: agg[name][0].append(float(np.mean(e)))
            a = torch.from_numpy(gt[:Tp]).permute(0,3,1,2).to(dev)*2-1
            b = torch.from_numpy(np.clip(dec[:Tp],0,1)).permute(0,3,1,2).to(dev)*2-1
            agg[name][1].append(lp(a,b).mean().item())
    print(f"seq{si} done", flush=True)
print("\n=== mp→ grip③ human-help (robot-only vs robot+human), e2e, n=32 ===")
for name in ["grip_ro", "grip_rh"]:
    c, l = agg[name]; print(f"{name}: cube_px {np.mean(c):.2f} (n={len(c)})  LPIPS {np.mean(l):.4f}")
c_ro, c_rh = np.mean(agg["grip_ro"][0]), np.mean(agg["grip_rh"][0])
l_ro, l_rh = np.mean(agg["grip_ro"][1]), np.mean(agg["grip_rh"][1])
print(f"★human效应: cube {c_ro:.2f}→{c_rh:.2f} ({'帮' if c_rh<c_ro else '害'} {c_ro-c_rh:+.2f}) | LPIPS {l_ro:.4f}→{l_rh:.4f} ({l_ro-l_rh:+.4f})")
