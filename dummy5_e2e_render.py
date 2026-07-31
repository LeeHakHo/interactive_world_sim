"""dummy5②→grip③ 完整 e2e replay(用户昨晚要的"dummy5版本").
① dummy5 ② rollout(真eef驱动)→ pred flow; 用 pred flow 替换 grip cond 的flow通道(grip-skel+warp保留)→ grip③ 渲染。
列 = GT | grip③(GT-flow天花板) | dummy5②→grip③(e2e). 256全分辨率. 对照 mp②→ro_80k 见 render_ablation。"""
import os, sys, numpy as np, torch, cv2, imageio
os.environ.setdefault("HF_HUB_OFFLINE", "1"); sys.path.insert(0, ".")
import exp_scel_dualview_wm as W
import exp_scel_dualview_dit as DIT
from eval_mh_render import sample, gt256
from video_multihead_wm import MultiHeadVideoWM, AuxHead
from wan_vae import WanVAE
for _c in [W.DualLWC, MultiHeadVideoWM, AuxHead]:
    setattr(sys.modules["__main__"], _c.__name__, _c)
dev = "cuda"; TLCAP = 6; NS = 20; GRID = 16; POOL = 128 // GRID
def pool16(x): return torch.nn.functional.avg_pool2d(torch.from_numpy(x[None]).float(), POOL)[0].numpy()
LAT = np.load("outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz")
lat, lat_low = LAT["lat"][:, :, :TLCAP], LAT["lat_low"][:, :, :TLCAP]
DS = np.load("outputs/flow_render_dataset_can_dual/clips_robot.npz")
GRIPCOND = np.load("outputs/video_arch_wm/cond_can_dual/cond_skel_grip_retrack.npz")["cond"]  # (N,2,7,tL,16,16) GTflow+grip-skel+warp
vae = WanVAE(device=dev)
wm = torch.load("outputs/cross_embodiment_wm/epsplit_L48/dummy5_rh_all/wm_dual.pt", map_location=dev, weights_only=False).eval()
m3 = torch.load("outputs/video_arch_wm/epsplit_L48_mh/grip_ro/mh_ema.pt", map_location=dev, weights_only=False).eval()
K, F = W.K, W.F; P = 48; L = 48
efA_full, efB_full = W.load_action_tokens("dummy5", "r", DS)
u8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
OUT = "outputs/video_arch_wm/dummy5_e2e_render"; os.makedirs(f"{OUT}/gifs", exist_ok=True)
SEQS = [int(x) for x in os.environ.get("SEQS", "332,418").split(",")]

for si in SEQS:
    vid = int(DS["vid"][si]); fidx = DS["fidx"][si]
    trg = [np.nan_to_num(DS["tracks"][si].astype(np.float32)), np.nan_to_num(DS["tracks_low"][si].astype(np.float32))]
    efg = [np.nan_to_num(DS["eef"][si].astype(np.float32)), np.nan_to_num(DS["eef_low"][si].astype(np.float32))]
    vsg = [np.nan_to_num(DS["vis"][si].astype(np.float32)), np.nan_to_num(DS["vis_low"][si].astype(np.float32))]
    trD = np.concatenate([trg[0][:K], trg[1][:K]], 1)[None]
    with torch.no_grad():
        pred = W.rollout_dual(wm, torch.from_numpy(trD).float().to(dev),
                              torch.from_numpy(efA_full[si][None]).float().to(dev),
                              torch.from_numpy(efB_full[si][None]).float().to(dev), L - K)[0].cpu().numpy()
    predtr = [np.concatenate([trg[v][:K], pred[:, v*P:(v+1)*P]], 0) for v in range(2)]
    za = torch.from_numpy(np.stack([lat[si], lat_low[si]])[None, :, :, :1].astype(np.float32)).to(dev)
    tL = GRIPCOND.shape[3]
    outs = {}
    for tag, use_pred in [("gtflow", False), ("e2e", True)]:
        cond = GRIPCOND[si].copy()[:, :, :TLCAP]      # (2,7,TLCAP,16,16): GTflow+grip-skel+warp
        if use_pred:
            for k in range(TLCAP):
                rf = 0 if k == 0 else min(4*k, L-1)
                for v in range(2):
                    fc = DIT.flow_cond(predtr[v][0], predtr[v][rf], efg[v][0], efg[v][rf], vsg[v][rf])  # (3,128,128)
                    cond[v, :3, k] = pool16(fc)      # 只换flow通道; grip-skel(3)+warp(4:7)保留GT
        c = torch.from_numpy(cond[None].astype(np.float32)).to(dev)
        with torch.no_grad():
            xs = sample(m3, za, c, NS)
            outs[tag] = vae.decode(xs[:, 0])[0].permute(1, 2, 3, 0).cpu().numpy()  # cam_high (Tp,256,256,3)
        print(f"seq{si} {tag} done", flush=True)
    Tp = min(outs["gtflow"].shape[0], outs["e2e"].shape[0], 40)
    gt = gt256(vid, fidx[:Tp], 0).astype(np.float32) / 255.
    cols = [("GT", gt), ("grip3 GTflow", outs["gtflow"]), ("dummy5->grip3 e2e", outs["e2e"])]
    frames = []
    for t in range(Tp):
        row = np.concatenate([u8(c[1][t]) for c in cols], 1)
        bar = np.ones((18, row.shape[1], 3), np.uint8) * 255
        for i, (lab, _) in enumerate(cols):
            cv2.putText(bar, lab, (i*256+3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,150), 1)
        frames.append(np.concatenate([bar, row], 0))
    imageio.mimsave(f"{OUT}/gifs/dummy5_e2e_seq{si}.gif", frames, fps=6)
    print(f"=== saved {OUT}/gifs/dummy5_e2e_seq{si}.gif ===", flush=True)
print("DONE")
