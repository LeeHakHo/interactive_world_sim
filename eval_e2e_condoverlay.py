"""mask/skel agent条件叠在各自e2e渲染上(类比 pred-flow overlay)。
每 seq 每视角 gif, 2行×2列:
  上行 = mask模型e2e渲染 | skel模型e2e渲染 (纯渲染, 参照)
  下行 = mask渲染+mask剪影叠加(红) | skel渲染+skel线画叠加(彩点+白线)
② rollout(GT eef)→pred-flow cond(mask版用gmask, skel版用骨架)→③各渲染。用 .venv_wan。
"""
import os, sys, numpy as np, torch, cv2, imageio
sys.path.insert(0, "."); os.environ.setdefault("RES", "128"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MASKGEN", "outputs/flow_wm/maskgen_caneef/maskgen_caneef.pt")
os.environ.setdefault("GMASK", "1"); os.environ.setdefault("GMASK_LOW", "1"); os.environ.setdefault("WARP", "1")
import exp_scel_dualview_gmaskcond as G
import exp_v3_human_helps_pixels as HP
import exp_scel_dualview_wm as W
import exp_scel_dualview_comb as DC
import exp_scel_dualview_dit as DIT
from video_dit import VideoDiT
from wan_vae import WanVAE
for _c in [W.DualLWC, DC.DualCombLWC, DIT.DualViewDiT, G.DualViewDiTG, VideoDiT]:
    setattr(sys.modules["__main__"], _c.__name__, _c)

DS = "outputs/flow_render_dataset_can_dual/clips_robot.npz"
SK = "outputs/flow_render_dataset_can_dual/skel_sidecar_robot.npz"
CK_MASK = "outputs/video_arch_wm/m4_video_dit/video_dit_ema.pt"
CK_SKEL = "outputs/video_arch_wm/m4_ablB_skel/video_dit_ema.pt"
WM = "outputs/cross_embodiment_wm/abs_vs_rel_humanhelps/wm_dummy5_rh_N3000.pt"
OUT = "outputs/video_arch_wm/m5_e2e_condoverlay"; os.makedirs(f"{OUT}/gifs", exist_ok=True)
SEQS = [int(x) for x in os.environ.get("SEQS", "332,418,442").split(",")]
NS = 20; dev = "cuda"; GRID = 16; POOL = 8; tL = 12; L = 48; S = 256
CROPS = {0: (60, 60, 390, 390), 1: (0, 0, 640, 480)}
VIDP = {v: f"human_play_data/play_robot_can_{{}}_eef/videos/chunk-000/observation.images.cam_{n}/episode_000000.mp4" for v, n in [(0, "high"), (1, "low")]}
_SK = np.load(SK); _SEG = _SK["segments"]
COL = [(255,80,80),(255,160,60),(255,230,60),(150,255,60),(60,255,160),(60,200,255),(120,120,255),(220,100,255),(255,100,180)]


def pool16(x): return torch.nn.functional.avg_pool2d(torch.from_numpy(x[None]).float(), POOL)[0].numpy()


def skel_chan(pts):
    img = np.zeros((128, 128), np.float32); P = (pts * 128).astype(np.int32)
    for a, b in _SEG:
        if np.all(np.abs(pts[a]) < 3) and np.all(np.abs(pts[b]) < 3): cv2.line(img, tuple(P[a]), tuple(P[b]), 1.0, 3, cv2.LINE_AA)
    for i, p in enumerate(P):
        if np.all(np.abs(pts[i]) < 3): cv2.circle(img, tuple(p), 3, 1.0, -1)
    return img


def cond_v(v, tr0, trt, ef0, eft, vis, jt, fr0, sk2d, agent):
    fc = DIT.flow_cond(tr0, trt, ef0, eft, vis)
    if agent == "skel": g = skel_chan(sk2d)
    else:
        g = (HP.gmask_imgs(jt[None], eft[None])[0] if v == 0 else G.gmask_low_imgs(jt[None], eft[None])[0])
        if g.shape[0] != 128: g = cv2.resize(g, (128, 128))
    return pool16(np.concatenate([fc, g[None], G.warp_preview(fr0, tr0, trt, vis)], 0).astype(np.float32))


@torch.no_grad()
def sample(model, za, cond, steps=NS):
    B, V, C, _, g, _ = za.shape; T = cond.shape[3]
    x = torch.randn(B, V, C, T, g, g, device=dev); x[:, :, :, :1] = za
    for i in range(steps):
        t = torch.full((B, T), i / steps, device=dev); t[:, 0] = 1.0
        x = x + model(x, t, cond) / steps; x[:, :, :, :1] = za
    return x


def gt256(vid, fidx, v):
    import av
    x, y, w, h = CROPS[v]; need = set(int(i) for i in fidx); got = {}
    c = av.open(VIDP[v].format(vid - 100 + 1))
    for i, fr in enumerate(c.decode(video=0)):
        if i in need: got[i] = cv2.resize(fr.to_ndarray(format="rgb24")[y:y+h, x:x+w], (S, S), interpolation=cv2.INTER_AREA)
        if i > max(need): break
    c.close()
    return np.stack([got[int(i)] for i in fidx])


def ov_mask(img, gm):                                 # gm 128 -> 叠红半透明在 render 上
    im = img.copy(); g = cv2.resize(gm, (S, S)) > 0.4
    im[g] = (0.45*im[g] + 0.55*np.array([255, 60, 60])).astype(np.uint8)
    return im


def ov_skel(img, pts):                                # 骨架线画叠在 render 上
    im = img.copy(); P = (pts * S).astype(np.int32)
    for a, b in _SEG:
        if np.all(np.abs(pts[a]) < 3) and np.all(np.abs(pts[b]) < 3): cv2.line(im, tuple(P[a]), tuple(P[b]), (255,255,255), 2, cv2.LINE_AA)
    for i, p in enumerate(P):
        if np.all(np.abs(pts[i]) < 3): cv2.circle(im, tuple(p), 3, COL[i], -1, cv2.LINE_AA)
    return im


def main():
    z = np.load(DS)
    m3M = torch.load(CK_MASK, map_location=dev, weights_only=False).eval()
    m3S = torch.load(CK_SKEL, map_location=dev, weights_only=False).eval()
    wm = torch.load(WM, map_location=dev, weights_only=False).eval()
    vae = WanVAE(device=dev)
    if not HP._G: HP.load_gmask()
    G.load_gmask_low()
    K = W.K; P = 48; u8 = lambda a: (np.clip(a, 0, 1)*255).astype(np.uint8)
    lab = lambda im, t: cv2.putText(cv2.copyMakeBorder(im,18,0,0,0,cv2.BORDER_CONSTANT,value=(0,0,0)), t,(3,13),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
    for si in SEQS:
        trg = [np.nan_to_num(z["tracks"][si].astype(np.float32)), np.nan_to_num(z["tracks_low"][si].astype(np.float32))]
        efg = [np.nan_to_num(z["eef"][si].astype(np.float32)), np.nan_to_num(z["eef_low"][si].astype(np.float32))]
        vsg = [np.nan_to_num(z["vis"][si].astype(np.float32)), np.nan_to_num(z["vis_low"][si].astype(np.float32))]
        jt = np.nan_to_num(z["joint"][si].astype(np.float32)); frs = [z["frames"][si], z["frames_low"][si]]
        sk = [_SK["skel2d_high"][si], _SK["skel2d_low"][si]]
        fidx = z["fidx"][si]; vid = int(z["vid"][si])
        trD = np.concatenate([trg[0][:K], trg[1][:K]], 1)[None]
        pr = W.rollout_dual(wm, torch.from_numpy(trD).float().to(dev), torch.from_numpy(efg[0][None]).float().to(dev), torch.from_numpy(efg[1][None]).float().to(dev), L-K)[0].cpu().numpy()
        predtr = [np.concatenate([trg[v][:K], pr[:, v*P:(v+1)*P]], 0) for v in range(2)]
        f0 = [gt256(vid, fidx[:1], v)[0] for v in range(2)]
        za = torch.stack([vae.encode(torch.from_numpy(f0[v].astype(np.float32).transpose(2, 0, 1)[None, :, None]/255.))[0, :, :1] for v in range(2)])[None]
        # 两模型各渲一版
        rends = {}
        for tag, model, agent in [("mask", m3M, "gmask"), ("skel", m3S, "skel")]:
            cond = np.zeros((2, 7, tL, GRID, GRID), np.float32)
            for k in range(tL):
                rf = 0 if k == 0 else min(4*k, L-1)
                for v in range(2): cond[v, :, k] = cond_v(v, trg[v][0], predtr[v][rf], efg[v][0], efg[v][rf], vsg[v][rf], jt[rf], frs[v][0], sk[v][rf], agent)
            xs = sample(model, za, torch.from_numpy(cond[None]).float().to(dev))
            rends[tag] = [vae.decode(xs[:, v])[0].permute(1, 2, 3, 0).cpu().numpy() for v in range(2)]
        # gmask@256 逐帧(批量)供叠加
        gmv = [np.stack((HP.gmask_imgs(jt, efg[0]) if v == 0 else G.gmask_low_imgs(jt, efg[1]))) for v in range(2)]
        for v in range(2):
            Tp = min(rends["mask"][v].shape[0], rends["skel"][v].shape[0])
            frames = []
            for t in range(Tp):
                rf = min(t, L-1)
                mR = u8(rends["mask"][v][t]); sR = u8(rends["skel"][v][t])
                top = np.concatenate([lab(mR, "mask e2e render"), lab(sR, "skel e2e render")], 1)
                bot = np.concatenate([lab(ov_mask(mR, gmv[v][rf]), "mask render + mask(silhouette)"),
                                      lab(ov_skel(sR, sk[v][rf]), "skel render + skel(lines)")], 1)
                frames.append(np.concatenate([top, bot], 0))
            imageio.mimsave(f"{OUT}/gifs/seq{si}_cam{'high' if v==0 else 'low'}.gif", frames, fps=8, loop=0)
            print(f"seq{si} cam{'high' if v==0 else 'low'}: cond-on-e2e overlay gif ({Tp}帧)", flush=True)
    open(f"{OUT}/README.txt", "w").write("mask/skel agent条件叠在各自e2e渲染上。上=纯e2e渲染(mask|skel), 下=渲染+条件叠加(mask红剪影 / skel彩线画)。看条件如何对应渲染出的agent。\n")
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
