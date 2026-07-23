"""一个标准 save_combined_gif 把 flow + mask/skel条件 叠加放一起。
列 = GT | mask | skel; Rendered行=渲染; Overlay行= 渲染 + ②pred-flow(红/绿) + agent条件(mask红剪影/skel白线)。
② rollout(GT eef)→pred-flow cond(mask版gmask/skel版骨架)→③各渲染。用 .venv_wan。
"""
import os, sys, numpy as np, torch, cv2
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
from viz_combined import save_combined_gif
for _c in [W.DualLWC, DC.DualCombLWC, DIT.DualViewDiT, G.DualViewDiTG, VideoDiT]:
    setattr(sys.modules["__main__"], _c.__name__, _c)

DS = "outputs/flow_render_dataset_can_dual/clips_robot.npz"
SK = "outputs/flow_render_dataset_can_dual/skel_sidecar_robot.npz"
CK_MASK = "outputs/video_arch_wm/m4_video_dit/video_dit_ema.pt"
CK_SKEL = "outputs/video_arch_wm/m4_ablB_skel/video_dit_ema.pt"
WM = "outputs/cross_embodiment_wm/abs_vs_rel_humanhelps/wm_dummy5_rh_N3000.pt"
OUT = "outputs/video_arch_wm/m5_e2e_combined"; os.makedirs(f"{OUT}/gifs", exist_ok=True)
SEQS = [int(x) for x in os.environ.get("SEQS", "332,418,442").split(",")]
NS = 20; dev = "cuda"; GRID = 16; POOL = 8; tL = 12; L = 48; IMG = 128
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
        if i in need: got[i] = cv2.resize(fr.to_ndarray(format="rgb24")[y:y+h, x:x+w], (256, 256), interpolation=cv2.INTER_AREA)
        if i > max(need): break
    c.close()
    return np.stack([got[int(i)] for i in fidx])


def ov(bg, gtobj, predobj, eef, gm=None, sk=None):
    """bg(128,3)uint8 -> 叠 flow(绿GT/红②pred点)+ agent条件(gm红剪影 or sk白线)+ eef黄。"""
    im = bg.copy(); s = IMG
    if gm is not None:
        m = cv2.resize(gm, (s, s)) > 0.4; im[m] = (0.5*im[m] + 0.5*np.array([255,60,60])).astype(np.uint8)
    if sk is not None:
        P = (sk*s).astype(int)
        for a, b in _SEG:
            if np.all(np.abs(sk[a])<3) and np.all(np.abs(sk[b])<3): cv2.line(im, tuple(P[a]), tuple(P[b]), (255,255,255), 1, cv2.LINE_AA)
        for i, p in enumerate(P):
            if np.all(np.abs(sk[i])<3): cv2.circle(im, tuple(p), 2, COL[i], -1)
    if gtobj is not None:
        for p in gtobj: cv2.circle(im, tuple((p*s).astype(int)), 1, (0,255,0), -1)
    if predobj is not None:
        for p in predobj: cv2.circle(im, tuple((p*s).astype(int)), 2, (255,60,60), -1)
    for p in eef: cv2.circle(im, tuple((p*s).astype(int)), 3, (255,230,0), -1)
    return im


def main():
    z = np.load(DS)
    m3M = torch.load(CK_MASK, map_location=dev, weights_only=False).eval()
    m3S = torch.load(CK_SKEL, map_location=dev, weights_only=False).eval()
    wm = torch.load(WM, map_location=dev, weights_only=False).eval()
    vae = WanVAE(device=dev)
    if not HP._G: HP.load_gmask()
    G.load_gmask_low()
    K = W.K; P = 48; u8 = lambda a: (np.clip(a,0,1)*255).astype(np.uint8); r128 = lambda im: cv2.resize(im, (IMG,IMG))
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
        rends = {}
        for tag, model, agent in [("mask", m3M, "gmask"), ("skel", m3S, "skel")]:
            cond = np.zeros((2, 7, tL, GRID, GRID), np.float32)
            for k in range(tL):
                rf = 0 if k == 0 else min(4*k, L-1)
                for v in range(2): cond[v, :, k] = cond_v(v, trg[v][0], predtr[v][rf], efg[v][0], efg[v][rf], vsg[v][rf], jt[rf], frs[v][0], sk[v][rf], agent)
            xs = sample(model, za, torch.from_numpy(cond[None]).float().to(dev))
            rends[tag] = [vae.decode(xs[:, v])[0].permute(1, 2, 3, 0).cpu().numpy() for v in range(2)]
        gmv = [np.stack((HP.gmask_imgs(jt, efg[0]) if v == 0 else G.gmask_low_imgs(jt, efg[1]))) for v in range(2)]
        for v in range(2):
            Tp = min(rends["mask"][v].shape[0], rends["skel"][v].shape[0])
            gtpx = gt256(vid, fidx[:Tp], v).astype(np.float32)/255.
            mR = np.stack([r128(u8(rends["mask"][v][t])) for t in range(Tp)])
            sR = np.stack([r128(u8(rends["skel"][v][t])) for t in range(Tp)])
            gR = np.stack([r128(u8(gtpx[t])) for t in range(Tp)])
            render_seq = np.stack([gR, mR, sR])                                     # (3,Tp,128,128,3)
            # overlay 行: GT(仅GT-flow绿) | mask(flow+mask剪影) | skel(flow+skel线)
            fl = np.stack([
                np.stack([ov(gR[t], trg[v][min(t,L-1)], None, efg[v][min(t,L-1)]) for t in range(Tp)]),
                np.stack([ov(mR[t], trg[v][min(t,L-1)], predtr[v][min(t,L-1)], efg[v][min(t,L-1)], gm=gmv[v][min(t,L-1)]) for t in range(Tp)]),
                np.stack([ov(sR[t], trg[v][min(t,L-1)], predtr[v][min(t,L-1)], efg[v][min(t,L-1)], sk=sk[v][min(t,L-1)]) for t in range(Tp)])])
            save_combined_gif(f"{OUT}/gifs/seq{si}_cam{'high' if v==0 else 'low'}.gif", render_seq, fl,
                              ["GT", "mask", "skel"], [None, None, None], 0,
                              caption=f"seq{si} cam{'high' if v==0 else 'low'} | e2e: overlay=②pred-flow(红)+agent条件(mask红剪影/skel白线)+GT(绿)")
            print(f"seq{si} cam{'high' if v==0 else 'low'} done", flush=True)
    open(f"{OUT}/README.txt", "w").write("标准save_combined_gif: 列GT|mask|skel; Rendered行+Overlay行(②pred-flow红+GT绿+agent条件[mask红剪影/skel白线]+eef黄)。flow与条件叠加放一起。\n")
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
