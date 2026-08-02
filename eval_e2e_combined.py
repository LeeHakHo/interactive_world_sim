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
from video_multihead_wm import MultiHeadVideoWM, AuxHead     # ★干净L48多头③(4ch cond, warp off)
from wan_vae import WanVAE
from viz_combined import save_combined_gif
for _c in [W.DualLWC, DC.DualCombLWC, DIT.DualViewDiT, G.DualViewDiTG, VideoDiT, MultiHeadVideoWM, AuxHead]:
    setattr(sys.modules["__main__"], _c.__name__, _c)

DS = os.environ.get("DS", "outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz")  # ★retrack新flow(与②eval eval_action_full一致); 旧clips_robot.npz的tracks差~6px是错的GT
SK = "outputs/flow_render_dataset_can_dual/skel_sidecar_robot.npz"
CK_MASK = os.environ.get("CK_MASK", "outputs/video_arch_wm/m4_video_dit/video_dit_ema.pt")
CK_SKEL = os.environ.get("CK_SKEL", "outputs/video_arch_wm/m4_ablB_skel/video_dit_ema.pt")
WM = os.environ.get("WM", "outputs/cross_embodiment_wm/abs_vs_rel_humanhelps/wm_dummy5_rh_N3000.pt")
OUT = os.environ.get("OUT", "outputs/video_arch_wm/m5_e2e_combined"); os.makedirs(f"{OUT}/gifs", exist_ok=True)
SEQS = [int(x) for x in os.environ.get("SEQS", "332,418,442").split(",")]
AGENTS_SEL = [a for a in os.environ.get("AGENTS", "mask,skel").split(",") if a]   # 选渲哪些③(默认both; AGENTS=skel=单列干净demo)
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
    action = getattr(wm, "action", "dummy5")                    # ★mp→4槽(含grip); dummy5→3点(向后兼容)
    efA_act, efB_act = W.load_action_tokens(action, "r", z)     # rollout 用的动作 token(cond/overlay 仍用 efg 3点)
    print(f"[e2e] WM={WM} action={action} act_tok_shape={efA_act.shape}", flush=True)
    ade_hi, ade_lo = [], []
    K = W.K; P = 48; u8 = lambda a: (np.clip(a,0,1)*255).astype(np.uint8); r128 = lambda im: cv2.resize(im, (IMG,IMG))
    for si in SEQS:
        trg = [np.nan_to_num(z["tracks"][si].astype(np.float32)), np.nan_to_num(z["tracks_low"][si].astype(np.float32))]
        efg = [np.nan_to_num(z["eef"][si].astype(np.float32)), np.nan_to_num(z["eef_low"][si].astype(np.float32))]
        vsg = [np.nan_to_num(z["vis"][si].astype(np.float32)), np.nan_to_num(z["vis_low"][si].astype(np.float32))]
        jt = np.nan_to_num(z["joint"][si].astype(np.float32)); frs = [z["frames"][si], z["frames_low"][si]]
        sk = [_SK["skel2d_high"][si], _SK["skel2d_low"][si]]
        fidx = z["fidx"][si]; vid = int(z["vid"][si])
        trD = np.concatenate([trg[0][:K], trg[1][:K]], 1)[None]
        pr = W.rollout_dual(wm, torch.from_numpy(trD).float().to(dev), torch.from_numpy(efA_act[si][None]).float().to(dev), torch.from_numpy(efB_act[si][None]).float().to(dev), L-K)[0].cpu().numpy()
        predtr = [np.concatenate([trg[v][:K], pr[:, v*P:(v+1)*P]], 0) for v in range(2)]
        ah = np.linalg.norm(pr[:, :P] - trg[0][K:L], axis=-1).mean()*IMG          # ②ADE cam_high px@128
        al = np.linalg.norm(pr[:, P:] - trg[1][K:L], axis=-1).mean()*IMG          # ②ADE cam_low
        ade_hi.append(ah); ade_lo.append(al); print(f"  seq{si} ②ADE high {ah:.2f} low {al:.2f} px", flush=True)
        f0 = [gt256(vid, fidx[:1], v)[0] for v in range(2)]
        za = torch.stack([vae.encode(torch.from_numpy(f0[v].astype(np.float32).transpose(2, 0, 1)[None, :, None]/255.))[0, :, :1] for v in range(2)])[None]
        rends = {}
        AG_SPEC = {"mask": (m3M, "gmask"), "skel": (m3S, "skel")}
        for tag in AGENTS_SEL:
            model, agent = AG_SPEC[tag]
            cond = np.zeros((2, 7, tL, GRID, GRID), np.float32)
            for k in range(tL):
                rf = 0 if k == 0 else min(4*k, L-1)
                for v in range(2): cond[v, :, k] = cond_v(v, trg[v][0], predtr[v][rf], efg[v][0], efg[v][rf], vsg[v][rf], jt[rf], frs[v][0], sk[v][rf], agent)
            cch = model.c_embed.in_features // 4   # ★从c_embed反推真Ccond(patch²=4): grip③=7/skel③=4; 别硬编码(误伤7ch grip多头)
            xs = sample(model, za, torch.from_numpy(cond[:, :cch][None]).float().to(dev))
            rends[tag] = [vae.decode(xs[:, v])[0].permute(1, 2, 3, 0).cpu().numpy() for v in range(2)]
        gmv = [np.stack((HP.gmask_imgs(jt, efg[0]) if v == 0 else G.gmask_low_imgs(jt, efg[1]))) for v in range(2)]
        for v in range(2):
            Tp = min(rends[tag][v].shape[0] for tag in AGENTS_SEL)
            gtpx = gt256(vid, fidx[:Tp], v).astype(np.float32)/255.
            gR = np.stack([r128(u8(gtpx[t])) for t in range(Tp)])
            cols = ["GT"]; render_rows = [gR]
            ov_rows = [np.stack([ov(gR[t], trg[v][min(t,L-1)], None, efg[v][min(t,L-1)]) for t in range(Tp)])]
            for tag in AGENTS_SEL:                                                   # 每个选中的③各一列(②-driven)
                R = np.stack([r128(u8(rends[tag][v][t])) for t in range(Tp)]); cols.append(tag); render_rows.append(R)
                ov_rows.append(np.stack([ov(R[t], trg[v][min(t,L-1)], predtr[v][min(t,L-1)], efg[v][min(t,L-1)],
                                            gm=(gmv[v][min(t,L-1)] if tag=="mask" else None),
                                            sk=(sk[v][min(t,L-1)] if tag=="skel" else None)) for t in range(Tp)]))
            render_seq = np.stack(render_rows); fl = np.stack(ov_rows)               # (ncol,Tp,128,128,3)
            save_combined_gif(f"{OUT}/gifs/seq{si}_cam{'high' if v==0 else 'low'}.gif", render_seq, fl,
                              cols, [None]*len(cols), 0,
                              caption=f"seq{si} cam{'high' if v==0 else 'low'} | mp②-driven → ③{'/'.join(AGENTS_SEL)}: overlay=②pred-flow(红)+GT(绿)+eef(黄)+agent条件")
            print(f"seq{si} cam{'high' if v==0 else 'low'} done", flush=True)
    mh, ml = float(np.mean(ade_hi)), float(np.mean(ade_lo))
    summ = (f"WM={WM} action={action} | ③渲染列={AGENTS_SEL} (mask={CK_MASK} skel={CK_SKEL})\n"
            f"②ADE(pred-flow vs GT, px@128, mean over {len(SEQS)} seqs): cam_high {mh:.2f}  cam_low {ml:.2f}\n"
            f"per-seq high {[round(a,2) for a in ade_hi]} low {[round(a,2) for a in ade_lo]}\n"
            f"save_combined_gif: 列 GT|{'|'.join(AGENTS_SEL)}; Rendered行+Overlay行(②pred-flow红+GT绿+eef黄+agent条件)。mp②-driven。\n")
    open(f"{OUT}/README.txt", "w").write(summ)
    print("=== ②ADE summary ===\n" + summ, flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
