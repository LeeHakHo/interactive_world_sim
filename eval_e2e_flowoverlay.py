"""聚焦: ②pred-flow 直接叠在 e2e 渲染帧上, 看 pred-flow 表现(render是否跟预测走 + pred偏GT多少)。
每 seq 2列(cam_high|cam_low), 上=e2e渲染, 下=e2e渲染+pred-flow叠加(红=②pred含尾迹 / 绿=GT / 黄=eef)。
② rollout(GT eef驱动)→pred tracks→pred-flow cond→③ e2e render。标准 save_combined_gif。用 .venv_wan/bin/python。
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
CK = "outputs/video_arch_wm/m4_video_dit/video_dit_ema.pt"
WM = "outputs/cross_embodiment_wm/abs_vs_rel_humanhelps/wm_dummy5_rh_N3000.pt"
OUT = "outputs/video_arch_wm/m5_e2e_flowoverlay"; os.makedirs(f"{OUT}/gifs", exist_ok=True)
SEQS = [int(x) for x in os.environ.get("SEQS", "332,418,442").split(",")]
NS = 20; dev = "cuda"; GRID = 16; POOL = 8; tL = 12; L = 48
CROPS = {0: (60, 60, 390, 390), 1: (0, 0, 640, 480)}
VIDP = {v: f"human_play_data/play_robot_can_{{}}_eef/videos/chunk-000/observation.images.cam_{n}/episode_000000.mp4" for v, n in [(0, "high"), (1, "low")]}


def pool16(x): return torch.nn.functional.avg_pool2d(torch.from_numpy(x[None]).float(), POOL)[0].numpy()


def cond7(v, tr0, trt, ef0, eft, vis, jt, fr0):
    fc = DIT.flow_cond(tr0, trt, ef0, eft, vis)
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


def draw_overlay(img, gt_obj_hist, pred_hist, eef):
    """img (256,256,3)uint8; gt_obj_hist/pred_hist = list of (P,2) 到当前帧(尾迹); eef (3,2)。红=pred 绿=gt 黄=eef。"""
    im = img.copy(); S = 256
    for h, obj in enumerate(gt_obj_hist):                      # GT 绿尾迹(质心)
        c = (obj.mean(0) * S).astype(int); cv2.circle(im, tuple(c), 2, (0, 220, 0), -1)
    for h, obj in enumerate(pred_hist):                        # ②pred 红尾迹(质心)
        c = (obj.mean(0) * S).astype(int); cv2.circle(im, tuple(c), 2, (255, 40, 40), -1)
    # 当前帧: pred 全点(红) + GT 全点(绿, 小)
    for p in pred_hist[-1]: cv2.circle(im, tuple((p*S).astype(int)), 2, (255, 80, 80), -1)
    for p in gt_obj_hist[-1]: cv2.circle(im, tuple((p*S).astype(int)), 1, (0, 255, 0), -1)
    for p in eef: cv2.circle(im, tuple((p*S).astype(int)), 3, (255, 230, 0), -1)
    return im


def main():
    z = np.load(DS); zl = np.load("outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz")
    m3 = torch.load(CK, map_location=dev, weights_only=False).eval()
    wm = torch.load(WM, map_location=dev, weights_only=False).eval()
    vae = WanVAE(device=dev)
    if not HP._G: HP.load_gmask()
    G.load_gmask_low()
    K = W.K; P = 48; u8 = lambda a: (np.clip(a, 0, 1)*255).astype(np.uint8)
    for si in SEQS:
        trg = [np.nan_to_num(z["tracks"][si].astype(np.float32)), np.nan_to_num(z["tracks_low"][si].astype(np.float32))]
        efg = [np.nan_to_num(z["eef"][si].astype(np.float32)), np.nan_to_num(z["eef_low"][si].astype(np.float32))]
        vsg = [np.nan_to_num(z["vis"][si].astype(np.float32)), np.nan_to_num(z["vis_low"][si].astype(np.float32))]
        jt = np.nan_to_num(z["joint"][si].astype(np.float32)); frs = [z["frames"][si], z["frames_low"][si]]
        fidx = z["fidx"][si]; vid = int(z["vid"][si])
        trD = np.concatenate([trg[0][:K], trg[1][:K]], 1)[None]
        pr = W.rollout_dual(wm, torch.from_numpy(trD).float().to(dev), torch.from_numpy(efg[0][None]).float().to(dev), torch.from_numpy(efg[1][None]).float().to(dev), L-K)[0].cpu().numpy()
        predtr = [np.concatenate([trg[v][:K], pr[:, v*P:(v+1)*P]], 0) for v in range(2)]
        f0 = [gt256(vid, fidx[:1], v)[0] for v in range(2)]
        za = torch.stack([vae.encode(torch.from_numpy(f0[v].astype(np.float32).transpose(2, 0, 1)[None, :, None]/255.))[0, :, :1] for v in range(2)])[None]
        cond = np.zeros((2, 7, tL, GRID, GRID), np.float32)
        for k in range(tL):
            rf = 0 if k == 0 else min(4*k, L-1)
            for v in range(2): cond[v, :, k] = cond7(v, trg[v][0], predtr[v][rf], efg[v][0], efg[v][rf], vsg[v][rf], jt[rf], frs[v][0])
        xs = sample(m3, za, torch.from_numpy(cond[None]).float().to(dev))
        rend = [vae.decode(xs[:, v])[0].permute(1, 2, 3, 0).cpu().numpy() for v in range(2)]
        Tp = min(rend[0].shape[0], rend[1].shape[0])
        # gif: 每帧 上行=两视角e2e渲染, 下行=两视角渲染+pred-flow叠加(红尾迹)
        frames = []
        for t in range(Tp):
            top = np.concatenate([u8(rend[0][t]), u8(rend[1][t])], 1)
            bot = np.concatenate([draw_overlay(u8(rend[v][t]), [trg[v][s] for s in range(t+1)], [predtr[v][s] for s in range(t+1)], efg[v][t]) for v in range(2)], 1)
            frames.append(np.concatenate([top, bot], 0))
        imageio.mimsave(f"{OUT}/gifs/seq{si}.gif", frames, fps=8, loop=0)
        print(f"seq{si}: e2e flow-overlay gif ({Tp}帧)", flush=True)
    open(f"{OUT}/README.txt", "w").write("②pred-flow叠在e2e渲染上。上=e2e渲染(high|low), 下=+pred-flow叠加(红=②pred尾迹+当前点/绿=GT/黄=eef)。\n看render是否跟pred走, pred偏GT多少。\n")
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
