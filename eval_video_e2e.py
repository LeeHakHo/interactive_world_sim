"""M5 e2e: ②-pred-flow 端到端 vs GT-flow 天花板, 标准 eval protocol。
每 seq: ② rollout_dual(dummy5, GT eef驱动)-> pred tracks; 建 pred-flow cond + GT-flow cond ->
  ③ 视频DiT 各渲一版 -> Wan decode。standard save_combined_gif: 列= GT | ③(GT-flow天花板) | ②→③(e2e);
  Rendered行 + Flow overlay行(绿=GT锚 / 红=②预测 / 黄=eef)。报两条 render LPIPS(含agent)。
用 .venv_wan/bin/python(Wan decode + ②/③ 都纯torch可import)。SEQS 默认同 replay。
"""
import os, sys, numpy as np, torch, cv2, lpips
sys.path.insert(0, "."); os.environ.setdefault("RES", "128"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MASKGEN", "outputs/flow_wm/maskgen_caneef/maskgen_caneef.pt")
os.environ.setdefault("GMASK", "1"); os.environ.setdefault("GMASK_LOW", "1"); os.environ.setdefault("WARP", "1")
import exp_scel_dualview_dit as DIT
import exp_scel_dualview_gmaskcond as G
import exp_scel_dualview_wm as W
import exp_scel_dualview_comb as DC
import exp_v3_human_helps_pixels as HP
from video_dit import VideoDiT
from video_multihead_wm import MultiHeadVideoWM       # ③ multihead(scarce e2e 用)
from wan_vae import WanVAE
from viz_combined import save_combined_gif, build_flow_cols
# ② WM / ③ ckpt 是整模型 pickle -> 注册类到 __main__ 供 torch.load
for _c in [W.DualLWC, DC.DualCombLWC, DIT.DualViewDiT, G.DualViewDiTG, VideoDiT, MultiHeadVideoWM]:
    setattr(sys.modules["__main__"], _c.__name__, _c)

DS = "outputs/flow_render_dataset_can_dual/clips_robot.npz"
CKPT = os.environ.get("CKPT", "outputs/video_arch_wm/m4_video_dit/video_dit_ema.pt")
WM = os.environ.get("WM_CKPT", "outputs/cross_embodiment_wm/abs_vs_rel_humanhelps/wm_dummy5_rh_N3000.pt")
OUT = os.environ.get("OUT", "outputs/video_arch_wm/m5_e2e"); os.makedirs(f"{OUT}/gifs", exist_ok=True)
SEQS = [int(x) for x in os.environ.get("SEQS", "332,59,418,442").split(",")]
NS = int(os.environ.get("STEPS_SAMPLE", "20")); dev = "cuda"; GRID = 16; POOL = 128 // GRID
CCOND = int(os.environ.get("CCOND", "0"))            # >0=切cond前CCOND通道(multihead ③=4: flow3+agent1); 0=全7ch(老VideoDiT)
ACTION = os.environ.get("ACTION", "dummy5")          # ②动作表示: mp/dhc需grip第4槽, 用load_action_tokens同训练口径建eef
CROPS = {0: (60, 60, 390, 390), 1: (0, 0, 640, 480)}
VIDP = {v: f"human_play_data/play_robot_can_{{}}_eef/videos/chunk-000/observation.images.cam_{n}/episode_000000.mp4"
        for v, n in [(0, "high"), (1, "low")]}


def pool16(x):
    return torch.nn.functional.avg_pool2d(torch.from_numpy(x[None]).float(), POOL)[0].numpy()


def cond7(tr0, trt, ef0, eft, vis, jt, fr0, v):        # (7,16,16)
    fc = DIT.flow_cond(tr0, trt, ef0, eft, vis)
    g = (HP.gmask_imgs(jt[None], eft[None])[0] if v == 0 else G.gmask_low_imgs(jt[None], eft[None])[0])
    if g.shape[0] != 128: g = cv2.resize(g, (128, 128))
    wp = G.warp_preview(fr0, tr0, trt, vis)
    return pool16(np.concatenate([fc, g[None], wp], 0).astype(np.float32))


@torch.no_grad()
def sample(model, z_anchor, cond, steps):
    B, V, C, _, g, _ = z_anchor.shape; T = cond.shape[3]
    x = torch.randn(B, V, C, T, g, g, device=dev); x[:, :, :, :1] = z_anchor
    for i in range(steps):
        t = torch.full((B, T), i / steps, device=dev); t[:, 0] = 1.0
        x = x + model(x, t, cond) / steps; x[:, :, :, :1] = z_anchor
    return x


def gt256(vid, fidx, v):
    import av
    x, y, w, h = CROPS[v]; need = set(int(i) for i in fidx); got = {}
    c = av.open(VIDP[v].format(vid - 100 + 1))
    for i, fr in enumerate(c.decode(video=0)):
        if i in need: got[i] = cv2.resize(fr.to_ndarray(format="rgb24")[y:y + h, x:x + w], (256, 256), interpolation=cv2.INTER_AREA)
        if i > max(need): break
    c.close()
    return np.stack([got[int(i)] for i in fidx])


def main():
    z = np.load(DS)
    # ★②rollout 用的 eef 必须与训练同款(mp/dhc = 3点+grip第4槽); dummy5 保持3点。
    # 全数据集一次性建(grip 归一化按全集 nanmax, 与训练一致), 再逐seq索引。
    efA_wm, efB_wm = W.load_action_tokens(ACTION, "r", z)
    m3 = torch.load(CKPT, map_location=dev, weights_only=False).eval()
    wm = torch.load(WM, map_location=dev, weights_only=False).eval()
    vae = WanVAE(device=dev); lp = lpips.LPIPS(net="alex").to(dev).eval()
    if not HP._G: HP.load_gmask()
    G.load_gmask_low()
    K, F = W.K, W.F; P = 48; tL = 12; L = 48
    u8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
    r128 = lambda im: cv2.resize(im, (128, 128))
    from eval_mh_render import can_pos_err            # cube_px 控制保真(含 det_rate/vanish 防nan陷阱)
    cube = {"gtflow": [], "e2e": []}                  # 每项: (cube_px, n_gtdet, det_rate, vanish)
    lines = []
    for si in SEQS:
        trg = [np.nan_to_num(z["tracks"][si].astype(np.float32)), np.nan_to_num(z["tracks_low"][si].astype(np.float32))]
        efg = [np.nan_to_num(z["eef"][si].astype(np.float32)), np.nan_to_num(z["eef_low"][si].astype(np.float32))]
        vsg = [np.nan_to_num(z["vis"][si].astype(np.float32)), np.nan_to_num(z["vis_low"][si].astype(np.float32))]
        jt = np.nan_to_num(z["joint"][si].astype(np.float32)); frs = [z["frames"][si], z["frames_low"][si]]
        fidx = z["fidx"][si]; vid = int(z["vid"][si])
        # ② rollout: GT首帧上下文 + GT eef 驱动 -> pred tracks (L,2P,2)
        trD = np.concatenate([trg[0][:K], trg[1][:K]], 1)[None]                       # (1,K,2P,2)
        efA = efA_wm[si][None]; efB = efB_wm[si][None]                                # ②rollout用同训练口径(mp含grip槽)
        pred = W.rollout_dual(wm, torch.from_numpy(trD).float().to(dev),
                              torch.from_numpy(efA).float().to(dev), torch.from_numpy(efB).float().to(dev), L - K)[0].cpu().numpy()
        predtr = [np.concatenate([trg[v][:K], pred[:, v * P:(v + 1) * P]], 0) for v in range(2)]   # (L,P,2) 每视角
        # 用 Wan 编码首帧作锚(两视角)
        f0 = [gt256(vid, fidx[:1], v)[0] for v in range(2)]
        za = torch.stack([vae.encode(torch.from_numpy(f0[v].astype(np.float32).transpose(2, 0, 1)[None, :, None] / 255.))[0, :, :1] for v in range(2)])[None]  # (1,2,48,1,16,16)
        outs = {}
        for tag, use_pred in [("gtflow", False), ("e2e", True)]:
            cond = np.zeros((2, 7, tL, GRID, GRID), np.float32)
            for k in range(tL):
                rf = 0 if k == 0 else min(4 * k, L - 1)
                for v in range(2):
                    src = predtr[v] if use_pred else trg[v]
                    cond[v, :, k] = cond7(trg[v][0], src[rf], efg[v][0], efg[v][rf], vsg[v][rf], jt[rf], frs[v][0], v)
            c = torch.from_numpy(cond[None]).float().to(dev)
            if CCOND: c = c[:, :, :CCOND]            # multihead ③ 训练时只吃前CCOND通道(flow3+agent1)
            xs = sample(m3, za, c, NS)
            outs[tag] = [vae.decode(xs[:, v])[0].permute(1, 2, 3, 0).cpu().numpy() for v in range(2)]  # per view (Tpix,256,256,3)
        # 指标 + gif (两视角各出一个 gif, 3列: GT | gtflow | e2e)
        for v in range(2):
            gtpx = gt256(vid, fidx[:outs["gtflow"][v].shape[0]], v).astype(np.float32) / 255.
            Tp = min(gtpx.shape[0], outs["gtflow"][v].shape[0], outs["e2e"][v].shape[0])
            for tag in ["gtflow", "e2e"]:
                a = torch.from_numpy(gtpx[:Tp]).permute(0, 3, 1, 2).to(dev) * 2 - 1
                b = torch.from_numpy(outs[tag][v][:Tp]).permute(0, 3, 1, 2).to(dev) * 2 - 1
                lpv = lp(a, b).mean().item()
                cm = can_pos_err(u8(outs[tag][v][:Tp]), u8(gtpx[:Tp]))      # 控制保真: render罐位 vs GT罐位
                cube[tag].append((cm["cube_px"], cm["n_gtdet"], cm["det_rate"], cm["vanish"]))
                lines.append(f"seq{si} cam{'high' if v==0 else 'low'} {tag:7s}: LPIPS {lpv:.4f} | cube_px {cm['cube_px']:.2f} (det{cm['det_rate']:.2f} van{cm['vanish']:.2f})")
                print(lines[-1], flush=True)
            # gif @128: 3 列
            rend = np.stack([np.stack([r128(u8(gtpx[t])) for t in range(Tp)]),
                             np.stack([r128(u8(outs["gtflow"][v][t])) for t in range(Tp)]),
                             np.stack([r128(u8(outs["e2e"][v][t])) for t in range(Tp)])])       # (3,Tp,128,128,3)
            gtobj = trg[v][:Tp]; eefv = efg[v][:Tp]
            fl = np.stack([build_flow_cols(rend[0], gtobj, [None], eefv)[0],                     # GT: 绿only
                           build_flow_cols(rend[1], gtobj, [None], eefv)[0],                     # 天花板: GT flow=绿only
                           build_flow_cols(rend[2], gtobj, [predtr[v][:Tp]], eefv)[0]])          # e2e: 绿GT+红②
            save_combined_gif(f"{OUT}/gifs/seq{si}_cam{'high' if v==0 else 'low'}.gif", rend, fl,
                              ["GT", "3-GTflow(ceil)", "2->3 e2e"], [None, None, None], 0,
                              caption=f"seq{si} cam{'high' if v==0 else 'low'} | replay天花板 vs ②pred-flow e2e")
    # 跨 seq/view 按检测帧加权汇总 cube_px + det/vanish
    cube_lines = ["", "=== cube_px 汇总(按检测帧加权; 越小=物体位置越准) ==="]
    for tag in ["gtflow", "e2e"]:
        rows = cube[tag]; N = sum(r[1] for r in rows)
        detc = sum(r[2] * r[1] for r in rows); vanc = sum(r[3] * r[1] for r in rows)
        valid = [(r[0], r[2] * r[1]) for r in rows if not np.isnan(r[0])]
        cpx = sum(c * w for c, w in valid) / max(sum(w for _, w in valid), 1e-9) if valid else float("nan")
        cube_lines.append(f"[cube_px {tag:7s}] {cpx:.2f} (det_rate={detc/max(N,1):.2f} vanish={vanc/max(N,1):.2f} n_gtdet={N})")
    open(f"{OUT}/summary.txt", "w").write(
        "M5 e2e: ②pred-flow端到端 vs GT-flow天花板 (标准protocol, 含agent LPIPS + cube_px控制保真)\n"
        f"③={CKPT}\n②={WM}\ngif列: GT | ③(GT-flow天花板) | ②→③(e2e); flow行 绿=GT锚/红=②预测/黄=eef\n\n"
        + "\n".join(lines) + "\n".join(cube_lines) + "\n\ne2e-天花板差 = ②误差穿过③掉多少。\n")
    print("=== M5 e2e DONE ===", flush=True)


if __name__ == "__main__":
    main()
