"""正式 render-LPIPS eval(2026-07-25夜): 多头ckpt 的 GT-flow列 render-LPIPS(含agent, ③天花板判据)。
★用训练同款预计算 cond_skel(4ch)+latents锚, 避开 eval_video_e2e 建7ch cond 的不兼容。只 gtflow 列(纯③, 不需②)。
Wan decode → gt256(与latent同crop的video源) → LPIPS。用 .venv_wan/bin/python。env: RUNS(逗号列run名)/SEQS。
产物 outputs/video_arch_wm/mh_eval/render_summary.txt + gifs。

2026-07-30 加 cube_px 控制保真度量(flow-cond vs naive-cond 对比, project FLOWCOND_VS_NAIVE_L48):
render-LPIPS 只测外观, 之前发现 flow 的增益集中在物体位置精度非外观(project_keyboard_3way_ablation:
cube 2.2 vs 6.0px 2.7x, PSNR/LPIPS 差不多) -> 加 HSV can 检测(复用 exp_keyboard_3way.detect_cube 同款
红色阈值改给 can 银顶红身调过)算渲染出的罐子位置 vs GT 帧罐子位置的像素误差。检测不到记 nan+det_rate/
vanish(见 feedback_cube_metric_nan_trap, 别信覆盖率, 检测不到不能算0)。
若 run 名含 "noflow"(=NOFLOW=1 训练的 naive baseline), eval 时同训练同款清零 cond 前3通道(flow: dx,dy,
footprint), 只留第4通道(agent)喂模型, 保持和训练时看到的输入分布一致。
"""
import os, sys, numpy as np, torch, cv2, lpips
sys.path.insert(0, "."); os.environ.setdefault("HF_HUB_OFFLINE", "1")
from wan_vae import WanVAE
from video_dit import VideoDiT
from video_multihead_wm import MultiHeadVideoWM, AuxHead
for _c in [VideoDiT, MultiHeadVideoWM, AuxHead]:            # torch.load unpickle 需类在 __main__
    setattr(sys.modules["__main__"], _c.__name__, _c)

LATP = "outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz"
CONDP = os.environ.get("CONDP", "outputs/video_arch_wm/cond_can_dual/cond_skel_all.npz")
DS = "outputs/flow_render_dataset_can_dual/clips_robot.npz"
RUNS = os.environ.get("RUNS", "ro_noaux,ro_dino,rh_noaux,rh_dino").split(",")
SEQS = [int(x) for x in os.environ.get("SEQS", "332,59,418,442").split(",")]
CCOND, TLCAP, NS = 4, 6, 20
Bdir = os.environ.get("BDIR", "outputs/video_arch_wm/mh_N100"); dev = "cuda"
OUT = os.environ.get("OUT", "outputs/video_arch_wm/mh_eval"); os.makedirs(f"{OUT}/gifs", exist_ok=True)
CROPS = {0: (60, 60, 390, 390), 1: (0, 0, 640, 480)}
VIDP = {v: f"human_play_data/play_robot_can_{{}}_eef/videos/chunk-000/observation.images.cam_{n}/episode_000000.mp4"
        for v, n in [(0, "high"), (1, "low")]}


@torch.no_grad()
def sample(model, z_anchor, cond, steps=NS):
    B, V, C, _, g, _ = z_anchor.shape; T = cond.shape[3]
    x = torch.randn(B, V, C, T, g, g, device=dev); x[:, :, :, :1] = z_anchor
    for i in range(steps):
        t = torch.full((B, T), i / steps, device=dev); t[:, 0] = 1.0
        x = x + model(x, t, cond) / steps; x[:, :, :, :1] = z_anchor
    return x


def detect_can(frame_u8):
    """HSV 红罐检测(银顶红身), 复用 exp_keyboard_3way.detect_cube 同款阈值风格。检测不到返 None(别当0)。"""
    hsv = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (((h > 168) | (h < 6)) & (s > 110) & (v > 50)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if mask.sum() < 8:
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def can_pos_err(pred_u8, gt_u8):
    """逐帧 render can 位置 vs GT can 位置的 px 误差。★必报 det_rate/vanish, 检测不到不计入均值(防 nan trap 奖励消失)。"""
    errs, det, vanish, n = [], 0, 0, 0
    for pf, gf in zip(pred_u8, gt_u8):
        gc = detect_can(gf)
        if gc is None:
            continue
        n += 1
        pc = detect_can(pf)
        if pc is None:
            vanish += 1; continue
        det += 1; errs.append(float(np.hypot(pc[0] - gc[0], pc[1] - gc[1])))
    return dict(cube_px=float(np.mean(errs)) if errs else float("nan"),
                det_rate=det / max(n, 1), vanish=vanish / max(n, 1), n_gtdet=n)


def gt256(vid, fidx, v):
    import av
    x, y, w, h = CROPS[v]; need = set(int(i) for i in fidx); got = {}
    c = av.open(VIDP[v].format(vid - 100 + 1))
    for i, fr in enumerate(c.decode(video=0)):
        if i in need: got[i] = cv2.resize(fr.to_ndarray(format="rgb24")[y:y+h, x:x+w], (256, 256), interpolation=cv2.INTER_AREA)
        if i > max(need): break
    c.close()
    return np.stack([got[int(i)] for i in fidx])


def main():
    zl = np.load(LATP); lat, lat_low = zl["lat"][:, :, :TLCAP], zl["lat_low"][:, :, :TLCAP]
    cond_full = np.load(CONDP)["cond"][:, :, :CCOND, :TLCAP]
    zc = np.load(DS)
    vae = WanVAE(device=dev); lp = lpips.LPIPS(net="alex").to(dev).eval()
    u8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
    lines = ["正式 render-LPIPS (GT-flow列=③天花板, 含agent) + cube_px(can 位置控制保真) | 越小越好 | held-out SEQS", ""]
    per_run = {}; per_run_cube = {}
    for r in RUNS:
        ck = f"{Bdir}/{r}/mh_ema.pt"
        if not os.path.exists(ck): lines.append(f"{r}: MISS"); continue
        m = torch.load(ck, map_location=dev, weights_only=False).eval()
        noflow = "noflow" in r        # ★naive baseline: 训练时清零了 cond 前3通道(flow), eval 也须清零保持同分布
        cond = cond_full.copy()
        if noflow: cond[:, :, :3] = 0.0
        lpvs = []; cube_errs = []; det_n = 0; vanish_n = 0; gtdet_n = 0
        for si in SEQS:
            fidx = zc["fidx"][si]; vid = int(zc["vid"][si])
            za = torch.from_numpy(np.stack([lat[si], lat_low[si]])[None, :, :, :1].astype(np.float32)).to(dev)  # (1,2,C,1,16,16)
            c = torch.from_numpy(cond[si][None].astype(np.float32)).to(dev)
            xs = sample(m, za, c)
            for v in range(2):
                dec = vae.decode(xs[:, v])[0].permute(1, 2, 3, 0).cpu().numpy()   # (Tp,256,256,3)
                Tp = dec.shape[0]; gt = gt256(vid, fidx[:Tp], v).astype(np.float32) / 255.
                Tp = min(Tp, gt.shape[0])
                a = torch.from_numpy(gt[:Tp]).permute(0, 3, 1, 2).to(dev) * 2 - 1
                b = torch.from_numpy(dec[:Tp]).permute(0, 3, 1, 2).to(dev) * 2 - 1
                lv = lp(a, b).mean().item(); lpvs.append(lv)
                gt_u8 = np.stack([u8(gt[t]) for t in range(Tp)]); dec_u8 = np.stack([u8(dec[t]) for t in range(Tp)])
                cm = can_pos_err(dec_u8, gt_u8)
                gtdet_n += cm["n_gtdet"]; det_n += round(cm["det_rate"] * cm["n_gtdet"]); vanish_n += round(cm["vanish"] * cm["n_gtdet"])
                if not np.isnan(cm["cube_px"]): cube_errs.append(cm["cube_px"])
                if v == 0:     # 存 gif 眼检(cam_high, 叠加检测点: 绿=GT can, 红=render can)
                    frames = []
                    for t in range(Tp):
                        gtf = cv2.resize(gt_u8[t], (128, 128)).copy(); pdf = cv2.resize(dec_u8[t], (128, 128)).copy()
                        gc = detect_can(gt_u8[t]); pc = detect_can(dec_u8[t])
                        if gc is not None: cv2.circle(gtf, (int(gc[0] * 128 / 256), int(gc[1] * 128 / 256)), 4, (0, 255, 0), -1)
                        if pc is not None: cv2.circle(pdf, (int(pc[0] * 128 / 256), int(pc[1] * 128 / 256)), 4, (255, 0, 0), -1)
                        frames.append(np.concatenate([gtf, pdf], 1))
                    import imageio; imageio.mimsave(f"{OUT}/gifs/{r}_seq{si}_high.gif", np.stack(frames), fps=6)
        per_run[r] = float(np.mean(lpvs))
        per_run_cube[r] = dict(cube_px=float(np.mean(cube_errs)) if cube_errs else float("nan"),
                                det_rate=det_n / max(gtdet_n, 1), vanish=vanish_n / max(gtdet_n, 1), n_gtdet=gtdet_n)
        pc = per_run_cube[r]
        lines.append(f"{r:12s}: render-LPIPS {per_run[r]:.4f} (n={len(lpvs)}) | cube_px {pc['cube_px']:.2f} "
                      f"(det_rate={pc['det_rate']:.2f} vanish={pc['vanish']:.2f} n_gtdet={pc['n_gtdet']}){'  [NOFLOW: cond flow-ch zeroed]' if noflow else ''}")
        print(lines[-1], flush=True)
    lines.append("")
    for aux in ["noaux", "dino", "mask", "hc", "all"]:
        ro, rh = per_run.get(f"ro_{aux}"), per_run.get(f"rh_{aux}")
        if ro and rh: lines.append(f"[{aux:5s}] ro={ro:.4f} rh={rh:.4f}  human-helps Δ%={100*(ro-rh)/ro:+.1f}")
    if "ro" in per_run and "rh" in per_run:      # human-helps ③ (robot-only vs robot+human mixed)
        lp_ro, lp_rh = per_run["ro"], per_run["rh"]; c_ro, c_rh = per_run_cube["ro"]["cube_px"], per_run_cube["rh"]["cube_px"]
        lines.append(f"[human-helps ro vs rh] render-LPIPS ro={lp_ro:.4f} rh={lp_rh:.4f} Δ%={100*(lp_ro-lp_rh)/lp_ro:+.1f} | "
                      f"cube_px ro={c_ro:.2f} rh={c_rh:.2f} Δ%={100*(c_ro-c_rh)/c_ro:+.1f}")
    for r in RUNS:                                # flow-cond vs naive-cond (r vs r_noflow), 本 report 主判据
        rn = f"{r}_noflow"
        if r in per_run and rn in per_run:
            lp_f, lp_n = per_run[r], per_run[rn]; c_f, c_n = per_run_cube[r]["cube_px"], per_run_cube[rn]["cube_px"]
            lines.append(f"[flow vs naive: {r} vs {rn}] render-LPIPS flow={lp_f:.4f} naive={lp_n:.4f} Δ%={100*(lp_n-lp_f)/lp_n:+.1f} | "
                          f"cube_px flow={c_f:.2f} naive={c_n:.2f} ratio(naive/flow)={c_n/max(c_f,1e-6):.2f}x")
    open(f"{OUT}/render_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[-8:]) + f"\nsaved {OUT}/render_summary.txt", flush=True)


if __name__ == "__main__":
    main()
