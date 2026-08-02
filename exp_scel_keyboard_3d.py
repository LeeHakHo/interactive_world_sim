"""Keyboard 3D 交互 demo (视频架构③). 三阶段:
  A precompute(.venv_wan, GPU): 键盘脚本→世界系3D eef(script_eef3d)→双视角REAL标定投影→②rollout
     +物体3D刚体跟随(object_track_dual)+②驱动兜底(blend_predtr)+IK joint. 存 precompute/<name>_si<si>.npz。
  B fk_skel2d.py(conda phantom): joint→skel2d 两视角(见该文件)。存 skel2d/<name>_si<si>.npz。
  C render(.venv_wan, GPU): 读 precompute+skel2d → skel cond(flow3+skel1+warp3) → VideoDiT(skel③)sample
     → 双视角标准 save_combined_gif(Rendered行 + Flow+Skel overlay行 + caption)。
控制=世界系3D(水平x/y + z上下, 修lift_high漂移: cam非top-down→纯z两视角一致投影)。抓取=功能性(grip闭+近物体→刚体跟随)。
组件: ②=wm_dummy5_rh_N3000 ③=m4_ablB_skel(skel, 消融赢家) IK=ik_adapter_can VAE=官方AutoencoderKLWan。
跑: .venv_wan/bin/python exp_scel_keyboard_3d.py --stage precompute
    conda run -n phantom python fk_skel2d.py
    .venv_wan/bin/python exp_scel_keyboard_3d.py --stage render
"""
import os, sys, argparse, numpy as np
sys.path.insert(0, ".")

FOLLOW_TH = float(os.environ.get("FOLLOW_TH", "0.4"))

# ---- 控制标度 (spec: DELTA0.012/帧, REPEAT3, z幅度≤~0.25m; cam_low>0.5m投影发散) ----
DELTA = float(os.environ.get("DELTA", "0.012"))
GRIP_DELTA = float(os.environ.get("GRIP_DELTA", "0.006"))
REPEAT = int(os.environ.get("REPEAT", "3"))
RENDER_H = int(os.environ.get("RENDER_H", "48"))   # ②rollout步数; 自回归无硬限制, env可调长(rollout_dual/③DiT T可变)
Z_LIFT_CAP = 0.28                # BOX3D z 上限 = z0 + 此值
GRID, POOL = 16, 8
tL = int(os.environ.get("TL", str(RENDER_H // 4)))  # ③ latent帧数(stride4采样② rollout), 随RENDER_H延长
DS = os.environ.get("DS", "outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz")  # ★retrack新flow(与②训练/eval一致)
OUT = os.environ.get("OUT", "outputs/video_arch_wm/keyboard_3d")
WM_CKPT = os.environ.get("WM_CKPT", "outputs/cross_embodiment_wm/abs_vs_rel_humanhelps/wm_dummy5_rh_N3000.pt")
IK_CKPT = os.environ.get("IK_CKPT", "outputs/cross_embodiment_wm/ik_adapter_can/ik_adapter.pt")
CK_SKEL = os.environ.get("CK_SKEL", "outputs/video_arch_wm/m4_ablB_skel/video_dit_ema.pt")
SEQS = [int(x) for x in os.environ.get("SEQS", "332,59,418").split(",")]

# 多命令组合 (每条 Σn·REPEAT 运动帧, precompute hold-pad 到 48; grip: gclose/gopen)
SCRIPTS = {
    "translate_LR":         [("left", 6), ("right", 6)],
    "square_xy":            [("left", 3), ("fwd", 3), ("right", 3), ("back", 3)],
    "lift_high":            [("up", 6), ("up", 6)],
    "z_wave":               [("up", 3), ("down", 3), ("up", 3), ("down", 3)],
    "grip_cycle":           [("gclose", 3), ("gopen", 3), ("gclose", 3), ("gopen", 3)],
    "pick_place_left":      [("gclose", 2), ("up", 3), ("left", 3), ("down", 2), ("gopen", 2)],
    "pick_place_right":     [("gclose", 2), ("up", 3), ("right", 3), ("down", 2), ("gopen", 2)],
    "carry_square":         [("gclose", 2), ("up", 2), ("left", 2), ("fwd", 2), ("down", 2), ("gopen", 2)],
    "lift_translate_lower": [("gclose", 1), ("up", 3), ("left", 4), ("down", 3), ("gopen", 1)],
}


def blend_predtr(pred2_A, pred2_B, objA, objB, grasp, efA, efB):
    H = len(pred2_A); trajA = pred2_A.copy(); trajB = pred2_B.copy(); fb = np.zeros(H, bool)
    for h in range(H):
        if grasp[h]:
            ef_disp = np.linalg.norm(np.concatenate([efA[h].mean(0)-efA[0].mean(0), efB[h].mean(0)-efB[0].mean(0)]))
            pr_disp = np.linalg.norm(np.concatenate([pred2_A[h].mean(0)-pred2_A[0].mean(0), pred2_B[h].mean(0)-pred2_B[0].mean(0)]))
            if ef_disp > 1e-3 and pr_disp / ef_disp < FOLLOW_TH:
                trajA[h] = objA[h]; trajB[h] = objB[h]; fb[h] = True
    return trajA, trajB, fb


# ======================= 阶段 A: precompute =======================
def load_control():
    """加载 ②(rollout) + IK; register WM 类到 __main__。返回 R dict。"""
    import torch
    import exp_scel_dualview_wm as W
    import exp_scel_dualview_comb as DC
    import exp_scel_dualview_dit as DIT
    import exp_scel_dualview_gmaskcond as G
    from exp_scel_ik_adapter import IKAdapter
    from video_dit import VideoDiT
    for _c in [W.DualLWC, DC.DualCombLWC, G.DualViewDiTG, DIT.DualViewDiT, IKAdapter, VideoDiT]:
        setattr(sys.modules["__main__"], _c.__name__, _c)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wm = torch.load(WM_CKPT, map_location=dev, weights_only=False).eval()
    ik = torch.load(IK_CKPT, map_location=dev, weights_only=False).eval()
    z = np.load(DS)
    return dict(wm=wm, ik=ik, W=W, z=z, dev=dev)


def precompute_script(si, script, R):
    """键盘脚本 → 世界系3D控制 → ②rollout + 物体刚体跟随 + 兜底 + IK。返回 dict(存 npz)。"""
    import torch
    from keyboard_3d_control import script_eef3d, project_eef_dual, object_track_dual
    from amplify_wm import K, F
    wm, ik, W, z, dev = R["wm"], R["ik"], R["W"], R["z"], R["dev"]
    P = 48; Lw = K + F
    eef3d0 = z["eef3d"][si, 0].astype(np.float64)
    grip0 = float(z["grip"][si, 0])
    obj3d0 = np.nan_to_num(z["tracks3d"][si, 0].astype(np.float64))
    valid0 = z["tracks3d_valid"][si, 0]
    z0 = eef3d0[:, 2].mean()
    BOX3D = (-0.5, 0.5, -0.5, 0.5, 0.0, z0 + Z_LIFT_CAP)
    # 纯运动帧 (F=0), 再 hold-pad 到 RENDER_H+Lw (世界帧 0..H-1 渲染 + rollout 前瞻窗口)
    traj_m, grip_m, Hm = script_eef3d(eef3d0, grip0, script, DELTA, GRIP_DELTA, REPEAT, 0, BOX3D)
    Ltot = RENDER_H + Lw
    pad = max(0, Ltot - len(traj_m))
    traj3d = np.concatenate([traj_m, np.repeat(traj_m[-1:], pad, 0)], 0)[:Ltot]
    grip_traj = np.concatenate([grip_m, np.repeat(grip_m[-1:], pad)], 0)[:Ltot]
    efA, efB = project_eef_dual(traj3d)                                   # (Ltot,3,2), efA[f]=世界帧f
    # ★mp/dhc/dummy5g 等需第4槽 grip: 合成eef指尖冻结, 故把 grip 命令(grip_traj)归一化注入第4槽(匹配训练 gripslot 语义)
    _action = getattr(wm, "action", "dummy5")
    def _add_gripslot(ef):
        # ★标定(r=0.967): 训练 gripslot = norm(指尖2D距离)/0.1137, 与 physical grip 关系 slot≈18.97*grip+0.006
        # (旧错误用 grip/0.04 偏开0.16 → mp误以为没夹紧 → fb虚高)
        gn = np.clip(18.9747 * grip_traj + 0.0057, 0.0, 1.0).astype(np.float32)  # (Ltot,) physical grip→训练gripslot空间
        return np.concatenate([ef, np.stack([gn, np.zeros_like(gn)], -1)[:, None]], 1)  # (Ltot,4,2)
    if _action in ("mp", "dhc", "dummy5g", "dummy5rs"):
        efA_act, efB_act = _add_gripslot(efA), _add_gripslot(efB)
    else:
        efA_act, efB_act = efA, efB
    # ② rollout: 初始 tracks 两视角 repeat K 当历史; 预测世界帧 K..H-1, 前 K 帧 object=静止起始(与 za 世界帧0锚对齐, 照 eval_e2e_combined)
    trA0 = np.nan_to_num(z["tracks"][si, 0].astype(np.float32))          # (P,2)
    trB0 = np.nan_to_num(z["tracks_low"][si, 0].astype(np.float32))
    trDseq = np.repeat(np.concatenate([trA0, trB0], 0)[None], K, 0)[None]  # (1,K,2P,2)
    t2 = lambda a: torch.from_numpy(a[None]).float().to(dev)
    with torch.no_grad():
        pr = W.rollout_dual(wm, torch.from_numpy(trDseq).float().to(dev),
                            t2(efA_act), t2(efB_act), RENDER_H - K)[0].cpu().numpy()   # (H-K,2P,2) 世界帧 K..H-1
    pred2_A = np.concatenate([np.repeat(trA0[None], K, 0), pr[:, :P]], 0)      # (H,P,2) 世界帧 0..H-1
    pred2_B = np.concatenate([np.repeat(trB0[None], K, 0), pr[:, P:2*P]], 0)
    # 渲染帧 = 世界帧 0..H-1 (与 za 锚对齐, cond rf 直接索引世界帧)
    efA_r, efB_r = efA[:RENDER_H], efB[:RENDER_H]                         # (H,3,2)
    traj3d_r, grip_r = traj3d[:RENDER_H], grip_traj[:RENDER_H]
    objA, objB, grasp = object_track_dual(traj3d_r, grip_r, obj3d0, valid0)   # (H,48,2),(H,),
    trajA, trajB, fb = blend_predtr(pred2_A, pred2_B, objA, objB, grasp, efA_r, efB_r)
    with torch.no_grad():
        joint = ik(torch.from_numpy(efA_r).float().to(dev)).cpu().numpy()    # (H,7)
    # VAE 首帧锚 (256 GT)
    from eval_e2e_combined import gt256
    vid = int(z["vid"][si]); fidx = z["fidx"][si]
    f0_high = gt256(vid, fidx[:1], 0)[0]; f0_low = gt256(vid, fidx[:1], 1)[0]
    return dict(si=si, efA=efA_r.astype(np.float32), efB=efB_r.astype(np.float32),
                trajA=trajA.astype(np.float32), trajB=trajB.astype(np.float32),
                joint=joint.astype(np.float32), grip=grip_r.astype(np.float32),
                grasp=grasp, fb=fb, f0_high=f0_high, f0_low=f0_low, H=RENDER_H)


def stage_precompute():
    os.makedirs(f"{OUT}/precompute", exist_ok=True)
    R = load_control()
    for si in SEQS:
        for name, script in SCRIPTS.items():
            d = precompute_script(si, script, R)
            np.savez(f"{OUT}/precompute/{name}_si{si}.npz", **d)
            print(f"[precompute] {name}_si{si}: H={d['H']} grasp={d['grasp'].mean():.2f} "
                  f"fallback={d['fb'].mean():.2f}", flush=True)
    print("=== precompute DONE ===", flush=True)


# ======================= 阶段 C: render =======================
def render_from_precompute(pre_npz, skel_npz, models):
    """读 precompute + skel2d → skel cond → VideoDiT sample → rend(2,H,128,128,3)。models=(m3S,vae,z)。"""
    import torch
    from eval_e2e_combined import sample, cond_v
    m3S, vae, z = models
    dev = next(m3S.parameters()).device
    d = np.load(pre_npz); sk = np.load(skel_npz)
    si = int(d["si"]); H = int(d["H"]); P = 48
    skel = [sk["skel2d_high"], sk["skel2d_low"]]                          # (H,9,2)
    traj = [d["trajA"], d["trajB"]]; ef = [d["efA"], d["efB"]]
    tr0 = [np.nan_to_num(z["tracks"][si, 0].astype(np.float32)), np.nan_to_num(z["tracks_low"][si, 0].astype(np.float32))]
    fr0 = [z["frames"][si, 0], z["frames_low"][si, 0]]
    joint = d["joint"]; f0 = [d["f0_high"], d["f0_low"]]
    vis = np.ones(P, np.float32)
    za = torch.stack([vae.encode(torch.from_numpy(f0[v].astype(np.float32).transpose(2, 0, 1)[None, :, None] / 255.).to(dev))[0, :, :1]
                      for v in range(2)])[None]
    tL_r = 1 + (H - 1) // 4                                               # ★latent帧数随实际H(支持早停后的短rollout)
    cond = np.zeros((2, 7, tL_r, GRID, GRID), np.float32)
    for k in range(tL_r):
        rf = 0 if k == 0 else min(4 * k, H - 1)
        for v in range(2):
            cond[v, :, k] = cond_v(v, tr0[v], traj[v][rf], ef[v][0], ef[v][rf], vis, joint[rf], fr0[v], skel[v][rf], "skel")
    from video_multihead_wm import MultiHeadVideoWM
    cch = m3S.c_embed.in_features // 4                                    # ★从c_embed反推真Ccond(patch²=4): grip③=7/skel③=4; 别硬编码
    with torch.no_grad():
        xs = sample(m3S, za, torch.from_numpy(cond[:, :cch][None]).float().to(dev))
        rend = np.stack([vae.decode(xs[:, v])[0].permute(1, 2, 3, 0).cpu().numpy() for v in range(2)])  # (2,H,128,128,3)
    return rend, d, skel


def stage_render():
    import torch, cv2
    from eval_e2e_combined import ov
    from wan_vae import WanVAE
    from viz_combined import save_combined_gif
    import exp_scel_dualview_wm as W, exp_scel_dualview_comb as DC, exp_scel_dualview_dit as DIT
    import exp_scel_dualview_gmaskcond as G
    from video_dit import VideoDiT
    from video_multihead_wm import MultiHeadVideoWM, AuxHead      # ★干净L48多头③
    from keyboard_3d_control import _AR
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for _c in [W.DualLWC, DC.DualCombLWC, G.DualViewDiTG, DIT.DualViewDiT, VideoDiT, MultiHeadVideoWM, AuxHead]:
        setattr(sys.modules["__main__"], _c.__name__, _c)
    m3S = torch.load(CK_SKEL, map_location=dev, weights_only=False).eval()
    vae = WanVAE(device=dev); z = np.load(DS)
    os.makedirs(f"{OUT}/gifs", exist_ok=True)
    u8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8); r128 = lambda im: cv2.resize(im, (128, 128))
    K = W.K
    for si in SEQS:
        for name, script in SCRIPTS.items():
            pre = f"{OUT}/precompute/{name}_si{si}.npz"; skf = f"{OUT}/skel2d/{name}_si{si}.npz"
            if not (os.path.exists(pre) and os.path.exists(skf)):
                print(f"[render] skip {name}_si{si} (missing precompute/skel2d)", flush=True); continue
            rend, d, skel = render_from_precompute(pre, skf, (m3S, vae, z))
            # Wan 因果VAE: 12 latent 帧 decode -> 45 像素帧 (1+(12-1)*4), 非 48 -> 取实际长度 Tp 对齐 overlay
            Tp = min(rend[0].shape[0], rend[1].shape[0]); traj = [d["trajA"], d["trajB"]]; ef = [d["efA"], d["efB"]]
            grasp = d["grasp"]; ref0 = [np.nan_to_num(z["tracks"][si, 0].astype(np.float32)),
                                        np.nan_to_num(z["tracks_low"][si, 0].astype(np.float32))]
            rcols, fcols = [], []
            for v in range(2):
                Rr = np.stack([r128(u8(rend[v][t])) for t in range(Tp)])
                # overlay: 静态起始(绿) + ②/兜底物体(红) + eef(黄) + skel(白)
                Fl = np.stack([ov(Rr[t], ref0[v], traj[v][t], ef[v][t], sk=skel[v][t]) for t in range(Tp)])
                rcols.append(Rr); fcols.append(Fl)
            cmdstr = " ".join(f"{n}{_AR.get(c, c)}" for c, n in script)
            gstate = "grip %d->%d%%" % (100 * d["grip"][0] / 0.04, 100 * d["grip"][Tp-1] / 0.04)
            gif_k0 = int(os.environ.get("GIF_K0", str(K)))     # 冷启动demo标签从0起(GIF_K0=0); keyboard默认K
            save_combined_gif(f"{OUT}/gifs/{name}_si{si}.gif", np.stack(rcols), np.stack(fcols),
                              ["cam_high", "cam_low"], [None, None], gif_k0,
                              caption=f"{name} | {cmdstr} | {gstate} | grasp{grasp[:Tp].mean():.0%} | H={Tp}")
            print(f"[render] {name}_si{si} done (Tp={Tp} grasp {grasp[:Tp].mean():.0%} fb {d['fb'].mean():.0%})", flush=True)
    open(f"{OUT}/README.txt", "w").write(
        "keyboard 3D demo (视频架构skel③). 双视角并列: Rendered行 + Flow+Skel overlay行(②/兜底物体红+静态起始绿+eef黄+skel白线).\n"
        f"scripts={list(SCRIPTS)} seqs={SEQS} DELTA={DELTA} REPEAT={REPEAT}. "
        f"②={os.path.basename(os.path.dirname(WM_CKPT))} ③={os.path.basename(os.path.dirname(CK_SKEL))} IK={os.path.basename(os.path.dirname(IK_CKPT))}.\n")
    print("=== render DONE ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["precompute", "render"], required=True)
    a = ap.parse_args()
    (stage_precompute if a.stage == "precompute" else stage_render)()
