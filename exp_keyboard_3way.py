"""Keyboard 三列对比 (GT | IWS-naive co-train | ours object-flow).

IWS-naive = hyeonhoo 的 H+R co-train stage-2 ckpt (checkpoints/epoch=3-step=250000.ckpt):
  2 view (camera_0=cam_high, camera_1=cam_right_wrist) + latent 32ch + 7D eef action.
  load 配方见 memory project_stage1_hr_checkpoint / spec 2026-06-27.

本文件先实现 Task1+2:load IWS WM + clip<->parquet world-action 对齐 (图像 MSE) + 2-view replay,
眼检 IWS replay 是否跟 GT 一致。ours 列与 keyboard 合成在后续 task 接入。
"""
import os, glob, importlib.util
import numpy as np, torch, cv2
from omegaconf import OmegaConf
from einops import rearrange

OmegaConf.register_new_resolver("eval", lambda e: eval(e, {"np": np}), replace=True)
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x), replace=True)

from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import LatentWorldModel
from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
# ours ③/②/IK unpickle 类必须在 __main__ namespace 才能 torch.load (见 feedback)
os.environ.setdefault("GRIP", "1"); os.environ.setdefault("REN_KIND", "detmem"); os.environ.setdefault("IK", "1")
from exp_scel_latent_detmem import DetMemRenderer, render_detmem        # noqa: F401
from exp_scel_grip_wm import GripLWC, rollout_grip                      # noqa: F401
from exp_scel_ik_adapter import IKAdapter                              # noqa: F401
from exp_detmem_eef_cotrain import render_eef as render_noflow, eef_splat  # 第三列: ours 架构去flow co-train
NOFLOW_PT = os.environ.get("NOFLOW_PT", "outputs/cross_embodiment_wm/detmem_eef_cotrain_rh/detmem_eef.pt")

DEVICE = "cuda"
IMG = 128
CKPT = "checkpoints/epoch=3-step=250000.ckpt"
NPZ = "outputs/flow_render_dataset_v3_grip/clips_robot.npz"
V3_CROP = (190, 225, 210, 205)            # x,y,w,h  cam_high workspace (ours 同款)
OUT = "outputs/cross_embodiment_wm/keyboard_3way_iws"
# action normalizer 区间 (sanity 用,校验构造的 action 落在区间内)
NORM_MIN = np.array([-0.274, -0.120, 0.099, -0.086, 0.102, 1.549, -0.001])
NORM_MAX = np.array([0.199, 0.305, 0.101, 0.037, 0.281, 1.658, 0.042])

# ── 借 eval_stage1_hr_alignment 的 build_cfg / normalizer helper ─────────────
_spec = importlib.util.spec_from_file_location("_hr", "scripts/eval/eval_stage1_hr_alignment.py")
_hr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_hr)


def load_iws_wm(ckpt=CKPT, device=DEVICE):
    """唯一能 load 这个裸 ckpt 的配方 (见 memory project_stage1_hr_checkpoint)."""
    cfg = _hr.build_cfg()
    cfg.training_stage = 2
    cfg.dynamo_ssl.encoder_backbone = "conv2d"     # ckpt 是 plain conv encoder.0..8
    model = LatentWorldModel(cfg)
    for k in ["camera_0_color", "camera_1_color"]:
        model.normalizer[k] = _hr.get_image_range_normalizer()
        model.normalizer[f"{k}_mask"] = _hr.get_image_range_normalizer()
    model.normalizer["action"] = _hr.SingleFieldLinearNormalizer.create_manual(
        scale=np.ones(7, np.float32), offset=np.zeros(7, np.float32),
        input_stats_dict={s: np.zeros(7, np.float32) if s in ("min", "mean")
                          else np.ones(7, np.float32) for s in ("min", "max", "mean", "std")})
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    res = model.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if "validation_fvd" not in k] + \
          [k for k in res.unexpected_keys if "validation_fvd" not in k]
    assert not bad, f"load not clean: {bad[:6]}"
    return model.eval().to(device)


# ── clip <-> parquet world-action 对齐 (图像 MSE) ───────────────────────────
def _crop128(img):
    x, y, w, h = V3_CROP
    return cv2.resize(img[y:y + h, x:x + w], (IMG, IMG))


def decode_cam(robot_num, cam="cam_high"):
    """解码 v3 robot 某相机视频 (robot_num=1..18). cam_high 用 V3_CROP→128; wrist 整图 resize→128."""
    import av
    d = f"human_play_data/play_robot_v3_{robot_num}_eef"
    key = "cam_high" if cam == "cam_high" else "cam_right_wrist"
    path = f"{d}/videos/chunk-000/observation.images.{key}/episode_000000.mp4"
    cont = av.open(path); out = []
    for fr in cont.decode(video=0):
        rgb = fr.to_ndarray(format="rgb24")
        out.append(_crop128(rgb) if cam == "cam_high" else cv2.resize(rgb, (IMG, IMG)))
    return np.stack(out)


def locate_clip_start(clip_frame0_u8, vid_video_u8):
    diff = vid_video_u8.astype(np.int32) - clip_frame0_u8.astype(np.int32)[None]
    mse = (diff * diff).reshape(len(vid_video_u8), -1).mean(1)
    return int(mse.argmin())


def clip_world_action(npz, i, cache):
    """npz clip i -> 同帧 (L,7) world action (pos3+euler3+grip1) + wrist clip + start."""
    import pandas as pd
    vid = int(npz["vid"][i])
    rn = vid - 100 + 1            # vid 100..117 -> play_robot_v3_1..18
    if vid not in cache:
        d = f"human_play_data/play_robot_v3_{rn}_eef"
        df = pd.read_parquet(sorted(glob.glob(f"{d}/data/**/*.parquet", recursive=True))[0])
        # IWS 训练用 action_right_* 列(z=0.1 匹配 normalizer;obs_right z=0.08 不匹配)
        pos = np.stack(df["action_right_ee_position"]).astype(np.float32)
        eul = np.stack(df["action_right_ee_euler_xyz"]).astype(np.float32)
        grp = np.asarray(df["action_right_gripper"]).astype(np.float32)[:, None]
        act_all = np.concatenate([pos, eul, grp], 1)
        cache[vid] = (decode_cam(rn, "cam_high"), decode_cam(rn, "wrist"), act_all)
    vid_high, vid_wrist, act_all = cache[vid]
    s = locate_clip_start(npz["frames"][i][0], vid_high)
    L = npz["frames"].shape[1]
    return act_all[s:s + L], vid_wrist[s:s + L], s


# ── IWS 2-view rollout ──────────────────────────────────────────────────────
@torch.no_grad()
def iws_rollout(model, cam_high0_u8, wrist0_u8, actions_raw, device=DEVICE):
    """init = frame0 (cam_high + wrist), actions_raw (T,7) world. -> cam_high pred (T,128,128,3)."""
    def prep(u8):
        return torch.from_numpy(u8).float().div(255).permute(2, 0, 1)[None, None].to(device)
    cam0 = model.normalizer["camera_0_color"].normalize(prep(cam_high0_u8))     # (1,1,3,H,W)
    cam1 = model.normalizer["camera_1_color"].normalize(prep(wrist0_u8))
    obs = torch.cat([cam0, cam1], dim=2)                                        # (1,1,6,H,W)
    z0 = model.encoder_forward(rearrange(obs, "b t c h w -> (b t) c h w"))
    z0 = rearrange(z0, "(b t) c h w -> b t c h w", b=1)                         # (1,1,32,Hl,Wl)
    act = model.normalizer["action"].normalize(
        torch.from_numpy(actions_raw).float()[None].to(device))                # (1,T,7)
    z_pred = model.dynamics_forward(z0, act)                                    # (1,T-1,32,Hl,Wl)
    z_all = torch.cat([z0, z_pred], dim=1)[0]                                   # (T,32,Hl,Wl)
    px = render_img_cm(model, z_all, IMG, model.normalizer, num_views=2, batch_size=50)  # (T,6,H,W)
    cam_high = px[:, :3].clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    return (cam_high * 255).astype(np.uint8)


# ── ours (object-flow ②③ detmem) replay ────────────────────────────────────
def load_ours(device=DEVICE):
    """复用 exp_scel_keyboard 的 grip② + detmem③ + IK adapter (GRIP/detmem 模式)."""
    os.environ.setdefault("GRIP", "1"); os.environ.setdefault("REN_KIND", "detmem")
    os.environ.setdefault("IK", "1")
    import exp_scel_keyboard as KB
    KB.load_gmask()
    ren = torch.load(KB.DETMEM_PT, map_location=device, weights_only=False).to(device).eval()
    wm = torch.load(KB.WM, map_location=device, weights_only=False).to(device).eval()
    adapter = (torch.load(KB.ADAPTER_PT, map_location=device, weights_only=False).to(device).eval()
               if os.path.exists(KB.ADAPTER_PT) else None)
    return KB, ren, wm, adapter


@torch.no_grad()
def ours_replay(KB, ren, wm, adapter, z, si, Hd, device=DEVICE):
    """真实 eef/grip → grip② cube flow → detmem③ → cam_high frames (Hd,128,128,3) u8, frame K..K+Hd."""
    K = KB.K
    lt = z["tracks"].astype(np.float32); le = z["eef"].astype(np.float32)
    lv = z["vis"].astype(np.float32); lf = z["frames"]; lg = z["grip"].astype(np.float32)
    ef_full = le[si, :K + Hd][None]; grip_full = lg[si, :K + Hd][None]
    pr = KB.rollout_grip(wm, torch.from_numpy(lt[si:si + 1, :K]).float().to(device),
                         torch.from_numpy(ef_full).float().to(device),
                         torch.from_numpy(grip_full).float().to(device), Hd).cpu().numpy()[0]
    ef_render = le[si, K:K + Hd]; vis_seq = lv[si, K:K + Hd]
    if adapter is not None:
        jt_use = adapter(torch.from_numpy(ef_render).float().to(device)).cpu().numpy()
    else:
        jt_use = z["joint"].astype(np.float32)[si, K:K + Hd]
    rseq = np.asarray(KB.render_detmem(ren, lf[si, 0], lt[si], ef_render, vis_seq, jt_use, pr))
    if rseq.dtype != np.uint8:
        rseq = (np.clip(rseq, 0, 1) * 255).astype(np.uint8)
    return rseq


# ── metric ──────────────────────────────────────────────────────────────────
def detect_cube(frame_u8):
    hsv = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (((h > 163) | (h < 10)) & (s > 90) & (v > 35) & (v < 210)).astype(np.uint8)
    if mask.sum() < 8:
        return None
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()), float(ys.mean()))


def cube_pos_err(pred, gt):
    """vs GT cube 位置误差; 必报检测率 + 消失率 (防 nan trap)."""
    errs = []; det = 0; vanish = 0; n = 0
    for pf, gf in zip(pred, gt):
        gc = detect_cube(gf)
        if gc is None:
            continue
        n += 1; pc = detect_cube(pf)
        if pc is None:
            vanish += 1; continue
        det += 1; errs.append(float(np.hypot(pc[0] - gc[0], pc[1] - gc[1])))
    return dict(cube_px=float(np.mean(errs)) if errs else float("nan"),
                det_rate=det / max(n, 1), vanish=vanish / max(n, 1))


def psnr(pred, gt):
    mse = np.mean((pred.astype(np.float32) / 255 - gt.astype(np.float32) / 255) ** 2)
    return float(10 * np.log10(1.0 / (mse + 1e-10)))


def label_cols(frame, names, w=IMG):
    """在拼好的多列帧每列左上角写列名 (RGB, 黄字)."""
    f = np.ascontiguousarray(frame)
    for i, nm in enumerate(names):
        cv2.putText(f, nm, (i * w + 3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(f, nm, (i * w + 3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    return f


def run_replay_3way(seq_idxs, out=OUT):
    """三列 GT | IWS-naive | ours, 对齐 frame K..L-1, 出 gif + 质量 metric (summary.txt)."""
    import imageio
    z = np.load(NPZ)
    iws_model = load_iws_wm()
    KB, ren, wm, adapter = load_ours()
    K = KB.K
    noflow_m = (torch.load(NOFLOW_PT, map_location=DEVICE, weights_only=False).to(DEVICE).eval()
                if os.path.exists(NOFLOW_PT) else None)
    print(f"[noflow third column] {'loaded '+NOFLOW_PT if noflow_m is not None else 'NOT FOUND, skip'}", flush=True)
    try:
        import lpips as _lp; lpfn = _lp.LPIPS(net="alex").to(DEVICE).eval()
    except Exception as e:
        print(f"[lpips skip] {e}"); lpfn = None

    def lpips_seq(pred, gt):
        if lpfn is None:
            return float("nan")
        with torch.no_grad():
            def t(x): return (torch.from_numpy(x).float().permute(0, 3, 1, 2) / 127.5 - 1).to(DEVICE)
            return float(lpfn(t(pred), t(gt)).mean())

    os.makedirs(f"{out}/gifs", exist_ok=True); cache = {}
    rows = []
    for si in seq_idxs:
        gt = z["frames"][si]; L = len(gt); Hd = L - K
        act, wrist_clip, s = clip_world_action(z, si, cache)
        iws = iws_rollout(iws_model, gt[K], wrist_clip[K], act[K:L])
        ours = ours_replay(KB, ren, wm, adapter, z, si, Hd)
        le_si = z["eef"].astype(np.float32)[si]
        cols = {"GT": gt[K:L], "IWS-naive": iws}
        if noflow_m is not None:
            cols["ours-noflow"] = (np.clip(render_noflow(noflow_m, gt[0], le_si[K:L]), 0, 1) * 255).astype(np.uint8)
        cols["ours"] = ours
        T = min(len(v) for v in cols.values())
        cols = {k: v[:T] for k, v in cols.items()}; names = list(cols.keys())
        frames = [label_cols(np.concatenate([cols[k][t] for k in names], 1), names) for t in range(T)]
        imageio.mimsave(f"{out}/gifs/seq{si}_replay3.gif", frames, fps=6)
        gtw = cols["GT"]
        md = lambda p: dict(psnr=psnr(p, gtw), lpips=lpips_seq(p, gtw), **cube_pos_err(p, gtw))
        m = {"si": si, "iws": md(cols["IWS-naive"]), "ours": md(cols["ours"])}
        if "ours-noflow" in cols: m["noflow"] = md(cols["ours-noflow"])
        rows.append(m)
        print(f"[seq{si}] IWS psnr={m['iws']['psnr']:.1f} cube={m['iws']['cube_px']:.1f}"
              + (f" | noflow psnr={m['noflow']['psnr']:.1f} cube={m['noflow']['cube_px']:.1f}" if "noflow" in m else "")
              + f" | ours psnr={m['ours']['psnr']:.1f} cube={m['ours']['cube_px']:.1f}", flush=True)

    # summary.txt
    def agg(side, key):
        vals = [r[side][key] for r in rows if not np.isnan(r[side][key])]
        return float(np.mean(vals)) if vals else float("nan")
    sides = ["iws"] + (["noflow"] if any("noflow" in r for r in rows) else []) + ["ours"]
    sname = {"iws": "IWS-naive", "noflow": "ours-noflow", "ours": "ours"}
    with open(f"{out}/summary.txt", "w") as f:
        f.write("Keyboard 四列 replay 质量对比 (GT | IWS-naive | ours-noflow | ours)\n")
        f.write(f"seqs={[r['si'] for r in rows]}  对齐 frame K..L-1, cam_high\n")
        f.write("IWS-naive = hyeonhoo 2view VAE+latent eef naive co-train | "
                "ours-noflow = 同 VAE+latent transition(detmem) eef 条件 NO flow, robot+human co-train | "
                "ours = object-flow ②③(detmem + flow)\n")
        f.write("★ ablation: ours vs ours-noflow 隔离 object-flow; ours-noflow vs IWS 同(无flow eef co-train)不同实现\n\n")
        f.write(f"{'metric':<12}" + "".join(f"{sname[s]:>14}" for s in sides) + "   (PSNR↑ LPIPS↓ cube_px↓ det↑ vanish↓)\n")
        for key in ["psnr", "lpips", "cube_px", "det_rate", "vanish"]:
            f.write(f"{key:<12}" + "".join(f"{agg(s,key):>14.3f}" for s in sides) + "\n")
        f.write("\nper-seq:\n")
        for r in rows:
            f.write(f"  seq{r['si']:<5}" + " | ".join(
                f"{sname[s]} psnr={r[s]['psnr']:.1f} lpips={r[s]['lpips']:.3f} cube={r[s]['cube_px']:.1f}" for s in sides if s in r) + "\n")
    print(f"=> {out}/summary.txt + {len(rows)} gifs", flush=True)


# ── keyboard 合成可控 demo (IWS-naive | ours) ───────────────────────────────
def calib_world_to_img(world_xy, img_uv):
    """从同一 eef 的 (world xy, 图像 uv) 拟合 world→image 2x2 (两表示同源,见对话)."""
    X = np.c_[world_xy, np.ones(len(world_xy))]
    A = np.linalg.lstsq(X, img_uv, rcond=None)[0]      # (3,2)
    return A[:2].T                                       # (2img,2world): img_delta = A2 @ world_delta


_IMG_DIRS = {"right": (1, 0), "left": (-1, 0), "up": (0, -1), "down": (0, 1)}   # image du,dv


def synth_iws_action(start7, script, Jinv, step_px=4.0, repeat=5, grip_open=0.040, grip_close=0.0):
    """keyboard 指令 → IWS world action (T,7). 平移经 Jinv 把图像方向转 world dim0/1; open/close ramp dim6."""
    cur = start7.astype(np.float32).copy(); out = [cur.copy()]
    for name, cnt in script:
        n = cnt * repeat
        for _ in range(n):
            cur = cur.copy()
            if name in _IMG_DIRS:
                du, dv = _IMG_DIRS[name]
                dxy = Jinv @ np.array([du * step_px, dv * step_px])
                cur[0] += dxy[0]; cur[1] += dxy[1]
            elif name == "open":
                cur[6] = min(grip_open, cur[6] + grip_open / n)
            elif name == "close":
                cur[6] = max(grip_close, cur[6] - grip_open / n)
            out.append(cur)
    return np.stack(out)


def ours_synth_eef(KB, ef_last3, script, step_img, g_start, g_close, g_open, repeat):
    """ours 合成 eef(图像空间, 用 _IMG_DIRS 与 IWS 同方向同量) + grip. 复用 KB 指尖开合."""
    base_c = ef_last3.mean(0); off0 = ef_last3 - base_c
    c = base_c.astype(np.float32).copy(); g = float(g_start); efs = []; grips = []
    for name, cnt in script:
        n = cnt * repeat
        for _ in range(n):
            if name in _IMG_DIRS:
                du, dv = _IMG_DIRS[name]; c = c + np.array([du, dv], np.float32) * step_img
            elif name == "open":
                g = min(g_open, g + (g_open - g_close) / n)
            elif name == "close":
                g = max(g_close, g - (g_open - g_close) / n)
            off = KB._fingertip_offset(off0, g, g_close, g_open)
            efs.append(c[None] + off); grips.append(g)
    return np.stack(efs).astype(np.float32), np.array(grips, np.float32)


@torch.no_grad()
def ours_drive(KB, ren, wm, adapter, z, si, ef_fut, grip_fut, device=DEVICE):
    """ours 合成: 真实 K 帧 history + 合成 ef_fut/grip_fut → grip② → detmem③ cam_high (Hd,128,128,3)."""
    K = KB.K
    lt = z["tracks"].astype(np.float32); le = z["eef"].astype(np.float32)
    lv = z["vis"].astype(np.float32); lf = z["frames"]; lg = z["grip"].astype(np.float32)
    Hd = len(ef_fut)
    ef_full = np.concatenate([le[si, :K], ef_fut], 0)[None]
    grip_full = np.concatenate([lg[si, :K], grip_fut], 0)[None]
    pr = KB.rollout_grip(wm, torch.from_numpy(lt[si:si + 1, :K]).float().to(device),
                         torch.from_numpy(ef_full).float().to(device),
                         torch.from_numpy(grip_full).float().to(device), Hd).cpu().numpy()[0]
    ef_render = ef_full[0, K:K + Hd]
    vis_seq = lv[si, K:K + Hd] if K + Hd <= lv.shape[1] else np.ones((Hd, lt.shape[2]), np.float32)
    jt = (adapter(torch.from_numpy(ef_render).float().to(device)).cpu().numpy()
          if adapter is not None else np.repeat(z["joint"].astype(np.float32)[si, K:K + 1], Hd, 0))
    rseq = np.asarray(KB.render_detmem(ren, lf[si, 0], lt[si], ef_render, vis_seq, jt, pr))
    if rseq.dtype != np.uint8:
        rseq = (np.clip(rseq, 0, 1) * 255).astype(np.uint8)
    return rseq, pr


SYNTH_SCRIPTS = {                                    # box 绕圈回原点 = 分布内 (避静止 OOD 自漂移, 见 memory)
    "box": [("right", 2), ("down", 2), ("left", 2), ("up", 2)],
    "box_grip": [("close", 1), ("right", 2), ("down", 2), ("left", 2), ("up", 2), ("open", 1)],
}


def _dir_follow(cube_uv, script):
    """cube 实际位移方向 vs 第一个移动指令方向的一致性 (cos)."""
    move = next((nm for nm, _ in script if nm in _IMG_DIRS), None)
    if move is None or len(cube_uv) < 2:
        return float("nan")
    d = cube_uv[-1] - cube_uv[0]
    if np.linalg.norm(d) < 1:
        return 0.0
    tgt = np.array(_IMG_DIRS[move], float)
    return float(np.dot(d, tgt) / (np.linalg.norm(d) * np.linalg.norm(tgt) + 1e-8))


def run_synth_2way(seq_idxs, out=OUT, step_px=4.0):
    """同键盘指令驱动 IWS-naive vs ours, 两列 gif + 可控 metric (cube travel + 方向跟随)."""
    import imageio
    z = np.load(NPZ)
    iws_model = load_iws_wm(); KB, ren, wm, adapter = load_ours()
    noflow_m = (torch.load(NOFLOW_PT, map_location=DEVICE, weights_only=False).to(DEVICE).eval()
                if os.path.exists(NOFLOW_PT) else None)
    K = KB.K; lg = z["grip"].astype(np.float32); le = z["eef"].astype(np.float32)
    g_close = float(np.percentile(lg, 5)); g_open = float(np.percentile(lg, 95))
    os.makedirs(f"{out}/gifs", exist_ok=True); cache = {}; lines = []
    for si in seq_idxs:
        act, wrist_clip, s = clip_world_action(z, si, cache)
        L = len(act)
        Jinv = np.linalg.inv(calib_world_to_img(act[:L, :2], le[si, :L].mean(1) * IMG))
        g_start = float(lg[si, K - 1])
        step_img = float(KB.DELTA)            # ours 图像归一步长; IWS 用同等像素量
        for sname, script in SYNTH_SCRIPTS.items():
            # IWS: init frame K + 合成 world action (图像位移 step_img*IMG px 经 Jinv 转 world)
            iws_act = synth_iws_action(act[K], script, Jinv, step_img * IMG, KB.REPEAT, g_open, g_close)
            iws = iws_rollout(iws_model, z["frames"][si][K], wrist_clip[K], iws_act)
            # ours: 合成 eef + grip (图像空间, 同 _IMG_DIRS 方向同量)
            ef_fut, grip_fut = ours_synth_eef(KB, le[si, K - 1], script, step_img, g_start, g_close, g_open, KB.REPEAT)
            ours, pr = ours_drive(KB, ren, wm, adapter, z, si, ef_fut, grip_fut)
            cols = {"IWS-naive": iws}
            if noflow_m is not None:
                cols["ours-noflow"] = (np.clip(render_noflow(noflow_m, z["frames"][si][0], ef_fut), 0, 1) * 255).astype(np.uint8)
            cols["ours"] = ours
            T = min(len(v) for v in cols.values())
            cols = {k: v[:T] for k, v in cols.items()}; names = list(cols.keys())
            frames = [label_cols(np.concatenate([cols[k][t] for k in names], 1), names) for t in range(T)]
            imageio.mimsave(f"{out}/gifs/seq{si}_{sname}.gif", frames, fps=6)
            # 可控 metric: box 绕圈→ net(回原点,小=跟手) + path(路程,大=动过) (IWS 像素检测 cube, ours ② flow)
            uvof = lambda imgs: np.array([detect_cube(f) if detect_cube(f) is not None else [np.nan, np.nan] for f in imgs], float)
            iws_uv = uvof(cols["IWS-naive"]); ours_uv = pr.mean(1)[:T] * IMG
            net = lambda uv: float(np.linalg.norm(uv[np.isfinite(uv).all(1)][-1] - uv[np.isfinite(uv).all(1)][0])) if np.isfinite(uv).all(1).sum() > 1 else float("nan")
            path = lambda uv: float(np.nansum(np.linalg.norm(np.diff(uv, axis=0), axis=1)))
            nf_str = (f" | noflow net={net(uvof(cols['ours-noflow'])):5.1f} path={path(uvof(cols['ours-noflow'])):5.1f}"
                      if "ours-noflow" in cols else "")
            lines.append(f"seq{si} {sname:<8} IWS net={net(iws_uv):5.1f} path={path(iws_uv):5.1f}{nf_str}"
                         f" | ours net={net(ours_uv):5.1f} path={path(ours_uv):5.1f}px  (box: net小+path大=跟手回原点)")
            print(lines[-1], flush=True)
    with open(f"{out}/summary_synth.txt", "w") as f:
        f.write("keyboard 合成可控 demo (IWS-naive | ours, 同指令). travel=cube首末像素位移; follow=cube方向 vs 指令方向 cos\n")
        f.write("IWS cube 从渲染像素 HSV 检测; ours cube 从 ② flow.\n\n" + "\n".join(lines) + "\n")
    print(f"=> {out}/summary_synth.txt + synth gifs", flush=True)


def _sanity(seq_idx=0, wrist_zero=False):
    z = np.load(NPZ)
    gt = z["frames"][seq_idx]                                  # (L,128,128,3) cam_high GT
    act, wrist_clip, s = clip_world_action(z, seq_idx, {})
    print(f"[seq{seq_idx}] vid={int(z['vid'][seq_idx])} start={s} L={len(gt)}")
    print(f"  action min/max per dim:\n   min {act.min(0).round(3)}\n   max {act.max(0).round(3)}")
    inside = (act.min(0) >= NORM_MIN - 1e-2).all() and (act.max(0) <= NORM_MAX + 1e-2).all()
    print(f"  action 落在 normalizer 区间内: {inside}")
    w0 = np.zeros_like(gt[0]) if wrist_zero else wrist_clip[0]
    model = load_iws_wm()
    pred = iws_rollout(model, gt[0], w0, act)
    os.makedirs(f"{OUT}/diag", exist_ok=True)
    K = min(8, len(pred))
    idxs = np.linspace(0, len(pred) - 1, K).astype(int)
    rows = [np.concatenate([gt[t] for t in idxs], 1),
            np.concatenate([pred[t] for t in idxs], 1)]
    img = cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR)
    p = f"{OUT}/diag/task1_replay_seq{seq_idx}{'_wz' if wrist_zero else ''}.png"
    cv2.imwrite(p, img); print(f"  saved {p}  (上 GT | 下 IWS replay, cam_high)")


if __name__ == "__main__":
    mode = os.environ.get("MODE", "replay3")
    if "SEQS" in os.environ:
        seqs = [int(x) for x in os.environ["SEQS"].split(",")]
    else:                                            # 默认: 复用 ours keyboard demo 的 seq (固定 layout)
        import exp_scel_keyboard as KB
        _z = np.load(NPZ); _lt = _z["tracks"].astype(np.float32); _le = _z["eef"].astype(np.float32)
        _ho = np.random.default_rng(0).permutation(len(_lt))[:KB.HELDOUT]
        _hi = sorted(set(KB.HO_IDX) | set(KB.pick_central(_lt, _ho, KB.NAUTO, _le)))
        seqs = [int(_ho[h]) for h in _hi]
        print(f"seqs (ours 选择 HO_IDX{KB.HO_IDX}+pick_central×{KB.NAUTO}) = {seqs}", flush=True)
    if mode == "sanity":
        _sanity(int(os.environ.get("SEQ", 0)), wrist_zero=bool(int(os.environ.get("WZ", "0"))))
    elif mode == "synth":
        run_synth_2way(seqs)
    else:
        run_replay_3way(seqs)
