"""组装 9 通道交互草图(2026-07-31 spec + world_dz)。SRC=robot|human。3D世界系算->投影双视角2D->pool16。
robot/human 的 skel2d+grip 都从各自现成 sidecar 读(不运行时跑 IK/FK/pinocchio)。在 .venv_wan 跑。
通道序: [flow_img(3), world_dz(1), skel(1), grip(1), trace(1), attachment(1), contact_splat(1)] = 9。
用法: SRC=robot .venv_wan/bin/python build_sketch.py   (env: SMOKE=1 只前4条)
"""
import os, numpy as np, cv2
import sketch_lib as S
from contact_detector import detect_contact_2d

IMG = 128; GRID = 16
WARP = os.environ.get("WARP") == "1"        # ★加warp(3): 12ch cond(9草图+warp3), 存 _warp.npz. grip+warp证明很重要
NCH = 12 if WARP else 9
_G = None
def _warp_mod():
    global _G
    if _G is None:
        import exp_scel_dualview_gmaskcond as G; _G = G
    return _G
RB = "outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz"
HM = "outputs/flow_render_dataset_can_dual_L48/clips_human_L48_retrack_realwrist_3d.npz"  # ★带tracks3d(augment加, depth反投影)
SKELF_R = "outputs/flow_render_dataset_can_dual/skel_sidecar_robot.npz"
SKELF_H = "outputs/flow_render_dataset_can_dual/skel_sidecar_human_ik.npz"
OUTDIR = "outputs/video_arch_wm/sketch_can_dual"; os.makedirs(OUTDIR, exist_ok=True)
_SKR = np.load(SKELF_R); _SEG = _SKR["segments"]
_SKH = np.load(SKELF_H)


def skel_chan(pts2d):
    img = np.zeros((IMG, IMG), np.float32); P = (pts2d * IMG).astype(np.int32)
    for a, b in _SEG:
        if np.all(np.abs(pts2d[a]) < 3) and np.all(np.abs(pts2d[b]) < 3):
            cv2.line(img, tuple(P[a]), tuple(P[b]), 1.0, 3, cv2.LINE_AA)
    for i, p in enumerate(P):
        if np.all(np.abs(pts2d[i]) < 3): cv2.circle(img, tuple(p), 3, 1.0, -1)
    return img


def load_robot_arrays():
    z = np.load(RB)
    keys = ["tracks", "tracks_low", "tracks3d", "tracks3d_valid", "eef", "eef_low", "vis", "vis_low", "eef3d", "grip", "frames", "frames_low", "vid"]
    A = {k: z[k][:] for k in keys}
    A["skel2d_high"] = _SKR["skel2d_high"][:]; A["skel2d_low"] = _SKR["skel2d_low"][:]
    A["_valid_idx"] = np.arange(A["tracks"].shape[0])
    return A


def load_human_arrays():
    z = np.load(HM)
    keys = ["tracks", "tracks_low", "tracks3d", "tracks3d_valid", "eef", "eef_low", "vis", "vis_low", "eef3d", "frames", "frames_low", "vid"]
    A = {k: z[k][:] for k in keys}
    A["skel2d_high"] = _SKH["skel2d_high"][:]; A["skel2d_low"] = _SKH["skel2d_low"][:]
    # ★human grip 用自身range重算(sidecar grip 是旧robot-range映射, 饱和near-open p50=0.04死信号);
    #   自身range后 mean~0.022 与 robot~0.026 一致, contact/attachment 才活。skel手指开合仍是sidecar的(cosmetic).
    heef = np.nan_to_num(z["eef3d"].astype(np.float64))
    gap = np.linalg.norm(heef[:, :, 1] - heef[:, :, 2], axis=-1)
    glo, ghi = np.nanpercentile(gap, [5, 95])
    A["grip"] = (np.clip((gap - glo) / (ghi - glo + 1e-6), 0, 1) * S.GRIP_MAX).astype(np.float32)
    e3 = z["eef3d"]; t2 = z["tracks"]
    valid = np.isfinite(e3.reshape(len(e3), e3.shape[1], -1)).all(-1).all(-1) & \
            np.isfinite(t2.reshape(len(t2), t2.shape[1], -1)).all(-1).all(-1)   # (N,) per-clip 全帧有效
    A["_valid_idx"] = np.where(valid)[0]
    return A


def build_clip_sketch(A, n, tL, L, src):
    """9通道: [flow_img(3), world_dz(1), skel(1), grip(1), trace(1), attach(1), contact_splat(1)]。
    ★object-flow 图像2D(两域都有tracks); ★world_dz 从tracks3d(depth反投影, 物理竖直, 把lift从平移分离);
    trace走eef3d投影; skel从sidecar。"""
    f32 = lambda a: np.nan_to_num(np.asarray(a, np.float32))
    tr2d = [f32(A["tracks"][n]), f32(A["tracks_low"][n])]
    tr3d = f32(A["tracks3d"][n]); tr3dv = np.asarray(A["tracks3d_valid"][n], bool)   # (L,K,3),(L,K)
    eef3d = f32(A["eef3d"][n]); grip = f32(A["grip"][n])
    ef = [f32(A["eef"][n]), f32(A["eef_low"][n])]; vs = [f32(A["vis"][n]), f32(A["vis_low"][n])]
    fr = [A["frames"][n], A["frames_low"][n]] if WARP else None
    contact = [detect_contact_2d(tr2d[v], ef[v], grip) for v in range(2)]   # 每视角 (att(L,), cpt(L,2))
    out = np.zeros((2, NCH, tL, GRID, GRID), np.float32)
    views = ["high", "low"]
    for k in range(tL):
        rf = 0 if k == 0 else min(4 * k, L - 1)
        dz = S.object_dz_scalar(tr3d[0], tr3d[rf], tr3dv[0] & tr3dv[rf])                               # 物体世界z位移(标量)
        for v in range(2):
            flow = S.object_flow_2d(tr2d[v][0], tr2d[v][rf], ef[v][0], ef[v][rf], vs[v][rf])           # (3,128,128)
            dzc = S.world_dz_channel(tr2d[v][rf], dz, vs[v][rf])                                       # (1,128,128) 物理竖直
            sk = skel_chan(A["skel2d_high" if v == 0 else "skel2d_low"][n, rf])[None]                  # (1,128,128)
            gp = S.grip_channel(grip[rf])                                                              # (1,128,128)
            trc = S.agent_trace_channel(eef3d[:rf + 1], views[v])                                      # (1,128,128)
            att_v, cpt_v = contact[v]
            ct = S.contact_channels_2d(att_v[rf], cpt_v[rf])                                           # (2,128,128) [attach,splat]
            chs = [flow, dzc, sk, gp, trc, ct]                                                         # 9ch 抽象草图
            if WARP:
                chs.append(_warp_mod().warp_preview(fr[v][0], tr2d[v][0], tr2d[v][rf], vs[v][rf]))     # +warp(3)=12ch
            out[v, :, k] = S.pool16(np.concatenate(chs, 0))
    return out.astype(np.float16)


def viz_clip(A, n, src, view_i, out_png):
    """出 8 通道 原帧|通道|叠加 三联(128分辨率, 取 rf 中段一帧)供眼检。"""
    f32 = lambda a: np.nan_to_num(np.asarray(a, np.float32))
    view = "high" if view_i == 0 else "low"
    frames = A["frames"] if view_i == 0 else A["frames_low"]
    L = A["tracks"].shape[1]; rf = min(20, L - 1)
    tr2d = f32(A["tracks"][n] if view_i == 0 else A["tracks_low"][n])
    eef3d = f32(A["eef3d"][n]); grip = f32(A["grip"][n])
    ef = f32(A["eef"][n] if view_i == 0 else A["eef_low"][n])
    vs = f32(A["vis"][n] if view_i == 0 else A["vis_low"][n])
    att, cpt = detect_contact_2d(tr2d, ef, grip)
    sk2d = A["skel2d_high" if view_i == 0 else "skel2d_low"][n, rf]
    tr3d = f32(A["tracks3d"][n]); tr3dv = np.asarray(A["tracks3d_valid"][n], bool)
    dz = S.object_dz_scalar(tr3d[0], tr3d[rf], tr3dv[0] & tr3dv[rf])
    chans = {
        "flow": S.object_flow_2d(tr2d[0], tr2d[rf], ef[0], ef[rf], vs[rf]),
        "world_dz": S.world_dz_channel(tr2d[rf], dz, vs[rf]),
        "skel": skel_chan(sk2d)[None], "grip": S.grip_channel(grip[rf]),
        "trace": S.agent_trace_channel(eef3d[:rf + 1], view),
        "contact": S.contact_channels_2d(att[rf], cpt[rf]),
    }
    bg = frames[n, rf].astype(np.uint8)
    rows = []
    for name, ch in chans.items():
        c = ch[0] if ch.shape[0] == 1 else (ch[:3] if ch.shape[0] >= 3 else ch[1:2][0])
        if isinstance(c, np.ndarray) and c.ndim == 3: c = c.transpose(1, 2, 0)
        cimg = (np.clip(np.abs(c), 0, 1) * 255).astype(np.uint8)
        if cimg.ndim == 2: cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2RGB)
        over = cv2.addWeighted(bg, 0.6, cv2.resize(cimg, (IMG, IMG)), 0.6, 0)
        lab = bg.copy(); cv2.putText(lab, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        rows.append(np.concatenate([lab, cimg, over], 1))
    grid = np.concatenate(rows, 0)
    cv2.imwrite(out_png, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[viz] {out_png}", flush=True)


def main():
    src = os.environ.get("SRC", "robot")
    if os.environ.get("VIZ") == "1":
        os.makedirs(f"{OUTDIR}/viz", exist_ok=True)
        A = load_human_arrays() if src == "human" else load_robot_arrays()
        n = int(A["_valid_idx"][0]) if src == "human" else 356
        for v in range(2):
            viz_clip(A, n, src, v, f"{OUTDIR}/viz/sketch_{src}_clip{n}_cam{'hi' if v == 0 else 'lo'}.png")
        return
    A = load_human_arrays() if src == "human" else load_robot_arrays()
    N = A["tracks"].shape[0]; L = A["tracks"].shape[1]; tL = 6
    idx = A["_valid_idx"]
    rows = idx[:4] if os.environ.get("SMOKE") == "1" else idx
    sketch = np.zeros((N, 2, NCH, tL, GRID, GRID), np.float16)
    import time; t0 = time.time()
    for i, n in enumerate(rows):
        sketch[int(n)] = build_clip_sketch(A, int(n), tL, L, src)
        if i % 100 == 0: print(f"sketch {i}/{len(rows)} (clip {n})  {(time.time()-t0)/max(i,1):.3f}s/clip", flush=True)
    suffix = "_warp" if WARP else ""
    np.savez(f"{OUTDIR}/sketch_{src}{suffix}.npz", sketch=sketch, tL=np.array(tL))
    print(f"saved {OUTDIR}/sketch_{src}{suffix}.npz {sketch.shape}", flush=True)


if __name__ == "__main__":
    main()
