"""组装 8 通道交互草图(2026-07-31 spec)。SRC=robot|human。3D世界系算->投影双视角2D->pool16。
robot/human 的 skel2d+grip 都从各自现成 sidecar 读(不运行时跑 IK/FK/pinocchio)。在 .venv_wan 跑。
通道序: [flow(3), skel(1), grip(1), trace(1), attachment(1), contact(1)] = 8。
用法: SRC=robot .venv_wan/bin/python build_sketch.py   (env: SMOKE=1 只前4条)
"""
import os, numpy as np, cv2
import sketch_lib as S
from contact_detector import detect_contact_2d

IMG = 128; GRID = 16
RB = "outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz"
HM = "outputs/flow_render_dataset_can_dual_L48/clips_human_L48_retrack_realwrist.npz"
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
    keys = ["tracks", "tracks_low", "eef", "eef_low", "vis", "vis_low", "eef3d", "grip", "frames", "frames_low", "vid"]
    A = {k: z[k][:] for k in keys}
    A["skel2d_high"] = _SKR["skel2d_high"][:]; A["skel2d_low"] = _SKR["skel2d_low"][:]
    A["_valid_idx"] = np.arange(A["tracks"].shape[0])
    return A


def load_human_arrays():
    z = np.load(HM)
    keys = ["tracks", "tracks_low", "eef", "eef_low", "vis", "vis_low", "eef3d", "frames", "frames_low", "vid"]
    A = {k: z[k][:] for k in keys}
    A["skel2d_high"] = _SKH["skel2d_high"][:]; A["skel2d_low"] = _SKH["skel2d_low"][:]
    A["grip"] = _SKH["grip"][:].astype(np.float32)                    # human sidecar grip(自身range已映射)
    e3 = z["eef3d"]; t2 = z["tracks"]
    valid = np.isfinite(e3.reshape(len(e3), e3.shape[1], -1)).all(-1).all(-1) & \
            np.isfinite(t2.reshape(len(t2), t2.shape[1], -1)).all(-1).all(-1)   # (N,) per-clip 全帧有效
    A["_valid_idx"] = np.where(valid)[0]
    return A


def build_clip_sketch(A, n, tL, L, src):
    """★object-flow+contact 走 2D per-view(human 无 tracks3d, 用两域都有的 2D tracks);
    agent-trace 走 eef3d(两域都有)投影; agent-skel 从 sidecar。"""
    f32 = lambda a: np.nan_to_num(np.asarray(a, np.float32))
    tr2d = [f32(A["tracks"][n]), f32(A["tracks_low"][n])]
    eef3d = f32(A["eef3d"][n]); grip = f32(A["grip"][n])
    ef = [f32(A["eef"][n]), f32(A["eef_low"][n])]; vs = [f32(A["vis"][n]), f32(A["vis_low"][n])]
    contact = [detect_contact_2d(tr2d[v], ef[v], grip) for v in range(2)]   # 每视角 (att(L,), cpt(L,2))
    out = np.zeros((2, 8, tL, GRID, GRID), np.float32)
    views = ["high", "low"]
    for k in range(tL):
        rf = 0 if k == 0 else min(4 * k, L - 1)
        for v in range(2):
            flow = S.object_flow_2d(tr2d[v][0], tr2d[v][rf], ef[v][0], ef[v][rf], vs[v][rf])           # (3,128,128)
            sk = skel_chan(A["skel2d_high" if v == 0 else "skel2d_low"][n, rf])[None]                  # (1,128,128)
            gp = S.grip_channel(grip[rf])                                                              # (1,128,128)
            trc = S.agent_trace_channel(eef3d[:rf + 1], views[v])                                      # (1,128,128)
            att_v, cpt_v = contact[v]
            ct = S.contact_channels_2d(att_v[rf], cpt_v[rf])                                           # (2,128,128) [attach,splat]
            chans = np.concatenate([flow, sk, gp, trc, ct], 0)                                         # (8,128,128)
            out[v, :, k] = S.pool16(chans)
    return out.astype(np.float16)


def main():
    src = os.environ.get("SRC", "robot")
    A = load_human_arrays() if src == "human" else load_robot_arrays()
    N = A["tracks"].shape[0]; L = A["tracks"].shape[1]; tL = 6
    idx = A["_valid_idx"]
    rows = idx[:4] if os.environ.get("SMOKE") == "1" else idx
    sketch = np.zeros((N, 2, 8, tL, GRID, GRID), np.float16)
    import time; t0 = time.time()
    for i, n in enumerate(rows):
        sketch[int(n)] = build_clip_sketch(A, int(n), tL, L, src)
        if i % 100 == 0: print(f"sketch {i}/{len(rows)} (clip {n})  {(time.time()-t0)/max(i,1):.3f}s/clip", flush=True)
    np.savez(f"{OUTDIR}/sketch_{src}.npz", sketch=sketch, tL=np.array(tL))
    print(f"saved {OUTDIR}/sketch_{src}.npz {sketch.shape}", flush=True)


if __name__ == "__main__":
    main()
