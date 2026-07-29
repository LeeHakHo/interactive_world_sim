"""human low视角真腕(2026-07-28, 用户要"给human low也建真腕, 复用phantom pipeline")。
phantom 已把手 lift 到 world 3D(parquet observation.eef.kpts21_3d_world, 21点, 腕=kpt0)。
本脚本: 腕world3D → REAL标定 project → cam_high(对比现有2D检测腕=验证投影正确) + cam_low(新)。
不覆盖 wrist_sidecar_human.npz; 写 wrist_sidecar_human_low.npz + 叠加图眼检(原帧|腕点, high&low)。
用 .venv_wan/bin/python。产物 outputs/human_wrist_low/。
"""
import os, sys, json, numpy as np, pandas as pd, cv2, av, imageio.v2 as imageio
from scipy.spatial.transform import Rotation
sys.path.insert(0, ".")
OUTDIR = os.environ.get("BASE", "outputs/flow_render_dataset_can_dual"); VIZ = "outputs/human_wrist_low"; os.makedirs(VIZ, exist_ok=True)
HCLIP = os.environ.get("HCLIP", "clips_human_L24")   # basename含clips_前缀; L48=clips_human_L48
HUMAN_DIRS = [f"human_play_eef_data/play_human_can_eef_{i}" for i in range(1, 13)]
CROPS = {"high": (60, 60, 390, 390), "low": (0, 0, 640, 480)}
CALF = {"high": "calib/rgb_cam_calib_can_REAL.json", "low": "calib/rgb_cam_calib_can_low_REAL.json"}
CAMK = {"high": "observation.images.cam_high", "low": "observation.images.cam_low"}


def calib(view):
    r = json.load(open(CALF[view]))
    R_wc = Rotation.from_quat(r["quat_xyzw"]).as_matrix(); t_wc = np.asarray(r["t_world"])
    T = np.eye(4); T[:3, :3] = R_wc.T; T[:3, 3] = -R_wc.T @ t_wc
    K = np.array([[r["f"], 0, r["cx"]], [0, r["f"], r["cy"]], [0, 0, 1.0]])
    return T, K


def project(pw, T, K):
    pc = T[:3, :3] @ np.asarray(pw, np.float64) + T[:3, 3]
    if pc[2] <= 1e-3: return None
    return np.array([K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]])


def to_norm(uv, view):
    if uv is None: return np.array([np.nan, np.nan], np.float32)
    x, y, w, h = CROPS[view]
    return np.array([(uv[0] - x) / w, (uv[1] - y) / h], np.float32)


def load_wrist_world(d):
    df = pd.read_parquet(f"{d}/data/chunk-000/episode_000000.parquet",
                         columns=["observation.eef.kpts21_3d_world", "observation.eef.detected_right",
                                  "observation.eef.lift_ok_right"])
    kp = np.stack(df["observation.eef.kpts21_3d_world"].values).astype(np.float64).reshape(-1, 21, 3)
    det = np.array([bool(np.asarray(x).ravel()[0]) for x in df["observation.eef.detected_right"].values])
    lok = np.array([bool(np.asarray(x).ravel()[0]) for x in df["observation.eef.lift_ok_right"].values])
    return kp[:, 0], det & lok       # (M,3) 腕world, valid


def gather(src, fidx):
    M = len(src); ok = fidx < M; fi = np.clip(fidx, 0, M - 1)
    out = src[fi].copy().astype(np.float64); out[~ok] = np.nan
    return out, ok


def main():
    z = np.load(f"{OUTDIR}/{HCLIP}.npz")
    VID, FIDX = z["vid"], z["fidx"]; N, L = FIDX.shape
    Th, Kh = calib("high"); Tl, Kl = calib("low")
    w2d_high = np.full((N, L, 2), np.nan, np.float32); w2d_low = np.full((N, L, 2), np.nan, np.float32)
    vld = np.zeros((N, L), bool)
    for vid in np.unique(VID):
        d = HUMAN_DIRS[int(vid)]; wworld, wok = load_wrist_world(d)
        rows = np.where(VID == vid)[0]
        ww, okg = gather(wworld, FIDX[rows]); wvalid, _ = gather(wok.astype(float), FIDX[rows])
        for ri, r in enumerate(rows):
            for t in range(L):
                pw = ww[ri, t]
                if not np.isfinite(pw).all() or wvalid[ri, t] < 0.5: continue
                hh = to_norm(project(pw, Th, Kh), "high"); ll = to_norm(project(pw, Tl, Kl), "low")
                w2d_high[r, t] = hh; w2d_low[r, t] = ll
                vld[r, t] = np.isfinite(ll).all() and (0 <= ll[0] <= 1) and (0 <= ll[1] <= 1)
        print(f"  vid {vid} ({d}): {len(rows)} clips", flush=True)
    outp = f"{OUTDIR}/wrist_sidecar_human_low.npz"
    np.savez(outp, wrist2d_low=w2d_low, wrist_valid_low=vld, wrist2d_high_proj=w2d_high)
    print(f"saved {outp}: low valid(inbounds) {vld.mean()*100:.1f}%", flush=True)
    # 验证: 3D投影high腕 vs 现有2D检测腕(应接近); L48无老sidecar则跳过
    _oldp = f"{OUTDIR}/wrist_sidecar_human.npz"
    old = np.load(_oldp) if os.path.exists(_oldp) else None
    both = None
    if old is not None:
        ohi = old["wrist2d_high"]; ov = old["wrist_valid"] > 0.5
        both = ov & np.isfinite(w2d_high).all(-1)
    if both is not None and both.sum():
        d_px = np.linalg.norm((w2d_high[both] - ohi[both]) * 128, axis=-1)
        print(f"验证 3D投影high腕 vs 2D检测腕: 中位差 {np.median(d_px):.1f}px 均值 {d_px.mean():.1f}px (n={both.sum()})", flush=True)
    # 眼检叠加: 4 clip, high|low 两行, 腕=3D投影(青), high再叠2D检测腕(黄)对比
    viz_clips(z, w2d_high, w2d_low, ohi, ov)
    print("=== DONE ===", flush=True)


def decode(d, view, fids):
    x, y, w, h = CROPS[view]; need = set(int(f) for f in fids); got = {}
    c = av.open(f"{d}/videos/chunk-000/{CAMK[view]}/episode_000000.mp4")
    for i, fr in enumerate(c.decode(video=0)):
        if i in need: got[i] = fr.to_ndarray(format="rgb24")[y:y + h, x:x + w].copy()
        if i > max(need): break
    c.close()
    return [got.get(int(f)) for f in fids]


def viz_clips(z, whi, wlo, ohi, ov, cis=(0, 300, 900, 1500)):
    VID, FIDX = z["vid"], z["fidx"]
    for ci in cis:
        if ci >= len(VID): continue
        d = HUMAN_DIRS[int(VID[ci])]; fids = FIDX[ci]
        fh = decode(d, "high", fids); fl = decode(d, "low", fids)
        gif = []
        for t in range(len(fids)):
            if fh[t] is None or fl[t] is None: continue
            hi = cv2.resize(fh[t], (256, 256)); lo = cv2.resize(fl[t], (256, 256))
            if np.isfinite(whi[ci, t]).all():
                p = (whi[ci, t] * 256).astype(int); cv2.circle(hi, tuple(p), 6, (0, 255, 255), 2)  # 3D投影腕 青
            if ov[ci, t] and np.isfinite(ohi[ci, t]).all():
                p = (ohi[ci, t] * 256).astype(int); cv2.circle(hi, tuple(p), 4, (255, 255, 0), -1)  # 2D检测腕 黄
            if np.isfinite(wlo[ci, t]).all():
                p = (wlo[ci, t] * 256).astype(int); cv2.circle(lo, tuple(p), 6, (0, 255, 255), 2)  # low 3D投影腕 青
            cv2.putText(hi, f"H{ci} high (cyan=3Dproj, yellow=2Ddet)", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(lo, "low (cyan=3Dproj wrist NEW)", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            gif.append(np.hstack([hi, lo]))
        if gif:
            p = f"{VIZ}/wrist_H{ci}.gif"; imageio.mimsave(p, gif, fps=6); print(f"  viz {p}", flush=True)


if __name__ == "__main__":
    main()
