"""M2b: 把 can dual 数据集(clips_robot.npz)的帧在 256 原生分辨率编码成 Wan2.2 latent 缓存。
不存 256 原始帧(51GB 太大), 直接存时序 latent = 视频 DiT 训练输入。
每 clip 每视角: 256 帧 (L,3,256,256) -> Wan causal 编码 -> (48, tL, 16,16), tL=1+(L-1)//4。
复用 gen_flow_render_dataset_caneef 的 CROP/ROBOT_DIRS/crop_to (帧提取与现管线一致)。
用 .venv_wan/bin/python 跑 (需 diffusers AutoencoderKLWan)。
产物: outputs/video_arch_wm/wan_latents_can_dual/latents.npz (lat, lat_low, tL) + summary。
env: SMOKE=1 只跑前 2 vid; VIDS=逗号列表 只跑部分(分 GPU 并行)。
"""
import os, sys, numpy as np, av, cv2, torch
sys.path.insert(0, ".")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from wan_vae import WanVAE

SRC = os.environ.get("SRC", "robot")             # robot(默认) | human
DS = os.environ.get("DS", "outputs/flow_render_dataset_can_dual/clips_robot.npz"
                    if SRC == "robot" else "outputs/flow_render_dataset_can_dual/clips_human_L24.npz")
OUTDIR = os.environ.get("OUTDIR", "outputs/video_arch_wm/wan_latents_can_dual"); os.makedirs(OUTDIR, exist_ok=True)   # L48重建用独立OUTDIR不覆盖L24
RES = 256
CROPS = {"high": (60, 60, 390, 390), "low": (0, 0, 640, 480)}
CAM = {"high": "observation.images.cam_high", "low": "observation.images.cam_low"}
ROBOT_DIRS = [f"human_play_data/play_robot_can_{i}_eef" for i in range(1, 19)]
HUMAN_DIRS = [f"human_play_eef_data/play_human_can_eef_{i}" for i in range(1, 13)]
# vid -> lerobot dir 映射: robot vid=100+(idx), human vid=0..11 直接 idx
DIRS = ROBOT_DIRS if SRC == "robot" else HUMAN_DIRS
VID_OFFSET = 100 if SRC == "robot" else 0
SMOKE = os.environ.get("SMOKE", "0") == "1"


def decode256(path, view, need):
    """decode 只保留 need(set of frame idx) 的帧 -> dict{idx:(256,256,3)uint8}。省内存/时间。"""
    x, y, w, h = CROPS[view]; out = {}; mx = max(need)
    c = av.open(path)
    for i, fr in enumerate(c.decode(video=0)):
        if i in need:
            rgb = fr.to_ndarray(format="rgb24")[y:y + h, x:x + w]
            out[i] = cv2.resize(rgb, (RES, RES), interpolation=cv2.INTER_AREA)
        if i >= mx: break
    c.close()
    return out


def main():
    z = np.load(DS)
    VID, FIDX = z["vid"], z["fidx"]
    N, L = FIDX.shape
    tL = 1 + (L - 1) // 4
    dev = "cuda"; v = WanVAE(device=dev)
    lat = np.zeros((N, v.z_dim, tL, RES // 16, RES // 16), np.float16)
    lat_low = np.zeros_like(lat)
    low_valid = z["low_valid"] if "low_valid" in z.files else np.ones(N, bool)
    vids = np.unique(VID)
    if SMOKE: vids = vids[:2]
    if os.environ.get("VIDS"): vids = [int(x) for x in os.environ["VIDS"].split(",")]

    def enc_clip(frames256):  # (L,256,256,3)uint8 -> (48,tL,16,16) f16
        x = torch.from_numpy(frames256.astype(np.float32) / 255.).permute(3, 0, 1, 2)[None]  # (1,3,L,H,W)
        zl = v.encode(x)[0].cpu().numpy().astype(np.float16)                                  # (48,tL,16,16)
        return zl

    done = 0
    for vid in vids:
        rows = np.where(VID == vid)[0]
        d = DIRS[vid - VID_OFFSET]
        need = set(int(i) for n in rows for i in FIDX[n])
        fh = decode256(f"{d}/videos/chunk-000/{CAM['high']}/episode_000000.mp4", "high", need)
        fl = decode256(f"{d}/videos/chunk-000/{CAM['low']}/episode_000000.mp4", "low", need)
        for n in rows:
            idxs = FIDX[n]
            lat[n] = enc_clip(np.stack([fh[i] for i in idxs]))
            if low_valid[n] and all(i in fl for i in idxs):
                lat_low[n] = enc_clip(np.stack([fl[i] for i in idxs]))
            done += 1
        print(f"vid {vid}: {len(rows)} clips encoded  (total {done}/{N if not (SMOKE or os.environ.get('VIDS')) else '~'})", flush=True)
    tag = "smoke" if SMOKE else (os.environ.get("VIDS", "all").replace(",", "_"))
    pref = "" if SRC == "robot" else f"{SRC}_"          # human_ 前缀, 不覆盖 robot latents_all.npz
    outp = f"{OUTDIR}/latents_{pref}{tag}.npz"
    np.savez(outp, lat=lat, lat_low=lat_low, tL=np.array(tL), vids=np.array(list(vids)), low_valid=low_valid)
    print(f"saved {outp}: lat{lat.shape} tL={tL}  (encoded rows for vids {list(vids)})", flush=True)
    print("=== M2b DONE ===", flush=True)


if __name__ == "__main__":
    main()
