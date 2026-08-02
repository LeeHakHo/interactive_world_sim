"""P1 cond builder (de-risk): 把 flow/skel cond 流渲成 256 视频 → 过同一 Wan VAE 编码 → (48ch,tL,16,16) per流,
concat 成 VAE-cond latent(细节留通道, 不 avg-pool)。对照 build_can_cond.py 的 avg-pool 7ch。
de-risk 用 flow(3ch)+skel(1→3ch) 两流 → Ccond=96。子集 via env CLIPS/NCLIP。用法(.venv_wan): python build_can_cond_vae.py"""
import os, sys, numpy as np, torch, cv2
sys.path.insert(0, "."); os.environ.setdefault("RES", "128"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import exp_scel_dualview_dit as DIT                # flow_cond (128)
import eval_e2e_combined as E                      # WanVAE

DS = os.environ.get("DS", "outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz")
SKF = os.environ.get("SKF", "outputs/flow_render_dataset_can_dual/skel_sidecar_robot_gripaware.npz")
OUT = os.environ.get("OUT", "outputs/video_arch_wm/cond_can_dual/cond_vae_flowskel.npz")
dev = "cuda"; L = 48; RES = 256; SMOKE = os.environ.get("SMOKE") == "1"
_SEG = np.load(SKF)["segments"]                   # 骨架连线 (从 sidecar 读, 别 import None)


def skel256(pts2d):                               # (9,2) crop-norm -> (256,256) 线画 [0,1]
    img = np.zeros((RES, RES), np.float32); P = (pts2d * RES).astype(np.int32)
    for a, b in _SEG:
        if np.all(np.abs(pts2d[a]) < 3) and np.all(np.abs(pts2d[b]) < 3):
            cv2.line(img, tuple(P[a]), tuple(P[b]), 1.0, 5, cv2.LINE_AA)
    for i, p in enumerate(P):
        if np.all(np.abs(pts2d[i]) < 3): cv2.circle(img, tuple(p), 5, 1.0, -1)
    return img


def stream_video(A, n, v, kind):                  # -> (3, L, 256, 256) float [0,1]
    f32 = lambda a: np.nan_to_num(np.asarray(a, np.float32), nan=0.0)
    tr = f32(A["tracks" if v == 0 else "tracks_low"][n]); ef = f32(A["eef" if v == 0 else "eef_low"][n])
    vs = f32(A["vis" if v == 0 else "vis_low"][n]); sk = A["skel2d_high" if v == 0 else "skel2d_low"][n]
    frames = np.zeros((L, 3, RES, RES), np.float32)
    for t in range(L):
        if kind == "flow":
            fc = DIT.flow_cond(tr[0], tr[t], ef[0], ef[t], vs[t])          # (3,128,128)
            frames[t] = np.stack([cv2.resize(fc[c], (RES, RES)) for c in range(3)])
        else:                                                             # skel -> 3ch 灰度
            g = skel256(f32(sk[t])); frames[t] = np.repeat(g[None], 3, 0)
    return frames


def main():
    z = np.load(DS); sk = np.load(SKF)
    A = {k: z[k] for k in ["tracks", "tracks_low", "eef", "eef_low", "vis", "vis_low"]}
    A["skel2d_high"] = sk["skel2d_high"]; A["skel2d_low"] = sk["skel2d_low"]
    low_valid = z["low_valid"]; N = len(z["tracks"])
    vae = E.WanVAE(device=dev)
    clips = [int(x) for x in os.environ["CLIPS"].split(",")] if os.environ.get("CLIPS") else list(range(N))
    if os.environ.get("NCLIP"): clips = clips[:int(os.environ["NCLIP"])]
    if SMOKE: clips = clips[:1]

    def enc(vid3):                                                        # (3,L,256,256) -> (48,tL,16,16)
        x = torch.from_numpy(vid3).permute(1, 0, 2, 3)[None].to(dev)      # (1,3,L,H,W)  ★permute到(C,L,H,W)
        return vae.encode(x)[0].cpu().numpy().astype(np.float16)

    tL = None; conds = []
    for i, n in enumerate(clips):
        cv = []
        for v in range(2):
            if v == 1 and not low_valid[n]:
                cv.append(np.zeros_like(cv[0])); continue
            fl = enc(stream_video(A, n, v, "flow")); sl = enc(stream_video(A, n, v, "skel"))  # (48,tL,16,16)x2
            cv.append(np.concatenate([fl, sl], 0))                        # (96,tL,16,16)
        conds.append(np.stack(cv))                                        # (2,96,tL,16,16)
        if i == 0:
            tL = conds[0].shape[2]; print(f"[shape] per-clip cond {conds[0].shape} (V,Ccond=96,tL={tL},16,16)", flush=True)
        if i % 20 == 0: print(f"  {i}/{len(clips)} clip{n}", flush=True)
    cond = np.stack(conds)                                                # (Nsel,2,96,tL,16,16)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, cond=cond, clips=np.array(clips), tL=tL)
    print(f"[save] {OUT} cond{cond.shape}", flush=True)


if __name__ == "__main__":
    main()
