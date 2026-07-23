"""诊断: residual latent (Δz) 是否比绝对 latent z 更跨域同域 (human vs robot)。
用户 residual 假设: human/robot 的 Wan latent z 不同域(582×可分), 但在 flow 类似下 Δz(delta latent)可能同域
→ 若成立, ③ 预测 Δz(而非绝对z) + robot 锚, 就能让 human data 帮到 ③ 的 dynamics。
测法(照 object-flow 诊断口径 + [[feedback_probe_must_converge]]收敛probe): 编码 human+robot 帧到 Wan latent,
对 z_abs / Δz_consec(z_{t+1}-z_t) / Δz_anchor(z_t-z_0) 各跑收敛 LogReg human-vs-robot 可分性(chance0.5)。
z→1.0 且 Δz→0.5 = 假设成立。用 .venv_wan + GPU。
"""
import os, sys, numpy as np, torch
sys.path.insert(0, "."); os.environ.setdefault("HF_HUB_OFFLINE", "1")
from wan_vae import WanVAE
import cv2

RH = "outputs/flow_render_dataset_can_dual/clips_robot.npz"
HH = "outputs/flow_render_dataset_can_dual/clips_human_L24.npz"
N = int(os.environ.get("N", "150"))         # 每域 clip 数
OUT = "outputs/cross_embodiment_wm/diag_residual_latent"; os.makedirs(OUT, exist_ok=True)
dev = "cuda"


def enc_clip(vae, frames):                   # frames (T,128,128,3) uint8 -> latent (C,tL,16,16)
    T = len(frames)
    x = np.stack([cv2.resize(frames[t], (256, 256)) for t in range(T)]).astype(np.float32) / 255.
    x = torch.from_numpy(x.transpose(3, 0, 1, 2)[None]).to(dev)   # (1,3,T,256,256)
    with torch.no_grad():
        z = vae.encode(x)[0].cpu().numpy()   # (C,tL,16,16)
    return z


def feats(zs, mode, pool):
    """zs: list of (C,tL,16,16). mode: abs/dconsec/danchor. pool: 1(spatial-mean) or 4(4x4). -> (Nsamp, F)."""
    out = []
    for z in zs:
        C, tL = z.shape[:2]
        if mode == "abs": w = z
        elif mode == "dconsec": w = z[:, 1:] - z[:, :-1]
        elif mode == "danchor": w = z[:, 1:] - z[:, :1]
        # w (C, t', 16,16) -> per-frame pooled feature
        t2 = w.shape[1]
        if pool == 1:
            f = w.reshape(C, t2, -1).mean(-1).T        # (t', C)
        else:
            wp = torch.nn.functional.avg_pool2d(torch.from_numpy(w).permute(1, 0, 2, 3), 16 // pool).numpy()  # (t',C,pool,pool)
            f = wp.reshape(t2, -1)                      # (t', C*pool*pool)
        out.append(f)
    return np.concatenate(out, 0)


def probe(Xh, Xr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler
    n = min(len(Xh), len(Xr)); X = np.concatenate([Xh[:n], Xr[:n]], 0); y = np.r_[np.zeros(n), np.ones(n)]
    X = StandardScaler().fit_transform(X)
    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    accs = []
    for tr, te in sss.split(X, y):
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def main():
    vae = WanVAE(device=dev)
    zr = np.load(RH); zh = np.load(HH)
    ridx = np.linspace(0, len(zr["frames"]) - 1, N).astype(int)
    hidx = np.linspace(0, len(zh["frames"]) - 1, N).astype(int)
    print(f"encoding {N} robot + {N} human clips ...", flush=True)
    zsr = [enc_clip(vae, zr["frames"][i][:24]) for i in ridx]     # robot 取前24帧与human对齐
    zsh = [enc_clip(vae, zh["frames"][i]) for i in hidx]
    print("done encode; probing ...", flush=True)
    lines = ["=== residual latent 跨域可分性 (human vs robot, chance=0.5) ==="]
    for pool in [1, 4]:
        for mode, name in [("abs", "z_abs(绝对latent)"), ("dconsec", "Δz_consec(z_t+1−z_t)"), ("danchor", "Δz_anchor(z_t−z_0)")]:
            acc = probe(feats(zsh, mode, pool), feats(zsr, mode, pool))
            lines.append(f"pool{pool} {name:24s} probe acc = {acc:.3f}")
        lines.append("")
    txt = "\n".join(lines)
    print(txt, flush=True)
    open(f"{OUT}/summary.txt", "w").write(txt + "\n判读: z_abs高(~1.0=不同域) + Δz低(~0.5=同域) => residual假设成立, ③预测Δz+robot锚可让human帮dynamics。\n")
    print(f"=== saved {OUT}/summary.txt ===", flush=True)


if __name__ == "__main__":
    main()
