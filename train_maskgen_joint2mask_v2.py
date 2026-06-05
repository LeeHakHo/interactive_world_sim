"""densev4d step 1b: sharpen g(joint -> agent mask). Three upgrades over v1 (IoU 0.64):
  (1) Dice loss + BCE  -> directly optimizes overlap, fights BCE's blurry edges.
  (2) bigger net.
  (3) input += 3 EEF 2D points (base + 2 fingertips, projected via K+T_cw -- these point
      projections ARE aligned to the real gripper, unlike the URDF arm mesh) -> strong
      localization cue on top of the 7 joints.
All inputs are rollout-known (joint + EEF-from-action) and never touch future frames -> no
leak. Held-out = video 3. Compare IoU to v1 (0.64) and densev4c URDF (0.39).
Output: outputs/densev4d_maskgen_joint2mask_v2/
"""
import os, sys, json, numpy as np, cv2, torch, torch.nn as nn, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import av
from scipy.spatial.transform import Rotation
sys.path.insert(0, ".")
from apply_robot_mask_v3 import robot_v3

OUT = "outputs/densev4d_maskgen_joint2mask_v2"; os.makedirs(OUT, exist_ok=True)
CX, CY, CW, CH = 195, 195, 256, 256
RES = 128; MAXK, STRIDE = 6000, 3
LINK6_TO_EE = np.array([0.156062, 0.0, 0.0]); FINGER_X = 0.0865
device = "cuda" if torch.cuda.is_available() else "cpu"
def crop128(im): return cv2.resize(im[CY:CY + CH, CX:CX + CW], (RES, RES))

from interactive_world_sim.algorithms.latent_decompose.ot_align import robot_world_to_cam
T_cw = np.asarray(robot_world_to_cam(), np.float64)
intr = json.load(open("phantom_human_play/intrinsics_cam_high.json"))["left"]
fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]


def eef3_feat(ee_p, ee_q, grip):
    R = Rotation.from_quat(ee_q).as_matrix(); base = ee_p - R @ LINK6_TO_EE
    pts = np.stack([base, base + R @ np.array([FINGER_X, grip / 2, 0]), base + R @ np.array([FINGER_X, -grip / 2, 0])])
    pc = (T_cw[:3, :3] @ pts.T).T + T_cw[:3, 3]; z = np.clip(pc[:, 2], 1e-3, None)
    uv = np.stack([fx * pc[:, 0] / z + cx, fy * pc[:, 1] / z + cy], 1)
    return ((uv - np.array([CX, CY])) / CW).reshape(-1).astype(np.float32)   # 6, crop-normalized


X, M, F, VID = [], [], [], []
for v in (1, 2, 3):
    base = f"play_robot_eef/play_robot_{v}_eef"
    df = pd.read_parquet(f"{base}/data/chunk-000/episode_000000.parquet")
    action = np.stack(df["action"].values).astype(np.float32)
    ee_p = np.stack(df["action_right_ee_position"].values).astype(np.float64)
    ee_q = np.stack(df["action_right_ee_quat_xyzw"].values).astype(np.float64)
    cont = av.open(f"{base}/videos/chunk-000/observation.images.cam_high/episode_000000.mp4")
    n = 0
    for k, fr in enumerate(cont.decode(cont.streams.video[0])):
        if k >= MAXK: break
        if k % STRIDE != 0 or k >= len(action): continue
        c = crop128(fr.to_ndarray(format="rgb24"))
        feat = np.concatenate([action[k, :7], eef3_feat(ee_p[k], ee_q[k], float(action[k, 6]))])  # 13
        X.append(feat); M.append((robot_v3(np.ascontiguousarray(c)) > 0).astype(np.float32))
        F.append(c.astype(np.uint8)); VID.append(v); n += 1
    cont.close(); print(f"video {v}: {n} frames", flush=True)
X = np.stack(X); M = np.stack(M); F = np.stack(F); VID = np.array(VID)
print(f"X{X.shape} M{M.shape}", flush=True)

tr = VID != 3; ho = VID == 3
xm = X[tr].mean(0); xs = X[tr].std(0) + 1e-6; Xn = (X - xm) / xs


class MaskGen(nn.Module):
    def __init__(s, indim=13):
        super().__init__()
        s.fc = nn.Sequential(nn.Linear(indim, 512), nn.ReLU(), nn.Linear(512, 512 * 4 * 4), nn.ReLU())
        s.dc = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1))
    def forward(s, x): return s.dc(s.fc(x).view(-1, 512, 4, 4))


def dice(logits, t):
    p = torch.sigmoid(logits); num = 2 * (p * t).sum((1, 2)) + 1; den = p.sum((1, 2)) + t.sum((1, 2)) + 1
    return (1 - num / den).mean()


g = MaskGen().to(device); opt = torch.optim.AdamW(g.parameters(), 1e-3)
bce = nn.BCEWithLogitsLoss()
Xt = torch.from_numpy(Xn[tr]).float().to(device); Mt = torch.from_numpy(M[tr]).float().to(device)
idx = np.arange(len(Xt)); rng = np.random.default_rng(0)
for ep in range(120):
    rng.shuffle(idx); tot = 0.0; nb = 0
    for i in range(0, len(idx), 64):
        b = idx[i:i + 64]; lg = g(Xt[b])[:, 0]
        loss = bce(lg, Mt[b]) + dice(lg, Mt[b])
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
    if ep % 15 == 0 or ep == 119: print(f"ep{ep:3d} loss {tot/nb:.4f}", flush=True)

g.eval()
Xh = torch.from_numpy(Xn[ho]).float().to(device); Mh = M[ho]; Fh = F[ho]
with torch.no_grad(): ph = g(Xh)[:, 0].cpu().numpy() > 0.0
ious = [((ph[i] & (Mh[i] > 0.5)).sum()) / max((ph[i] | (Mh[i] > 0.5)).sum(), 1) for i in range(len(Mh))]
print(f"HELD-OUT (video3) mean IoU = {np.mean(ious):.3f}  (v1=0.64, URDF=0.39)", flush=True)

sel = np.random.RandomState(1).choice(len(Mh), 8, replace=False)
fig, ax = plt.subplots(3, 8, figsize=(20, 7.6))
for j, i in enumerate(sel):
    ov = Fh[i].copy(); pm = ph[i]; ov[pm] = (0.4 * ov[pm] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
    ax[0, j].imshow(Fh[i]); ax[0, j].axis("off"); ax[0, j].set_title("real frame", fontsize=8)
    ax[1, j].imshow(Mh[i], cmap="gray"); ax[1, j].axis("off"); ax[1, j].set_title("real mask", fontsize=8)
    ax[2, j].imshow(ov); ax[2, j].axis("off"); ax[2, j].set_title(f"g pred IoU={ious[i]:.2f}", fontsize=8)
fig.suptitle("densev4d g v2 (Dice+BCE, bigger net, +EEF 2D inputs). row3 green = g(joint,eef) on real frame. "
             "DEPLOYABLE + NON-LEAKING.", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/qc.png", dpi=120)
torch.save({"g": g.state_dict(), "xm": xm, "xs": xs, "indim": 13}, f"{OUT}/maskgen.pt")
open(f"{OUT}/summary.txt", "w").write(f"densev4d g v2 (Dice+BCE+EEF)\nHELD-OUT IoU = {np.mean(ious):.3f} (v1=0.64 URDF=0.39)\n")
print(f"saved {OUT}/qc.png + maskgen.pt + summary.txt\n=== DONE ===", flush=True)
