"""User: "学到的 mask 可能有些偏差和噪声". Decompose g-mask error vs robot_v3 into
BIAS (systematic) vs NOISE (variance), because the fix differs:
  - centroid offset (g - v3): mean != 0 => systematic position BIAS (correctable).
  - centroid offset variance + per-frame IoU spread => random NOISE.
  - area ratio (g/v3): systematic over/under-coverage BIAS.
  - temporal jitter: IoU(g_t, g_{t+1}) vs IoU(v3_t, v3_{t+1}) -> does g add jitter beyond
    what the supervision already has? (g has no temporal constraint.)
Held-out = video 3 (g never trained on it). Output: outputs/diag_gmask_error_structure/
"""
import os, sys, json, numpy as np, cv2, torch, torch.nn as nn, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import av
from scipy.spatial.transform import Rotation
sys.path.insert(0, ".")
from apply_robot_mask_v3 import robot_v3

OUT = "outputs/flow_wm/diag_gmask_error_structure"; os.makedirs(OUT, exist_ok=True)
GCK = "outputs/densev4d_maskgen_joint2mask_v2/maskgen.pt"
CX, CY, CW, CH = 195, 195, 256, 256
RES, STRIDE, MAXK = 128, 3, 6000
LINK6_TO_EE = np.array([0.156062, 0.0, 0.0]); FINGER_X = 0.0865
device = "cuda" if torch.cuda.is_available() else "cpu"
_otl = torch.load
torch.load = lambda *a, **k: _otl(*a, **{**k, "weights_only": False})
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
    return ((uv - np.array([CX, CY])) / CW).reshape(-1).astype(np.float32)


class MaskGen(nn.Module):
    def __init__(s, indim=13):
        super().__init__()
        s.fc = nn.Sequential(nn.Linear(indim, 512), nn.ReLU(), nn.Linear(512, 512 * 4 * 4), nn.ReLU())
        s.dc = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1))
    def forward(s, x): return s.dc(s.fc(x).view(-1, 512, 4, 4))


def centroid(m):
    ys, xs = np.where(m)
    return (np.array([xs.mean(), ys.mean()]) if len(xs) else None)


def main():
    gck = torch.load(GCK, map_location="cpu")
    g = MaskGen(gck["indim"]).to(device); g.load_state_dict(gck["g"]); g.eval()
    xm = np.asarray(gck["xm"], np.float32); xs_ = np.asarray(gck["xs"], np.float32)

    v = 3  # held-out
    base = f"play_robot_eef/play_robot_{v}_eef"
    df = pd.read_parquet(f"{base}/data/chunk-000/episode_000000.parquet")
    action = np.stack(df["action"].values).astype(np.float32)
    ee_p = np.stack(df["action_right_ee_position"].values).astype(np.float64)
    ee_q = np.stack(df["action_right_ee_quat_xyzw"].values).astype(np.float64)
    cont = av.open(f"{base}/videos/chunk-000/observation.images.cam_high/episode_000000.mp4")
    G, V3, FR = [], [], []
    for k, fr in enumerate(cont.decode(cont.streams.video[0])):
        if k >= MAXK: break
        if k % STRIDE != 0 or k >= len(action): continue
        c = crop128(fr.to_ndarray(format="rgb24"))
        feat = np.concatenate([action[k, :7], eef3_feat(ee_p[k], ee_q[k], float(action[k, 6]))])
        fn = (feat - xm) / xs_
        with torch.no_grad():
            pm = (torch.sigmoid(g(torch.from_numpy(fn[None]).float().to(device))[0, 0]) > 0.5).cpu().numpy()
        G.append(pm); V3.append(robot_v3(np.ascontiguousarray(c)) > 0); FR.append(c.astype(np.uint8))
    cont.close()
    G = np.stack(G); V3 = np.stack(V3); FR = np.stack(FR); n = len(G)
    print(f"video{v}: {n} frames", flush=True)

    # ---- per-frame metrics ----
    iou, dcent, area_ratio = [], [], []
    for i in range(n):
        inter = (G[i] & V3[i]).sum(); uni = (G[i] | V3[i]).sum()
        iou.append(inter / max(uni, 1))
        cg, cv = centroid(G[i]), centroid(V3[i])
        if cg is not None and cv is not None: dcent.append(cg - cv)  # (dx,dy) px@128
        ag, av_ = G[i].sum(), V3[i].sum()
        if av_ > 0: area_ratio.append(ag / av_)
    iou = np.array(iou); dcent = np.array(dcent); area_ratio = np.array(area_ratio)

    # ---- temporal jitter (consecutive-frame self IoU) ----
    def selfiou(M): return np.array([(M[i] & M[i+1]).sum() / max((M[i] | M[i+1]).sum(), 1) for i in range(len(M)-1)])
    jg, jv = selfiou(G), selfiou(V3)

    mean_off = dcent.mean(0); std_off = dcent.std(0)
    L = [f"=== g-mask ERROR STRUCTURE vs robot_v3 (held-out video3, n={n}) ===", "",
         f"IoU            mean {iou.mean():.3f}  std {iou.std():.3f}  worst {iou.min():.3f}  best {iou.max():.3f}", "",
         "--- BIAS (systematic) ---",
         f"centroid offset (g-v3)  mean=({mean_off[0]:+.2f},{mean_off[1]:+.2f})px  (|mean|={np.linalg.norm(mean_off):.2f})",
         f"area ratio (g/v3)       mean {area_ratio.mean():.3f}  (>1 = g over-covers, <1 = under)", "",
         "--- NOISE (variance) ---",
         f"centroid offset  std=({std_off[0]:.2f},{std_off[1]:.2f})px   (random frame-to-frame wobble)",
         f"area ratio std          {area_ratio.std():.3f}",
         f"temporal self-IoU  g={jg.mean():.3f}  robot_v3={jv.mean():.3f}  "
         f"(lower=more jitter; if g<<v3, g adds jitter beyond supervision)", "",
         "--- verdict ---",
         f"bias |mean centroid offset| = {np.linalg.norm(mean_off):.2f}px vs noise std = {np.linalg.norm(std_off):.2f}px",
         "  -> if |mean|<<std: NOISE-dominated (temporal smoothing / better supervision).",
         "  -> if |mean|~std : real BIAS too (correctable shift / area calibration)."]
    print("\n".join(L), flush=True)
    open(f"{OUT}/summary.txt", "w").write("\n".join(L) + "\n")

    # ---- figures ----
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 6)
    # row0: worst/median/best frame overlays (g green, v3 red, overlap yellow)
    order = np.argsort(iou); picks = [order[0], order[len(order)//2], order[-1]]
    for c, i in enumerate(picks):
        ov = FR[i].copy()
        ov[V3[i] & ~G[i]] = [255, 0, 0]; ov[G[i] & ~V3[i]] = [0, 255, 0]; ov[G[i] & V3[i]] = [255, 255, 0]
        ax = fig.add_subplot(gs[0, c*2:c*2+2]); ax.imshow(ov); ax.axis("off")
        ax.set_title(f"frame{i} IoU={iou[i]:.2f} (R=v3only G=g-only Y=overlap)", fontsize=9)
    # row1: centroid offset scatter (bias=center of cloud, noise=spread)
    ax = fig.add_subplot(gs[1, 0:2]); ax.scatter(dcent[:, 0], dcent[:, 1], s=6, alpha=.4)
    ax.scatter([mean_off[0]], [mean_off[1]], c="r", s=80, marker="x"); ax.axhline(0, c="k", lw=.5); ax.axvline(0, c="k", lw=.5)
    ax.set_title(f"centroid offset (g-v3) px\nred X = mean bias ({mean_off[0]:+.1f},{mean_off[1]:+.1f})", fontsize=9)
    ax.set_xlabel("dx"); ax.set_ylabel("dy"); ax.set_aspect("equal")
    ax = fig.add_subplot(gs[1, 2:4]); ax.hist(iou, bins=30); ax.axvline(iou.mean(), c="r")
    ax.set_title(f"per-frame IoU (mean {iou.mean():.2f})", fontsize=9)
    ax = fig.add_subplot(gs[1, 4:6]); ax.hist(area_ratio, bins=30); ax.axvline(1.0, c="k"); ax.axvline(area_ratio.mean(), c="r")
    ax.set_title(f"area ratio g/v3 (mean {area_ratio.mean():.2f}, k=1 ideal)", fontsize=9)
    # row2: temporal jitter + IoU over time
    ax = fig.add_subplot(gs[2, 0:3]); ax.plot(jg, label=f"g self-IoU ({jg.mean():.2f})"); ax.plot(jv, label=f"v3 self-IoU ({jv.mean():.2f})", alpha=.7)
    ax.set_title("temporal stability (higher=steadier)", fontsize=9); ax.legend(fontsize=8); ax.set_xlabel("frame")
    ax = fig.add_subplot(gs[2, 3:6]); ax.plot(iou); ax.set_title("IoU over time", fontsize=9); ax.set_xlabel("frame")
    fig.suptitle("g-mask error structure: BIAS (mean centroid offset / area ratio) vs NOISE (spread / temporal jitter)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/diag.png", dpi=120)
    print(f"saved {OUT}/diag.png + summary.txt\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
