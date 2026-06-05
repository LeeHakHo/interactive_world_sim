"""Build a PROPER (not-overfit) training set for the flow-conditioned decoder.

Per clip we keep: frames@128 (uint8), object tracks (CoTracker), 3 EEF points.
FULL data, LARGE time step (L=24, S=4 -> clip spans ~96 frames) so that when we
train the flow-cond decoder on thousands of samples it CANNOT memorize z0->target
and is forced to actually USE flow -- the exact 走捷径 failure the overfit derisk
(job45964: spread 0.25, decoder ignored flow) exposed.

Output: outputs/flow_render_dataset/clips_{human,robot}.npz
  frames (N,L,128,128,3 uint8), tracks (N,L,P,2 f16), eef (N,L,3,2 f16), vid (N,)
Reuses the verified geometry/track logic from gen_flow_dataset_v3.py.
"""
import os, json, numpy as np, cv2, torch, pandas as pd

CHUNKS = "phantom/data/processed_play_human_aytsai_chunks/play_human_aytsai_chunks"
HUMAN_CHUNKS = ["1001", "2000", "3000"]
ROBOT_DIRS = [f"play_robot_eef/play_robot_{i}_eef" for i in (1, 2, 3)]
INTR = "phantom_human_play/intrinsics_cam_high.json"
CX, CY, CW, CH = 195, 195, 256, 256
TRACK_RES, IMG_RES = 224, 128
L, S, P = 24, 4, 48                 # L=24,S=4 -> clip spans 96 frames (big time step)
MAXF, CLIP_STRIDE, MOVE_MIN, TARGET_PER_VID = 9000, 18, 10.0, 200
LINK6_TO_EE = np.array([0.156062, 0.0, 0.0]); FINGER_X = 0.0865


def crop224(img): return cv2.resize(img[CY:CY + CH, CX:CX + CW], (TRACK_RES, TRACK_RES))
def crop128(img): return cv2.resize(img[CY:CY + CH, CX:CX + CW], (IMG_RES, IMG_RES))
def to_crop(uv): return None if uv is None else np.array([(uv[0]-CX)/CW*TRACK_RES, (uv[1]-CY)/CH*TRACK_RES], np.float32)


def cube_mask(c):
    hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV); h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h < 12) | (h > 168)) & (s > 90) & (v > 35) & (v < 175)
    m = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1 or st[1:, cv2.CC_STAT_AREA].max() < 30: return np.zeros_like(m)
    return (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def quat_xyzw_to_R(q):
    x, y, z, w = q; n = x*x+y*y+z*z+w*w; s = 2.0/max(n, 1e-12)
    xx, yy, zz = s*x*x, s*y*y, s*z*z; xy, xz, yz = s*x*y, s*x*z, s*y*z
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    return np.array([[1-(yy+zz), xy-wz, xz+wy], [xy+wz, 1-(xx+zz), yz-wx],
                     [xz-wy, yz+wx, 1-(xx+yy)]], np.float64)


def load_K():
    intr = json.load(open(INTR))["left"]
    return np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])


def project(p_world, T_cw, K):
    pc = T_cw[:3, :3] @ np.asarray(p_world, np.float64) + T_cw[:3, 3]
    if pc[2] <= 1e-3: return None
    return np.array([K[0, 0]*pc[0]/pc[2]+K[0, 2], K[1, 1]*pc[1]/pc[2]+K[1, 2]])


def robot_eef3_loader(robot_dir, T_cw, K):
    df = pd.read_parquet(f"{robot_dir}/data/chunk-000/episode_000000.parquet")
    pos = np.stack(df["action_right_ee_position"].values).astype(np.float64)
    quat = np.stack(df["action_right_ee_quat_xyzw"].values).astype(np.float64)
    grip = np.clip(df["action_right_gripper"].values.astype(np.float64), 0, None)

    def fn(i):
        if i >= len(pos): return None
        R = quat_xyzw_to_R(quat[i]); base = pos[i] - R @ LINK6_TO_EE; w = grip[i]
        pts3d = [base, base + R @ np.array([FINGER_X, +w/2, 0.0]), base + R @ np.array([FINGER_X, -w/2, 0.0])]
        pts = [to_crop(project(p, T_cw, K)) for p in pts3d]
        if any(p is None for p in pts): return None
        return np.stack(pts)
    return fn, len(pos)


def human_eef3_loader(ch):
    hd = np.load(f"{CHUNKS}/{ch}/hand_processor/hand_data_right.npz")
    kp, det = hd["kpts_2d"], hd["hand_detected"]

    def fn(i):
        if i >= len(kp) or not bool(det[i]): return None
        return np.stack([to_crop(kp[i, j]) for j in (0, 4, 8)]).astype(np.float32)
    return fn, len(kp)


def decode(domain, path):
    c224, c128 = [], []
    if domain == "human":
        cap = cv2.VideoCapture(path)
        while len(c224) < MAXF:
            ok, fr = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB); c224.append(crop224(rgb)); c128.append(crop128(rgb))
        cap.release()
    else:
        import av; cont = av.open(path); st = cont.streams.video[0]
        for frame in cont.decode(st):
            if len(c224) >= MAXF: break
            rgb = frame.to_ndarray(format="rgb24"); c224.append(crop224(rgb)); c128.append(crop128(rgb))
        cont.close()
    return c224, c128


@torch.no_grad()
def track(ct, frames_u8, q_xy, device):
    vid = torch.from_numpy(np.stack(frames_u8)).permute(0, 3, 1, 2)[None].float().to(device)
    q = np.concatenate([np.zeros((len(q_xy), 1), np.float32), q_xy], 1)
    tr, vs = ct(vid, queries=torch.from_numpy(q)[None].to(device))
    return tr[0].cpu().numpy(), vs[0].cpu().numpy().astype(np.float32)


def collect(ct, domain, items, device, rng, vid_base):
    F, T, V, E, VID = [], [], [], [], []
    for vj, (path, eef_fn) in enumerate(items):
        c224, c128 = decode(domain, path); got = 0
        starts = list(range(0, len(c224) - L * S, CLIP_STRIDE)); rng.shuffle(starts)
        for s0 in starts:
            if got >= TARGET_PER_VID: break
            idxs = [s0 + k * S for k in range(L)]
            fr224 = [c224[i] for i in idxs]
            m0 = cube_mask(fr224[0]); ys, xs = np.where(m0 > 0)
            if len(xs) < P: continue
            sel = rng.choice(len(xs), P, replace=False)
            q = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
            tr, vs = track(ct, fr224, q, device)
            c = tr.mean(1)
            if float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1))) < MOVE_MIN: continue
            eef3 = [eef_fn(idxs[t]) for t in range(L)]
            if any(e is None for e in eef3): continue
            F.append(np.stack([c128[i] for i in idxs]).astype(np.uint8))     # (L,128,128,3)
            T.append((tr / TRACK_RES).astype(np.float16))                    # (L,P,2)
            E.append((np.stack(eef3) / TRACK_RES).astype(np.float16))        # (L,3,2)
            V.append(vs.astype(np.float16)); VID.append(vid_base + vj); got += 1
        print(f"  {domain} {path.split('/')[-4] if domain=='robot' else path.split('/')[-2]}: kept {got}", flush=True)
    return F, T, E, V, VID


def save(tag, F, T, E, V, VID):
    os.makedirs("outputs/flow_render_dataset", exist_ok=True)
    np.savez(f"outputs/flow_render_dataset/clips_{tag}.npz",
             frames=np.stack(F), tracks=np.stack(T), eef=np.stack(E),
             vis=np.stack(V), vid=np.array(VID, np.int64))
    print(f"saved clips_{tag}.npz N={len(F)} frames={np.stack(F).shape}", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ct = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device).eval()
    from interactive_world_sim.algorithms.latent_decompose.ot_align import robot_world_to_cam
    T_cw = np.asarray(robot_world_to_cam(), np.float64); K = load_K()
    rng = np.random.default_rng(0)

    print("=== human ===", flush=True)
    h_items = [(f"{CHUNKS}/{c}/video_rgb_imgs.mkv", human_eef3_loader(c)[0]) for c in HUMAN_CHUNKS]
    Fh, Th, Eh, Vh, IDh = collect(ct, "human", h_items, device, rng, 0)
    save("human", Fh, Th, Eh, Vh, IDh)

    print("=== robot ===", flush=True)
    r_items = [(f"{d}/videos/chunk-000/observation.images.cam_high/episode_000000.mp4",
                robot_eef3_loader(d, T_cw, K)[0]) for d in ROBOT_DIRS]
    Fr, Tr, Er, Vr, IDr = collect(ct, "robot", r_items, device, rng, 10)
    save("robot", Fr, Tr, Er, Vr, IDr)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
