"""Step 2a-v3: object-flow tracks + 3 EEF points (center + 2 tips) as the action.

Replaces v2's single contact-proxy point with 3 semantically-aligned EEF points:
  robot: from parquet EEF pose (action_right_ee_position/quat + gripper width),
         place center + left/right jaw via URDF offsets -> world -> cam
         (ot_align URDF-FK T_cam_world, no ROS swap) -> project D405 K -> 224 crop.
  human: hand_processor kpts_2d -> wrist(0), thumb_tip(4), index_tip(8) -> 224 crop.
Both verified by eyeball in derisk_eef3_qc.py.

Output: outputs/flow_dataset/flow_ds_v4.npz (tracks, vis, eef3 (N,L,3,2), domain, vid, path)
"""
import os, json, numpy as np, cv2, torch, pandas as pd

CHUNKS = "phantom/data/processed_play_human_aytsai_chunks/play_human_aytsai_chunks"
HUMAN_CHUNKS = ["1001", "2000", "3000"]
ROBOT_DIRS = [f"play_robot_eef/play_robot_{i}_eef" for i in (1, 2, 3)]
INTR = "phantom_human_play/intrinsics_cam_high.json"
CX, CY, CW, CH = 195, 195, 256, 256
RES, L, S, P = 224, 16, 3, 48
MAXF, CLIP_STRIDE, MOVE_THRESH, TARGET_PER_VID = 9000, 6, 6.0, 700
# v4: include STATIC cube clips (v3 dropped everything < MOVE_THRESH, so the model never
# saw a still cube and hallucinated motion / the contact-gate had no "off" examples).
# Stratified keep: up to STATIC_FRAC of each video's quota are static clips (cube cpath
# < MOVE_THRESH), the rest moving. MOVE_THRESH now only labels static-vs-moving, not a filter.
STATIC_FRAC = 0.4
LINK6_TO_EE = np.array([0.156062, 0.0, 0.0]); FINGER_X = 0.0865


def crop_resize(img): return cv2.resize(img[CY:CY + CH, CX:CX + CW], (RES, RES))
def to_crop(uv): return None if uv is None else np.array([(uv[0]-CX)/CW*RES, (uv[1]-CY)/CH*RES], np.float32)


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
        # base(link6/wrist) + 2 jaw tips -> non-degenerate triangle, aligned with
        # human (wrist + thumb_tip + index_tip); does NOT collapse when gripper closed.
        pts3d = [base, base + R @ np.array([FINGER_X, +w/2, 0.0]), base + R @ np.array([FINGER_X, -w/2, 0.0])]
        pts = [to_crop(project(p, T_cw, K)) for p in pts3d]
        if any(p is None for p in pts): return None
        return np.stack(pts)  # (3,2) in 224px

    return fn, len(pos)


def human_eef3_loader(ch):
    hd = np.load(f"{CHUNKS}/{ch}/hand_processor/hand_data_right.npz")
    kp, det = hd["kpts_2d"], hd["hand_detected"]

    def fn(i):
        if i >= len(kp) or not bool(det[i]): return None
        pts = [to_crop(kp[i, j]) for j in (0, 4, 8)]  # wrist, thumb_tip, index_tip
        return np.stack(pts).astype(np.float32)

    return fn, len(kp)


def decode(domain, path):
    crops = []
    if domain == "human":
        cap = cv2.VideoCapture(path); i = 0
        while len(crops) < MAXF:
            ok, fr = cap.read()
            if not ok: break
            crops.append(crop_resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))); i += 1
        cap.release()
    else:
        import av; cont = av.open(path); st = cont.streams.video[0]
        for frame in cont.decode(st):
            if len(crops) >= MAXF: break
            crops.append(crop_resize(frame.to_ndarray(format="rgb24")))
        cont.close()
    return crops


@torch.no_grad()
def track(ct, frames_u8, q_xy, device):
    vid = torch.from_numpy(np.stack(frames_u8)).permute(0, 3, 1, 2)[None].float().to(device)
    q = np.concatenate([np.zeros((len(q_xy), 1), np.float32), q_xy], 1)
    tr, vs = ct(vid, queries=torch.from_numpy(q)[None].to(device))
    return tr[0].cpu().numpy(), vs[0].cpu().numpy().astype(np.float32)


def collect(ct, domain, items, device, rng, dom_label, vid_base):
    T, V, E, D, PA, VI = [], [], [], [], [], []
    for vj, (path, eef_fn) in enumerate(items):
        crops = decode(domain, path)
        max_static = int(round(TARGET_PER_VID * STATIC_FRAC)); max_move = TARGET_PER_VID - max_static
        n_static = n_move = 0
        starts = list(range(0, len(crops) - L * S, CLIP_STRIDE)); rng.shuffle(starts)
        for s0 in starts:
            if n_static >= max_static and n_move >= max_move: break
            idxs = [s0 + k * S for k in range(L)]
            fr = [crops[i] for i in idxs]
            m0 = cube_mask(fr[0]); ys, xs = np.where(m0 > 0)
            if len(xs) < P: continue
            sel = rng.choice(len(xs), P, replace=False)
            q = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
            tr, vs = track(ct, fr, q, device)
            c = tr.mean(1)
            move = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
            is_static = move < MOVE_THRESH
            if is_static and n_static >= max_static: continue   # static bucket full
            if (not is_static) and n_move >= max_move: continue  # moving bucket full
            eef3 = [eef_fn(idxs[t]) for t in range(L)]
            if any(e is None for e in eef3): continue
            T.append((tr / RES).astype(np.float32)); VI.append(vs.astype(np.float32))
            E.append((np.stack(eef3) / RES).astype(np.float32))  # (L,3,2)
            D.append(dom_label); PA.append(move); V.append(vid_base + vj)
            if is_static: n_static += 1
            else: n_move += 1
        nm = path.split('/')[-4] if domain == 'robot' else path.split('/')[-2]
        print(f"  {domain} {nm}: kept {n_move} moving + {n_static} static", flush=True)
    return T, VI, E, D, PA, V


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ct = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device).eval()
    from interactive_world_sim.algorithms.latent_decompose.ot_align import robot_world_to_cam
    T_cw = np.asarray(robot_world_to_cam(), np.float64); K = load_K()
    rng = np.random.default_rng(0)

    print("=== human ===", flush=True)
    human_items = [(f"{CHUNKS}/{c}/video_rgb_imgs.mkv", human_eef3_loader(c)[0]) for c in HUMAN_CHUNKS]
    Th, Vh, Eh, Dh, Ph, Wh = collect(ct, "human", human_items, device, rng, 0, 0)
    print("=== robot ===", flush=True)
    robot_items = [(f"{d}/videos/chunk-000/observation.images.cam_high/episode_000000.mp4",
                    robot_eef3_loader(d, T_cw, K)[0]) for d in ROBOT_DIRS]
    Tr, Vr, Er, Dr, Pr, Wr = collect(ct, "robot", robot_items, device, rng, 1, 10)

    tracks = np.stack(Th + Tr); vis = np.stack(Vh + Vr); eef3 = np.stack(Eh + Er)
    domain = np.array(Dh + Dr, np.int64); path = np.array(Ph + Pr, np.float32); vid = np.array(Wh + Wr, np.int64)
    os.makedirs("outputs/flow_dataset", exist_ok=True)
    np.savez("outputs/flow_dataset/flow_ds_v4.npz", tracks=tracks, vis=vis, eef3=eef3, domain=domain, path=path, vid=vid)
    print(f"\nsaved flow_ds_v4.npz tracks={tracks.shape} eef3={eef3.shape} "
          f"human={int((domain==0).sum())} robot={int((domain==1).sum())}", flush=True)


if __name__ == "__main__":
    main()
