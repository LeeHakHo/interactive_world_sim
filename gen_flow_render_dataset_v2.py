"""Regenerate ROBOT clips WITH joint action (+ frame index), for densev4d (g: joint->mask
feeds ③). The original clips_robot.npz lacks joint, and clips don't store frame indices,
so g(joint) can't be aligned. This robot-only re-gen stores everything densev4d needs:
frames@128 + object tracks + 3 EEF points + JOINT action(7) + fidx.
Reuses the verified CoTracker/geometry logic from gen_flow_render_dataset.py.
Output: outputs/flow_render_dataset/clips2_robot.npz
"""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, ".")
from gen_flow_render_dataset import (cube_mask, quat_xyzw_to_R, load_K, project, to_crop, decode, track,
                                     ROBOT_DIRS, TRACK_RES, L, S, P, CLIP_STRIDE, MOVE_MIN, TARGET_PER_VID,
                                     LINK6_TO_EE, FINGER_X)
from interactive_world_sim.algorithms.latent_decompose.ot_align import robot_world_to_cam

device = "cuda" if torch.cuda.is_available() else "cpu"


def robot_loader(robot_dir, T_cw, K):
    df = pd.read_parquet(f"{robot_dir}/data/chunk-000/episode_000000.parquet")
    pos = np.stack(df["action_right_ee_position"].values).astype(np.float64)
    quat = np.stack(df["action_right_ee_quat_xyzw"].values).astype(np.float64)
    grip = np.clip(df["action_right_gripper"].values.astype(np.float64), 0, None)
    joint = np.stack(df["action"].values).astype(np.float32)[:, :7]

    def eef_fn(i):
        if i >= len(pos): return None
        R = quat_xyzw_to_R(quat[i]); base = pos[i] - R @ LINK6_TO_EE; w = grip[i]
        pts3d = [base, base + R @ np.array([FINGER_X, +w / 2, 0.0]), base + R @ np.array([FINGER_X, -w / 2, 0.0])]
        pts = [to_crop(project(p, T_cw, K)) for p in pts3d]
        if any(p is None for p in pts): return None
        return np.stack(pts)
    return eef_fn, len(pos), joint


def main():
    ct = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device).eval()
    T_cw = np.asarray(robot_world_to_cam(), np.float64); K = load_K()
    rng = np.random.default_rng(0)
    F, T, E, V, VID, JOINT, FIDX = [], [], [], [], [], [], []
    for vj, d in enumerate(ROBOT_DIRS):
        path = f"{d}/videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
        eef_fn, _, joint = robot_loader(d, T_cw, K)
        c224, c128 = decode("robot", path); got = 0
        starts = list(range(0, len(c224) - L * S, CLIP_STRIDE)); rng.shuffle(starts)
        for s0 in starts:
            if got >= TARGET_PER_VID: break
            idxs = [s0 + k * S for k in range(L)]
            fr224 = [c224[i] for i in idxs]
            m0 = cube_mask(fr224[0]); ys, xs = np.where(m0 > 0)
            if len(xs) < P: continue
            sel = rng.choice(len(xs), P, replace=False)
            q = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
            tr, vs = track(ct, fr224, q, device); cc = tr.mean(1)
            if float(np.sum(np.linalg.norm(np.diff(cc, axis=0), axis=1))) < MOVE_MIN: continue
            eef3 = [eef_fn(idxs[t]) for t in range(L)]
            if any(e is None for e in eef3): continue
            F.append(np.stack([c128[i] for i in idxs]).astype(np.uint8))
            T.append((tr / TRACK_RES).astype(np.float16))
            E.append((np.stack(eef3) / TRACK_RES).astype(np.float16))
            V.append(vs.astype(np.float16)); VID.append(10 + vj)
            JOINT.append(joint[idxs].astype(np.float32)); FIDX.append(np.array(idxs, np.int32))
            got += 1
        print(f"robot {d.split('/')[-1]}: kept {got}", flush=True)
    os.makedirs("outputs/flow_render_dataset", exist_ok=True)
    np.savez("outputs/flow_render_dataset/clips2_robot.npz",
             frames=np.stack(F), tracks=np.stack(T), eef=np.stack(E), vis=np.stack(V),
             vid=np.array(VID, np.int64), joint=np.stack(JOINT), fidx=np.stack(FIDX))
    print(f"saved clips2_robot.npz N={len(F)} joint={np.stack(JOINT).shape}\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
