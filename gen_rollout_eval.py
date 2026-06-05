"""Generate LONG object-point sequences for the compounding-error test. Clips are only L=24
(K=4+F=20), too short for autoregressive rollout. Here: held-out robot episodes, decode long
runs, CoTracker the same P=48 cube points + 3 EEF points over SEQ=64 sample frames (S=4 stride
-> spans 256 raw frames), so we can roll ② out to horizon H=40 with GT eef available each step.
Reuses the verified gen_flow_render_dataset CoTracker/EEF logic. Output:
outputs/flow_wm/rollout_eval/seqs.npz
"""
import os, sys, numpy as np, torch
sys.path.insert(0, ".")
from gen_flow_render_dataset import cube_mask, track, decode, ROBOT_DIRS, TRACK_RES, P, S, MOVE_MIN, load_K
from gen_flow_render_dataset_v2 import robot_loader
from interactive_world_sim.algorithms.latent_decompose.ot_align import robot_world_to_cam

OUT = "outputs/flow_wm/rollout_eval"; os.makedirs(OUT, exist_ok=True)
K, F, H = 4, 20, 40
SEQ = H + K + F                 # 64 sample frames -> eef available for every rollout step
BIG_STRIDE = 150               # raw-frame stride between sequence starts
TARGET_SEQ = 60
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ct = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device).eval()
    T_cw = np.asarray(robot_world_to_cam(), np.float64); Kintr = load_K()
    rng = np.random.default_rng(0)
    TR, EF, VS, FR, JN = [], [], [], [], []
    for d in ROBOT_DIRS:
        if len(TR) >= TARGET_SEQ: break
        path = f"{d}/videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
        eef_fn, _, joint = robot_loader(d, T_cw, Kintr)
        c224, c128 = decode("robot", path)
        starts = list(range(0, len(c224) - SEQ * S, BIG_STRIDE)); rng.shuffle(starts)
        got = 0
        for s0 in starts:
            if len(TR) >= TARGET_SEQ or got >= 12: break
            idxs = [s0 + k * S for k in range(SEQ)]
            if idxs[-1] >= len(c224): continue
            fr224 = [c224[i] for i in idxs]
            m0 = cube_mask(fr224[0]); ys, xs = np.where(m0 > 0)
            if len(xs) < P: continue
            sel = rng.choice(len(xs), P, replace=False)
            q = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
            tr, vs = track(ct, fr224, q, device); cc = tr.mean(1)
            if float(np.sum(np.linalg.norm(np.diff(cc, axis=0), axis=1))) < MOVE_MIN: continue
            eef = [eef_fn(idxs[t]) for t in range(SEQ)]
            if any(e is None for e in eef): continue
            TR.append((tr / TRACK_RES).astype(np.float32))
            EF.append((np.stack(eef) / TRACK_RES).astype(np.float32))
            VS.append(vs.astype(np.float32))
            FR.append(np.stack([c128[i] for i in idxs]).astype(np.uint8))
            JN.append(joint[idxs].astype(np.float32)); got += 1
        print(f"{d.split('/')[-1]}: kept {got} (total {len(TR)})", flush=True)
    np.savez(f"{OUT}/seqs.npz", tracks=np.stack(TR), eef=np.stack(EF), vis=np.stack(VS),
             frames=np.stack(FR), joint=np.stack(JN), K=K, F=F, H=H, SEQ=SEQ)
    print(f"saved {OUT}/seqs.npz tracks={np.stack(TR).shape}\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
