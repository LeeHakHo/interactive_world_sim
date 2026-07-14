"""组合新臂对比 gif: GT | flow | flowskel | flowskel3 | flowwarp (seed0, 复用各 run eval 已存的
seq*_render.npy, 不重跑渲染). 输出 outputs/cross_embodiment_wm/dualview_dit_formal/compare_newarms/gifs/.
iws env; 只需 CPU + 一次 clips frames 读取."""
import json
import os

import numpy as np

from viz_combined import save_combined_gif, build_flow_cols

ROOT = "outputs/cross_embodiment_wm/dualview_dit_formal"
ARMS = ["flow", "flowskel", "flowskel3", "flowwarp"]
K, H, NSEQ = 4, 20, 6
OUT = f"{ROOT}/compare_newarms"; os.makedirs(f"{OUT}/gifs", exist_ok=True)

z = np.load("outputs/flow_render_dataset_can_dual/clips_robot.npz")
chosen = json.load(open(f"{ROOT}/eval_seqs_n24.json"))[:NSEQ]
frames = {0: z["frames"], 1: z["frames_low"]}
tracks = {0: z["tracks"].astype(np.float32), 1: np.nan_to_num(z["tracks_low"].astype(np.float32), nan=0.5)}
eef = {0: z["eef"].astype(np.float32), 1: np.nan_to_num(z["eef_low"].astype(np.float32), nan=0.5)}
vname = {0: "high", 1: "low"}

for si in chosen:
    rrs = {}
    for c in ARMS:
        p = f"{ROOT}/{c}_cv1_s0/gifs/seq{si}_render.npy"
        rrs[c] = np.load(p) if os.path.exists(p) else None
        if rrs[c] is None: print(f"missing {p}, skip col", flush=True)
    for v in range(2):
        gt = frames[v][si, K:K + H].astype(np.uint8)
        cols = [gt]; labels = [f"GT {vname[v]}"]
        for c in ARMS:
            if rrs[c] is not None:
                cols.append(rrs[c][v][:H]); labels.append(c)
        cols = np.stack(cols)
        fc = build_flow_cols(gt, tracks[v][si, K:K + H], [None] * len(cols), eef[v][si, K:K + H])
        save_combined_gif(f"{OUT}/gifs/seq{si}_cam{vname[v]}.gif", cols, fc, labels, [None] * len(cols), K,
                          caption=f"new arms seed0 | GT|flow|flowskel|flowskel3|flowwarp | cam_{vname[v]}")
print(f"saved -> {OUT}/gifs/", flush=True)
