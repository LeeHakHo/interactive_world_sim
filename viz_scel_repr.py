# viz_scel_repr.py
"""M0 eyeball: overlay object_node (centroid+scale circle) and grasp_frame (pos+orient arrow+width)
on real frames, robot & human. Output: outputs/cross_embodiment_wm/scel_m0_viz/."""
import os, numpy as np, cv2, scel_repr as S

DS = "outputs/flow_render_dataset_v3"
OUT = "outputs/cross_embodiment_wm/scel_m0_viz"; os.makedirs(OUT, exist_ok=True)
IMG = 128


def draw(frame, node, gf):
    im = frame.copy()
    cx, cy, sc = node
    p = (int(cx * IMG), int(cy * IMG))
    cv2.circle(im, p, max(2, int(sc * IMG)), (0, 255, 0), 1)      # scale circle (green)
    cv2.circle(im, p, 2, (0, 255, 0), -1)                         # centroid
    gx, gy, c, s, w = gf
    gp = (int(gx * IMG), int(gy * IMG))
    tip = (int((gx + 0.15 * c) * IMG), int((gy + 0.15 * s) * IMG))
    cv2.arrowedLine(im, gp, tip, (0, 0, 255), 1, tipLength=0.3)   # grasp orient (red)
    cv2.putText(im, f"w{w:.2f}", (gp[0] + 3, gp[1]), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 1)
    return im


def run(tag):
    z = np.load(f"{DS}/clips_{tag}.npz")
    fr, tr, ef = z["frames"], z["tracks"].astype(np.float32), z["eef"].astype(np.float32)
    nodes = S.object_node(tr); gfs = S.grasp_frame(ef)
    rng = np.random.default_rng(0); picks = rng.choice(len(fr), 4, replace=False)
    rows = []
    for i in picks:
        t = tr.shape[1] // 2
        rows.append(draw(fr[i, t].astype(np.uint8), nodes[i, t], gfs[i, t]))
    grid = np.concatenate(rows, 1)
    cv2.imwrite(f"{OUT}/overlay_{tag}.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return f"{tag}: {len(fr)} clips, drew {len(picks)} frames"


if __name__ == "__main__":
    lines = [run("robot"), run("human")]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True); print(f"saved {OUT}/", flush=True)
