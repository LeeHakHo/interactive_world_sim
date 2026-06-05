"""Step 1/2 for robot_arm_mask supervision: the 128 video-cache is DESATURATED (S~25 vs
original ~60), which breaks robot_arm_mask's HSV plate/dark detection (it grabs the plate).
So decode the ORIGINAL robot videos (full color) on g's frames (video 1/2/3, STRIDE=3),
crop to g's region [195:451,195:451] (256), store frames + fidx. Step 2 (phantom env) reads
these and runs robot_arm_mask. env: iws (has PyAV). Output: outputs/flow_wm/robot_arm_mask_g_cache/
"""
import os, numpy as np, av

OUT = "outputs/flow_wm/robot_arm_mask_g_cache"; os.makedirs(OUT, exist_ok=True)
STRIDE, MAXK = 3, 6000
for v in (1, 2, 3):
    base = f"play_robot_eef/play_robot_{v}_eef"
    cont = av.open(f"{base}/videos/chunk-000/observation.images.cam_high/episode_000000.mp4")
    fidx, frames = [], []
    for k, fr in enumerate(cont.decode(cont.streams.video[0])):
        if k >= MAXK: break
        if k % STRIDE != 0: continue
        rgb = fr.to_ndarray(format="rgb24")
        frames.append(np.ascontiguousarray(rgb[195:451, 195:451]).copy()); fidx.append(k)
    cont.close()
    fr = np.stack(frames).astype(np.uint8)
    np.savez(f"{OUT}/frames256_v{v}.npz", fidx=np.array(fidx, np.int32), frames=fr)
    import cv2
    s = cv2.cvtColor(fr[len(fr) // 2], cv2.COLOR_RGB2HSV)[..., 1].mean()
    print(f"video {v}: {len(fr)} frames @256 saved, mid-frame mean S={s:.1f} (should be ~60)", flush=True)
print("=== DONE ===", flush=True)
