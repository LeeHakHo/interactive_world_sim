"""ROBOT agent mask v3 — deterministic, NO SAM2. SAM2 structurally drops the gripper
when it sits over the blue plate (low contrast / tracking loss); the gripper is the
most important part of the embodiment, so we abandon SAM2 for robot and build the
mask directly from the dark structure of the arm:

  dark = (V < vthr) minus blue plate minus red cube
  arm  = dark components connected (via a dilation BRIDGE) to the darkest pixel
         (the gripper/arm is always the darkest thing in frame) — the bridge spans
         the small gaps (around the plate, claw-to-rod) without inflating the mask
         (we intersect the bridged-connectivity back with `dark`).
  close (3x3).

This reliably includes the gripper-over-plate (the dark gripper is in `dark` and
bridges to the darkest pixel), stays tight on dark pixels (no "大了一圈"), and rejects
the table/keyboard (light wood is not dark; not connected). Overwrites the canonical
agent_mask_res128_crop195-195-256-256.npy. The raw SAM2 backup *_sam2raw.npy (made by
apply_robot_mask_fix.py) is left intact for rollback. Human masks untouched. iws env.
"""
import glob
import os

import cv2
import numpy as np

RES = 128
VID = "observation.images.cam_high"
TAG = "crop195-195-256-256"
ROBOT_CACHE = "/scr2/yusenluo/interactive_world_sim/play_robot_eef/_video_cache"


def K(n):
    return np.ones((n, n), np.uint8)


def dark_mask(f, vthr=90):
    hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = (h > 90) & (h < 130) & (s > 70) & (v > 60)
    red = ((h < 12) | (h > 168)) & (s > 90) & (v > 35) & (v < 175)
    return ((v < vthr) & ~blue & ~red).astype(np.uint8)


def plate_rim(f):
    """Thin band around the blue plate's filled disk — its dark low-V edge gets caught
    by the dark mask, so subtract it (removes the plate-edge points). A gripper that
    genuinely crosses the rim only loses a ~4px sliver there."""
    hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = ((h > 90) & (h < 130) & (s > 70) & (v > 60)).astype(np.uint8)
    cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(blue)
    big = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(big) < 80:
        return np.zeros_like(blue)
    filled = np.zeros_like(blue)
    cv2.drawContours(filled, [big], -1, 1, -1)
    return ((cv2.dilate(filled, K(5)) > 0) & ~(cv2.erode(filled, K(5)) > 0)).astype(np.uint8)


def robot_v3(f, vthr=90, bridge=7):
    dark = dark_mask(f, vthr)
    if dark.sum() == 0:
        return np.zeros(f.shape[:2], np.uint8)
    lum = cv2.GaussianBlur(f.mean(2).astype(np.float32), (9, 9), 0)
    yx = np.unravel_index(int(np.argmin(lum)), lum.shape)
    seed = np.zeros(f.shape[:2], np.uint8)
    cv2.circle(seed, (yx[1], yx[0]), 3, 1, -1)
    br = cv2.dilate(dark, np.ones((bridge, bridge), np.uint8))
    nlab, lab = cv2.connectedComponents(br)
    keep = np.zeros_like(dark)
    for c in range(1, nlab):
        if ((lab == c) & (seed > 0)).any():
            keep[lab == c] = 1
    arm = cv2.morphologyEx((dark & keep), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return (arm & ~(plate_rim(f) > 0)).astype(np.uint8)


def main():
    for p in sorted(glob.glob(f"{ROBOT_CACHE}/*/{VID}_res{RES}_{TAG}.npy")):
        name = os.path.basename(os.path.dirname(p))
        mpath = p.replace(f"{VID}_res{RES}_{TAG}.npy", f"agent_mask_res{RES}_{TAG}.npy")
        n = os.path.getsize(p) // (3 * RES * RES)
        vid = np.memmap(p, mode="r", dtype=np.uint8, shape=(n, 3, RES, RES))
        out = np.zeros((n, RES, RES), np.uint8)
        for i in range(n):
            f = np.ascontiguousarray(vid[i].transpose(1, 2, 0))
            out[i] = (robot_v3(f) > 0).astype(np.uint8)
        np.save(mpath, out)
        cov = out.reshape(n, -1).mean(1)
        print(f"[{name}] N={n}  cov med={np.median(cov):.3f} max={cov.max():.3f} "
              f"zero={float((cov < 1e-4).mean()):.3f}", flush=True)


if __name__ == "__main__":
    main()
