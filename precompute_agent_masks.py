"""Precompute whole-agent masks for the EEF play data — DETERMINISTIC, no SAM2.

SAM2 point-seeding over-segments to the whole table in some frames, so we use the
validated deterministic masks from inpaint_gap_test.py (the "image-2" the user
accepted):
  human: hand (curated masks_arm) ∪ dark-sleeve CONNECTED to the hand → whole arm;
         close; subtract the dark-red cube. The table isn't dark and isn't connected
         to the hand, so it's excluded.
  robot: dark region CONNECTED to the darkest pixel (gripper) via a dilation bridge
         (catches thin rods/claw); close; protect the blue plate; subtract the cube.

Runs in the `iws` env (cv2 + numpy only, no GPU/SAM2). Reads the iws video-cache
memmaps (the exact crop+resize frames the model trains on → masks align 1:1). The
human HAND seed (masks_arm) is frame-aligned by concatenating the aytsai chunks that
make up each eef episode (verified: play_human_eef_1 N=35830 == chunks 1000+1001+
1002+1003). Native 128 px; kernels scaled down from the 224-tuned originals. Caches
`agent_mask_res128_<croptag>.npy` (N,H,W) uint8 next to each episode's video cache.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import cv2
import numpy as np

RES = 128
CROPTAG = "crop195-195-256-256"
CROP = (195, 195, 256, 256)  # x,y,w,h on the 480x640 cam_high
ROBOT_CACHE = "/scr2/yusenluo/interactive_world_sim/play_robot_eef/_video_cache"
HUMAN_CACHE = "/scr2/yusenluo/interactive_world_sim/human_play_eef_data/_video_cache"
CHUNKS = "/scr2/yusenluo/interactive_world_sim/phantom/data/processed_play_human_aytsai_chunks/play_human_aytsai_chunks"
VID = "observation.images.cam_high"


def K(n):
    return np.ones((n, n), np.uint8)


def hsv_of(c):
    return cv2.cvtColor(c, cv2.COLOR_RGB2HSV)


def dark_mask(c, thr=75):
    return cv2.morphologyEx((hsv_of(c)[..., 2] < thr).astype(np.uint8),
                            cv2.MORPH_OPEN, K(3))


def cube_mask(c):
    h, s, v = (hsv_of(c)[..., k] for k in range(3))
    red = ((h < 12) | (h > 168)) & (s > 90) & (v > 35) & (v < 175)
    m = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_OPEN, K(3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K(5))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1 or st[1:, cv2.CC_STAT_AREA].max() < 15:
        return np.zeros_like(m)
    return (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def connected_to(mask, seed):
    """Keep connected components of `mask` that overlap `seed`."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8))
    keep = np.zeros_like(mask, np.uint8)
    for c in range(1, n):
        comp = lab == c
        if (comp & (seed > 0)).any():
            keep[comp] = 1
    return keep


def _pt_mask(c, pt):
    m = np.zeros(c.shape[:2], np.uint8)
    cv2.circle(m, (int(pt[0]), int(pt[1])), 3, 1, -1)
    return m


def plate_blue(c):
    h, s, v = (hsv_of(c)[..., k] for k in range(3))
    return ((h > 90) & (h < 130) & (s > 70) & (v > 60)).astype(np.uint8)


def plate_rim(c):
    """Band around the blue plate's filled disk — its dark low-V edge gets caught by
    the dark mask, so subtract this thin band from the agent mask."""
    blue = plate_blue(c)
    cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(blue)
    big = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(big) < 80:
        return np.zeros_like(blue)
    filled = np.zeros_like(blue)
    cv2.drawContours(filled, [big], -1, 1, -1)
    return ((cv2.dilate(filled, K(5)) > 0) & ~(cv2.erode(filled, K(5)) > 0)).astype(np.uint8)


def human_arm_mask(c, hand):
    """c: (128,128,3) uint8 RGB. hand: (128,128) bool masks_arm HAND seed, aligned.

    Deterministic (no SAM2): hand ∪ dark-sleeve connected to the hand, − cube.
    """
    hand = cv2.dilate(hand.astype(np.uint8), K(3))
    if hand.sum() == 0:
        return np.zeros(c.shape[:2], np.uint8)   # hand out of view → no agent mask
    sleeve = connected_to(dark_mask(c), hand)
    arm = cv2.morphologyEx(((hand | sleeve) > 0).astype(np.uint8), cv2.MORPH_CLOSE, K(5))
    cube = cv2.dilate(cube_mask(c), K(3))
    return (arm & ~(cube > 0) & ~(plate_rim(c) > 0)).astype(np.uint8)


def robot_arm_mask(c):
    """Deterministic (no SAM2): dark region connected (via dilation bridge) to the
    darkest pixel (gripper) → whole arm incl. thin rods; protect plate; − cube."""
    hsv = hsv_of(c)
    hh, ss, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    dark = (v < 85).astype(np.uint8)
    plate = ((hh > 90) & (hh < 130) & (ss > 70) & (v > 60)).astype(np.uint8)
    lum = cv2.GaussianBlur(c.mean(2).astype(np.float32), (9, 9), 0)
    yx = np.unravel_index(int(np.argmin(lum)), lum.shape)
    seed = _pt_mask(c, (int(yx[1]), int(yx[0])))
    bridged = cv2.dilate(dark, K(7))            # bridge claw→arm gap
    arm = (dark & (connected_to(bridged, seed) > 0)).astype(np.uint8)
    arm = cv2.morphologyEx(arm, cv2.MORPH_CLOSE, K(5))
    arm = cv2.dilate(arm, K(3))
    dark_d = cv2.dilate(dark, K(3))
    # drop PURE-plate pixels but keep dark-arm lying over the plate
    arm = (arm & ~((plate > 0) & (dark_d == 0))).astype(np.uint8)
    cube = cv2.dilate(cube_mask(c), K(3))
    return (arm & ~(cube > 0) & ~(plate_rim(c) > 0)).astype(np.uint8)


# ---------- masks_arm alignment (human) ----------
def crop_resize_mask(m480):
    x, y, w, h = CROP
    return cv2.resize(m480[y:y + h, x:x + w].astype(np.uint8), (RES, RES),
                      interpolation=cv2.INTER_NEAREST)


class HumanHandSeed:
    """Lazy masks_arm for an eef episode: concat of aytsai chunks (i*1000 + j),
    cropped+resized to 128, indexed by eef frame."""

    def __init__(self, ep_i):
        chunks = sorted(int(d) for d in os.listdir(CHUNKS)
                        if d.isdigit() and int(d) // 1000 == ep_i)
        self.mms, self.offs, off = [], [], 0
        for ch in chunks:
            mm = np.load(f"{CHUNKS}/{ch}/segmentation_processor/masks_arm.npy", mmap_mode="r")
            self.mms.append(mm); self.offs.append((off, off + mm.shape[0])); off += mm.shape[0]
        self.total = off

    def __getitem__(self, idx):
        for mm, (a, b) in zip(self.mms, self.offs):
            if a <= idx < b:
                return crop_resize_mask(np.asarray(mm[idx - a]))
        return np.zeros((RES, RES), np.uint8)


def load_frames(npy_path):
    n = os.path.getsize(npy_path) // (3 * RES * RES)
    arr = np.memmap(npy_path, mode="r", dtype=np.uint8, shape=(n, 3, RES, RES))
    return np.ascontiguousarray(arr.transpose(0, 2, 3, 1)), n


def ep_index_from_dir(d):
    m = re.search(r"play_human_eef_(\d+)", d)
    return int(m.group(1)) if m else None


def episode_cache_paths(domain):
    root = ROBOT_CACHE if domain == "robot" else HUMAN_CACHE
    return sorted(glob.glob(f"{root}/*/{VID}_res{RES}_{CROPTAG}.npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["robot", "human", "both"], default="both")
    ap.add_argument("--max-frames", type=int, default=0, help="0=all (testing)")
    ap.add_argument("--max-eps", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for domain in (["robot", "human"] if args.domain == "both" else [args.domain]):
        for p in (episode_cache_paths(domain)[: args.max_eps] if args.max_eps
                  else episode_cache_paths(domain)):
            out = p.replace(f"{VID}_res{RES}_{CROPTAG}.npy", f"agent_mask_res{RES}_{CROPTAG}.npy")
            if os.path.exists(out) and not args.overwrite:
                print(f"[skip] {out}", flush=True); continue
            frames, n = load_frames(p)
            if args.max_frames:
                frames = frames[: args.max_frames]
            hand = (HumanHandSeed(ep_index_from_dir(os.path.basename(os.path.dirname(p))))
                    if domain == "human" else None)
            masks = np.zeros((frames.shape[0], RES, RES), np.uint8)
            for i in range(frames.shape[0]):
                c = np.ascontiguousarray(frames[i])
                masks[i] = robot_arm_mask(c) if domain == "robot" else human_arm_mask(c, hand[i] > 0)
            np.save(out, masks)
            cov = masks.reshape(masks.shape[0], -1).mean(1)
            print(f"[{domain}] {os.path.basename(os.path.dirname(p))} N={masks.shape[0]} "
                  f"cov med={np.median(cov):.3f} max={cov.max():.3f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
