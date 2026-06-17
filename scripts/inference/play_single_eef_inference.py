"""
Interactive EEF keyboard inference for play_single latent world model.

Keys:
  W / S      : EEF x  +/-
  A / D      : EEF y  +/-
  Q / E      : EEF z  +/-
  I / K      : pitch  +/-
  J / L      : yaw    +/-
  U / O      : roll   +/-
  G          : gripper toggle (open <-> closed)
  R          : reset EEF to initial state
  Space      : hold (no movement, advance 1 step)
  ESC        : quit

Usage:
    python scripts/inference/play_single_eef_inference.py \
        --ckpt outputs/.../checkpoints/best.ckpt \
        --dataset_dir data/play_single_hdf5 \
        --episode 0
"""

import argparse
import os
import sys
import termios
import tty
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import LatentWorldModel


def getch():
    """Read a single character from stdin without pressing Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


OBS_KEYS = ["camera_0_color", "camera_1_color"]

# EEF action indices: [x, y, z, roll, pitch, yaw, gripper]
STEP = 0.01   # meters or radians per keypress
GRIPPER_MAX = 0.4
GRIPPER_MIN = 0.0

KEY_MAP = {
    ord("w"): (0,  +STEP),   # x+
    ord("s"): (0,  -STEP),   # x-
    ord("a"): (1,  +STEP),   # y+
    ord("d"): (1,  -STEP),   # y-
    ord("q"): (2,  +STEP),   # z+
    ord("e"): (2,  -STEP),   # z-
    ord("u"): (3,  +STEP),   # roll+
    ord("o"): (3,  -STEP),   # roll-
    ord("i"): (4,  +STEP),   # pitch+
    ord("k"): (4,  -STEP),   # pitch-
    ord("j"): (5,  +STEP),   # yaw+
    ord("l"): (5,  -STEP),   # yaw-
}


def load_model(ckpt_path: str, device: str, dec_infer_steps: int = 3) -> LatentWorldModel:
    cfg_path = Path(ckpt_path).parent.parent / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.algorithm.load_ae = None
    cfg.algorithm.dec_infer_steps = dec_infer_steps
    algo = LatentWorldModel.load_from_checkpoint(
        ckpt_path, cfg=cfg.algorithm, map_location=device,
        strict=False, weights_only=False,
    )
    algo.eval()
    return algo


def load_episode(dataset_dir: str, episode_idx: int):
    for subdir in ["train", "val", ""]:
        path = os.path.join(dataset_dir, subdir, f"episode_{episode_idx}.hdf5")
        if os.path.exists(path):
            break
    else:
        raise FileNotFoundError(f"episode_{episode_idx}.hdf5 not found under {dataset_dir}")

    with h5py.File(path, "r") as f:
        cam0 = f["obs"]["images"]["camera_0_color"][:]  # (T, H, W, 3)
        cam1 = f["obs"]["images"]["camera_1_color"][:]  # (T, H, W, 3)
        actions = f["action"][:]                         # (T, 7) EEF
    return cam0, cam1, actions.astype(np.float32)


def compute_action_bounds(dataset_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-dim min/max across all train episodes."""
    import glob
    files = sorted(glob.glob(os.path.join(dataset_dir, "train", "episode_*.hdf5")))
    all_actions = []
    for f_path in files:
        with h5py.File(f_path, "r") as f:
            all_actions.append(f["action"][:])
    all_actions = np.concatenate(all_actions, axis=0)
    return all_actions.min(axis=0), all_actions.max(axis=0)


def encode_frame(model, normalizer, cam0_np, cam1_np, device, dtype):
    def to_tensor(img):
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    img0 = normalizer["camera_0_color"].normalize(to_tensor(cam0_np))
    img1 = normalizer["camera_1_color"].normalize(to_tensor(cam1_np))
    with torch.no_grad():
        z = model.encoder_forward(torch.cat([img0, img1], dim=1))
    return z  # (1, C, H_lat, W_lat)


def decode_latent(model, z, resolution, normalizer, num_views):
    with torch.no_grad():
        xs = render_img_cm(model, z, resolution, normalizer, num_views=num_views)
    frames = []
    for i in range(num_views):
        ch = xs[0, i*3:(i+1)*3]
        frames.append((ch.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8))
    return frames


def make_video_frame(gt0, gt1, pred0, pred1, h, w):
    def to_bgr(img):
        return cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_RGB2BGR)
    row = np.concatenate([to_bgr(gt0), to_bgr(gt1), to_bgr(pred0), to_bgr(pred1)], axis=1)
    labels = ["GT cam0", "GT cam1", "Pred cam0", "Pred cam1"]
    for i, label in enumerate(labels):
        cv2.putText(row, label, (i*w+5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return row


def print_status(eef_state, step):
    x, y, z, roll, pitch, yaw, gripper = eef_state
    print(f"\r[step {step:4d}] x:{x:+.3f} y:{y:+.3f} z:{z:+.3f} | "
          f"roll:{roll:+.3f} pitch:{pitch:+.3f} yaw:{yaw:+.3f} | "
          f"gripper:{'OPEN  ' if gripper > 0.01 else 'CLOSED'}", end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_dir", default="data/play_single_hdf5")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--dec_infer_steps", type=int, default=3)
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Frame index within the episode to start inference from (initial frame)")
    parser.add_argument("--step_size", type=float, default=STEP, help="EEF delta per keypress")
    parser.add_argument("--output_dir", default="outputs/inference")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    step_size = args.step_size
    gripper_step = GRIPPER_MAX / 10  # 10 steps to fully open/close
    key_map = {
        ord("w"): (0,  +step_size),
        ord("s"): (0,  -step_size),
        ord("a"): (1,  +step_size),
        ord("d"): (1,  -step_size),
        ord("q"): (2,  +step_size),
        ord("e"): (2,  -step_size),
        ord("u"): (3,  +step_size),
        ord("o"): (3,  -step_size),
        ord("i"): (4,  +step_size),
        ord("k"): (4,  -step_size),
        ord("j"): (5,  +step_size),
        ord("l"): (5,  -step_size),
        ord("g"): (6,  +gripper_step),  # gripper open
        ord("h"): (6,  -gripper_step),  # gripper close
    }

    os.makedirs(args.output_dir, exist_ok=True)
    h = w = args.resolution
    device = args.device

    print("Computing action bounds from dataset...")
    action_min, action_max = compute_action_bounds(args.dataset_dir)
    print(f"  xyz  : [{action_min[:3]}] ~ [{action_max[:3]}]")
    print(f"  euler: [{action_min[3:6]}] ~ [{action_max[3:6]}]")
    print(f"  grip : [{action_min[6]:.3f}] ~ [{action_max[6]:.3f}]")

    print("Loading model...")
    model = load_model(args.ckpt, device, args.dec_infer_steps)
    model = model.to(device)
    normalizer = model.normalizer
    dtype = model.dtype
    num_views = len(model.obs_keys)
    n_tokens = model.n_tokens

    print(f"Loading episode {args.episode}...")
    cam0_frames, cam1_frames, gt_actions = load_episode(args.dataset_dir, args.episode)

    # Clamp start frame to valid range
    start = max(0, min(args.start_frame, len(gt_actions) - 1))
    if start != args.start_frame:
        print(f"  start_frame {args.start_frame} out of range, clamped to {start}")
    print(f"  starting from frame {start} (episode length {len(gt_actions)})")

    # Initialize EEF state from the start-frame GT action
    eef_state = gt_actions[start].copy()  # [x, y, z, roll, pitch, yaw, gripper]
    init_eef = eef_state.copy()

    # Encode initial frame
    z0 = encode_frame(model, normalizer, cam0_frames[start], cam1_frames[start], device, dtype)
    curr_latent = z0.unsqueeze(1)  # (1, 1, C, H_lat, W_lat)

    action_tensor = torch.from_numpy(eef_state).to(device=device, dtype=dtype)
    action_norm = normalizer["action"].normalize(action_tensor.unsqueeze(0))  # (1, 7)
    past_actions = action_norm.unsqueeze(0)  # (1, 1, 7)

    # Decode initial
    pred_frames = decode_latent(model, z0, h, normalizer, num_views)

    step = 0
    recorded_frames = []

    print("\nControls: W/S=x  A/D=y  Q/E=z  U/O=roll  I/K=pitch  J/L=yaw  G=gripper open  H=gripper close  R=reset  ESC=quit")
    print("Press any key to step forward...\n")

    try:
        while True:
            gt_idx = min(start + step, len(cam0_frames) - 1)
            frame = make_video_frame(
                cam0_frames[gt_idx], cam1_frames[gt_idx],
                pred_frames[0], pred_frames[1], h, w,
            )
            recorded_frames.append(frame)
            print_status(eef_state, step)

            ch = getch()

            if ch == "\x1b":  # ESC
                break
            elif ch == "r":
                eef_state = init_eef.copy()
                continue
            elif ord(ch) in key_map:
                idx, delta = key_map[ord(ch)]
                eef_state[idx] += delta
                eef_state[idx] = np.clip(eef_state[idx], action_min[idx], action_max[idx])
            elif ch == " ":
                pass
            else:
                continue

            # Feed action to model
            action_tensor = torch.from_numpy(eef_state).to(device=device, dtype=dtype)
            action_norm = normalizer["action"].normalize(action_tensor.unsqueeze(0))
            future_action = action_norm.unsqueeze(0)  # (1, 1, 7)
            action_input = torch.cat([past_actions, future_action], dim=1)

            with torch.no_grad():
                z_pred = model.dynamics_forward(curr_latent, action_input)

            pred_frames = decode_latent(model, z_pred[:, -1], h, normalizer, num_views)

            curr_latent = torch.cat([curr_latent, z_pred[:, -1:]], dim=1)[:, -n_tokens:]
            past_actions = torch.cat([past_actions, future_action], dim=1)[:, -n_tokens:]
            step += 1

    except KeyboardInterrupt:
        pass

    print()

    # Save recording with PyAV (AV1-free h264)
    if recorded_frames:
        out_path = os.path.join(args.output_dir, f"eef_inference_ep{args.episode}.mp4")
        container = av.open(out_path, mode="w")
        stream = container.add_stream("h264", rate=10)
        stream.width, stream.height = w * 4, h
        stream.pix_fmt = "yuv420p"
        for f in recorded_frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for pkt in stream.encode(av_frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))
    main()
