"""
Non-interactive batch evaluation for the play_single EEF latent world model.

Replays a set of *scripted* teleop sequences (defined in SCENARIOS below) from
different initial block locations (episode + start_frame) and saves one mp4 per
scenario. Scripted interaction (push / grasp / pull / ...) probes slightly-OOD
generalization more reliably than GT action replay.

Each scenario key string is fed one character per dynamics step, using the same
key->delta mapping as the interactive script:

    w / s : x +/-     a / d : y +/-     q / e : z +/-
    u / o : roll +/-  i / k : pitch +/- j / l : yaw +/-
    g     : gripper open (+)    h : gripper close (-)
    ' '   : hold (no movement, advance 1 step)

Usage:
    python scripts/inference/play_single_eef_eval.py \
        --ckpt outputs/.../checkpoints/best.ckpt \
        --dataset_dir data/play_robot_v3_hdf5 \
        --output_dir outputs/eval_videos
"""

import argparse
import os
import sys

import av
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

# reuse helpers/constants from the interactive script (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from play_single_eef_inference import (  # noqa: E402
    GRIPPER_MAX,
    STEP,
    compute_action_bounds,
    decode_latent,
    encode_frame,
    load_episode,
    load_model,
)

# Register OmegaConf resolvers used in checkpoint configs (the interactive script
# does this inside main(), which we don't call here).
OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}), replace=True)
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x), replace=True)


# ---------------------------------------------------------------------------
# Scenarios: edit these. Each replays `keys` (1 char = 1 step) from the given
# episode's start_frame. Pick a few different initial block locations.
# ---------------------------------------------------------------------------
SCENARIOS = [
    # name, source episode, frame to start from, scripted key sequence
    {"name": "episode_0",  "episode": 0, "start_frame": 0,  "keys": "ggaaahhddggaaahhddssaaddwwwwaassssddwwdd"},
    {"name": "episode_1",  "episode": 0, "start_frame": 0,  "keys": "dddddddddddd"},
    {"name": "episode_2",  "episode": 1, "start_frame": 0,  "keys": "wwwwwwwwwwww"},
    {"name": "episode_3",  "episode": 1, "start_frame": 30, "keys": "eeeee" + "hhhhh" + "qqqqq"},
    {"name": "episode_4",   "episode": 2, "start_frame": 0,  "keys": "hhhhh" + "ssssssss"},
    {"name": "episode_5", "episode": 3, "start_frame": 0,  "keys": "eeeeeeee" + "          "},
]


def build_key_map(step_size: float) -> dict:
    """Same mapping as the interactive script's key_map."""
    gripper_step = GRIPPER_MAX / 10  # 10 steps to fully open/close
    return {
        "w": (0, +step_size), "s": (0, -step_size),   # x
        "a": (1, +step_size), "d": (1, -step_size),   # y
        "q": (2, +step_size), "e": (2, -step_size),   # z
        "u": (3, +step_size), "o": (3, -step_size),   # roll
        "i": (4, +step_size), "k": (4, -step_size),   # pitch
        "j": (5, +step_size), "l": (5, -step_size),   # yaw
        "g": (6, +gripper_step), "h": (6, -gripper_step),  # gripper open/close
    }


def compose_frame(pred0, pred1, name, step, h, w):
    """Side-by-side cam0 | cam1 prediction with a caption."""
    def to_bgr(img):
        return cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_RGB2BGR)

    row = np.concatenate([to_bgr(pred0), to_bgr(pred1)], axis=1)
    cv2.putText(row, f"{name}  step {step}", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return row


def save_mp4(frames, out_path, h, w, fps=10):
    """Save BGR frames as h264 mp4 (AV1-free), matching the interactive script."""
    container = av.open(out_path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width, stream.height = w * 2, h  # two views side by side
    stream.pix_fmt = "yuv420p"
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for pkt in stream.encode(av_frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


@torch.no_grad()
def run_scenario(model, scn, dataset_dir, action_min, action_max, key_map,
                 h, w, device, dtype, output_dir, key_repeat=1, bound_margin=0.0):
    """Replay one scripted scenario and write an mp4. Returns the output path."""
    episode = scn["episode"]
    # clip bounds widened by bound_margin * range (0.0 = strict dataset range)
    span = action_max - action_min
    lo = action_min - bound_margin * span
    hi = action_max + bound_margin * span
    # repeat each key (action) key_repeat times, e.g. "wh" -> "wwww...hhhh..."
    keys = "".join(ch * key_repeat for ch in scn["keys"])
    num_views = len(model.obs_keys)
    n_tokens = model.n_tokens
    normalizer = model.normalizer

    cam0_frames, cam1_frames, gt_actions = load_episode(dataset_dir, episode)
    start = max(0, min(scn.get("start_frame", 0), len(gt_actions) - 1))

    # initial state + latent from the start frame
    eef_state = gt_actions[start].copy()
    z0 = encode_frame(model, normalizer, cam0_frames[start], cam1_frames[start], device, dtype)
    curr_latent = z0.unsqueeze(1)  # (1, 1, C, H, W)

    action_norm = normalizer["action"].normalize(
        torch.from_numpy(eef_state).to(device=device, dtype=dtype).unsqueeze(0)
    )
    past_actions = action_norm.unsqueeze(0)  # (1, 1, 7)

    pred_frames = decode_latent(model, z0, h, normalizer, num_views)
    recorded = [compose_frame(pred_frames[0], pred_frames[1], scn["name"], 0, h, w)]

    for step, ch in enumerate(keys, start=1):
        if ch in key_map:
            idx, delta = key_map[ch]
            eef_state[idx] = np.clip(eef_state[idx] + delta, lo[idx], hi[idx])
        # else: ' ' or unknown -> hold (no movement)

        future_action = normalizer["action"].normalize(
            torch.from_numpy(eef_state).to(device=device, dtype=dtype).unsqueeze(0)
        ).unsqueeze(0)  # (1, 1, 7)
        action_input = torch.cat([past_actions, future_action], dim=1)

        z_pred = model.dynamics_forward(curr_latent, action_input)
        pred_frames = decode_latent(model, z_pred[:, -1], h, normalizer, num_views)
        recorded.append(compose_frame(pred_frames[0], pred_frames[1], scn["name"], step, h, w))

        curr_latent = torch.cat([curr_latent, z_pred[:, -1:]], dim=1)[:, -n_tokens:]
        past_actions = torch.cat([past_actions, future_action], dim=1)[:, -n_tokens:]

    out_path = os.path.join(output_dir, f"{scn['name']}_ep{episode}_sf{start}.mp4")
    save_mp4(recorded, out_path, h, w)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Batch scripted-teleop eval videos")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_dir", default="data/play_robot_v3_hdf5")
    parser.add_argument("--output_dir", default="outputs/eval_videos")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--dec_infer_steps", type=int, default=3)
    parser.add_argument("--step_size", type=float, default=STEP, help="EEF delta per key step")
    parser.add_argument("--key_repeat", type=int, default=5,
                        help="Repeat each key (action) N times, e.g. 10 makes every move 10x longer")
    parser.add_argument("--bound_margin", type=float, default=0.2,
                        help="Widen action clip bounds by this fraction of the range "
                             "(0.0=strict dataset range, 0.2=allow 20%% beyond). Large values diverge.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    h = w = args.resolution
    device = args.device

    print("Computing action bounds from dataset...")
    action_min, action_max = compute_action_bounds(args.dataset_dir)

    print("Loading model...")
    model = load_model(args.ckpt, device, args.dec_infer_steps).to(device)
    dtype = model.dtype
    key_map = build_key_map(args.step_size)

    print(f"\nRunning {len(SCENARIOS)} scenarios -> {args.output_dir}\n")
    for scn in SCENARIOS:
        try:
            out = run_scenario(model, scn, args.dataset_dir, action_min, action_max,
                               key_map, h, w, device, dtype, args.output_dir,
                               key_repeat=args.key_repeat, bound_margin=args.bound_margin)
            print(f"  [ok]   {scn['name']:14s} ({len(scn['keys'])} steps) -> {out}")
        except Exception as e:
            print(f"  [fail] {scn['name']:14s}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
