"""Headless inference script for LIBERO latent world model.

Predefines an action sequence as a numpy array and feeds it into the world model
to generate a predicted video.

Usage (Stage 1 -- reconstruction, encode initial frame then decode):
    python scripts/inference/libero_inference.py \
        --ckpt outputs/2026-03-13/15-18-43/checkpoints/epoch=0-step=90000.ckpt \
        --stage 1 --init_episode 99

Usage (Stage 2 -- future prediction with custom actions):
    python scripts/inference/libero_inference.py \
        --ckpt outputs/.../checkpoints/epoch=0-step=200000.ckpt \
        --stage 2 --init_episode 99

Action definition (edit define_actions() at the bottom of this file):
    - shape: (T, 7) -- T steps, 7-DOF joint position delta
    - value range: approximately -1 to 1 (normalizer scale)
"""

import argparse
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)


# ============================================================
# Define your action sequence here
# ============================================================
def define_actions() -> np.ndarray:
    """Return action sequence of shape (T, 7).

    Each column is a 7-DOF joint position.
    Values are in raw action scale before normalizer (same scale as dataset).
    """
    segments = []

    # Example: increase joint 0 by 0.05 for 30 steps
    seg = np.zeros((30, 7))
    seg[:, 0] = 0.05
    segments.append(seg)

    # Example: decrease joint 1 by 0.05 for 20 steps
    seg = np.zeros((20, 7))
    seg[:, 1] = -0.05
    segments.append(seg)

    # Example: stay still for 10 steps
    segments.append(np.zeros((10, 7)))

    actions = np.concatenate(segments, axis=0).astype(np.float32)
    return actions  # (T, 7)
# ============================================================


def load_model(ckpt_path: str, device: str, dec_infer_steps: int = 3) -> LatentWorldModel:
    cfg_path = Path(ckpt_path).parent.parent / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.algorithm.load_ae = None
    cfg.algorithm.dec_infer_steps = dec_infer_steps
    algo = LatentWorldModel.load_from_checkpoint(
        ckpt_path,
        cfg=cfg.algorithm,
        map_location=device,
        strict=False,
        weights_only=False,
    )
    algo.eval()
    return algo


def decode_image(img_dict: dict, h: int, w: int) -> np.ndarray:
    img = Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB")
    img_np = np.array(img)
    if img_np.shape[:2] != (h, w):
        img_np = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_AREA)
    return img_np


def load_init_frame(dataset_dir: str, episode_idx: int, obs_key: str, h: int, w: int) -> np.ndarray:
    """Load only the first frame of an episode."""
    with open(os.path.join(dataset_dir, "meta", "info.json")) as f:
        info = json.load(f)
    chunk_idx = episode_idx // info["chunks_size"]
    parquet_path = os.path.join(
        dataset_dir,
        info["data_path"].format(episode_chunk=chunk_idx, episode_index=episode_idx),
    )
    df = pd.read_parquet(parquet_path, columns=[obs_key])
    return decode_image(df[obs_key].iloc[0], h, w)


def load_episode_sequence(
    dataset_dir: str, episode_idx: int, obs_key: str, h: int, w: int, n_frames: int = 32
):
    """Load n_frames frames and actions from an episode."""
    with open(os.path.join(dataset_dir, "meta", "info.json")) as f:
        info = json.load(f)
    chunk_idx = episode_idx // info["chunks_size"]
    parquet_path = os.path.join(
        dataset_dir,
        info["data_path"].format(episode_chunk=chunk_idx, episode_index=episode_idx),
    )
    df = pd.read_parquet(parquet_path)
    n = min(n_frames, len(df))
    frames = [decode_image(df[obs_key].iloc[i], h, w) for i in range(n)]
    actions = np.stack([np.array(df["actions"].iloc[i]) for i in range(n)], axis=0)
    return frames, actions  # list of (H,W,3) uint8, (n, action_dim)


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float [0,1] → (H, W, 3) uint8."""
    return (t.permute(1, 2, 0).cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_dir", default="/scr2/shared/world_model/libero")
    parser.add_argument("--init_episode", type=int, default=99, help="episode to source the initial frame from")
    parser.add_argument("--obs_key", default="image")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--hist_context", type=int, default=1)
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2])
    parser.add_argument("--output_dir", default="outputs/inference")
    parser.add_argument("--dec_infer_steps", type=int, default=3, help="number of decoding steps (higher = better quality, slower)")
    parser.add_argument("--flow_frames", type=int, default=32, help="number of frames to use for optical flow inference")
    parser.add_argument("--flow_dir", default=None, help="directory containing GT optical flow (skipped if not provided)")
    parser.add_argument("--use_gt_action", action="store_true", help="use GT actions instead of define_actions()")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    h = w = args.resolution

    print("Loading model...")
    model = load_model(args.ckpt, device, dec_infer_steps=args.dec_infer_steps)
    model = model.to(device)
    normalizer = model.normalizer
    dtype = model.dtype
    num_views = len(model.obs_keys)

    # Define action sequence
    actions_np = define_actions()  # (T, 7)
    T = len(actions_np)
    print(f"Action sequence: {T} steps")

    # Load GT frames
    print(f"Loading GT frames from episode {args.init_episode}...")
    gt_frames, gt_actions_np = load_episode_sequence(
        args.dataset_dir, args.init_episode, args.obs_key, h, w, n_frames=T + 1
    )
    init_frame = gt_frames[0]

    if args.use_gt_action:
        # action[t] = action that causes transition z[t-1] -> z[t], so shift by 1
        actions_np = gt_actions_np[1:T + 1].astype(np.float32)
        T = len(actions_np)
        print(f"Using GT actions: {T} steps")

    actions = torch.from_numpy(actions_np).to(device=device, dtype=dtype)
    actions_norm = normalizer["action"].normalize(actions)  # (T, 7)

    # Encode initial frame
    img0 = torch.from_numpy(init_frame.astype(np.float32) / 255.0)
    img0 = img0.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    img0_norm = normalizer[args.obs_key].normalize(img0)
    with torch.no_grad():
        z0 = model.encoder_forward(img0_norm)  # (1, C, H_lat, W_lat)

    curr_latent = z0.unsqueeze(1)  # (1, 1, C, H_lat, W_lat)

    pred_frames = [tensor_to_uint8(
        render_img_cm(model, curr_latent[:, -1], h, normalizer, num_views=num_views)[0]
    )]

    if args.stage == 1:
        # Stage 1: decode initial frame repeatedly to verify reconstruction
        print("Stage 1: decoding initial frame only...")
        for _ in tqdm(range(T)):
            with torch.no_grad():
                xs_pred = render_img_cm(model, curr_latent[:, -1], h, normalizer, num_views=num_views)
            pred_frames.append(tensor_to_uint8(xs_pred[0]))

        # Optical flow video (only when use_optical_flow is enabled)
        if model.use_optical_flow:
            print("Stage 1: predicting optical flow...")
            frames_seq, actions_seq = load_episode_sequence(
                args.dataset_dir, args.init_episode, args.obs_key, h, w,
                n_frames=args.flow_frames,
            )
            n_seq = len(frames_seq)

            # Encode frame sequence
            imgs_seq = torch.stack([
                torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1)
                for f in frames_seq
            ], dim=0).to(device=device, dtype=dtype)  # (n_seq, 3, H, W)
            imgs_seq_norm = normalizer[args.obs_key].normalize(imgs_seq)
            with torch.no_grad():
                z_seq, z_raw_seq, feat0_seq, feat1_seq = model.encoder_forward(
                    imgs_seq_norm, return_feats=True
                )

            # Even frame indices
            even_idx = list(range(0, n_seq, 2))
            T_even = len(even_idx)

            # Action averaging: average even and odd frame actions per even step
            actions_t = torch.from_numpy(actions_seq.astype(np.float32)).to(device=device, dtype=dtype)
            actions_norm_seq = normalizer["action"].normalize(actions_t)  # (n_seq, A)
            a1 = actions_norm_seq[::2][:T_even]
            a2 = actions_norm_seq[1::2]
            min_ta = min(a1.shape[0], a2.shape[0])
            action_even = torch.zeros_like(a1)
            if min_ta > 0:
                action_even[:min_ta, :-1] = (a1[:min_ta, :-1] + a2[:min_ta, :-1]) / 2.0
                action_even[:min_ta, -1:] = a1[:min_ta, -1:]
                action_even[min_ta:] = a1[min_ta:]
            else:
                action_even = a1

            z_raw_even = z_raw_seq[::2][:T_even]
            feat0_even = feat0_seq[::2][:T_even]
            feat1_even = feat1_seq[::2][:T_even]

            with torch.no_grad():
                flow_preds = model.flow_predictor(z_raw_even, feat1_even, feat0_even, action_even)

            # Load GT flow (if flow_dir is provided)
            gt_flows = None
            if args.flow_dir is not None:
                flow_path = os.path.join(args.flow_dir, "train", str(args.init_episode), "0.pt")
                if os.path.exists(flow_path):
                    episode_flow = torch.load(flow_path, map_location="cpu", weights_only=True)  # (T_flow, 2, H, W)
                    # flow file is pre-computed at stride-2: flow[k] = motion of original frame 2k -> 2k+2
                    gt_flows = episode_flow[:T_even].float()
                    # resize if needed
                    if gt_flows.shape[-1] != w or gt_flows.shape[-2] != h:
                        gt_flows = torch.nn.functional.interpolate(gt_flows, size=(h, w), mode="bilinear", align_corners=False)
                else:
                    print(f"GT flow not found at {flow_path}, skipping GT.")

            # Save flow video
            has_gt = gt_flows is not None
            video_w = w * 3 if has_gt else w * 2
            flow_out_path = os.path.join(
                args.output_dir, f"ep{args.init_episode}_stage1_flow.mp4"
            )
            out_flow = cv2.VideoWriter(
                flow_out_path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (video_w, h)
            )

            if has_gt:
                gt_norm = torch.sign(gt_flows) * torch.sqrt(torch.abs(gt_flows) / 20.0 + 1e-7)

            n_vis = min(T_even, gt_flows.shape[0]) if has_gt else T_even
            for i in range(n_vis):
                frame_rgb = frames_seq[even_idx[i]]
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # per-frame vmax
                vmax = flow_preds[i].abs().max().item()
                if has_gt:
                    vmax = max(vmax, gt_norm[i].abs().max().item())
                vmax = max(vmax, 1e-3)

                pred_rgb = LatentWorldModel._flow_to_rgb(flow_preds[i], vmax=vmax)
                pred_np = (pred_rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)

                if has_gt:
                    gt_rgb = LatentWorldModel._flow_to_rgb(gt_norm[i], vmax=vmax)
                    gt_np = (gt_rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
                    combined = np.concatenate([frame_bgr, gt_bgr, pred_bgr], axis=1)
                    cv2.putText(combined, "Frame", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.putText(combined, "GT Flow", (w + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.putText(combined, "Pred Flow", (w * 2 + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                else:
                    combined = np.concatenate([frame_bgr, pred_bgr], axis=1)
                    cv2.putText(combined, "Frame", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.putText(combined, "Pred Flow", (w + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                out_flow.write(combined)
            out_flow.release()
            print(f"Flow video saved to {flow_out_path}")
    else:
        # Stage 2: dynamics rollout with custom actions
        print("Stage 2: rolling out with custom actions...")
        for t in tqdm(range(T)):
            hist_actions = torch.zeros(
                1, curr_latent.shape[1], actions_norm.shape[-1],
                device=device, dtype=dtype,
            )
            future_action = actions_norm[t].unsqueeze(0).unsqueeze(0)  # (1, 1, 7)
            action_input = torch.cat([hist_actions, future_action], dim=1)  # (1, 2, 7)

            with torch.no_grad():
                z_pred = model.dynamics_forward(curr_latent, action_input)
                xs_pred = render_img_cm(model, z_pred[:, -1], h, normalizer, num_views=num_views)

            pred_frames.append(tensor_to_uint8(xs_pred[0]))

            curr_latent = torch.cat([curr_latent, z_pred[:, -1:]], dim=1)
            curr_latent = curr_latent[:, -args.hist_context:]

    # Save video (left: GT, right: prediction)
    out_path = os.path.join(args.output_dir, f"ep{args.init_episode}_stage{args.stage}.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w * 2, h))
    for i, pred_f in enumerate(pred_frames):
        gt_frame = gt_frames[min(i, len(gt_frames) - 1)]
        gt_bgr = cv2.cvtColor(cv2.resize(gt_frame, (w, h)), cv2.COLOR_RGB2BGR)
        pred_bgr = cv2.cvtColor(pred_f, cv2.COLOR_RGB2BGR)
        combined = np.concatenate([gt_bgr, pred_bgr], axis=1)
        cv2.putText(combined, "GT", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(combined, "Pred", (w + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        out.write(combined)
    out.release()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))
    main()
