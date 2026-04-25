"""Real-time inference script using live robot observations.

Connects to TrossenAIStationary robot, reads joint positions and camera frames
at 30 Hz, runs world model inference, and displays predicted frames alongside GT.

Usage:
    python scripts/inference/realtime_inference.py \
        --ckpt outputs/.../checkpoints/epoch=X-step=Y.ckpt \
        [--device cuda:0] [--dec_infer_steps 3] [--hist_context 10]

Press 'q' to quit.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x), replace=True)

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import LatentWorldModel
from interactive_world_sim.utils.aloha_conts import (
    MASTER_GRIPPER_JOINT_UNNORMALIZE_FN,
    PUPPET_GRIPPER_JOINT_NORMALIZE_FN,
)
from yixuan_utilities.draw_utils import center_crop

from lerobot.common.robot_devices.robots.configs import TrossenAIStationaryRobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config


def load_model(ckpt_path: str, device: str, dec_infer_steps: int = 3) -> LatentWorldModel:
    cfg_path = Path(ckpt_path).parent.parent / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.algorithm.load_ae = None
    cfg.algorithm.dec_infer_steps = dec_infer_steps
    model = LatentWorldModel.load_from_checkpoint(
        ckpt_path,
        cfg=cfg.algorithm,
        map_location=device,
        strict=False,
        weights_only=False,
    )
    model.eval()
    return model.to(device)


def process_joint_pos(joint_pos: np.ndarray) -> np.ndarray:
    """Apply gripper normalization matching RealAlohaDataset preprocessing."""
    out = joint_pos.copy()
    for r_i in range(2):
        g = r_i * 7 + 6
        out[g] = MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(
            PUPPET_GRIPPER_JOINT_NORMALIZE_FN(out[g])
        )
    return out


def obs_to_frame(obs_img) -> np.ndarray:
    """Convert LeRobot observation image to (H, W, 3) uint8 numpy RGB."""
    if isinstance(obs_img, torch.Tensor):
        img = obs_img.cpu().numpy()
    else:
        img = np.array(obs_img)
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    if img.shape[0] == 3:  # (C, H, W) → (H, W, C)
        img = img.transpose(1, 2, 0)
    return img


def crop_resize(frame: np.ndarray, resolution: int) -> np.ndarray:
    h = w = resolution
    return cv2.resize(center_crop(frame, (h, w)), (w, h), interpolation=cv2.INTER_AREA)


def encode_frame(model, normalizer, cam0_np, cam1_np, device, dtype):
    """Encode (H, W, 3) uint8 frames from both cameras → latent (1, C, Hl, Wl)."""
    def to_t(img):
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = torch.cat([
        normalizer["camera_0_color"].normalize(to_t(cam0_np)),
        normalizer["camera_1_color"].normalize(to_t(cam1_np)),
    ], dim=1)
    with torch.no_grad():
        return model.encoder_forward(x)


def decode_latent(model, z, resolution, normalizer, num_views):
    """Decode latent → list of (H, W, 3) uint8 numpy frames."""
    with torch.no_grad():
        xs = render_img_cm(model, z, resolution, normalizer, num_views=num_views)
    frames = []
    for i in range(num_views):
        ch = xs[0, i * 3:(i + 1) * 3]
        frames.append((ch.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8))
    return frames


def make_display_frame(gt0, gt1, pred0, pred1, w, h):
    """GT cam0 | GT cam1 | Pred cam0 | Pred cam1 → BGR display frame."""
    def to_bgr(img):
        return cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_RGB2BGR)
    row = np.concatenate([to_bgr(gt0), to_bgr(gt1), to_bgr(pred0), to_bgr(pred1)], axis=1)
    for i, lbl in enumerate(["GT cam0", "GT cam1", "Pred cam0", "Pred cam1"]):
        cv2.putText(row, lbl, (i * w + 4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--dec_infer_steps", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    h = w = args.resolution
    device = args.device

    # ── Load world model ────────────────────────────────────────────────────
    print("Loading model...")
    model = load_model(args.ckpt, device, dec_infer_steps=args.dec_infer_steps)
    normalizer = model.normalizer
    dtype = model.dtype
    num_views = len(model.obs_keys)
    n_tokens = model.n_tokens
    print(f"Model loaded. n_tokens={n_tokens}, num_views={num_views}")

    # ── Connect robot ────────────────────────────────────────────────────────
    print("Connecting to robot...")
    robot_cfg = TrossenAIStationaryRobotConfig()
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    print("Robot connected.")

    try:
        # ── Initial observation ──────────────────────────────────────────────
        print("Capturing initial frame...")
        obs = robot.capture_observation()

        cam0_np = crop_resize(obs_to_frame(obs["observation.images.cam_high"]), h)
        cam1_np = crop_resize(obs_to_frame(obs["observation.images.cam_low"]), h)
        joint_pos = obs["observation.state"].numpy() if isinstance(obs["observation.state"], torch.Tensor) else np.array(obs["observation.state"])
        joint_pos = process_joint_pos(joint_pos)

        z0 = encode_frame(model, normalizer, cam0_np, cam1_np, device, dtype)
        curr_latent = z0.unsqueeze(1)  # (1, 1, C, Hl, Wl)

        action_t = torch.from_numpy(joint_pos).to(device=device, dtype=dtype)
        action_norm = normalizer["action"].normalize(action_t.unsqueeze(0))
        past_actions = action_norm.unsqueeze(0)  # (1, 1, A)

        # Decode initial frame
        preds = decode_latent(model, z0, h, normalizer, num_views)

        print("Running. Press 'q' to quit.")
        step = 0
        while True:
            t0 = time.perf_counter()

            # ── Read robot observation ───────────────────────────────────────
            obs = robot.capture_observation()
            cam0_gt = crop_resize(obs_to_frame(obs["observation.images.cam_high"]), h)
            cam1_gt = crop_resize(obs_to_frame(obs["observation.images.cam_low"]), h)
            joint_pos = obs["observation.state"].numpy() if isinstance(obs["observation.state"], torch.Tensor) else np.array(obs["observation.state"])
            joint_pos = process_joint_pos(joint_pos)

            # ── World model inference ────────────────────────────────────────
            action_t = torch.from_numpy(joint_pos).to(device=device, dtype=dtype)
            action_norm = normalizer["action"].normalize(action_t.unsqueeze(0))
            future_action = action_norm.unsqueeze(0).unsqueeze(0)  # (1, 1, A)
            action_input = torch.cat([past_actions, future_action], dim=1)

            with torch.no_grad():
                z_pred = model.dynamics_forward(curr_latent, action_input)

            preds = decode_latent(model, z_pred[:, -1], h, normalizer, num_views)

            curr_latent = torch.cat([curr_latent, z_pred[:, -1:]], dim=1)[:, -n_tokens:]
            past_actions = torch.cat([past_actions, future_action], dim=1)[:, -n_tokens:]

            # ── Display ──────────────────────────────────────────────────────
            frame = make_display_frame(cam0_gt, cam1_gt, preds[0], preds[1], w, h)
            elapsed = time.perf_counter() - t0
            cv2.putText(frame, f"step={step} {1/elapsed:.1f}fps", (4, h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
            cv2.imshow("IWS Realtime Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            step += 1

    finally:
        robot.disconnect()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
