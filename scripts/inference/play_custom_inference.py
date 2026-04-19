"""Headless inference script for play_custom latent world model (2-camera, 14-DOF).

Usage (Stage 1 -- reconstruction):
    python scripts/inference/play_custom_inference.py \
        --ckpt outputs/.../checkpoints/epoch=X-step=Y.ckpt \
        --stage 1 --episode 0

Usage (Stage 2/3 -- future prediction with GT or custom actions):
    python scripts/inference/play_custom_inference.py \
        --ckpt outputs/.../checkpoints/epoch=X-step=Y.ckpt \
        --stage 2 --episode 0 --use_gt_action

Action definition (edit define_actions() below for custom actions):
    - shape: (T, 14) -- bimanual joint positions (7 per arm)
    - values: same scale as dataset (raw, before normalizer)
"""

import argparse
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from yixuan_utilities.draw_utils import center_crop

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import LatentWorldModel
from interactive_world_sim.utils.aloha_conts import (
    MASTER_GRIPPER_JOINT_UNNORMALIZE_FN,
    PUPPET_GRIPPER_JOINT_NORMALIZE_FN,
)


OBS_KEYS = ["camera_0_color", "camera_1_color"]


def process_joint_pos(joint_pos: np.ndarray) -> np.ndarray:
    """Apply gripper normalization matching RealAlohaDataset preprocessing.

    joint_pos: (T, 14) -- raw joint positions from HDF5
    Gripper joints are at index 6 and 13 (one per arm).
    """
    out = joint_pos.copy()
    for r_i in range(2):
        gripper_idx = r_i * 7 + 6
        out[:, gripper_idx] = MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(
            PUPPET_GRIPPER_JOINT_NORMALIZE_FN(out[:, gripper_idx])
        )
    return out


# ============================================================
# Keyboard-to-action mapping
#
# Left arm  (indices 7-13): w=상 s=하 a=좌 d=우 f=그리퍼
# Right arm (indices 0-6):  i=상 k=하 j=좌 l=우 h=그리퍼
#
# 각 키 한 번 = step_size(rad) 만큼 해당 joint 이동
# 그리퍼: 누를 때마다 open/closed 토글
#
# 사용법: --keys "wwwsssiikk" 형태로 키 시퀀스 입력
#         각 문자 = 1 timestep
# ============================================================

# joint index mapping: (arm_offset, joint_idx_within_arm)
# 상/하 = joint 0 (base rotation), 좌/우 = joint 1 (shoulder pitch)
_KEY_MAP = {
    # left arm (offset 7)
    "w": (7, 0, +1),   # 상
    "s": (7, 0, -1),   # 하
    "a": (7, 1, +1),   # 좌
    "d": (7, 1, -1),   # 우
    # right arm (offset 0)
    "i": (0, 0, +1),   # 상
    "k": (0, 0, -1),   # 하
    "j": (0, 1, +1),   # 좌
    "l": (0, 1, -1),   # 우
}
_GRIPPER_KEYS = {"f": 7 + 6, "h": 0 + 6}  # key -> gripper joint index


def keys_to_actions(keys: str, init_joint_pos: np.ndarray, step_size: float = 0.05) -> np.ndarray:
    """Convert a key string into an action sequence (T, 14).

    Each character in keys = 1 timestep.
    Movement keys accumulate joint position changes from init_joint_pos.
    Gripper keys toggle between open (0.0) and closed (1.0).
    """
    base = init_joint_pos.copy()
    gripper_state = {6: base[6], 13: base[13]}  # track gripper state per arm
    actions = []
    for key in keys:
        if key in _KEY_MAP:
            offset, joint, sign = _KEY_MAP[key]
            base[offset + joint] += sign * step_size
        elif key in _GRIPPER_KEYS:
            g_idx = _GRIPPER_KEYS[key]
            gripper_state[g_idx] = 1.0 if gripper_state[g_idx] < 0.5 else 0.0
            base[g_idx] = gripper_state[g_idx]
        # unknown keys = hold still (1 timestep)
        actions.append(base.copy())
    return np.array(actions, dtype=np.float32)


def define_actions(init_joint_pos: np.ndarray = None) -> np.ndarray:
    """Fallback when --keys is not provided. Edit this for scripted sequences."""
    base = init_joint_pos.copy() if init_joint_pos is not None else np.zeros(14)
    # default: hold still for 30 steps
    return np.tile(base, (30, 1)).astype(np.float32)
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


def load_episode(dataset_dir: str, episode_idx: int, resolution: int, n_frames: int = None):
    """Load frames and actions from a play_custom HDF5 episode.

    Returns:
        cam0_frames: (T, H, W, 3) uint8
        cam1_frames: (T, H, W, 3) uint8
        actions:     (T, 14) float32
    """
    # try train/ subdir first, then root
    for subdir in ["val", ""]:
        path = os.path.join(dataset_dir, subdir, f"episode_{episode_idx}.hdf5")
        if os.path.exists(path):
            break
    else:
        raise FileNotFoundError(f"episode_{episode_idx}.hdf5 not found under {dataset_dir}")

    h = w = resolution
    with h5py.File(path, "r") as f:
        joint_pos = f["obs"]["joint_pos"][:]  # (T, 14)
        cam0 = f["obs"]["images"]["camera_0_color"][:]  # (T, H_orig, W_orig, 3)
        cam1 = f["obs"]["images"]["camera_1_color"][:]  # (T, H_orig, W_orig, 3)
    actions = process_joint_pos(joint_pos)

    if n_frames is not None:
        actions = actions[:n_frames]
        cam0 = cam0[:n_frames]
        cam1 = cam1[:n_frames]

    def resize_frames(frames):
        return np.stack(
            [cv2.resize(center_crop(f, (h, w)), (w, h), interpolation=cv2.INTER_AREA) for f in frames], axis=0
        )

    cam0 = resize_frames(cam0)
    cam1 = resize_frames(cam1)
    return cam0, cam1, actions.astype(np.float32)


def encode_init_frame(model, normalizer, cam0_np, cam1_np, device, dtype):
    """Encode a single frame from both cameras into a latent. Returns (1, C, H_lat, W_lat)."""
    def to_tensor(img_np):
        return torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

    img0 = to_tensor(cam0_np)
    img1 = to_tensor(cam1_np)
    img0_norm = normalizer["camera_0_color"].normalize(img0)
    img1_norm = normalizer["camera_1_color"].normalize(img1)
    x = torch.cat([img0_norm, img1_norm], dim=1)  # (1, 6, H, W)
    with torch.no_grad():
        z = model.encoder_forward(x)  # (1, C, H_lat, W_lat)
    return z


def decode_latent(model, z, resolution, normalizer, num_views):
    """Decode latent → (num_views, 3, H, W) uint8 numpy arrays."""
    with torch.no_grad():
        xs_pred = render_img_cm(model, z, resolution, normalizer, num_views=num_views)
    # xs_pred: (1, 3*num_views, H, W) in [0, 1]
    frames = []
    for i in range(num_views):
        ch = xs_pred[0, i*3:(i+1)*3]  # (3, H, W)
        frame = (ch.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        frames.append(frame)
    return frames  # list of (H, W, 3) uint8


def make_video_frame(gt_cam0, gt_cam1, pred_cam0, pred_cam1, h, w):
    """Combine GT and predicted frames for both cameras into one wide frame."""
    def to_bgr(img):
        return cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_RGB2BGR)

    row = np.concatenate([to_bgr(gt_cam0), to_bgr(gt_cam1), to_bgr(pred_cam0), to_bgr(pred_cam1)], axis=1)
    labels = ["GT cam0", "GT cam1", "Pred cam0", "Pred cam1"]
    for i, label in enumerate(labels):
        cv2.putText(row, label, (i * w + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_dir", default="/scr2/hyeonhoo/play_custom_robot")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--n_steps", type=int, default=60, help="number of rollout steps")
    parser.add_argument("--hist_context", type=int, default=1)
    parser.add_argument("--dec_infer_steps", type=int, default=3)
    parser.add_argument("--use_gt_action", action="store_true", help="use GT actions from the episode")
    parser.add_argument("--keys", type=str, default=None, help="key sequence e.g. 'wwwsssiikk' (w/s/a/d/f=left arm, i/k/j/l/h=right arm)")
    parser.add_argument("--step_size", type=float, default=0.05, help="joint delta per keypress (radians)")
    parser.add_argument("--output_dir", default="outputs/inference")
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

    print(f"Loading episode {args.episode}...")
    cam0_frames, cam1_frames, gt_actions = load_episode(
        args.dataset_dir, args.episode, h, n_frames=args.n_steps + 1
    )
    T = min(args.n_steps, len(gt_actions) - 1)

    if args.use_gt_action:
        actions_np = gt_actions[1:T + 1]
        print(f"Using GT actions: {T} steps")
    elif args.keys:
        actions_np = keys_to_actions(args.keys, gt_actions[0], step_size=args.step_size)
        T = len(actions_np)
        print(f"Using key sequence '{args.keys}': {T} steps")
    else:
        actions_np = define_actions(init_joint_pos=gt_actions[0])[:T]
        T = len(actions_np)
        print(f"Using scripted actions: {T} steps")

    actions = torch.from_numpy(actions_np).to(device=device, dtype=dtype)
    actions_norm = normalizer["action"].normalize(actions)  # (T, 14)

    # n_tokens: context window matching training (= training horizon)
    n_tokens = model.n_tokens

    # Encode initial frame
    z0 = encode_init_frame(model, normalizer, cam0_frames[0], cam1_frames[0], device, dtype)
    curr_latent = z0.unsqueeze(1)  # (1, 1, C, H_lat, W_lat)

    # Track past actions to pass as history (matches training convention)
    past_actions = actions_norm[:1].unsqueeze(0)  # (1, 1, A) — action at t=0

    pred_cam0_frames = []
    pred_cam1_frames = []

    # Decode initial frame
    init_preds = decode_latent(model, z0, h, normalizer, num_views)
    pred_cam0_frames.append(init_preds[0])
    pred_cam1_frames.append(init_preds[1])

    if args.stage == 1:
        print("Stage 1: reconstruction only...")
        for _ in tqdm(range(T)):
            preds = decode_latent(model, curr_latent[:, -1], h, normalizer, num_views)
            pred_cam0_frames.append(preds[0])
            pred_cam1_frames.append(preds[1])
    else:
        print(f"Stage {args.stage}: dynamics rollout...")
        for t in tqdm(range(T)):
            future_action = actions_norm[t].unsqueeze(0).unsqueeze(0)  # (1, 1, A)
            action_input = torch.cat([past_actions, future_action], dim=1)  # (1, T_hist+1, A)

            with torch.no_grad():
                z_pred = model.dynamics_forward(curr_latent, action_input)

            preds = decode_latent(model, z_pred[:, -1], h, normalizer, num_views)
            pred_cam0_frames.append(preds[0])
            pred_cam1_frames.append(preds[1])

            curr_latent = torch.cat([curr_latent, z_pred[:, -1:]], dim=1)
            curr_latent = curr_latent[:, -n_tokens:]

            past_actions = torch.cat([past_actions, future_action], dim=1)
            past_actions = past_actions[:, -n_tokens:]

    # Save video: GT cam0 | GT cam1 | Pred cam0 | Pred cam1
    out_path = os.path.join(args.output_dir, f"ep{args.episode}_stage{args.stage}.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w * 4, h))
    n_out = len(pred_cam0_frames)
    for i in range(n_out):
        gt0 = cam0_frames[min(i, len(cam0_frames) - 1)]
        gt1 = cam1_frames[min(i, len(cam1_frames) - 1)]
        frame = make_video_frame(gt0, gt1, pred_cam0_frames[i], pred_cam1_frames[i], h, w)
        out.write(frame)
    out.release()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))
    main()
