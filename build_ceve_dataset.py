"""Build cross-embodiment video-edit dataset (Plan 1).
Usage:
  python build_ceve_dataset.py --domain human --limit 2     # 小规模试跑
  python build_ceve_dataset.py --domain robot --limit 2
  python build_ceve_dataset.py --domain all                 # 全量
"""
import argparse
import os
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from interactive_world_sim.cross_embod_ve.data.video_io import (
    read_robot_frames, read_human_frames, crop_resize)
from interactive_world_sim.cross_embod_ve.data.geometry import (
    load_intrinsics, transform_points, project_points_to_pixels)
from interactive_world_sim.cross_embod_ve.data.agent_mask import (
    human_arm_mask_from_hand, robot_gripper_mask)
from interactive_world_sim.cross_embod_ve.data.object_track import object_traj_3d
from interactive_world_sim.cross_embod_ve.data.embodiment_image import crop_around
from interactive_world_sim.cross_embod_ve.data.clip_record import (
    assemble_clip, write_clip, append_index)
from interactive_world_sim.algorithms.latent_decompose.ot_align import robot_world_to_cam

CFG = OmegaConf.load("configurations/dataset/cross_embod_ve.yaml")
K = load_intrinsics(CFG.intrinsics)
INDEX = os.path.join(CFG.out_root, "index.json")


def _clip_bounds(n):
    return [(s, s + CFG.t_clip) for s in range(0, n - CFG.t_clip + 1, CFG.stride)]


def build_human_chunk(chunk_id):
    base = f"{CFG.human.chunks_root}/{chunk_id}"
    act_path = f"{base}/smoothing_processor/smoothed_actions_right_single_arm.npz"
    masks_path = f"{base}/segmentation_processor/masks_arm.npy"
    vid_path = f"{base}/video_rgb_imgs.mkv"
    missing = [p for p in (act_path, masks_path, vid_path) if not os.path.exists(p)]
    if missing:
        print(f"[skip] human chunk {chunk_id}: missing {missing}")
        return
    frames_full = read_human_frames(vid_path)                              # (T,480,640,3)
    masks_arm = np.load(masks_path)                                        # (T,480,640) bool
    act = np.load(act_path)
    ee_pts, ee_oris, ee_w = act["ee_pts"], act["ee_oris"], act["ee_widths"]
    n = min(len(frames_full), len(masks_arm), len(ee_pts))
    depth_path = f"{base}/depth.npy"
    depth = np.load(depth_path) if os.path.exists(depth_path) else None
    for ci, (a, b) in enumerate(_clip_bounds(n)):
        fr = frames_full[a:b]
        quat = Rotation.from_matrix(ee_oris[a:b]).as_quat()                 # xyzw
        eef = np.concatenate([ee_pts[a:b], quat, ee_w[a:b, None]], axis=1).astype(np.float32)
        obj = (object_traj_3d(fr, depth[a:b], K) if depth is not None
               else np.full((b - a, CFG.k_obj, 3), np.nan, np.float32))
        mid = (b - a) // 2
        uv_mid = project_points_to_pixels(K, eef[mid:mid + 1, :3])[0]
        cimg = crop_around(fr[mid], uv_mid, CFG.emb_crop)
        frames = np.stack([crop_resize(f, list(CFG.crop), CFG.res) for f in fr])
        # arm mask 在裁好的桌面图上算（桌外暗背景天然排除，避免连通域外溢到背景）
        hand_crop = [crop_resize(masks_arm[a + t].astype(np.uint8) * 255, list(CFG.crop), CFG.res) > 127
                     for t in range(b - a)]
        amask_rs = np.stack([human_arm_mask_from_hand(frames[t], hand_crop[t]) for t in range(b - a)])
        rec = assemble_clip(frames, amask_rs, eef, obj, cimg, "human",
                            {"source_id": f"chunk_{chunk_id}", "frame_start": int(a),
                             "fps": 30, "crop": list(CFG.crop)})
        out = f"{CFG.out_root}/human/chunk_{chunk_id}/clip_{ci:04d}.npz"
        write_clip(rec, out)
        append_index(INDEX, {"path": out, "domain": "human",
                             "source_id": f"chunk_{chunk_id}", "n_frames": int(b - a)})
        print("wrote", out)


def build_robot_episode(ep):
    base = f"{CFG.robot.dataset_root}/play_robot_{ep}_eef"
    mp4 = f"{base}/videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
    pq = f"{base}/data/chunk-000/episode_000000.parquet"
    frames_full = read_robot_frames(mp4)
    df = pd.read_parquet(pq)
    pos = np.stack(df["action_right_ee_position"].to_numpy())               # (T,3) world
    quat = np.stack(df["action_right_ee_quat_xyzw"].to_numpy())             # (T,4)
    grip = np.stack(df["action_right_gripper"].to_numpy()).reshape(-1, 1)
    T_cw = robot_world_to_cam(CFG.robot.urdf)
    pos_cam = transform_points(T_cw, pos)
    n = min(len(frames_full), len(pos_cam))
    for ci, (a, b) in enumerate(_clip_bounds(n)):
        fr = frames_full[a:b]
        eef = np.concatenate([pos_cam[a:b], quat[a:b], grip[a:b]], axis=1).astype(np.float32)
        obj = np.full((b - a, CFG.k_obj, 3), np.nan, np.float32)            # robot 无 VGGT 深度 → NaN
        mid = (b - a) // 2
        uv_mid = project_points_to_pixels(K, eef[mid:mid + 1, :3])[0]
        cimg = crop_around(fr[mid], uv_mid, CFG.emb_crop)
        frames = np.stack([crop_resize(f, list(CFG.crop), CFG.res) for f in fr])
        # 夹爪 mask 在 crop 桌面图上算，用每帧 EEF 投影像素当 SAM2 种子（比最暗点可靠，
        # 避免夹爪移位时种到边缘阴影）；种子越界则 robot_gripper_mask 内部退回最暗点
        cx, cy, cw, ch = list(CFG.crop)
        uv_full = project_points_to_pixels(K, eef[:, :3])                   # (T,2) 全帧像素
        seeds = [((uv_full[t, 0] - cx) * CFG.res / cw,
                  (uv_full[t, 1] - cy) * CFG.res / ch) for t in range(b - a)]
        amask_rs = np.stack([robot_gripper_mask(frames[t], seed=seeds[t])
                             for t in range(b - a)])
        rec = assemble_clip(frames, amask_rs, eef, obj, cimg, "robot",
                            {"source_id": f"robot_{ep}", "frame_start": int(a),
                             "fps": 30, "crop": list(CFG.crop)})
        out = f"{CFG.out_root}/robot/robot_{ep}/clip_{ci:04d}.npz"
        write_clip(rec, out)
        append_index(INDEX, {"path": out, "domain": "robot",
                             "source_id": f"robot_{ep}", "n_frames": int(b - a)})
        print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["human", "robot", "all"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个 source")
    args = ap.parse_args()
    os.makedirs(CFG.out_root, exist_ok=True)
    if args.domain in ("human", "all"):
        for cid in list(CFG.human.chunk_ids)[: args.limit]:
            try:
                build_human_chunk(cid)
            except Exception as e:
                print(f"[error] human chunk {cid}: {type(e).__name__}: {e}")
    if args.domain in ("robot", "all"):
        for ep in list(CFG.robot.episodes)[: args.limit]:
            try:
                build_robot_episode(ep)
            except Exception as e:
                print(f"[error] robot ep {ep}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
