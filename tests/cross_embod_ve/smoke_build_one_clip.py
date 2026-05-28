"""Smoke: 读 1 human + 1 robot 视频首帧，验证解码与 crop。直接 `python` 运行。"""
import cv2
import numpy as np
from omegaconf import OmegaConf
from interactive_world_sim.cross_embod_ve.data.video_io import (
    read_robot_frames, read_human_frames, crop_resize,
)

cfg = OmegaConf.load("configurations/dataset/cross_embod_ve.yaml")


def main():
    rb = f"{cfg.robot.dataset_root}/play_robot_1_eef/videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
    hm = f"{cfg.human.chunks_root}/{cfg.human.chunk_ids[0]}/video_rgb_imgs.mkv"
    rf = read_robot_frames(rb)
    hf = read_human_frames(hm)
    print("robot frames", rf.shape, rf.dtype)
    print("human frames", hf.shape, hf.dtype)
    rc = crop_resize(rf[0], list(cfg.crop), cfg.res)
    hc = crop_resize(hf[0], list(cfg.crop), cfg.res)
    assert rc.shape == (cfg.res, cfg.res, 3) and hc.shape == (cfg.res, cfg.res, 3)
    assert rf.ndim == 4 and hf.ndim == 4
    print("OK: decode + crop_resize")

    # --- O_s smoke (human) ---
    from interactive_world_sim.cross_embod_ve.data.object_track import object_pixels_one_frame
    uv = object_pixels_one_frame(hf[0])
    print("object pixels (blue, red):", uv)
    vis = hf[0].copy()
    for k, color in enumerate([(0, 255, 0), (255, 255, 0)]):
        if not np.isnan(uv[k]).any():
            cv2.circle(vis, (int(uv[k, 0]), int(uv[k, 1])), 8, color, 2)
    import os; os.makedirs("outputs/ceve_qc", exist_ok=True)
    cv2.imwrite("outputs/ceve_qc/obj_centroids.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print("wrote outputs/ceve_qc/obj_centroids.png")


if __name__ == "__main__":
    main()
