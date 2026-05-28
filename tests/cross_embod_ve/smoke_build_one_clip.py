"""Smoke: 读 1 human + 1 robot 视频首帧，验证解码与 crop。直接 `python` 运行。"""
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


if __name__ == "__main__":
    main()
