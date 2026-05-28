import cv2
import numpy as np
from .geometry import lift_pixels_to_3d


def _hsv_centroid(img_rgb, lo, hi, min_px=40):
    """颜色区间 mask 的最大连通域 centroid (u,v)；不足 min_px 返回 None。"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, np.array(lo), np.array(hi))
    num, labels, stats, cents = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
    if num <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < min_px:
        return None
    return cents[idx].astype(np.float32)   # (u,v)


# 蓝盘 / 红块的两段红色 HSV 先验（OpenCV H∈[0,180]）
_BLUE = ([95, 80, 40], [130, 255, 255])
_RED1 = ([0, 90, 60], [12, 255, 255])
_RED2 = ([168, 90, 60], [180, 255, 255])


def object_pixels_one_frame(img_rgb):
    """返回 (K_OBJ,2) 像素 centroid，未检出为 NaN。顺序: [blue_plate, red_cube]。"""
    blue = _hsv_centroid(img_rgb, *_BLUE)
    r1 = _hsv_centroid(img_rgb, *_RED1)
    r2 = _hsv_centroid(img_rgb, *_RED2)
    red = r1 if r1 is not None else r2
    out = np.full((2, 2), np.nan, np.float32)
    if blue is not None:
        out[0] = blue
    if red is not None:
        out[1] = red
    return out


def object_traj_3d(frames_rgb, depth_clip, K):
    """frames_rgb:(T,H,W,3) full-res；depth_clip:(T,H,W) 米；K:(3,3)。
    返回 (T,K_OBJ,3) 相机系 3D（米），未检出 NaN。"""
    T = frames_rgb.shape[0]
    out = np.full((T, 2, 3), np.nan, np.float32)
    for t in range(T):
        uv = object_pixels_one_frame(frames_rgb[t])   # (2,2)
        for k in range(2):
            if np.isnan(uv[k]).any():
                continue
            u, v = int(round(uv[k, 0])), int(round(uv[k, 1]))
            if 0 <= v < depth_clip.shape[1] and 0 <= u < depth_clip.shape[2]:
                d = float(depth_clip[t, v, u])
                out[t, k] = lift_pixels_to_3d(K, uv[k:k+1], np.array([d], np.float32))[0]
    return out
