import json
import numpy as np


def load_intrinsics(path: str) -> np.ndarray:
    """读 cam_high 内参 JSON → 3x3 K (float32)。兼容 {fx,fy,cx,cy} 或 {K:[9]} 两种格式。"""
    with open(path) as f:
        d = json.load(f)
    if "K" in d:
        return np.asarray(d["K"], dtype=np.float32).reshape(3, 3)
    return np.array([[d["fx"], 0, d["cx"]],
                     [0, d["fy"], d["cy"]],
                     [0, 0, 1]], dtype=np.float32)


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """T: (4,4) 齐次变换；pts: (N,3) → (N,3)。"""
    pts = np.asarray(pts, dtype=np.float32)
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), np.float32)], axis=1)  # (N,4)
    out = hom @ np.asarray(T, dtype=np.float32).T
    return out[:, :3]


def project_points_to_pixels(K: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    """pts_cam: (N,3) 相机系 → (N,2) 像素 (u,v)。"""
    pts_cam = np.asarray(pts_cam, dtype=np.float32)
    z = pts_cam[:, 2:3]
    uvw = pts_cam @ np.asarray(K, dtype=np.float32).T
    return uvw[:, :2] / z


def lift_pixels_to_3d(K: np.ndarray, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """uv: (N,2) 像素；depth: (N,) 米 → (N,3) 相机系。depth=NaN 则该点全 NaN。"""
    K = np.asarray(K, dtype=np.float32)
    uv = np.asarray(uv, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (uv[:, 0] - cx) / fx * depth
    y = (uv[:, 1] - cy) / fy * depth
    return np.stack([x, y, depth], axis=1)
