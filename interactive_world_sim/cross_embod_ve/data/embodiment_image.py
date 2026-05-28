import numpy as np


def crop_around(img_rgb: np.ndarray, center_uv, size: int) -> np.ndarray:
    """围绕像素 center_uv=(u,v) crop size×size，越界则 clamp 到图内。返回 (size,size,3)。"""
    H, W = img_rgb.shape[:2]
    half = size // 2
    u, v = int(round(center_uv[0])), int(round(center_uv[1]))
    x0 = int(np.clip(u - half, 0, W - size))
    y0 = int(np.clip(v - half, 0, H - size))
    return img_rgb[y0:y0 + size, x0:x0 + size].copy()
