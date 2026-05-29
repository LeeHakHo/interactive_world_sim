import cv2
import numpy as np


def red_cube_mask(img_rgb: np.ndarray) -> np.ndarray:
    """红块 mask：HSV 双区间红 + 高饱和。返回 (H,W) bool。"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((h < 12) | (h > 168)) & (s > 90) & (v > 60))
    return red


def human_arm_mask_from_hand(img_rgb: np.ndarray, hand_mask: np.ndarray,
                             dilate: int = 15, dark_thr: int = 60) -> np.ndarray:
    """手部 mask → 膨胀 + 并上与手相连的暗袖连通域 − 红块。返回 (H,W) bool。"""
    hand = hand_mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    hand_d = cv2.dilate(hand, k)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    dark = (gray < dark_thr).astype(np.uint8)
    # 只保留与手相连的暗区（连通域）
    num, labels = cv2.connectedComponents(((dark | hand_d) > 0).astype(np.uint8))
    keep = np.zeros_like(dark, bool)
    hand_labels = set(labels[hand_d > 0].tolist()) - {0}
    for lb in hand_labels:
        keep |= (labels == lb)

    arm = keep & ~red_cube_mask(img_rgb)
    return arm.astype(bool)


def build_sam2_predictor(cfg_name: str, ckpt: str, device: str = "cuda"):
    import os
    import sys
    # sam2 是 vendored submodule（未 pip 安装），按仓库惯例加入 path
    if not any(p.endswith("submodules/sam2") for p in sys.path):
        for cand in ("phantom/submodules/sam2",
                     os.path.join(os.path.dirname(__file__),
                                  "../../../phantom/submodules/sam2")):
            cand = os.path.abspath(cand)
            if os.path.isdir(cand) and cand not in sys.path:
                sys.path.insert(0, cand)
                break
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor(build_sam2(cfg_name, ckpt, device=device))


def robot_gripper_mask(img_rgb: np.ndarray, dilate: int = 11,
                       seed=None, v_thr: int = 95, radius: int = 80) -> np.ndarray:
    """robot 夹爪 mask：以 EEF 投影像素 seed 为中心，取「半径 radius 内 ∩ 暗(减蓝盘+红块)」。
    用 seed 半径而非全图连通域，是因为桌面木纹暗线/阴影也 < 阈值（纯阈值/连通域会被噪点
    和反光碎裂骗到）；夹爪一定在 EEF 投影附近，半径内的暗即夹爪+近臂。
    无 seed 时退回最大暗连通域。返回 (H,W) bool。"""
    H, W = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = (h >= 90) & (h <= 135) & (s > 60)                    # 蓝盘
    dark = (v < v_thr) & ~blue & ~red_cube_mask(img_rgb)        # 黑夹爪(含反光)，排除盘/块
    if seed is not None and 0 <= seed[0] < W and 0 <= seed[1] < H:
        yy, xx = np.ogrid[:H, :W]
        disk = (xx - seed[0]) ** 2 + (yy - seed[1]) ** 2 <= radius ** 2
        m = (dark & disk).astype(np.uint8)
    else:
        d8 = dark.astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(d8)
        if num <= 1:
            return np.zeros((H, W), bool)
        m = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    return cv2.dilate(m, k).astype(bool)
