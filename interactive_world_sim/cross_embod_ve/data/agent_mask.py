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


def robot_gripper_mask(img_rgb: np.ndarray, dilate: int = 15,
                       seed=None, dark_thr: int = 70) -> np.ndarray:
    """robot 夹爪 mask：桌面 crop 上靠近 seed 的暗连通域（gripper 是浅色桌面上唯一的
    黑色物体）。seed=(x,y) 来自 EEF 投影用于选对的连通域；无 seed 取最大暗域。
    比点种 SAM2 稳——SAM2 会从夹爪点抓到整条连通 mount/盘子。返回 (H,W) bool。"""
    H, W = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    dark = (gray < dark_thr).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    if num <= 1:
        return np.zeros((H, W), bool)
    if seed is not None and 0 <= seed[0] < W and 0 <= seed[1] < H:
        # 选与 seed 周围小圆盘重叠最多的暗连通域（比"质心最近"鲁棒：
        # 避免 seed 旁的小噪声域质心更近而漏掉大夹爪域）
        disk = np.zeros((H, W), np.uint8)
        cv2.circle(disk, (int(seed[0]), int(seed[1])), 14, 1, -1)
        overlaps = [int(((labels == i) & (disk > 0)).sum()) for i in range(1, num)]
        idx = (1 + int(np.argmax(overlaps))) if max(overlaps) > 0 \
            else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    else:
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (labels == idx)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    return cv2.dilate(m.astype(np.uint8), k).astype(bool)
