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


def robot_gripper_mask(predictor, img_rgb: np.ndarray, dilate: int = 21) -> np.ndarray:
    """robot 夹爪 mask：在图内最暗点种 SAM2，取面积合理的 mask，膨胀。返回 (H,W) bool。"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    yx = np.unravel_index(np.argmin(gray), gray.shape)
    pt = np.array([[yx[1], yx[0]]], dtype=np.float32)   # (x,y)
    predictor.set_image(img_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=pt, point_labels=np.array([1]), multimask_output=True)
    area = img_rgb.shape[0] * img_rgb.shape[1]
    cand = [(s, m) for s, m in zip(scores, masks)
            if 0.01 * area < m.sum() < 0.5 * area]
    if not cand:
        best = masks[int(np.argmax(scores))]
    else:
        best = max(cand, key=lambda sm: sm[0])[1]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    return cv2.dilate(best.astype(np.uint8), k).astype(bool)
