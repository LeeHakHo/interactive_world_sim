import av
import cv2
import numpy as np


def crop_resize(frame_rgb: np.ndarray, crop, res: int) -> np.ndarray:
    """frame_rgb: (H,W,3) uint8 → crop (x,y,w,h) → resize res×res。"""
    x, y, w, h = crop
    c = frame_rgb[y:y + h, x:x + w]
    return cv2.resize(c, (res, res), interpolation=cv2.INTER_AREA)


def read_robot_frames(mp4_path: str) -> np.ndarray:
    """AV1 robot 视频 → (T,H,W,3) uint8 RGB（全分辨率，未 crop）。"""
    container = av.open(mp4_path)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames, axis=0)


def read_human_frames(mkv_path: str) -> np.ndarray:
    """human video_rgb_imgs.mkv → (T,H,W,3) uint8 RGB（全分辨率，未 crop）。"""
    cap = cv2.VideoCapture(mkv_path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0)
