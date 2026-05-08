"""
Visualize end-effector (EEF) pose as 3D coordinate axes overlaid on video.

Projects the EEF frame (position + quaternion) from world coordinates into
the camera image using obs_head_camera_position/quat as extrinsics.
Draws X(red) / Y(green) / Z(blue) axes and a position trail.
"""

import argparse
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


# Estimated intrinsics for 640x480 RGB camera (~60 deg FOV)
FX = FY = 600.0
CX, CY = 320.0, 240.0

AXIS_LEN  = 0.05   # meters — length of each drawn axis
TRAIL_LEN = 60     # frames
TRAIL_RADIUS = 3
DOT_RADIUS   = 6

# BGR colors
COLOR_X     = (0,   50, 255)   # red
COLOR_Y     = (0,  200,   0)   # green
COLOR_Z     = (255,  80,   0)  # blue
COLOR_DOT   = (255, 255, 255)
COLOR_TRAIL = (200, 200,   0)  # cyan-ish trail


def world_to_image(
    p_world: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot: Rotation,
    fx: float, fy: float, cx: float, cy: float,
) -> tuple[float, float, float]:
    """Project a 3-D world point to (u, v, z_cam)."""
    p_cam = cam_rot.inv().apply(p_world - cam_pos)
    z = p_cam[2]
    if z <= 1e-4:
        return None
    return fx * p_cam[0] / z + cx, fy * p_cam[1] / z + cy, z


def draw_axes(
    img: np.ndarray,
    eef_pos: np.ndarray,
    eef_rot: Rotation,
    origin_uv: tuple,
    cam_pos: np.ndarray,
    cam_rot: Rotation,
    fx: float, fy: float, cx: float, cy: float,
    axis_len: float = AXIS_LEN,
) -> np.ndarray:
    """Draw projected X/Y/Z axes of the EEF frame."""
    axes = {
        COLOR_X: eef_rot.apply([axis_len, 0, 0]),
        COLOR_Y: eef_rot.apply([0, axis_len, 0]),
        COLOR_Z: eef_rot.apply([0, 0, axis_len]),
    }
    out = img
    pt1 = (int(origin_uv[0]), int(origin_uv[1]))
    for color, offset in axes.items():
        tip = world_to_image(eef_pos + offset, cam_pos, cam_rot, fx, fy, cx, cy)
        if tip is None:
            continue
        pt2 = (int(tip[0]), int(tip[1]))
        cv2.line(out, pt1, pt2, color, 3, cv2.LINE_AA)
        cv2.circle(out, pt2, 4, color, -1, cv2.LINE_AA)
    return out


def draw_frame(
    bgr: np.ndarray,
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot: Rotation,
    uv_trail: list,
    frame_idx: int,
    fx: float, fy: float, cx: float, cy: float,
    axis_len: float = AXIS_LEN,
) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]

    # Project EEF origin
    proj = world_to_image(eef_pos, cam_pos, cam_rot, fx, fy, cx, cy)
    if proj is None:
        return out
    u, v, z = proj

    # Trail
    for i, (tu, tv) in enumerate(uv_trail):
        if 0 <= tu < w and 0 <= tv < h:
            alpha = (i + 1) / max(len(uv_trail), 1)
            c = int(200 * alpha)
            cv2.circle(out, (int(tu), int(tv)), TRAIL_RADIUS, (c, c, 0), -1, cv2.LINE_AA)

    # 3D axes
    eef_rot = Rotation.from_quat(eef_quat)
    draw_axes(out, eef_pos, eef_rot, (u, v), cam_pos, cam_rot, fx, fy, cx, cy, axis_len)

    # Origin dot
    if 0 <= u < w and 0 <= v < h:
        cv2.circle(out, (int(u), int(v)), DOT_RADIUS, COLOR_DOT, -1, cv2.LINE_AA)
        cv2.circle(out, (int(u), int(v)), DOT_RADIUS + 2, (0, 0, 0), 1, cv2.LINE_AA)

    # Text
    x, y, z_pos = eef_pos
    for i, t in enumerate([
        f"frame {frame_idx}",
        f"x {x:+.3f}  y {y:+.3f}  z {z_pos:+.3f}",
    ]):
        cv2.putText(out, t, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, t, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def visualize(
    dataset_dir: Path,
    camera: str,
    episode: int,
    out_path: Path,
    fx: float, fy: float, cx: float, cy: float,
    axis_len: float = AXIS_LEN,
):
    parquet_path = dataset_dir / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    video_path   = dataset_dir / "videos" / "chunk-000" \
                   / f"observation.images.{camera}" / f"episode_{episode:06d}.mp4"

    print(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    eef_positions = np.stack(df["obs_right_ee_position"].values)    # (N, 3)
    eef_quats     = np.stack(df["obs_right_ee_quat_xyzw"].values)   # (N, 4)
    cam_positions = np.stack(df["obs_head_camera_position"].values)  # (N, 3)
    cam_quats     = np.stack(df["obs_head_camera_quat_xyzw"].values) # (N, 4)

    print(f"Opening video: {video_path}")
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    fps    = float(stream.average_rate) or 30.0
    width, height = stream.width, stream.height
    n_frames = stream.frames
    print(f"Video: {width}x{height} @ {fps:.1f} fps, {n_frames} frames")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_container = av.open(str(out_path), mode="w")
    out_stream = out_container.add_stream("h264", rate=int(fps))
    out_stream.width, out_stream.height = width, height
    out_stream.pix_fmt = "yuv420p"

    uv_history: list[tuple[float, float]] = []
    frame_idx = 0

    for packet in container.demux(stream):
        for av_frame in packet.decode():
            bgr = av_frame.to_ndarray(format="bgr24")
            i   = min(frame_idx, len(df) - 1)

            cam_rot = Rotation.from_quat(cam_quats[i])
            proj    = world_to_image(eef_positions[i], cam_positions[i], cam_rot,
                                     fx, fy, cx, cy)
            if proj:
                uv_history.append((proj[0], proj[1]))

            trail = uv_history[max(0, len(uv_history) - TRAIL_LEN): -1]

            out_bgr = draw_frame(bgr, eef_positions[i], eef_quats[i],
                                 cam_positions[i], cam_rot, trail, frame_idx,
                                 fx, fy, cx, cy, axis_len)

            rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            out_av = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for pkt in out_stream.encode(out_av):
                out_container.mux(pkt)

            frame_idx += 1
            if frame_idx % 300 == 0:
                print(f"  {frame_idx}/{n_frames} frames")

    for pkt in out_stream.encode():
        out_container.mux(pkt)
    out_container.close()
    container.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/play_human_single_0_eef")
    parser.add_argument("--camera", default="cam_high",
                        choices=["cam_high", "cam_low", "cam_right_wrist"])
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fx", type=float, default=FX)
    parser.add_argument("--fy", type=float, default=FY)
    parser.add_argument("--cx", type=float, default=CX)
    parser.add_argument("--cy", type=float, default=CY)
    parser.add_argument("--axis-len", type=float, default=AXIS_LEN,
                        help="Axis length in meters (default: 0.05)")
    args = parser.parse_args()

    axis_len = args.axis_len
    out_path = Path(args.out) if args.out else \
        Path(f"outputs/eef_{args.camera}_ep{args.episode:03d}.mp4")

    visualize(Path(args.dataset_dir), args.camera, args.episode, out_path,
              args.fx, args.fy, args.cx, args.cy, axis_len)


if __name__ == "__main__":
    main()
