import numpy as np
from interactive_world_sim.cross_embod_ve.data.geometry import (
    transform_points, lift_pixels_to_3d, project_points_to_pixels, load_intrinsics,
)


def test_transform_points_identity():
    pts = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    out = transform_points(np.eye(4, dtype=np.float32), pts)
    np.testing.assert_allclose(out, pts, atol=1e-6)


def test_transform_points_translation():
    T = np.eye(4, dtype=np.float32); T[:3, 3] = [10, 0, -5]
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = transform_points(T, pts)
    np.testing.assert_allclose(out, [[11.0, 2.0, -2.0]], atol=1e-6)


def test_lift_then_project_roundtrip():
    # 已知内参，点在相机前方 → 投影到像素 → 用同 depth lift 回来应一致
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    pts_cam = np.array([[0.1, -0.05, 0.8], [0.0, 0.0, 1.2]], dtype=np.float32)
    uv = project_points_to_pixels(K, pts_cam)
    depth = pts_cam[:, 2]
    back = lift_pixels_to_3d(K, uv, depth)
    np.testing.assert_allclose(back, pts_cam, atol=1e-4)


def test_lift_pixels_nan_depth_propagates():
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    uv = np.array([[100.0, 100.0]], dtype=np.float32)
    back = lift_pixels_to_3d(K, uv, np.array([np.nan], dtype=np.float32))
    assert np.isnan(back).all()


def test_robot_world_to_cam_is_rigid():
    from interactive_world_sim.algorithms.latent_decompose.ot_align import robot_world_to_cam
    T = robot_world_to_cam()  # 默认 URDF 或 fallback
    assert T.shape == (4, 4)
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-3)   # 正交
    assert abs(abs(np.linalg.det(R)) - 1.0) < 1e-3              # det≈±1
