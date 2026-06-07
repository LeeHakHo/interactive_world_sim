import numpy as np
import pytest

from pin_eef_to_plane import intrinsics_matrix, pin_points_to_plane


def test_pinned_points_lie_on_plane_and_reproject_to_same_pixel():
    K = intrinsics_matrix(381.092, 381.092, 310.085, 245.318)
    # Identity cam<-world so robot-world == cam frame: plane {z=z_plane} is flat in cam.
    T_cam_world = np.eye(4, dtype=np.float64)
    z_plane = 0.10
    # Three arbitrary cam-frame eef points at various depths.
    p_cam = np.array([[0.05, -0.02, 0.9],
                      [-0.10, 0.03, 1.3],
                      [0.0, 0.0, 0.7]], dtype=np.float64)
    p_world, p_cam_pin = pin_points_to_plane(p_cam, K, T_cam_world, z_plane)

    # 1) every pinned world point sits exactly on the robot plane z=z_plane
    np.testing.assert_allclose(p_world[:, 2], z_plane, atol=1e-9)

    # 2) pinned point reprojects to the SAME pixel as the original eef point
    def proj(p):
        uvw = (K @ p.T).T
        return uvw[:, :2] / uvw[:, 2:3]
    np.testing.assert_allclose(proj(p_cam), proj(p_cam_pin), atol=1e-6)


def test_nonidentity_extrinsic_keeps_points_on_world_plane():
    K = intrinsics_matrix(381.092, 381.092, 310.085, 245.318)
    T_cam_world = np.array([[0.0, -1.0, 0.0, 0.0090],
                            [-0.9063, 0.0, -0.4226, 0.1486],
                            [0.4226, 0.0, -0.9063, 1.0868],
                            [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    z_plane = 0.10
    p_cam = np.array([[0.0, 0.05, 0.95], [0.08, -0.03, 1.1]], dtype=np.float64)
    p_world, _ = pin_points_to_plane(p_cam, K, T_cam_world, z_plane)
    np.testing.assert_allclose(p_world[:, 2], z_plane, atol=1e-6)
