"""Pin phantom human EEF (cam_high optical frame) onto the robot's locked
tabletop plane, expressed in the robot-world frame.

The hand's 2D image position is reliable; the monocular depth z is not. We keep
the eef pixel (ray direction) and replace depth by intersecting that camera ray
with the robot's operating plane {z_world = z_plane}. Output is in robot-world
frame, so human and robot EEF live in ONE coordinate system on the same plane.
"""
import numpy as np


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def pin_points_to_plane(p_cam, K, T_cam_world, z_plane):
    """Args:
        p_cam: (N,3) eef points in cam_high optical frame.
        K: (3,3) intrinsics.
        T_cam_world: (4,4) robot-world -> cam transform.
        z_plane: scalar robot-world plane height.
    Returns:
        p_world: (N,3) eef in robot-world frame, p_world[:,2] == z_plane.
        p_cam_pin: (N,3) the same points in cam frame (reproject == original pixel).
    """
    p_cam = np.asarray(p_cam, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T_cam_world = np.asarray(T_cam_world, dtype=np.float64)

    # Pixel of each eef point, then the camera ray direction through that pixel.
    uvw = (K @ p_cam.T).T                       # (N,3)
    uv1 = uvw / uvw[:, 2:3]                      # (N,3) homogeneous pixel [u,v,1]
    d = (np.linalg.inv(K) @ uv1.T).T            # (N,3) ray dir in cam frame

    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]
    x0_cam = R @ np.array([0.0, 0.0, z_plane]) + t   # a point on the plane, cam frame
    n_cam = R @ np.array([0.0, 0.0, 1.0])            # plane normal, cam frame

    # Ray from camera origin s*d intersects plane: n.(s d) = n.x0  ->  s = n.x0 / n.d
    s = (n_cam @ x0_cam) / (d @ n_cam)               # (N,)
    p_cam_pin = d * s[:, None]                       # (N,3)

    T_world_cam = np.linalg.inv(T_cam_world)
    Rw = T_world_cam[:3, :3]
    tw = T_world_cam[:3, 3]
    p_world = (Rw @ p_cam_pin.T).T + tw              # (N,3); z == z_plane by construction
    return p_world, p_cam_pin


def rotate_oris_to_world(ee_oris, T_cam_world):
    """Rotate (N,3,3) cam-frame orientation matrices into robot-world frame."""
    ee_oris = np.asarray(ee_oris, dtype=np.float64)
    R_world_cam = np.linalg.inv(np.asarray(T_cam_world, dtype=np.float64))[:3, :3]
    return np.einsum("ij,njk->nik", R_world_cam, ee_oris)
