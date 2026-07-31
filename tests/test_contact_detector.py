import numpy as np
from contact_detector import detect_contact_geometric


def _traj(T, obj0, ef_off, grip_val, move):
    """物体从obj0沿x移动move; 夹爪在物体+ef_off处同步移动; grip=grip_val。"""
    t = np.arange(T)[:, None]
    obj = obj0[None] + np.concatenate([move * t / T, np.zeros((T, 2))], 1)
    tracks3d = obj[:, None, :] + np.zeros((T, 6, 3))
    center = obj + ef_off
    eef = np.stack([center, center + [0, 0.014, 0], center + [0, -0.014, 0]], 1)
    grip = np.full(T, grip_val)
    return tracks3d, eef, grip


def test_grasping_gives_high_attachment():
    tr, ef, g = _traj(12, np.array([-0.05, 0., 0.17]), np.array([0., 0., 0.]), 0.005, np.array([0.06]))
    att, cp = detect_contact_geometric(tr, ef, g)
    assert att.shape == (12,) and cp.shape == (12, 3)
    assert att.mean() > 0.6, f"grasping attachment {att.mean():.2f} too low"


def test_open_hand_far_gives_low_attachment():
    tr, ef, g = _traj(12, np.array([-0.05, 0., 0.17]), np.array([0.12, 0., 0.]), 0.04, np.array([0.0]))
    att, _ = detect_contact_geometric(tr, ef, g)
    assert att.mean() < 0.3, f"non-grasp attachment {att.mean():.2f} too high"
