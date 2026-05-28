import torch

from interactive_world_sim.algorithms.latent_decompose.agent_mask import (
    agent_mask_from_rgb,
)


def _frame(h=16, w=16):
    # background = tan wood (R>G>B, mid luminance), not agent
    f = torch.zeros(3, h, w)
    f[0], f[1], f[2] = 0.55, 0.45, 0.30
    return f


def test_dark_region_is_agent_for_robot():
    f = _frame()
    f[:, 4:8, 4:8] = 0.05  # dark gripper block
    m = agent_mask_from_rgb(f, "robot")  # (1,H,W)
    assert m.shape == (1, 16, 16)
    assert m[0, 5, 5] == 1.0
    assert m[0, 0, 0] == 0.0  # wood background not masked


def test_skin_region_is_agent_for_human_only():
    f = _frame()
    f[0, 4:8, 4:8] = 0.95
    f[1, 4:8, 4:8] = 0.65
    f[2, 4:8, 4:8] = 0.65  # pink hand
    m_h = agent_mask_from_rgb(f, "human")
    m_r = agent_mask_from_rgb(f, "robot")
    assert m_h[0, 5, 5] == 1.0  # human: skin is agent
    assert m_r[0, 5, 5] == 0.0  # robot: skin NOT agent (dark-only)


def test_blue_plate_rejected():
    f = _frame()
    f[2, 10:14, 10:14] = 0.9
    f[0, 10:14, 10:14] = 0.1
    f[1, 10:14, 10:14] = 0.1  # blue
    m = agent_mask_from_rgb(f, "human")
    assert m[0, 12, 12] == 0.0


def test_batched_and_temporal_shapes():
    f = _frame()
    batched = f.unsqueeze(0).unsqueeze(0).expand(2, 4, 3, 16, 16).contiguous()
    m = agent_mask_from_rgb(batched, "human")
    assert m.shape == (2, 4, 1, 16, 16)
