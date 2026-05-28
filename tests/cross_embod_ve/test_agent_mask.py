import numpy as np
from interactive_world_sim.cross_embod_ve.data.agent_mask import (
    red_cube_mask, human_arm_mask_from_hand,
)


def test_red_cube_mask_picks_red():
    img = np.zeros((10, 10, 3), np.uint8)
    img[2:5, 2:5] = [200, 20, 20]      # 红块
    img[6:9, 6:9] = [20, 20, 200]      # 蓝盘
    m = red_cube_mask(img)
    assert m[3, 3] and not m[7, 7]


def test_human_arm_mask_dilates_hand_and_drops_cube():
    hand = np.zeros((20, 20), bool); hand[8:12, 8:12] = True   # 手部 mask
    img = np.zeros((20, 20, 3), np.uint8)
    img[2:5, 2:5] = [200, 20, 20]                              # 远处红块
    m = human_arm_mask_from_hand(img, hand, dilate=3)
    assert m[10, 10]                  # 手保留
    assert not m[3, 3]                # 红块被剔除
    assert m.dtype == bool and m.shape == (20, 20)
