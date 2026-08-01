"""预测 joint -> 3 点 eef(2D 双视角)via fk_skel2d_dual。EEF_SKEL_IDX 由测试校验(vs GT eef)。"""
import numpy as np
from fk_skel2d import fk_skel2d_dual

EEF_SKEL_IDX = [5, 6, 7]   # link_6(腕), carriage_left, carriage_right (Task3 测试校验)


def fk_eef2d(joint7, grip):
    """joint7 (T,7)或(T,6), grip (T,) -> (eef_high, eef_low) 各 (T,3,2) crop-norm。"""
    j = np.asarray(joint7, np.float64)
    if j.shape[-1] == 6:
        j = np.concatenate([j, np.zeros((*j.shape[:-1], 1))], -1)
    sh, sl, _ = fk_skel2d_dual(j, np.asarray(grip, np.float64))   # (T,9,2) x2
    return sh[:, EEF_SKEL_IDX], sl[:, EEF_SKEL_IDX]
