import numpy as np


def test_build_conds_pred_uses_pred_tracks():
    """pred 版 cond 必须真用预测 tracks:与 GT 版在 trt 不同处 cond 不同、trt 相同时一致。"""
    import exp_scel_dualview_e2e as E2
    import exp_scel_dualview_dit as DIT
    rng = np.random.default_rng(0)
    P = 48
    tr0 = rng.random((P, 2)).astype(np.float32) * 0.5 + 0.25
    gt_t = tr0 + 0.05
    pred_t = tr0 + 0.10                                       # 与 GT 不同的预测
    ef0 = rng.random((3, 2)).astype(np.float32) * 0.5 + 0.25
    eft = ef0 + 0.02
    vis = np.ones(P, np.float32)
    c_gt = DIT.flow_cond(tr0, gt_t, ef0, eft, vis)
    c_pred = E2.pred_flow_cond(tr0, pred_t, ef0, eft, vis)
    c_same = E2.pred_flow_cond(tr0, gt_t, ef0, eft, vis)
    assert c_pred.shape == c_gt.shape == (3, 128, 128)
    assert np.allclose(c_same, c_gt)                          # trt 相同 -> 与 dit.flow_cond 完全一致
    assert not np.allclose(c_pred, c_gt)                      # trt 不同 -> cond 必须不同
