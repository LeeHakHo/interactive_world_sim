import torch, numpy as np
import exp_scel_dualview_comb as DC
from amplify_wm import K, F, device

P = 48  # per-view points; dual tokens = 2P


def _dummy(B=4):
    hist = torch.randn(B, 2 * P, K, 2, device=device) * 0.05 + 0.5
    efA = torch.randn(B, K + F, 3, 2, device=device) * 0.05 + 0.5
    efB = torch.randn(B, K + F, 3, 2, device=device) * 0.05 + 0.5
    return hist, efA, efB


def _teacher():
    t = DC.DualCombLWC(P, combine="skel", Dm=384, layers=3, W=15, vel_half=0.06).to(device)
    return t.eval()


def test_forward_shapes_all_combines():
    hist, efA, efB = _dummy()
    for comb in ["dummy5", "skel", "a1"]:
        m = DC.DualCombLWC(P, combine=comb, Dm=384, layers=3, W=15, vel_half=0.06).to(device)
        logits, anchor = m.fwd_dual(hist, efA, efB)
        assert logits.shape == (4, 2 * P, F, m.W * m.W), comb
        assert anchor.shape == (4, 2 * P, 2), comb
    m = DC.DualCombLWC(P, combine="align_wm", teacher=_teacher(), Dm=384, layers=3, W=15, vel_half=0.06).to(device)
    logits, anchor = m.fwd_dual(hist, efA, efB)
    assert logits.shape == (4, 2 * P, F, m.W * m.W)


def test_a1_human_masks_world_residual():
    hist, efA, efB = _dummy()
    m = DC.DualCombLWC(P, combine="a1", Dm=384, layers=3, W=15, vel_half=0.06).to(device).eval()
    with torch.no_grad():
        m.fwd_dual(hist, efA, efB, torch.ones(4, dtype=torch.bool, device=device))
        assert m.aux["res_sq"].item() < 1e-12          # 全 human -> 残差 0
        m.fwd_dual(hist, efA, efB, torch.zeros(4, dtype=torch.bool, device=device))
        assert m.aux["res_sq"].item() > 0              # 全 robot -> 残差开
        m.fwd_dual(hist, efA, efB)                     # 缺省 = 全 robot(评价路径)
        assert m.aux["res_sq"].item() > 0


def test_align_wm_loss_and_teacher_frozen():
    hist, efA, efB = _dummy()
    m = DC.DualCombLWC(P, combine="align_wm", teacher=_teacher(), Dm=384, layers=3, W=15, vel_half=0.06).to(device).train()
    logits, _ = m.fwd_dual(hist, efA, efB)
    assert torch.isfinite(m.aux["align"]) and 0.0 <= m.aux["align"].item() <= 2.0
    (logits.mean() + m.aux["align"]).backward()
    assert all(p.grad is None for p in m.teacher.parameters())


def test_train_dcomb_smoke():
    import eval_scheduled_sampling as SSm
    SSm.WM_EPOCHS, SSm.WM_BS = 2, 16
    rng = np.random.default_rng(0)
    Nr = 20
    trD = rng.random((30, 24, 2 * P, 2)).astype(np.float32)
    vsD = np.ones((30, 24, 2 * P), np.float32)
    efA = rng.random((30, 24, 3, 2)).astype(np.float32)
    efB = rng.random((30, 24, 3, 2)).astype(np.float32)
    idx = torch.arange(30)
    t = DC.train_dcomb(trD, vsD, efA, efB, idx, "skel", Nr, seed=0)
    snap = [p.clone() for p in t.parameters()]
    for comb, tch in [("dummy5", None), ("a1", None), ("align_wm", t)]:
        m = DC.train_dcomb(trD, vsD, efA, efB, idx, comb, Nr, seed=0, teacher=tch)
        logits, _ = m.fwd_dual(torch.from_numpy(trD[:2, :K]).permute(0, 2, 1, 3).float().to(device),
                               torch.from_numpy(efA[:2]).float().to(device),
                               torch.from_numpy(efB[:2]).float().to(device))
        assert torch.isfinite(logits).all(), comb
    assert all(torch.equal(a, b) for a, b in zip(snap, t.parameters()))  # 教师没被 align_wm 训练动
