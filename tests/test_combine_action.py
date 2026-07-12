import torch, numpy as np
import exp_scel_combine_action as C
from amplify_wm import K, F, device

def _dummy(B=8, P=48):
    hist = torch.randn(B, P, K, 2, device=device) * 0.05 + 0.5
    eef3 = torch.randn(B, K + F, 3, 2, device=device) * 0.05 + 0.5
    return hist, eef3

def test_forward_shape():
    hist, eef3 = _dummy()
    m = C.CombLWC(48, combine="a1").to(device)
    logits, anchor = m(hist, eef3)
    assert logits.shape == (8, 48, F, m.W * m.W)
    assert anchor.shape == (8, 48, 2)

def test_human_only_shared_a1():
    # human 样本 (is_h=True) 的 world 残差必须被 mask 成 0 -> 与纯 shared 前向一致
    hist, eef3 = _dummy(B=4)
    m = C.CombLWC(48, combine="a1", alpha=0.5).to(device).eval()
    is_h = torch.tensor([False, False, True, True], device=device)
    with torch.no_grad():
        m(hist, eef3, is_h)
        res = m.aux["res_sq"]  # residual^2 mean over ALL samples; human rows contribute 0
        # 直接核验:强制全 human -> 残差全 0
        m(hist, eef3, torch.ones(4, dtype=torch.bool, device=device))
        assert m.aux["res_sq"].item() < 1e-12
        # 全 robot -> 残差非 0
        m(hist, eef3, torch.zeros(4, dtype=torch.bool, device=device))
        assert m.aux["res_sq"].item() > 0

def test_default_is_robot():
    # 评价路径 model(hist,eef3) 不传 is_h -> 视作全 robot(world 残差全开)
    hist, eef3 = _dummy(B=4)
    m = C.CombLWC(48, combine="a1").to(device).eval()
    with torch.no_grad():
        m(hist, eef3)
        assert m.aux["res_sq"].item() > 0

def test_align_produces_loss():
    hist, eef3 = _dummy(B=4)
    m = C.CombLWC(48, combine="align").to(device)
    is_h = torch.tensor([False, False, True, True], device=device)
    m(hist, eef3, is_h)
    assert torch.isfinite(m.aux["align"]) and m.aux["align"].item() >= 0

def test_train_comb_smoke():
    import eval_scheduled_sampling as SSm
    SSm.WM_EPOCHS, SSm.R_SS, SSm.WM_BS = 2, 4, 16
    import exp_scel_combine_action as C
    rng = np.random.default_rng(0)
    Nr = 20
    tr = (rng.random((30, 24, 48, 2)).astype(np.float32))
    vs = np.ones((30, 24, 48), np.float32)
    ef = (rng.random((30, 24, 3, 2)).astype(np.float32))
    idx = torch.arange(30)  # 0..19 robot, 20..29 human
    for comb in ["a1", "align", "warm"]:
        m = C.train_comb(tr, vs, ef, idx, comb, Nr, seed=0)
        logits, anchor = m(torch.from_numpy(tr[:4, :K]).permute(0, 2, 1, 3).float().to(device),
                           torch.from_numpy(ef[:4]).float().to(device))
        assert torch.isfinite(logits).all()


# ---- align_wm(LaST-HD faithful:冻结混训 skel-WM 教师 + cosine latent 对齐) ----

def _teacher(P=48):
    import exp_scel_velocity_action as V
    t = V.VelLWC(P, feat="skel", Dm=384, layers=3, W=15, vel_half=V.VEL_HALF).to(device)
    return t.eval()

def test_align_wm_forward_and_loss():
    # train 模式 + teacher:aux["align"] = mean(1-cos) ∈ [0,2] 有限
    hist, eef3 = _dummy(B=4)
    m = C.CombLWC(48, combine="align_wm", teacher=_teacher()).to(device).train()
    logits, anchor = m(hist, eef3, torch.tensor([False, False, True, True], device=device))
    assert logits.shape == (4, 48, F, m.W * m.W)
    assert torch.isfinite(m.aux["align"]) and 0.0 <= m.aux["align"].item() <= 2.0

def test_align_wm_teacher_frozen():
    # 反传后教师参数不能有梯度(冻结目标,非移动靶)
    hist, eef3 = _dummy(B=4)
    m = C.CombLWC(48, combine="align_wm", teacher=_teacher()).to(device).train()
    logits, _ = m(hist, eef3)
    (logits.mean() + m.aux["align"]).backward()
    assert all(p.grad is None for p in m.teacher.parameters())
    assert any(p.grad is not None for p in m.act_world.parameters())

def test_align_wm_eval_without_teacher():
    # 评价路径(ade_world/rollout)在 eval 模式下不依赖教师前向,teacher=None 也能跑
    hist, eef3 = _dummy(B=4)
    m = C.CombLWC(48, combine="align_wm", teacher=None).to(device).eval()
    with torch.no_grad():
        logits, anchor = m(hist, eef3)
    assert torch.isfinite(logits).all() and anchor.shape == (4, 48, 2)

def test_train_comb_align_wm_smoke():
    import eval_scheduled_sampling as SSm
    SSm.WM_EPOCHS, SSm.R_SS, SSm.WM_BS = 2, 4, 16
    rng = np.random.default_rng(0)
    Nr = 20
    tr = rng.random((30, 24, 48, 2)).astype(np.float32)
    vs = np.ones((30, 24, 48), np.float32)
    ef = rng.random((30, 24, 3, 2)).astype(np.float32)
    idx = torch.arange(30)
    t = _teacher()
    snap = [p.clone() for p in t.parameters()]
    m = C.train_comb(tr, vs, ef, idx, "align_wm", Nr, seed=0, teacher=t)
    assert all(torch.equal(a, b) for a, b in zip(snap, t.parameters()))  # 教师训练后不变
    logits, _ = m(torch.from_numpy(tr[:4, :K]).permute(0, 2, 1, 3).float().to(device),
                  torch.from_numpy(ef[:4]).float().to(device))
    assert torch.isfinite(logits).all()
