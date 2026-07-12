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
