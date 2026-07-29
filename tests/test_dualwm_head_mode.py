import pytest
import torch
import exp_scel_dualview_wm as W


def _model(head_mode):
    torch.manual_seed(0)
    return W.DualLWC(3, action="mp", head_mode=head_mode, Dm=32, layers=1, W=5, vel_half=0.12)


def test_single_readout_is_plain_head():
    m = _model("single")
    x = torch.randn(2, 6, 32)
    assert torch.equal(m._readout(x, "r"), m.head(x))
    assert torch.equal(m._readout(x, "h"), m.head(x))          # single 忽略 dom


def test_single_has_no_human_head():
    assert not hasattr(_model("single"), "head_h")


def test_two_has_human_head():
    assert hasattr(_model("two"), "head_h")


def test_two_warmstart_equal_at_init():
    m = _model("two")
    x = torch.randn(2, 6, 32)
    assert torch.allclose(m._readout(x, "r"), m._readout(x, "h"))   # head_h=deepcopy(head)


def test_two_routes_by_domain_after_divergence():
    m = _model("two")
    with torch.no_grad():
        for p in m.head_h.parameters():
            p.add_(1.0)                                         # 扰动 human 头
    x = torch.randn(2, 6, 32)
    assert torch.equal(m._readout(x, "r"), m.head(x))          # robot 头未动
    assert torch.equal(m._readout(x, "h"), m.head_h(x))        # human 头被选
    assert not torch.allclose(m._readout(x, "r"), m._readout(x, "h"))


def test_readout_shape_matches_head():
    m = _model("two")
    x = torch.randn(2, 6, 32)
    assert m._readout(x, "h").shape == m.head(x).shape


def test_film_raises_not_implemented():
    m = _model("film")
    with pytest.raises(NotImplementedError):
        m._readout(torch.randn(1, 6, 32), "r")


def test_fwd_dual_accepts_dom_and_routes():
    m = _model("two")
    B, P1 = 2, 3
    hist = torch.randn(B, 2 * P1, W.K, 2)
    eef = torch.randn(B, W.K + W.F, 4, 2)                       # n_raw=4 (mp: 3 点 + grip 槽)
    # 避免 mp_constellation 对手指轴归一化时因随机点退化(模长≈0)产生 NaN:
    # 给各点加不同常数偏移, 保证接触点/两指点明显不同、非退化。
    offsets = torch.tensor([0.0, 1.0, -1.0, 0.3]).view(1, 1, 4, 1)
    eef = eef + offsets
    lr, _ = m.fwd_dual(hist, eef, eef, dom="r")
    lh, _ = m.fwd_dual(hist, eef, eef, dom="h")
    assert lr.shape == lh.shape
    assert torch.allclose(lr, lh, atol=1e-5)                   # 暖启 init 两头相等
