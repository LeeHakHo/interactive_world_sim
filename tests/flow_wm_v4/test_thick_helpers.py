import torch
import train_flow_wm_scarcity_v4 as v4


def test_constants_present():
    for name in ("K", "F", "L", "Dm", "P", "LAMBDA_GATE", "NOISE_STD",
                 "LAMBDA_CONSIST", "STATIC_TAU"):
        assert hasattr(v4, name), f"missing constant {name}"
    assert v4.K + v4.F == v4.L == 16


def test_thin_forward_shape():
    B = 5
    hist = torch.randn(B, v4.P, v4.K, 2)
    eef3 = torch.randn(B, v4.L, 3, 2)
    g = torch.rand(B, v4.L)
    m = v4.FlowWMThick(v4.P, thin=True)
    pred, alpha = m(hist, eef3, g)
    assert pred.shape == (B, v4.P, v4.F, 2)
    assert alpha is None  # thin path has no gate
