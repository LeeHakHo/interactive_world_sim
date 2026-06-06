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


def test_grasp_openness_known_distance():
    eef3 = torch.zeros(2, v4.L, 3, 2)
    eef3[:, :, 1, 0] = 0.3   # tip1 x
    eef3[:, :, 2, 0] = -0.1  # tip2 x  -> distance 0.4
    g = v4.grasp_openness(eef3)
    assert g.shape == (2, v4.L)
    assert torch.allclose(g, torch.full((2, v4.L), 0.4), atol=1e-5)


def test_normalize_grasp_per_domain_range_and_scale():
    # robot grasp ~[0,0.08], human pinch ~[0,0.5]; per-domain norm must map both to [0,1]
    g_raw = torch.cat([torch.linspace(0, 0.08, 50), torch.linspace(0, 0.5, 50)])
    dom = torch.cat([torch.ones(50, dtype=torch.long), torch.zeros(50, dtype=torch.long)])
    stats = v4.fit_grasp_stats(g_raw, dom)
    out = v4.normalize_grasp(g_raw, dom, stats)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # both domains should span most of [0,1] after per-domain normalization
    assert out[dom == 1].max() > 0.9 and out[dom == 0].max() > 0.9


def test_contact_gate_features_shape_and_distance():
    B = 3
    anchor = torch.zeros(B, v4.P, 2)            # object at origin
    eef3 = torch.zeros(B, v4.L, 3, 2)
    eef3[:, :, 1, 0] = 0.5                       # future tip1 at distance 0.5
    eef3[:, :, 2, 1] = 0.5                       # future tip2 at distance 0.5
    g = torch.full((B, v4.L), 0.3)
    feat = v4.contact_gate_features(anchor, eef3, g)
    assert feat.shape == (B, v4.P, v4.F, 3)     # [d_tip1, d_tip2, grasp]
    assert torch.allclose(feat[..., 0], torch.full((B, v4.P, v4.F), 0.5), atol=1e-5)
    assert torch.allclose(feat[..., 1], torch.full((B, v4.P, v4.F), 0.5), atol=1e-5)
    assert torch.allclose(feat[..., 2], torch.full((B, v4.P, v4.F), 0.3), atol=1e-5)


def test_thick_forward_shape_and_gate_static_invariant():
    B = 4
    hist = torch.randn(B, v4.P, v4.K, 2)
    eef3 = torch.randn(B, v4.L, 3, 2)
    g = torch.rand(B, v4.L)
    m = v4.FlowWMThick(v4.P, thin=False)
    pred, alpha = m(hist, eef3, g)
    assert pred.shape == (B, v4.P, v4.F, 2)
    assert alpha.shape == (B, v4.P, v4.F)

    # Force the gate fully closed -> predicted future == anchor repeated (cube static)
    with torch.no_grad():
        for layer in m.gate:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
        m.gate[-1].bias.fill_(-50.0)
    pred0, alpha0 = m(hist, eef3, g)
    anchor = hist[:, :, -1, :]
    assert alpha0.max().item() < 1e-3
    assert torch.allclose(pred0, anchor[:, :, None, :].expand(B, v4.P, v4.F, 2), atol=1e-4)
