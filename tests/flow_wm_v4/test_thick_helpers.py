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
    dom = torch.zeros(B, dtype=torch.long)
    m = v4.FlowWMThick(v4.P, thin=True)
    pred, alpha = m(hist, eef3, g, dom)
    assert pred.shape == (B, v4.P, v4.F, 2)
    assert alpha is None  # thin path has no gate


def test_grasp_openness_known_distance():
    eef3 = torch.zeros(2, v4.L, 3, 2)
    eef3[:, :, 1, 0] = 0.3   # tip1 x
    eef3[:, :, 2, 0] = -0.1  # tip2 x  -> distance 0.4
    g = v4.grasp_openness(eef3)
    assert g.shape == (2, v4.L)
    assert torch.allclose(g, torch.full((2, v4.L), 0.4), atol=1e-5)


def test_normalize_grasp_per_domain_zscore():
    g_raw = torch.cat([torch.linspace(0, 0.08, 50), torch.linspace(0, 0.5, 50)])
    dom = torch.cat([torch.ones(50, dtype=torch.long), torch.zeros(50, dtype=torch.long)])
    stats = v4.fit_grasp_stats(g_raw, dom)
    out = v4.normalize_grasp(g_raw, dom, stats)
    for d in (0, 1):
        assert abs(out[dom == d].mean().item()) < 1e-4
        assert abs(out[dom == d].std().item() - 1.0) < 0.05


def test_contact_gate_features_shape_and_distance():
    B = 3
    anchor = torch.zeros(B, v4.P, 2)            # object at origin
    eef3 = torch.zeros(B, v4.L, 3, 2)
    eef3[:, :, 1, 0] = 0.5                       # future tip1 at distance 0.5
    eef3[:, :, 2, 1] = 0.5                       # future tip2 at distance 0.5
    g = torch.full((B, v4.L), 0.3)
    dom = torch.zeros(B, dtype=torch.long)
    dist_stats = {0: (0.0, 1.0)}
    feat = v4.contact_gate_features(anchor, eef3, g, dom, dist_stats)
    assert feat.shape == (B, v4.P, v4.F, 3)     # [d_tip1, d_tip2, grasp]
    assert torch.allclose(feat[..., 0], torch.full((B, v4.P, v4.F), 0.5), atol=1e-5)
    assert torch.allclose(feat[..., 1], torch.full((B, v4.P, v4.F), 0.5), atol=1e-5)
    assert torch.allclose(feat[..., 2], torch.full((B, v4.P, v4.F), 0.3), atol=1e-5)


def test_thick_forward_shape_and_gate_static_invariant():
    B = 4
    hist = torch.randn(B, v4.P, v4.K, 2)
    eef3 = torch.randn(B, v4.L, 3, 2)
    g = torch.rand(B, v4.L)
    dom = torch.zeros(B, dtype=torch.long)
    m = v4.FlowWMThick(v4.P, thin=False)
    m.dist_stats = {0: (0.0, 1.0)}
    pred, alpha = m(hist, eef3, g, dom)
    assert pred.shape == (B, v4.P, v4.F, 2)
    assert alpha.shape == (B, v4.P, v4.F)

    # Force the gate fully closed -> predicted future == anchor repeated (cube static)
    with torch.no_grad():
        for layer in m.gate:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
        m.gate[-1].bias.fill_(-50.0)
    pred0, alpha0 = m(hist, eef3, g, dom)
    anchor = hist[:, :, -1, :]
    assert alpha0.max().item() < 1e-3
    assert torch.allclose(pred0, anchor[:, :, None, :].expand(B, v4.P, v4.F, 2), atol=1e-4)


def test_inject_state_noise_shape_and_determinism():
    hist = torch.zeros(6, v4.P, v4.K, 2)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    a = v4.inject_state_noise(hist, v4.NOISE_STD, g1)
    b = v4.inject_state_noise(hist, v4.NOISE_STD, g2)
    assert a.shape == hist.shape
    assert torch.allclose(a, b)                       # same seed -> same noise
    assert a.abs().mean() > 0                          # noise actually added
    # std in the right ballpark (normalized coords)
    assert 0.3 * v4.NOISE_STD < a.std().item() < 3 * v4.NOISE_STD


def test_multi_step_consistency_shape():
    B = 4
    tr = torch.randn(B, v4.L, v4.P, 2)            # full clip (B,L,P,2)
    eef3 = torch.randn(B, v4.L, 3, 2)
    g = torch.rand(B, v4.L)
    dom = torch.zeros(B, dtype=torch.long)
    m = v4.FlowWMThick(v4.P, thin=False)
    m.dist_stats = {0: (0.0, 1.0)}
    cons_pred, cons_tgt = v4.multi_step_consistency(m, tr, eef3, g, dom)
    overlap = v4.L - 2 * v4.K                       # 16 - 8 = 8
    assert cons_pred.shape == (B, v4.P, overlap, 2)
    assert cons_tgt.shape == (B, v4.P, overlap, 2)


def test_rollout_drift_static():
    B, Pn = 10, v4.P
    # GT: all static (centroid never moves); pred: half static, half drifting
    gt = torch.zeros(B, v4.F, Pn, 2)
    pred = torch.zeros(B, v4.F, Pn, 2)
    pred[5:, :, :, 0] = torch.linspace(0, 0.1, v4.F)[None, :, None]  # drift in last 5
    drift, n_static = v4.rollout_drift_static(pred, gt, tau=v4.STATIC_TAU)
    assert n_static == B                              # all GT clips are static
    assert drift > 0                                  # drifting preds counted
    # a fully-static pred -> ~0 drift
    drift0, _ = v4.rollout_drift_static(torch.zeros_like(gt), gt, tau=v4.STATIC_TAU)
    assert drift0 < 1e-4


def test_train_eval_smoke_tiny(tmp_path):
    # tiny synthetic dataset -> train_eval runs and returns finite ADE/FDE/drift
    torch.manual_seed(0)
    N = 40
    tr = torch.rand(N, v4.L, v4.P, 2) * 0.5 + 0.25
    vis = torch.ones(N, v4.L, v4.P)
    eef3 = torch.rand(N, v4.L, 3, 2)
    dom = torch.ones(N, dtype=torch.long)            # robot-only smoke
    gstats = v4.fit_grasp_stats(v4.grasp_openness(eef3), dom)
    out = v4.train_eval(tr, vis, eef3, dom, gstats,
                        train_idx=torch.arange(0, 30), test_idx=torch.arange(30, 40),
                        thin=False, seed=0, epochs=2)
    for k in ("ade", "fde", "drift"):
        assert k in out and out[k] == out[k]         # finite (not NaN)


def test_cg_descriptor_shape():
    N = 7
    tr = torch.rand(N, v4.L, v4.P, 2)
    eef3 = torch.rand(N, v4.L, 3, 2)
    g = torch.rand(N, v4.L)
    dom = torch.zeros(N, dtype=torch.long)
    desc = v4.cg_descriptor(tr, eef3, g, dom)
    assert desc.shape == (N, v4.L * 2 + v4.L)        # 2 tip-distances per frame + grasp per frame


def test_chained_rollout_shapes():
    B = 4
    tr = torch.randn(B, v4.L, v4.P, 2)
    eef3 = torch.randn(B, v4.L, 3, 2)
    g = torch.rand(B, v4.L)
    dom = torch.zeros(B, dtype=torch.long)
    m = v4.FlowWMThick(v4.P, thin=False)
    m.dist_stats = {0: (0.0, 1.0)}
    p_open, p_chain, gt = v4.chained_rollout(m, tr, eef3, g, dom)
    overlap = v4.L - 2 * v4.K
    for x in (p_open, p_chain, gt):
        assert x.shape == (B, v4.P, overlap, 2)


def test_multihop_tail_ade_returns_curve():
    B = 4
    tr = torch.randn(B, v4.L, v4.P, 2)
    vis = torch.ones(B, v4.L, v4.P)
    eef3 = torch.randn(B, v4.L, 3, 2)
    g = torch.rand(B, v4.L)
    dom = torch.zeros(B, dtype=torch.long)
    m = v4.FlowWMThick(v4.P, thin=False)
    m.dist_stats = {0: (0.0, 1.0)}
    ades = v4.multihop_tail_ade(m, tr, vis, eef3, g, dom, hops=2)
    assert len(ades) == 3                       # hop 0,1,2
    assert all(a == a and a >= 0 for a in ades) # finite, non-negative
