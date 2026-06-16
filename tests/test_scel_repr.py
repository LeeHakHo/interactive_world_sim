import numpy as np
import scel_repr as S


def test_object_node_centroid_and_scale():
    pts = np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]], np.float32)
    tracks = pts[None, None]
    node = S.object_node(tracks)
    assert node.shape == (1, 1, 3)
    assert np.allclose(node[0, 0, :2], [0.5, 0.5], atol=1e-6)
    assert np.isclose(node[0, 0, 2], 0.1 * np.sqrt(2), atol=1e-6)


def test_object_node_batch_shapes():
    tracks = np.random.rand(5, 24, 48, 2).astype(np.float32)
    node = S.object_node(tracks)
    assert node.shape == (5, 24, 3)
    assert (node[..., 2] >= 0).all()


def test_grasp_frame_known_geometry():
    eef = np.array([[0.5, 0.5], [0.7, 0.6], [0.7, 0.4]], np.float32)
    eef = eef[None, None]
    gf = S.grasp_frame(eef)
    assert gf.shape == (1, 1, 5)
    assert np.allclose(gf[0, 0, :2], [0.5, 0.5], atol=1e-6)
    assert np.allclose(gf[0, 0, 2:4], [1.0, 0.0], atol=1e-6)
    assert np.isclose(gf[0, 0, 4], 0.2, atol=1e-6)


def test_grasp_frame_orientation_unit_norm():
    eef = np.random.rand(3, 24, 3, 2).astype(np.float32)
    gf = S.grasp_frame(eef)
    n = np.sqrt(gf[..., 2] ** 2 + gf[..., 3] ** 2)
    assert np.allclose(n, 1.0, atol=1e-5)


def test_agent_frame_known_value():
    gf = np.array([1.0, 1.0, 0.0, 1.0, 0.2], np.float32)[None, None]
    p = np.array([2.0, 1.0], np.float32)[None, None, None]
    rel = S.to_agent_frame(p, gf)
    assert np.allclose(rel[0, 0, 0], [0.0, -1.0], atol=1e-6)


def test_agent_frame_roundtrip_identity():
    rng = np.random.default_rng(0)
    p = rng.random((4, 24, 48, 2)).astype(np.float32)
    eef = rng.random((4, 24, 3, 2)).astype(np.float32)
    gf = S.grasp_frame(eef)
    rel = S.to_agent_frame(p, gf)
    back = S.from_agent_frame(rel, gf)
    assert np.allclose(back, p, atol=1e-5)
    assert rel.shape == p.shape


def test_vel_stats_per_domain_zscore():
    v = np.concatenate([np.linspace(0, 0.02, 50), np.linspace(0, 0.2, 50)]).astype(np.float32)[:, None]
    dom = np.concatenate([np.ones(50, int), np.zeros(50, int)])
    stats = S.fit_vel_stats(v, dom)
    out = S.normalize_vel(v, dom, stats)
    for d in (0, 1):
        m = out[dom == d]
        assert abs(m.mean()) < 1e-4
        assert abs(m.std() - 1.0) < 0.05


def test_vel_normalize_roundtrip():
    rng = np.random.default_rng(1)
    v = rng.standard_normal((100, 2)).astype(np.float32)
    dom = (rng.random(100) < 0.5).astype(int)
    stats = S.fit_vel_stats(v, dom)
    back = S.denormalize_vel(S.normalize_vel(v, dom, stats), dom, stats)
    assert np.allclose(back, v, atol=1e-5)


def test_transform_tracks_roundtrip():
    import exp_scel_agentframe as X
    rng = np.random.default_rng(2)
    tr = rng.random((3, 24, 48, 2)).astype(np.float32)
    ef = rng.random((3, 24, 3, 2)).astype(np.float32)
    rel = X.transform_tracks(tr, ef, fwd=True)
    back = X.transform_tracks(rel, ef, fwd=False)
    assert np.allclose(back, tr, atol=1e-5)
