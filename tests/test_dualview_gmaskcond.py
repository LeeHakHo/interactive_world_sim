import numpy as np, torch


def test_cond_builder_channels():
    """GMASK=1 -> 4ch(flow3+gmask);v1 与 human 的 gmask 通道必须为 0;GMASK=0 -> 3ch 且与 DIT.flow_cond 一致。"""
    import exp_scel_dualview_gmaskcond as G
    import exp_scel_dualview_dit as DIT
    rng = np.random.default_rng(0)
    P = 48
    R = {"tr": [rng.random((2, 24, P, 2)).astype(np.float32)] * 2,
         "ef": [rng.random((2, 24, 3, 2)).astype(np.float32)] * 2,
         "vs": [np.ones((2, 24, P), np.float32)] * 2}
    joint = rng.random((2, 24, 7)).astype(np.float32)
    c4, m = G.build_conds_g(R, joint, "r", 0, 5, True)
    assert c4.shape == (2, 4, 128, 128) and m.shape == (2, 128, 128)
    assert np.abs(c4[1, 3]).max() == 0.0                       # v1 无 maskgen -> 0
    assert np.abs(c4[0, 3]).max() > 0.0                        # v0 robot 有剪影
    ch, _ = G.build_conds_g(R, joint, "h", 0, 5, True)
    assert np.abs(ch[:, 3]).max() == 0.0                       # human 无 joint -> 0
    c3, _ = G.build_conds_g(R, joint, "r", 0, 5, False)
    ref = DIT.flow_cond(R["tr"][0][0, 0], R["tr"][0][0, 5], R["ef"][0][0, 0], R["ef"][0][0, 5], R["vs"][0][0, 5])
    assert c3.shape == (2, 3, 128, 128) and np.allclose(c3[0], ref)


def test_model_forward_4ch():
    import exp_scel_dualview_gmaskcond as G
    from exp_v3_human_helps_pixels import device
    from exp_scel_latent_renderer import latent_ch
    m = G.DualViewDiTG(gmask=True).to(device)
    z0 = torch.randn(2, 2, latent_ch(), 16, 16, device=device)
    cond = torch.randn(2, 2, 4, 128, 128, device=device)
    out = m(z0, z0.clone(), cond)
    assert out.shape == (2, 2, latent_ch(), 16, 16)
    m0 = G.DualViewDiTG(gmask=False).to(device)
    out0 = m0(z0, z0.clone(), cond[:, :, :3])
    assert out0.shape == out.shape
