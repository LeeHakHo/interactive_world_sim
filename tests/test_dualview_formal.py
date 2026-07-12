import os, sys
sys.path.insert(0, "/scr2/yusenluo/interactive_world_sim")
os.environ.setdefault("VAE_NAME", "ostris/vae-kl-f8-d16"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import pytest


def _block_mask(n, half):
    m = torch.full((n, n), float("-inf")); m[:half, :half] = 0; m[half:, half:] = 0
    return m


def test_block_attn_mask_blocks_info():
    """mask 阻断后,前半 token 输出对后半 token 扰动不变(信息不泄漏)。"""
    from exp_scel_latent_dit import Block
    torch.manual_seed(0)
    b = Block(32, 4, ada=False).eval()
    n, half = 8, 4
    mask = _block_mask(n, half)
    x1 = torch.randn(2, n, 32); x2 = x1.clone(); x2[:, half:] = torch.randn(2, half, 32)
    with torch.no_grad():
        y1 = b(x1, attn_mask=mask); y2 = b(x2, attn_mask=mask)
    assert torch.allclose(y1[:, :half], y2[:, :half], atol=1e-6)
    with torch.no_grad():
        y3 = b(x2)                                   # 无 mask 应受影响(对照)
    assert not torch.allclose(y1[:, :half], y3[:, :half], atol=1e-4)


def _mk(mode, crossview):
    from exp_scel_dualview_dit_formal import DualViewDiTFormal
    torch.manual_seed(0)
    return DualViewDiTFormal(mode=mode, crossview=crossview, D=64, depth=2, heads=2).eval()


def _rand_inputs(mode, Cz):
    z0 = torch.randn(2, 2, Cz, 16, 16); prev = torch.randn(2, 2, Cz, 16, 16)
    cond = torch.randn(2, 2, 3, 128, 128) if mode in ("flow", "eefsp") else torch.randn(2, 2, 12)
    return z0, prev, cond


@pytest.mark.parametrize("mode", ["flow", "eefsp", "eeffilm"])
def test_formal_forward_shapes(mode):
    from exp_scel_latent_renderer import latent_ch
    Cz = latent_ch()
    m = _mk(mode, crossview=True)
    z0, prev, cond = _rand_inputs(mode, Cz)
    with torch.no_grad():
        out = m(z0, prev, cond)
    assert out.shape == (2, 2, Cz, 16, 16) and torch.isfinite(out).all()


def test_crossview_off_no_leak_flow():
    """crossview=False 时 view0 输出对 view1 的 z0/prev/cond 扰动完全不变(spec §5 判定单测)。"""
    from exp_scel_latent_renderer import latent_ch
    Cz = latent_ch()
    m = _mk("flow", crossview=False)
    z0, prev, cond = _rand_inputs("flow", Cz)
    z0b, prevb, condb = z0.clone(), prev.clone(), cond.clone()
    z0b[:, 1] = torch.randn_like(z0b[:, 1]); prevb[:, 1] = torch.randn_like(prevb[:, 1])
    condb[:, 1] = torch.randn_like(condb[:, 1])
    with torch.no_grad():
        y1 = m(z0, prev, cond); y2 = m(z0b, prevb, condb)
    assert torch.allclose(y1[:, 0], y2[:, 0], atol=1e-6)
    assert not torch.allclose(y1[:, 1], y2[:, 1], atol=1e-4)


def test_crossview_off_no_leak_eeffilm_tokens():
    """eeffilm 臂:视觉 token 不泄漏;cond 向量双视角 concat 进全局 FiLM 属 action 级设计,只扰动 z0/prev。"""
    from exp_scel_latent_renderer import latent_ch
    Cz = latent_ch()
    m = _mk("eeffilm", crossview=False)
    z0, prev, cond = _rand_inputs("eeffilm", Cz)
    z0b, prevb = z0.clone(), prev.clone()
    z0b[:, 1] = torch.randn_like(z0b[:, 1]); prevb[:, 1] = torch.randn_like(prevb[:, 1])
    with torch.no_grad():
        y1 = m(z0, prev, cond); y2 = m(z0b, prevb, cond)
    assert torch.allclose(y1[:, 0], y2[:, 0], atol=1e-6)


def test_crossview_on_does_leak():
    from exp_scel_latent_renderer import latent_ch
    Cz = latent_ch()
    m = _mk("flow", crossview=True)
    z0, prev, cond = _rand_inputs("flow", Cz)
    z0b = z0.clone(); z0b[:, 1] = torch.randn_like(z0b[:, 1])
    with torch.no_grad():
        y1 = m(z0, prev, cond); y2 = m(z0b, prev, cond)
    assert not torch.allclose(y1[:, 0], y2[:, 0], atol=1e-4)


def test_eefsp_cond_shape():
    import numpy as np
    from exp_scel_dualview_dit_formal import eefsp_cond
    ef0 = np.random.rand(3, 2).astype(np.float32); eft = ef0 + 0.05
    c = eefsp_cond(ef0, eft)
    assert c.shape == (3, 128, 128) and np.isfinite(c).all() and c[2].max() > 0
