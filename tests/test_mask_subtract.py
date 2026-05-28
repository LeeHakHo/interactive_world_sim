import torch

from interactive_world_sim.algorithms.latent_decompose.agent_mask import (
    agent_mask_from_rgb,
)


def _frame(h=16, w=16):
    # background = tan wood (R>G>B, mid luminance), not agent
    f = torch.zeros(3, h, w)
    f[0], f[1], f[2] = 0.55, 0.45, 0.30
    return f


def test_dark_region_is_agent_for_robot():
    f = _frame()
    f[:, 4:8, 4:8] = 0.05  # dark gripper block
    m = agent_mask_from_rgb(f, "robot")  # (1,H,W)
    assert m.shape == (1, 16, 16)
    assert m[0, 5, 5] == 1.0
    assert m[0, 0, 0] == 0.0  # wood background not masked


def test_skin_region_is_agent_for_human_only():
    f = _frame()
    f[0, 4:8, 4:8] = 0.95
    f[1, 4:8, 4:8] = 0.65
    f[2, 4:8, 4:8] = 0.65  # pink hand
    m_h = agent_mask_from_rgb(f, "human")
    m_r = agent_mask_from_rgb(f, "robot")
    assert m_h[0, 5, 5] == 1.0  # human: skin is agent
    assert m_r[0, 5, 5] == 0.0  # robot: skin NOT agent (dark-only)


def test_blue_plate_rejected():
    f = _frame()
    f[2, 10:14, 10:14] = 0.9
    f[0, 10:14, 10:14] = 0.1
    f[1, 10:14, 10:14] = 0.1  # blue
    m = agent_mask_from_rgb(f, "human")
    assert m[0, 12, 12] == 0.0


def test_batched_and_temporal_shapes():
    f = _frame()
    batched = f.unsqueeze(0).unsqueeze(0).expand(2, 4, 3, 16, 16).contiguous()
    m = agent_mask_from_rgb(batched, "human")
    assert m.shape == (2, 4, 1, 16, 16)


# --- mask_subtract modules ---
from interactive_world_sim.algorithms.latent_decompose.mask_subtract import (
    ortho_proj_remove,
    MaskSubtractHead,
    AgentReconHead,
    supcon_anchor,
)


def test_ortho_proj_removes_the_direction():
    B, C, H, W = 2, 4, 8, 8
    Fm = torch.randn(B, C, H, W)
    u = torch.randn(B, C * H * W)
    u = u / u.norm(dim=-1, keepdim=True)
    z = ortho_proj_remove(Fm, u)
    assert z.shape == Fm.shape
    comp = (z.flatten(1) * u).sum(-1)
    assert torch.allclose(comp, torch.zeros(B), atol=1e-5)
    z2 = ortho_proj_remove(z, u)             # idempotent → info preserved
    assert torch.allclose(z, z2, atol=1e-5)


def test_mask_subtract_head_shapes_and_detach():
    B, C, Hl, Wl = 3, 4, 8, 8
    head = MaskSubtractHead(c_scene=C, latent_hw=Hl, d_emb=16, removal="ortho_proj")
    Fl = torch.randn(B, C, Hl, Wl, requires_grad=True)
    mask_lat = torch.zeros(B, 1, Hl, Wl)
    mask_lat[:, :, 2:5, 2:5] = 1.0
    out = head(Fl, mask_lat)
    assert out["z_emb_spatial"].shape == (B, C, Hl, Wl)
    assert out["e"].shape == (B, 16)
    assert out["z_scene"].shape == (B, C, Hl, Wl)
    assert out["z_emb_spatial"][:, :, 0, 0].abs().max() == 0.0   # gated outside mask
    # embodiment path detached from F: grad of e w.r.t. F is None
    g = torch.autograd.grad(out["e"].sum(), Fl, retain_graph=True, allow_unused=True)[0]
    assert g is None
    # z_scene DOES carry grad to F
    gz = torch.autograd.grad(out["z_scene"].sum(), Fl, allow_unused=True)[0]
    assert gz is not None


def test_agent_recon_head_upsamples_to_image():
    head = AgentReconHead(c_scene=4, out_hw=32)
    rgb = head(torch.randn(2, 4, 8, 8))
    assert rgb.shape == (2, 3, 32, 32)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_supcon_anchor_lower_when_clustered_by_domain():
    # same points (two tight clusters at ±2); labels that match the clusters give
    # a LOW anchor loss, labels that split each cluster give a HIGH loss.
    e = torch.tensor([[2.0, 0.0], [2.1, 0.1], [-2.0, 0.0], [-2.1, -0.1]])
    dl_good = torch.tensor([0, 0, 1, 1])   # positives are the close points
    dl_bad = torch.tensor([0, 1, 0, 1])    # positives are the far points
    assert supcon_anchor(e, dl_good) < supcon_anchor(e, dl_bad)
    # no positives (every sample a distinct domain) → 0 (guard against NaN)
    assert float(supcon_anchor(torch.randn(4, 2), torch.tensor([0, 1, 2, 3]))) == 0.0
