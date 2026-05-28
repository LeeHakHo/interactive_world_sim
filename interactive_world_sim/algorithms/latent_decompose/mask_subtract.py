"""mask_subtract: agent-mask-grounded embodiment readout + residual z_scene.

z_emb = head(stop_grad(F)) ⊙ mask   (agent-region spatial code)
e     = masked_pool(z_emb) → projection → anchored embedding
û     = normalize(map(e))            (embodiment direction, flattened-latent space)
z_scene = F − proj_û(F)              (ortho_proj default; û detached)

The embodiment path reads stop_grad(F): the anchor (domain-discriminative by design)
and the agent reconstruction never reshape the shared encoder latent. The removal
direction is detached so the residual cannot corrupt the embodiment code. No CLUB,
no adversary. See docs/superpowers/specs/2026-05-28-mask-subtract-residual-decompose-design.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ortho_proj_remove(feat: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Remove the unit direction u (B, C*H*W) from feat (B, C, H, W).

    z = feat - <feat, u> u, computed in the flattened latent. Bounded rank-1
    removal (rank-k if called per direction) → cannot collapse z_scene.
    """
    B = feat.shape[0]
    flat = feat.reshape(B, -1)                       # (B, D)
    u = F.normalize(u, dim=-1, eps=1e-8)
    comp = (flat * u).sum(-1, keepdim=True)          # (B, 1)
    out = flat - comp * u
    return out.reshape_as(feat)


def gate_remove(feat: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Multiplicative suppression: feat * (1 - sigmoid(g)). g broadcasts to feat."""
    return feat * (1.0 - torch.sigmoid(g))


def _masked_pool(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of z (B,C,H,W) over mask (B,1,H,W) interior → (B,C). Safe if empty."""
    denom = mask.sum(dim=(2, 3)).clamp_min(1.0)      # (B,1)
    return (z * mask).sum(dim=(2, 3)) / denom         # (B,C)


class MaskSubtractHead(nn.Module):
    """Detached embodiment readout + residual scene latent.

    forward(F_latent, mask_lat) -> dict(z_emb_spatial, e, z_scene). The embodiment
    path reads stop_grad(F_latent); the removal direction is detached so it cannot
    corrupt e. z_scene DOES backprop into F (the encoder is shaped by it).
    """

    def __init__(self, c_scene: int, latent_hw: int, d_emb: int = 64,
                 removal: str = "ortho_proj", hidden: int = 64):
        super().__init__()
        self.c_scene = int(c_scene)
        self.latent_hw = int(latent_hw)
        self.removal = removal
        self.d_flat = self.c_scene * self.latent_hw * self.latent_hw
        self.emb_conv = nn.Sequential(
            nn.Conv2d(c_scene, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, c_scene, 3, padding=1),
        )
        self.proj = nn.Sequential(
            nn.Linear(c_scene, hidden), nn.GELU(), nn.Linear(hidden, d_emb),
        )
        self.dir_map = nn.Linear(d_emb, self.d_flat)

    def forward(self, F_latent: torch.Tensor, mask_lat: torch.Tensor) -> dict:
        Fsg = F_latent.detach()
        z_emb_spatial = self.emb_conv(Fsg) * mask_lat            # gated to agent
        e = self.proj(_masked_pool(z_emb_spatial, mask_lat))     # (B, d_emb)
        if self.removal == "gate":
            g = self.dir_map(e).reshape_as(F_latent)
            z_scene = gate_remove(F_latent, g.detach())
        else:  # ortho_proj
            u = self.dir_map(e)                                  # (B, d_flat)
            z_scene = ortho_proj_remove(F_latent, u.detach())
        return {"z_emb_spatial": z_emb_spatial, "e": e, "z_scene": z_scene}


class AgentReconHead(nn.Module):
    """z_emb_spatial (B,C,Hl,Wl) → agent RGB (B,3,out_hw,out_hw) in [0,1].

    Used only for the mask-interior agent-reconstruction loss (the z_emb grounding
    signal that pins z_emb to the agent). Not the diffusion decoder.
    """

    def __init__(self, c_scene: int, out_hw: int, hidden: int = 64):
        super().__init__()
        self.out_hw = int(out_hw)
        self.stem = nn.Conv2d(c_scene, hidden, 3, padding=1)
        self.block = nn.Sequential(
            nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.to_rgb = nn.Conv2d(hidden, 3, 3, padding=1)

    def forward(self, z_emb_spatial: torch.Tensor) -> torch.Tensor:
        x = self.stem(z_emb_spatial)
        while x.shape[-1] < self.out_hw:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = x + self.block(x)
        if x.shape[-1] != self.out_hw:
            x = F.interpolate(x, size=(self.out_hw, self.out_hw),
                              mode="bilinear", align_corners=False)
        return torch.sigmoid(self.to_rgb(x))


def supcon_anchor(e: torch.Tensor, domain_label: torch.Tensor,
                  temperature: float = 0.1) -> torch.Tensor:
    """Supervised-contrastive (InfoNCE) anchor on embeddings e (B,d) by
    domain_label (B,). Positives = same domain, negatives = other. Returns 0 if no
    anchor has a same-domain positive (e.g. single-domain batch)."""
    z = F.normalize(e, dim=-1)
    sim = z @ z.t() / temperature                    # (B,B)
    B = z.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    same = domain_label[:, None] == domain_label[None, :]
    pos = same & (~eye)
    if pos.sum() == 0:
        return e.new_zeros(())
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # torch.where (not multiply): diagonal log_prob is -inf and pos is 0 there,
    # and -inf * 0 = NaN. where() zeroes non-positive entries safely.
    pos_log = torch.where(pos, log_prob, torch.zeros_like(log_prob))
    pos_count = pos.sum(1).clamp_min(1)
    loss = -pos_log.sum(1) / pos_count
    has_pos = pos.sum(1) > 0
    return loss[has_pos].mean()
