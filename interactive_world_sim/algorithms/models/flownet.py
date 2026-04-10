import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMResBlock(nn.Module):
    """ResBlock with FiLM action conditioning at every level."""

    def __init__(self, in_channels: int, out_channels: int, action_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        # FiLM: action -> gamma, beta per channel
        self.film = nn.Linear(action_emb_dim, out_channels * 2)

    def forward(self, x: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        # apply FiLM: scale + shift per channel
        gamma_beta = self.film(action_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)           # (B, C), (B, C)
        x = gamma[:, :, None, None] * x + beta[:, :, None, None]
        x = self.act(x)
        return x + identity


class ResBlock(nn.Module):
    """Standard residual block without action conditioning."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x + identity


class FlowPredictor(nn.Module):
    """Decoder-only flow predictor using encoder skip connections.

    Takes encoder intermediate features (feat0 at full res, feat1 at half res,
    z_raw at quarter res) and predicts optical flow at full resolution.

    Args:
        latent_ch:      number of channels in encoder features
        action_dim:     action vector dimension
        base_channels:  base channel width (default 64)
        action_emb_dim: dimension of action embedding for FiLM (default 128)
    """

    def __init__(
        self,
        latent_ch: int,
        action_dim: int,
        base_channels: int = 64,
        action_emb_dim: int = 512,
    ):
        super().__init__()
        C = base_channels

        # Action embedding shared across all FiLM layers
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, action_emb_dim),
            nn.SiLU(),
        )

        # Project z_raw (quarter-res) to working channels
        self.z_proj = nn.Conv2d(latent_ch, C * 4, 1)
        self.z_res = FiLMResBlock(C * 4, C * 4, action_emb_dim)   # bottleneck FiLM at H/4

        # Level 1: quarter-res -> half-res, fuse with feat1 skip
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C * 4, C * 2, 3, padding=1),
        )
        self.feat1_proj = nn.Conv2d(latent_ch, C * 2, 1)
        self.dec1 = FiLMResBlock(C * 4, C * 2, action_emb_dim)   # cat(up, feat1) -> C*4 in

        # Level 2: half-res -> full-res, fuse with feat0 skip
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C * 2, C, 3, padding=1),
        )
        self.feat0_proj = nn.Conv2d(latent_ch, C, 1)
        self.dec2 = FiLMResBlock(C * 2, C, action_emb_dim)        # cat(up, feat0) -> C*2 in

        # Output head
        self.outc = nn.Conv2d(C, 2, 1)

    def forward(
        self,
        z_raw: torch.Tensor,    # (B, latent_ch, H/4, W/4)
        feat1: torch.Tensor,    # (B, latent_ch, H/2, W/2)
        feat0: torch.Tensor,    # (B, latent_ch, H,   W)
        action: torch.Tensor,   # (B, action_dim)
    ) -> torch.Tensor:
        """
        Returns:
            flow: (B, 2, H, W), range (-1, 1) via tanh
        """
        action_emb = self.action_proj(action)   # (B, action_emb_dim)

        x = self.z_proj(z_raw)                                          # (B, C*4, H/4, W/4)
        x = self.z_res(x, action_emb)                                   # (B, C*4, H/4, W/4)

        # Upsample to half-res and fuse feat1
        x = self.up1(x)                                                  # (B, C*2, H/2, W/2)
        f1 = self.feat1_proj(feat1)                                      # (B, C*2, H/2, W/2)
        x = self.dec1(torch.cat([x, f1], dim=1), action_emb)            # (B, C*2, H/2, W/2)

        # Upsample to full-res and fuse feat0
        x = self.up2(x)                                                  # (B, C,   H,   W)
        f0 = self.feat0_proj(feat0)                                      # (B, C,   H,   W)
        x = self.dec2(torch.cat([x, f0], dim=1), action_emb)            # (B, C,   H,   W)

        return torch.tanh(self.outc(x))                                  # (B, 2, H, W)
