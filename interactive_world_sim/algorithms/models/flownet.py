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
        # FiLM: action → γ, β per channel
        self.film = nn.Linear(action_emb_dim, out_channels * 2)

    def forward(self, x: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        # apply FiLM: scale + shift per channel
        γ, β = self.film(action_emb).chunk(2, dim=-1)          # (B, C), (B, C)
        x = γ[:, :, None, None] * x + β[:, :, None, None]
        x = self.act(x)
        return x + identity


class FlowPredictor(nn.Module):
    """UNet-based flow predictor with FiLM action conditioning at every level.

    Args:
        latent_ch:      number of input channels (e.g. 3 for RGB, or num_latent_channel)
        action_dim:     action vector dimension
        base_channels:  base channel width (default 32)
        action_emb_dim: dimension of action embedding for FiLM (default 128)
    """

    def __init__(
        self,
        latent_ch: int,
        action_dim: int,
        base_channels: int = 32,
        action_emb_dim: int = 128,
    ):
        super().__init__()
        C = base_channels

        # Action embedding shared across all FiLM layers
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, action_emb_dim),
            nn.SiLU(),
        )

        # Encoder
        self.inc  = nn.Conv2d(latent_ch, C, 3, padding=1)
        self.res0 = FiLMResBlock(C,     C,     action_emb_dim)   # (B, C,   H,   W)
        self.pool1 = nn.MaxPool2d(2)
        self.res1 = FiLMResBlock(C,     C * 2, action_emb_dim)   # (B, 2C,  H/2, W/2)
        self.pool2 = nn.MaxPool2d(2)
        self.res2 = FiLMResBlock(C * 2, C * 4, action_emb_dim)   # (B, 4C,  H/4, W/4)
        self.pool3 = nn.MaxPool2d(2)
        self.res3 = FiLMResBlock(C * 4, C * 8, action_emb_dim)   # (B, 8C,  H/8, W/8)

        # Decoder (bilinear upsample + conv to avoid checkerboard)
        self.up3  = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(C * 8, C * 4, 3, padding=1))
        self.dec3 = FiLMResBlock(C * 8, C * 4, action_emb_dim)   # skip from res2

        self.up2  = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(C * 4, C * 2, 3, padding=1))
        self.dec2 = FiLMResBlock(C * 4, C * 2, action_emb_dim)   # skip from res1

        self.up1  = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(C * 2, C,     3, padding=1))
        self.dec1 = FiLMResBlock(C * 2, C,     action_emb_dim)   # skip from res0

        self.outc = nn.Conv2d(C, 2, 1)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_ch, H, W)
            action: (B, action_dim)
        Returns:
            flow:   (B, 2, H, W), range (-1, 1)
        """
        action_emb = self.action_proj(action)   # (B, action_emb_dim)

        # Encoder
        x0 = self.res0(self.inc(latent), action_emb)    # (B, C,   H,   W)
        x1 = self.res1(self.pool1(x0),   action_emb)    # (B, 2C,  H/2, W/2)
        x2 = self.res2(self.pool2(x1),   action_emb)    # (B, 4C,  H/4, W/4)
        x3 = self.res3(self.pool3(x2),   action_emb)    # (B, 8C,  H/8, W/8)

        # Decoder
        x = self.dec3(torch.cat([self.up3(x3), x2], dim=1), action_emb)  # (B, 4C, H/4, W/4)
        x = self.dec2(torch.cat([self.up2(x),  x1], dim=1), action_emb)  # (B, 2C, H/2, W/2)
        x = self.dec1(torch.cat([self.up1(x),  x0], dim=1), action_emb)  # (B, C,  H,   W)

        return torch.tanh(self.outc(x))                 # (B, 2, H, W)
