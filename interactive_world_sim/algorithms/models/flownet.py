import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.dim_head = query_dim // heads
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, N, query_dim)
            context: (B, M, context_dim)
        Returns:
            out: (B, N, query_dim)
        """
        B, N, _ = x.shape
        q = self.to_q(x).reshape(B, N, self.heads, self.dim_head).transpose(1, 2)         # (B, H, N, K)
        k = self.to_k(context).reshape(B, -1, self.heads, self.dim_head).transpose(1, 2)  # (B, H, M, K)
        v = self.to_v(context).reshape(B, -1, self.heads, self.dim_head).transpose(1, 2)  # (B, H, M, K)
        out = F.scaled_dot_product_attention(q, k, v)                                      # (B, H, N, K)
        out = out.transpose(1, 2).reshape(B, N, self.heads * self.dim_head)                # (B, N, query_dim)
        return self.to_out(out)


class ResBlock(nn.Module):
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
    """UNet-based flow predictor with cross-attention action conditioning.

    Trained in Stage 1 as auxiliary predictor to improve image representations.

    Args:
        latent_ch:      number of channels in the encoder latent (e.g. 4)
        action_dim:     action vector dimension (e.g. 7)
        base_channels:  base channel width (default 32)
        action_emb_dim: dimension to project action into for cross-attention (default 128)
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

        # Action projection: (B, action_dim) → (B, 1, action_emb_dim)
        self.action_proj = nn.Linear(action_dim, action_emb_dim)

        # Encoder
        self.inc   = nn.Conv2d(latent_ch, C, 3, padding=1)                         # (B, C,   64, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResBlock(C,     C * 2))         # (B, 2C,  32, 32)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(C * 2, C * 4))         # (B, 4C,  16, 16)
        self.attn2 = CrossAttention(C * 4, action_emb_dim, heads=4)
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResBlock(C * 4, C * 8))         # (B, 8C,   8,  8)
        self.attn3 = CrossAttention(C * 8, action_emb_dim, heads=4)

        # Decoder
        self.up3  = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(C * 8, C * 4, 3, padding=1))  # (B, 4C,  16, 16)
        self.dec3 = ResBlock(C * 8, C * 4)                                          # skip from down2

        self.up2  = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(C * 4, C * 2, 3, padding=1))  # (B, 2C,  32, 32)
        self.dec2 = ResBlock(C * 4, C * 2)                                          # skip from down1

        self.up1  = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(C * 2, C, 3, padding=1))      # (B, C,   64, 64)
        self.dec1 = ResBlock(C * 2, C)                                              # skip from inc

        self.outc = nn.Conv2d(C, 2, 1)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_ch, H, W)
            action: (B, action_dim)
        Returns:
            flow:   (B, 2, H, W)
        """
        # Action context: (B, action_emb_dim) → (B, 1, action_emb_dim)
        context = self.action_proj(action).unsqueeze(1)

        # Encoder
        x1 = self.inc(latent)    # (B, C,  64, 64)
        x2 = self.down1(x1)      # (B, 2C, 32, 32)
        x3 = self.down2(x2)      # (B, 4C, 16, 16)
        B, C4, H3, W3 = x3.shape
        x3 = x3 + self.attn2(
            x3.reshape(B, C4, -1).transpose(-1, -2), context
        ).transpose(-1, -2).reshape(B, C4, H3, W3)

        x4 = self.down3(x3)      # (B, 8C,  8,  8)
        B, C8, H4, W4 = x4.shape
        x4 = x4 + self.attn3(
            x4.reshape(B, C8, -1).transpose(-1, -2), context
        ).transpose(-1, -2).reshape(B, C8, H4, W4)

        # Decoder
        x = self.dec3(torch.cat([self.up3(x4), x3], dim=1))  # (B, 4C, 16, 16)
        x = self.dec2(torch.cat([self.up2(x),  x2], dim=1))  # (B, 2C, 32, 32)
        x = self.dec1(torch.cat([self.up1(x),  x1], dim=1))  # (B, C,  64, 64)

        return torch.tanh(self.outc(x))                        # (B, 2,  H, W), range (-1, 1)
