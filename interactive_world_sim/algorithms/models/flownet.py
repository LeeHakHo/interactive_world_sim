import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """표준 Residual Block"""
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


class SpatialSelfAttention(nn.Module):
    """이미지 피처맵 내의 전역적 문맥을 파악하는 Self-Attention"""
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_proj = nn.Conv2d(channels, channels * 3, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        identity = x
        x = self.norm(x)
        
        # Q, K, V 추출 및 형태 변환: (B, 3, Heads, H*W, Head_dim)
        qkv = self.qkv_proj(x).view(B, 3, self.num_heads, C // self.num_heads, H * W).transpose(-1, -2)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # 각각 (B, Heads, N, Head_dim)
        
        # PyTorch 2.0 이상: 빠르고 메모리 효율적인 Attention 연산
        attn_out = F.scaled_dot_product_attention(q, k, v)
        
        # 원래 공간 형태로 복원
        attn_out = attn_out.transpose(-1, -2).reshape(B, C, H, W)
        return identity + self.out_proj(attn_out)


class ActionCrossAttention(nn.Module):
    """이미지 피처(Q)와 Action 임베딩(K, V) 간의 Cross-Attention"""
    def __init__(self, img_channels: int, action_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Conv2d(img_channels, img_channels, 1)
        self.kv_proj = nn.Linear(action_dim, img_channels * 2)
        self.out_proj = nn.Conv2d(img_channels, img_channels, 1)
        self.norm = nn.GroupNorm(8, img_channels)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        identity = x
        x = self.norm(x)

        # 액션이 (B, action_dim) 1D 형태라면 시퀀스 차원 추가 -> (B, 1, action_dim)
        if action.ndim == 2:
            action = action.unsqueeze(1)
            
        L = action.shape[1] # Action Sequence Length

        # Query: 이미지 피처맵
        q = self.q_proj(x).view(B, self.num_heads, C // self.num_heads, H * W).transpose(-1, -2)
        
        # Key, Value: 액션 임베딩
        kv = self.kv_proj(action) # (B, L, C * 2)
        kv = kv.view(B, L, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (2, B, Heads, L, Head_dim)
        k, v = kv[0], kv[1]

        # Cross-Attention 연산
        attn_out = F.scaled_dot_product_attention(q, k, v)
        
        # 원래 공간 형태로 복원
        attn_out = attn_out.transpose(-1, -2).reshape(B, C, H, W)
        return identity + self.out_proj(attn_out)


class FlowPredictor(nn.Module):
    """Self-Attention과 Action Cross-Attention이 결합된 대형 Flow Predictor"""
    def __init__(
        self,
        latent_ch: int,
        action_dim: int,
        base_channels: int = 128,   # 채널 수 2배 증가 (기존 64)
        num_res_blocks: int = 2     # 레벨 당 ResBlock 개수 증가
    ):
        super().__init__()
        C = base_channels

        # 1. Bottleneck (H/4 해상도)
        self.z_proj = nn.Conv2d(latent_ch, C * 4, 1)
        
        # Bottleneck 연산: ResBlock -> Self-Attention -> Cross-Attention
        self.bottleneck_res = nn.Sequential(*[ResBlock(C * 4, C * 4) for _ in range(num_res_blocks)])
        self.self_attn = SpatialSelfAttention(C * 4)
        self.cross_attn1 = ActionCrossAttention(C * 4, action_dim)

        # 2. Level 1 (H/4 -> H/2 해상도)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C * 4, C * 2, 3, padding=1),
        )
        self.feat1_proj = nn.Conv2d(latent_ch, C * 2, 1)
        
        # Skip Connection 결합 후 연산
        self.dec1_res = nn.Sequential(*[ResBlock(C * 4, C * 2) if i==0 else ResBlock(C * 2, C * 2) for i in range(num_res_blocks)])
        self.cross_attn2 = ActionCrossAttention(C * 2, action_dim)

        # 3. Level 2 (H/2 -> Full 해상도)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C * 2, C, 3, padding=1),
        )
        self.feat0_proj = nn.Conv2d(latent_ch, C, 1)
        
        self.dec2_res = nn.Sequential(*[ResBlock(C * 2, C) if i==0 else ResBlock(C, C) for i in range(num_res_blocks)])
        self.cross_attn3 = ActionCrossAttention(C, action_dim)

        # 4. Output Head (3x3 Conv를 2번 거쳐서 디테일 보존)
        self.out_head = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(C // 2, 2, 3, padding=1)
        )

    def forward(
        self,
        z_raw: torch.Tensor,    # (B, latent_ch, H/4, W/4)
        feat1: torch.Tensor,    # (B, latent_ch, H/2, W/2)
        feat0: torch.Tensor,    # (B, latent_ch, H, W)
        action: torch.Tensor,   # (B, action_dim) 또는 (B, Seq_len, action_dim)
    ) -> torch.Tensor:
        
        # --- Bottleneck ---
        x = self.z_proj(z_raw)
        x = self.bottleneck_res(x)
        x = self.self_attn(x)                # 전역적 문맥 파악
        x = self.cross_attn1(x, action)      # 액션 정보 주입

        # --- Level 1 (H/2) ---
        x = self.up1(x)
        f1 = self.feat1_proj(feat1)
        x = torch.cat([x, f1], dim=1)        # Skip connection
        x = self.dec1_res(x)
        x = self.cross_attn2(x, action)      # 해상도를 키운 후 다시 액션 주입

        # --- Level 2 (Full Res) ---
        x = self.up2(x)
        f0 = self.feat0_proj(feat0)
        x = torch.cat([x, f0], dim=1)        # Skip connection
        x = self.dec2_res(x)
        x = self.cross_attn3(x, action)

        # --- Output ---
        # tanh를 제거하고 Linear하게 Flow 예측 (스케일링은 Loss 단에서 처리)
        flow = self.out_head(x)
        return flow


# class FiLMResBlock(nn.Module):
#     """ResBlock with FiLM action conditioning at every level."""

#     def __init__(self, in_channels: int, out_channels: int, action_emb_dim: int):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
#         self.norm1 = nn.GroupNorm(8, out_channels)
#         self.norm2 = nn.GroupNorm(8, out_channels)
#         self.act = nn.SiLU()
#         self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
#         # FiLM: action -> gamma, beta per channel
#         self.film = nn.Linear(action_emb_dim, out_channels * 2)

#     def forward(self, x: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
#         identity = self.shortcut(x)
#         x = self.act(self.norm1(self.conv1(x)))
#         x = self.norm2(self.conv2(x))
#         # apply FiLM: scale + shift per channel
#         gamma_beta = self.film(action_emb)
#         gamma, beta = gamma_beta.chunk(2, dim=-1)           # (B, C), (B, C)
#         x = gamma[:, :, None, None] * x + beta[:, :, None, None]
#         x = self.act(x)
#         return x + identity


# class ResBlock(nn.Module):
#     """Standard residual block without action conditioning."""

#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
#         self.norm1 = nn.GroupNorm(8, out_channels)
#         self.norm2 = nn.GroupNorm(8, out_channels)
#         self.act = nn.SiLU()
#         self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         identity = self.shortcut(x)
#         x = self.act(self.norm1(self.conv1(x)))
#         x = self.act(self.norm2(self.conv2(x)))
#         return x + identity


# class FlowPredictor(nn.Module):
#     """Decoder-only flow predictor using encoder skip connections.

#     Takes encoder intermediate features (feat0 at full res, feat1 at half res,
#     z_raw at quarter res) and predicts optical flow at full resolution.

#     Args:
#         latent_ch:      number of channels in encoder features
#         action_dim:     action vector dimension
#         base_channels:  base channel width (default 64)
#         action_emb_dim: dimension of action embedding for FiLM (default 128)
#     """

#     def __init__(
#         self,
#         latent_ch: int,
#         action_dim: int,
#         base_channels: int = 64,
#         action_emb_dim: int = 512,
#     ):
#         super().__init__()
#         C = base_channels

#         # Action embedding shared across all FiLM layers
#         self.action_proj = nn.Sequential(
#             nn.Linear(action_dim, action_emb_dim),
#             nn.SiLU(),
#         )

#         # Project z_raw (quarter-res) to working channels
#         self.z_proj = nn.Conv2d(latent_ch, C * 4, 1)
#         self.z_res = FiLMResBlock(C * 4, C * 4, action_emb_dim)   # bottleneck FiLM at H/4

#         # Level 1: quarter-res -> half-res, fuse with feat1 skip
#         self.up1 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(C * 4, C * 2, 3, padding=1),
#         )
#         self.feat1_proj = nn.Conv2d(latent_ch, C * 2, 1)
#         self.dec1 = FiLMResBlock(C * 4, C * 2, action_emb_dim)   # cat(up, feat1) -> C*4 in

#         # Level 2: half-res -> full-res, fuse with feat0 skip
#         self.up2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(C * 2, C, 3, padding=1),
#         )
#         self.feat0_proj = nn.Conv2d(latent_ch, C, 1)
#         self.dec2 = FiLMResBlock(C * 2, C, action_emb_dim)        # cat(up, feat0) -> C*2 in

#         # Output head
#         self.outc = nn.Conv2d(C, 2, 1)

#     def forward(
#         self,
#         z_raw: torch.Tensor,    # (B, latent_ch, H/4, W/4)
#         feat1: torch.Tensor,    # (B, latent_ch, H/2, W/2)
#         feat0: torch.Tensor,    # (B, latent_ch, H,   W)
#         action: torch.Tensor,   # (B, action_dim)
#     ) -> torch.Tensor:
#         """
#         Returns:
#             flow: (B, 2, H, W), range (-1, 1) via tanh
#         """
#         action_emb = self.action_proj(action)   # (B, action_emb_dim)

#         x = self.z_proj(z_raw)                                          # (B, C*4, H/4, W/4)
#         x = self.z_res(x, action_emb)                                   # (B, C*4, H/4, W/4)

#         # Upsample to half-res and fuse feat1
#         x = self.up1(x)                                                  # (B, C*2, H/2, W/2)
#         f1 = self.feat1_proj(feat1)                                      # (B, C*2, H/2, W/2)
#         x = self.dec1(torch.cat([x, f1], dim=1), action_emb)            # (B, C*2, H/2, W/2)

#         # Upsample to full-res and fuse feat0
#         x = self.up2(x)                                                  # (B, C,   H,   W)
#         f0 = self.feat0_proj(feat0)                                      # (B, C,   H,   W)
#         x = self.dec2(torch.cat([x, f0], dim=1), action_emb)            # (B, C,   H,   W)

#         return torch.tanh(self.outc(x))                                  # (B, 2, H, W)


class TwoFrameFlowPredictor(nn.Module):
    """Flow predictor that takes two consecutive frames' encoder features (concatenated).

    No action conditioning. Concatenates features from frame t and frame t+1 at each
    resolution level and predicts optical flow from frame t to frame t+1.

    Args:
        latent_ch:     number of channels in encoder features (per frame)
        base_channels: base channel width (default 64)
    """

    def __init__(self, latent_ch: int, base_channels: int = 64):
        super().__init__()
        C = base_channels
        in_ch = 2 * latent_ch  # two frames concatenated

        # Project z_raw (quarter-res) to working channels
        self.z_proj = nn.Conv2d(in_ch, C * 4, 1)
        self.z_res = ResBlock(C * 4, C * 4)

        # Level 1: quarter-res -> half-res, fuse with feat1 skip
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C * 4, C * 2, 3, padding=1),
        )
        self.feat1_proj = nn.Conv2d(in_ch, C * 2, 1)
        self.dec1 = ResBlock(C * 4, C * 2)   # cat(up, feat1) -> C*4 in

        # Level 2: half-res -> full-res, fuse with feat0 skip
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C * 2, C, 3, padding=1),
        )
        self.feat0_proj = nn.Conv2d(in_ch, C, 1)
        self.dec2 = ResBlock(C * 2, C)        # cat(up, feat0) -> C*2 in

        # Output head
        self.outc = nn.Conv2d(C, 2, 1)

    def forward(
        self,
        z_raw_t: torch.Tensor,   # (B, latent_ch, H/4, W/4)
        feat1_t: torch.Tensor,   # (B, latent_ch, H/2, W/2)
        feat0_t: torch.Tensor,   # (B, latent_ch, H,   W)
        z_raw_t2: torch.Tensor,  # (B, latent_ch, H/4, W/4)
        feat1_t2: torch.Tensor,  # (B, latent_ch, H/2, W/2)
        feat0_t2: torch.Tensor,  # (B, latent_ch, H,   W)
    ) -> torch.Tensor:
        """
        Returns:
            flow: (B, 2, H, W), range (-1, 1) via tanh
        """
        z_cat  = torch.cat([z_raw_t, z_raw_t2], dim=1)   # (B, 2*latent_ch, H/4, W/4)
        f1_cat = torch.cat([feat1_t, feat1_t2], dim=1)   # (B, 2*latent_ch, H/2, W/2)
        f0_cat = torch.cat([feat0_t, feat0_t2], dim=1)   # (B, 2*latent_ch, H,   W)

        x = self.z_proj(z_cat)                                    # (B, C*4, H/4, W/4)
        x = self.z_res(x)                                         # (B, C*4, H/4, W/4)

        x = self.up1(x)                                           # (B, C*2, H/2, W/2)
        f1 = self.feat1_proj(f1_cat)                              # (B, C*2, H/2, W/2)
        x = self.dec1(torch.cat([x, f1], dim=1))                  # (B, C*2, H/2, W/2)

        x = self.up2(x)                                           # (B, C,   H,   W)
        f0 = self.feat0_proj(f0_cat)                              # (B, C,   H,   W)
        x = self.dec2(torch.cat([x, f0], dim=1))                  # (B, C,   H,   W)

        return torch.tanh(self.outc(x))                           # (B, 2, H, W)
