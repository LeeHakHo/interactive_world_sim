import torch
import torch.nn as nn


class FlowDecoder(nn.Module):
    """CNN decoder: (B, flow_emb_dim) → (B, 2, 256, 256)

    Trained in Stage 1 alongside FlowPredictor as auxiliary supervision.
    Not used at inference time.
    """

    def __init__(self, flow_emb_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(flow_emb_dim, 128 * 8 * 8)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),  # (128, 16,  16)
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # (64,  32,  32)
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # (32,  64,  64)
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),    # (16,  128, 128)
            nn.SiLU(),
            nn.ConvTranspose2d(16, 2, 4, stride=2, padding=1),     # (2,   256, 256)
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb: (..., flow_emb_dim)
        Returns:
            flow: (..., 2, 256, 256)
        """
        shape = emb.shape[:-1]
        x = self.fc(emb.reshape(-1, emb.shape[-1])).reshape(-1, 128, 8, 8)
        out = self.net(x)
        return out.reshape(*shape, *out.shape[1:])


class FlowPredictor(nn.Module):
    """(image_latent, action) → flow_emb

    Trained in Stage 1 as auxiliary predictor to improve image representations.
    image_latent: (B, latent_ch, H, W) spatial feature map from encoder
    action: (B, action_dim)
    """

    def __init__(self, latent_ch: int, action_dim: int, flow_emb_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(latent_ch, hidden_dim),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, flow_emb_dim),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_ch, H, W)
            action: (B, action_dim)
        Returns:
            flow_emb: (B, flow_emb_dim)
        """
        z_vec = self.latent_proj(latent)            # (B, hidden_dim)
        x = torch.cat([z_vec, action], dim=-1)      # (B, hidden_dim + action_dim)
        return self.net(x)                          # (B, flow_emb_dim)
