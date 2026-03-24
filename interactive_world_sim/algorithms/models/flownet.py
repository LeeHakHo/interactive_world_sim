import torch
import torch.nn as nn


class FlowPredictor(nn.Module):
    """(image_latent, action) → flow (B, 2, H, W)

    Trained in Stage 1 as auxiliary predictor to improve image representations.
    Action is tiled spatially and concatenated with the latent feature map,
    then conv layers predict flow at the same spatial resolution as the latent.

    latent: (B, latent_ch, H, W) spatial feature map from encoder
    action: (B, action_dim)
    """

    def __init__(self, latent_ch: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_ch + action_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 2, 3, padding=1),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_ch, H, W)
            action: (B, action_dim)
        Returns:
            flow: (B, 2, H, W)
        """
        H, W = latent.shape[-2:]
        action_tiled = action[:, :, None, None].expand(-1, -1, H, W)  # (B, action_dim, H, W)
        x = torch.cat([latent, action_tiled], dim=1)                   # (B, latent_ch + action_dim, H, W)
        return self.net(x)                                              # (B, 2, H, W)
