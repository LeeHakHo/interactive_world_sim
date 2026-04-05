import torch
import torch.nn as nn
from omegaconf import DictConfig

from .cm_controlnet import CMControlledUnetModel, CMControlNet


class CMDecoder(nn.Module):
    """CMDecoder for decoding the latent space into the image space."""

    def __init__(
        self,
        x_shape: torch.Size,
        external_cond_dim: int,
        cfg: DictConfig,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.cfg = cfg

        self.x_shape = x_shape
        self.external_cond_dim = external_cond_dim
        self.dtype = dtype

        self._build_model()

    def _build_model(self) -> None:
        use_scale_shift_norm = getattr(self.cfg, "use_scale_shift_norm", False)
        self.model = CMControlledUnetModel(
            in_channels=self.x_shape[0],
            model_channels=self.cfg.model_channels,
            out_channels=self.x_shape[0],
            num_res_blocks=self.cfg.num_res_blocks,
            attention_resolutions=self.cfg.attention_resolutions,
            t_emb_dim=self.external_cond_dim,
            cond_dim=self.external_cond_dim,
            dropout=self.cfg.dropout,
            channel_mult=self.cfg.channel_mult,
            num_head_channels=self.cfg.num_head_channels,
            resblock_updown=self.cfg.resblock_updown,
            use_scale_shift_norm=use_scale_shift_norm,
            dtype=self.dtype,
        )
        self.control_net = CMControlNet(
            in_channels=self.x_shape[0],
            model_channels=self.cfg.model_channels,
            out_channels=self.x_shape[0],
            num_res_blocks=self.cfg.num_res_blocks,
            attention_resolutions=self.cfg.attention_resolutions,
            t_emb_dim=self.external_cond_dim,
            cond_dim=self.external_cond_dim,
            dropout=self.cfg.dropout,
            channel_mult=self.cfg.channel_mult,
            num_head_channels=self.cfg.num_head_channels,
            resblock_updown=self.cfg.resblock_updown,
            use_scale_shift_norm=use_scale_shift_norm,
            num_cond_upsamples=self.cfg.num_latent_downsample,
            num_cond_channel=(
                self.cfg.num_latent_channel
                if "num_latent_channel" in self.cfg
                else self.x_shape[0]
            ),
            dtype=self.dtype,
        )
        self.control_scales = [1.0] * 10

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        external_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the model."""
        controls = self.control_net(x, t, s, external_cond)
        controls = [
            c * scale for c, scale in zip(controls, self.control_scales, strict=False)
        ]
        model_output = self.model(x, t, s, controls=controls)

        return model_output


class CMFlowDecoder(nn.Module):
    """CM-based optical flow decoder.

    Identical structure to CMDecoder but predicts 2-channel optical flow.
    Action is conditioned by projecting to latent channel space and adding
    to z as a global spatial bias before passing to the ControlNet.
    """

    def __init__(
        self,
        flow_shape: torch.Size,
        external_cond_dim: int,
        action_dim: int,
        num_latent_channel: int,
        cfg: DictConfig,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.flow_shape = flow_shape
        self.external_cond_dim = external_cond_dim
        self.action_dim = action_dim
        self.num_latent_channel = num_latent_channel
        self.cfg = cfg
        self.dtype = dtype
        self._build_model()

    def _build_model(self) -> None:
        use_scale_shift_norm = getattr(self.cfg, "use_scale_shift_norm", False)
        self.model = CMControlledUnetModel(
            in_channels=self.flow_shape[0],
            model_channels=self.cfg.model_channels,
            out_channels=self.flow_shape[0],
            num_res_blocks=self.cfg.num_res_blocks,
            attention_resolutions=self.cfg.attention_resolutions,
            t_emb_dim=self.external_cond_dim,
            cond_dim=self.external_cond_dim,
            dropout=self.cfg.dropout,
            channel_mult=self.cfg.channel_mult,
            num_head_channels=self.cfg.num_head_channels,
            resblock_updown=self.cfg.resblock_updown,
            use_scale_shift_norm=use_scale_shift_norm,
            dtype=self.dtype,
        )
        self.control_net = CMControlNet(
            in_channels=self.flow_shape[0],
            model_channels=self.cfg.model_channels,
            out_channels=self.flow_shape[0],
            num_res_blocks=self.cfg.num_res_blocks,
            attention_resolutions=self.cfg.attention_resolutions,
            t_emb_dim=self.external_cond_dim,
            cond_dim=self.external_cond_dim,
            dropout=self.cfg.dropout,
            channel_mult=self.cfg.channel_mult,
            num_head_channels=self.cfg.num_head_channels,
            resblock_updown=self.cfg.resblock_updown,
            use_scale_shift_norm=use_scale_shift_norm,
            num_cond_upsamples=self.cfg.num_latent_downsample,
            num_cond_channel=self.num_latent_channel,
            dtype=self.dtype,
        )
        self.control_scales = [1.0] * 10
        # Project action to latent channel space as global spatial bias on z
        self.action_proj = nn.Linear(self.action_dim, self.num_latent_channel)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        external_cond,  # tuple: (z, action)
    ) -> torch.Tensor:
        z, action = external_cond
        action_bias = self.action_proj(action).view(action.shape[0], self.num_latent_channel, 1, 1)
        z_conditioned = z + action_bias
        controls = self.control_net(x, t, s, z_conditioned)
        controls = [c * scale for c, scale in zip(controls, self.control_scales, strict=False)]
        return self.model(x, t, s, controls=controls)
