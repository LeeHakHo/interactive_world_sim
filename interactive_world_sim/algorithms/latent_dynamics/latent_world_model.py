import os
import tracemalloc
from typing import Any, Callable

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR, LinearLR, ReduceLROnPlateau

# import matplotlib.pyplot as plt
from interactive_world_sim.algorithms.common.base_pytorch_algo import BasePytorchAlgo
from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.common.metrics import (
    FrechetInceptionDistance,
    FrechetVideoDistance,
    LearnedPerceptualImagePatchSimilarity,
)
from interactive_world_sim.algorithms.latent_dynamics.dynamo_ssl_module import (
    AdaLNSpatialProjection,
    DynaMoSSLModule,
    PlainSpatialProjection,
    ResNet18SpatialEncoder,
    SpatialProjectionHead,
    SpatialToVectorHead,
    ViTSpatialEncoder,
    ViTVectorHead,
)
from interactive_world_sim.algorithms.models.cm_decoder import CMDecoder
from interactive_world_sim.algorithms.models.utils import EinopsWrapper
from interactive_world_sim.utils.cm_utils import DDPMScheduler
from interactive_world_sim.utils.logging_utils import (
    get_validation_metrics_for_videos,
    log_video,
)
from interactive_world_sim.utils.normalizer import LinearNormalizer


class LatentWorldModel(BasePytorchAlgo):
    """StudentV1_0"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.metrics = cfg.metrics
        self.num_latent_channel = cfg.num_latent_channel
        self.num_latent_downsample = cfg.num_latent_downsample
        self.training_stage = cfg.training_stage
        assert self.training_stage in [1, 2, 3], "Invalid training stage"
        self.load_ae = cfg.load_ae if "load_ae" in cfg else None
        self.obs_keys = cfg.obs_keys
        self.num_views = len(self.obs_keys)
        self.latent_resolution = cfg.latent_resolution

        # Encoder backbone + SSL config (must be set before super().__init__ which
        # calls _build_model). The two axes are decoupled:
        #   - encoder_backbone in {vit, resnet, conv2d}: which encoder architecture
        #     to use. Default is `vit` (Plan-2 §3 Phase 0 task 1).
        #   - dynamo_ssl.enabled: whether to attach the DynaMo IDM/FDM SSL heads on
        #     top of the encoder during Stage 1 training. Disabled by default in
        #     HUMAN_ROBOT_ALIGN_PLAN-2 (the align module replaces SSL).
        # The legacy name `use_resnet_encoder` is kept but now means "use a real
        # backbone (ResNet18 OR ViT-S)" rather than "ResNet18 specifically".
        ssl_cfg_in = cfg.get("dynamo_ssl", None)
        self.encoder_backbone = (
            ssl_cfg_in.get("encoder_backbone", "vit") if ssl_cfg_in is not None else "vit"
        )
        assert self.encoder_backbone in ("vit", "resnet", "conv2d"), (
            f"encoder_backbone must be one of vit|resnet|conv2d, got {self.encoder_backbone}"
        )
        self.use_resnet_encoder = self.encoder_backbone in ("vit", "resnet")
        self.use_vit_encoder = (self.encoder_backbone == "vit")
        self.use_dynamo_ssl = (
            ssl_cfg_in is not None
            and ssl_cfg_in.get("enabled", False)
            and self.use_resnet_encoder
            and self.training_stage == 1
        )
        if self.use_dynamo_ssl:
            self.ssl_loss_coef = cfg.dynamo_ssl.loss_coef
            self.detach_rec_from_encoder = cfg.dynamo_ssl.get("detach_rec_from_encoder", True)
            import warnings
            warnings.warn(
                "[LatentWorldModel] dynamo_ssl.enabled=true: DynaMo SSL is being used. "
                "Per HUMAN_ROBOT_ALIGN_PLAN-2 Phase 0, SSL is supposed to be off in this "
                "plan iteration; the align module replaces it. Override only if you know "
                "what you're doing.",
                stacklevel=2,
            )

        # Phase 0 latent decomposition flag. Mirrors dynamo_ssl pattern.
        # Active only in Stage 1; Stages 2/3 ignore it.
        ld_cfg = cfg.get("latent_decompose", None)
        _decompose_on = bool(
            ld_cfg is not None
            and ld_cfg.get("enabled", False)
            and self.training_stage == 1
            and self.use_vit_encoder
        )
        # method: "channel_split" (v3, GRL adversary) | "dual_head" (v4, two
        # independent projection heads + CLUB MI penalty, decoder untouched) |
        # "emb_film" (v5, Phase 0a, global appearance code appended to the
        # decoder conditioning; scene latent + dynamics untouched).
        self.decompose_method = (
            ld_cfg.get("method", "channel_split")
            if ld_cfg is not None else "channel_split"
        )
        # emb_film is a separate path: it does NOT use the split machinery
        # (z_task_list/z_emb_list) or the GRL/CLUB classifiers, and runs under
        # automatic optimisation. Keep `use_latent_decompose` meaning the split
        # paths only, so the existing branches stay untouched.
        self.use_emb_film = bool(_decompose_on and self.decompose_method == "emb_film")
        self.use_latent_decompose = bool(
            _decompose_on and self.decompose_method in ("channel_split", "dual_head")
        )
        if self.use_latent_decompose and self.decompose_method == "channel_split":
            assert (
                int(ld_cfg.d_task) + int(ld_cfg.d_emb)
                == int(cfg.dynamo_ssl.vit.embed_dim)
            ), "channel_split requires d_task + d_emb == ViT embed_dim"
        self.use_dual_head = (
            self.use_latent_decompose and self.decompose_method == "dual_head"
        )
        # mask_subtract (2026-05-28): separate path like emb_film — runs under
        # automatic optimisation, decoder unchanged, embodiment path detached.
        self.use_mask_subtract = bool(
            _decompose_on and self.decompose_method == "mask_subtract"
        )
        # emb_film: per-view global appearance-code width appended as constant
        # channels to the decoder conditioning. Total appended = c_emb*num_views.
        self.c_emb = int(ld_cfg.get("c_emb", 16)) if (ld_cfg is not None) else 16
        self.c_emb_total = self.c_emb * self.num_views

        # Phase 0b: EgoBridge-style OT alignment on z_scene (only with emb_film).
        ot_cfg = ld_cfg.get("ot_align", None) if ld_cfg is not None else None
        self.use_ot_align = bool(
            self.use_emb_film and ot_cfg is not None and ot_cfg.get("enabled", False)
        )
        self.ot_cfg = ot_cfg
        # per-domain memory banks (plain attrs, not in state_dict): tuples of
        # (flattened z_scene, action traj, mean action), kept on the model device.
        self._ot_bank = {0: None, 1: None}
        # robot world->cam_high extrinsic for unifying EEF position frames in the
        # OT cost (robot EEF is world-frame, human EEF is cam-frame). Lazily
        # materialised on the model device on first use.
        self._T_cam_world_t = None

        super().__init__(cfg)

        # dual_head + CLUB is a two-player game (encoder vs variational q-net),
        # so it needs manual optimisation with two optimisers — same machinery
        # the SSL path already uses.
        if self.use_dynamo_ssl or self.use_dual_head:
            self.automatic_optimization = False

        self.normalizer = LinearNormalizer()
        self.validation_step_outputs: list = []
        self.validation_metrics: dict = {}
        self.timesteps: int = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.val_render = cfg.val_render
        self.max_val_render_batches: int = cfg.get("max_val_render_batches", 16)
        self.clip_noise = self.cfg.diffusion.clip_noise
        self.guidance_scale = self.cfg.guidance_scale
        self.n_tokens = self.cfg.n_frames
        self.mask_prev_action = (
            cfg.mask_prev_action if "mask_prev_action" in cfg else False
        )

        self.noise_scheduler: DDPMScheduler = hydra.utils.instantiate(
            cfg.noise_scheduler
        )

        self.debug = False
        self.lr_scheduler = cfg.lr_scheduler if "lr_scheduler" in cfg else "linear"
        self.sampling_strategy = (
            cfg.sampling_strategy if "sampling_strategy" in cfg else "uniform"
        )
        self.prev_frame_noise_scale = (
            cfg.prev_frame_noise_scale if "prev_frame_noise_scale" in cfg else 0.1
        )
        self.dyn_infer_steps = cfg.dyn_infer_steps if "dyn_infer_steps" in cfg else 1
        self.dec_infer_steps = cfg.dec_infer_steps if "dec_infer_steps" in cfg else 1
        self.last_frame_loss_only = (
            cfg.last_frame_loss_only if "last_frame_loss_only" in cfg else False
        )
        self.robust_latent = cfg.robust_latent if "robust_latent" in cfg else False

    def _build_model(self) -> None:
        # decoder. For emb_film we append c_emb_total constant "appearance"
        # channels to the conditioning latent, so the control_net must accept
        # (num_latent_channel + c_emb_total) cond channels. Only the decoder's
        # cond width changes; the dynamics latent stays num_latent_channel.
        dec_diff_cfg = self.cfg.diffusion
        if getattr(self, "use_emb_film", False):
            from omegaconf import OmegaConf
            dec_diff_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg.diffusion, resolve=True))
            dec_diff_cfg.num_latent_channel = (
                int(self.cfg.num_latent_channel) + int(self.c_emb_total)
            )
        self.decoder: CMDecoder = CMDecoder(
            self.cfg.x_shape,
            self.cfg.latent_dim,
            dec_diff_cfg,
            dtype=self.dtype,
        )

        # dynamics
        self.dynamics: EinopsWrapper = EinopsWrapper(
            from_shape="f b c h w",
            to_shape="b c f h w",
            module=hydra.utils.instantiate(self.cfg.dynamics),
        )

        # encoder
        latent_ch = self.num_latent_channel
        if self.use_resnet_encoder:
            c_per_view = latent_ch // self.num_views

            if self.use_vit_encoder:
                # ViT backbone + spatial projection
                vit_cfg = self.cfg.dynamo_ssl.get("vit", {})
                self.vit_encoder = ViTSpatialEncoder(
                    img_size=vit_cfg.get("img_size", self.cfg.x_shape[1]),
                    patch_size=vit_cfg.get("patch_size", 8),
                    embed_dim=vit_cfg.get("embed_dim", 384),
                    depth=vit_cfg.get("depth", 12),
                    num_heads=vit_cfg.get("num_heads", 6),
                    mlp_ratio=vit_cfg.get("mlp_ratio", 4.0),
                    drop_rate=vit_cfg.get("drop_rate", 0.0),
                )
                vit_grid = self.vit_encoder.grid_size
                vit_dim = self.vit_encoder.embed_dim
                proj_mode = self.cfg.dynamo_ssl.get("spatial_proj_mode", "plain")
                _proj_cls = AdaLNSpatialProjection if proj_mode == "adaln" else PlainSpatialProjection
                self.spatial_proj = _proj_cls(
                    embed_dim=vit_dim,
                    out_channels=c_per_view,
                    in_spatial=vit_grid,
                    out_spatial=self.latent_resolution,
                )
                if self.use_latent_decompose and self.decompose_method == "channel_split":
                    from interactive_world_sim.algorithms.latent_decompose.split_encoder import (
                        SplitEncoder,
                    )
                    from interactive_world_sim.algorithms.latent_decompose.domain_heads import (
                        PooledClassifier,
                    )
                    ld = self.cfg.latent_decompose
                    self.split_encoder = SplitEncoder(
                        self.vit_encoder, int(ld.d_task), int(ld.d_emb),
                    )
                    self.clf_emb = PooledClassifier(int(ld.d_emb))
                    self.clf_adv = PooledClassifier(int(ld.d_task))
                elif self.use_dual_head:
                    from interactive_world_sim.algorithms.latent_decompose.dual_head import (
                        DualHead,
                    )
                    from interactive_world_sim.algorithms.latent_decompose.club import (
                        CLUB,
                    )
                    from interactive_world_sim.algorithms.latent_decompose.domain_heads import (
                        PooledClassifier,
                    )
                    ld = self.cfg.latent_decompose
                    d_task, d_emb = int(ld.d_task), int(ld.d_emb)
                    self.dual_head = DualHead(
                        in_dim=vit_dim, d_task=d_task, d_emb=d_emb,
                    )
                    # z_emb must carry embodiment info -> domain classifier.
                    # PooledClassifier expects (B, D, H, W); here z_emb is
                    # already pooled (B, d_emb), so we use a plain MLP instead.
                    self.clf_emb = nn.Sequential(
                        nn.Linear(d_emb, 128), nn.GELU(), nn.Linear(128, 2),
                    )
                    # CLUB variational q(z_emb | z_task) for the MI penalty.
                    self.club = CLUB(
                        x_dim=d_task, y_dim=d_emb,
                        hidden=int(ld.get("club_hidden", 256)),
                    )
                if self.use_emb_film:
                    from interactive_world_sim.algorithms.latent_decompose.emb_film import (
                        EmbHead,
                        DomainProbe,
                    )
                    # Global appearance code from the [CLS] token (shared across
                    # views); broadcast + appended to the decoder conditioning.
                    self.emb_head = EmbHead(in_dim=vit_dim, c_emb=self.c_emb)
                    # Detached diagnostic probes (measure-first): how domain-
                    # separable are the scene latent vs the appearance code?
                    self.probe_scene = DomainProbe(int(self.cfg.num_latent_channel))
                    self.probe_emb = DomainProbe(int(self.c_emb_total))
                if getattr(self, "use_mask_subtract", False):
                    from interactive_world_sim.algorithms.latent_decompose.mask_subtract import (
                        MaskSubtractHead,
                        AgentReconHead,
                    )
                    from interactive_world_sim.algorithms.latent_decompose.emb_film import (
                        DomainProbe,
                    )
                    ld = self.cfg.latent_decompose
                    c_scene = int(self.cfg.num_latent_channel)
                    d_emb = int(ld.get("d_emb_mask", 64))
                    self.mask_subtract = MaskSubtractHead(
                        c_scene=c_scene,
                        latent_hw=int(self.latent_resolution),
                        d_emb=d_emb,
                        removal=str(ld.get("removal_op", "ortho_proj")),
                    )
                    self.agent_recon = AgentReconHead(
                        c_scene=c_scene, out_hw=int(self.cfg.x_shape[1]),
                    )
                    # detached diagnostics: z_full / z_scene / z_emb separability
                    self.probe_full = DomainProbe(c_scene)
                    self.probe_scene = DomainProbe(c_scene)
                    self.probe_emb = DomainProbe(d_emb)
            else:
                # ResNet18 backbone + spatial projection for decoder
                self.resnet_encoder = ResNet18SpatialEncoder(
                    pretrained=self.cfg.dynamo_ssl.get("pretrained_encoder", False)
                )
                resnet_spatial = self.cfg.x_shape[1] // 32  # e.g., 128//32 = 4
                self.spatial_proj = SpatialProjectionHead(
                    in_channels=512,
                    out_channels=c_per_view,
                    in_spatial=resnet_spatial,
                    out_spatial=self.latent_resolution,
                )
                self.encoder = nn.Sequential(self.resnet_encoder, self.spatial_proj)

            if self.use_dynamo_ssl:
                ssl_cfg = self.cfg.dynamo_ssl
                if self.use_vit_encoder:
                    self.vector_head = ViTVectorHead(
                        embed_dim=vit_dim, feature_dim=ssl_cfg.feature_dim,
                    )
                    encoder_for_ema = self.vit_encoder
                else:
                    self.vector_head = SpatialToVectorHead(
                        in_channels=512, feature_dim=ssl_cfg.feature_dim,
                    )
                    encoder_for_ema = self.resnet_encoder
                self.dynamo_ssl = DynaMoSSLModule(
                    encoder_for_ema=encoder_for_ema,
                    vector_head_for_ema=self.vector_head,
                    num_views=self.num_views,
                    window_size=ssl_cfg.window_size,
                    feature_dim=ssl_cfg.feature_dim,
                    projection_dim=ssl_cfg.projection_dim,
                    n_layer=ssl_cfg.n_layer,
                    n_head=ssl_cfg.n_head,
                    n_embd=ssl_cfg.n_embd,
                    dropout=ssl_cfg.dropout,
                    covariance_reg_coef=ssl_cfg.covariance_reg_coef,
                    dynamics_loss_coef=ssl_cfg.dynamics_loss_coef,
                    ema_beta=ssl_cfg.ema_beta,
                    beta_scheduling=ssl_cfg.beta_scheduling,
                    lr=ssl_cfg.lr,
                    weight_decay=ssl_cfg.weight_decay,
                    betas=tuple(ssl_cfg.betas),
                    separate_single_views=ssl_cfg.separate_single_views,
                    use_sparse_idm=ssl_cfg.get("use_sparse_idm", False),
                    sparse_lambda=ssl_cfg.get("sparse_lambda", 0.01),
                    sparse_mask_init=ssl_cfg.get("sparse_mask_init", 0.0),
                    use_sigreg=ssl_cfg.get("use_sigreg", False),
                    sigreg_weight=ssl_cfg.get("sigreg_weight", 0.09),
                    sigreg_knots=ssl_cfg.get("sigreg_knots", 17),
                    sigreg_num_proj=ssl_cfg.get("sigreg_num_proj", 1024),
                )
        else:
            encoder_module_ls = [nn.Conv2d(self.cfg.x_shape[0], latent_ch, 3, padding=1)]
            for _ in range(self.num_latent_downsample):
                encoder_module_ls.extend(
                    [
                        nn.SiLU(),
                        nn.Conv2d(latent_ch, latent_ch, kernel_size=3, padding=1),
                        nn.SiLU(),
                        nn.Conv2d(latent_ch, latent_ch, kernel_size=3, padding=1, stride=2),
                    ]
                )
            self.encoder = nn.Sequential(*encoder_module_ls)

        # load previous trained model
        if self.load_ae is not None:
            cfg_cp = self.cfg.copy()
            load_ae_dir = os.path.dirname(os.path.dirname(self.load_ae))
            cfg_path = f"{load_ae_dir}/.hydra/config.yaml"
            cfg_cp = OmegaConf.load(cfg_path)
            cfg_cp.load_ae = None
            diffae = LatentWorldModel.load_from_checkpoint(
                self.load_ae,
                cfg=cfg_cp.algorithm,
                map_location=self.device,
                weights_only=False,
            )
            if self.use_resnet_encoder:
                if self.use_vit_encoder:
                    self.vit_encoder.load_state_dict(diffae.vit_encoder.state_dict())
                else:
                    self.resnet_encoder.load_state_dict(diffae.resnet_encoder.state_dict())
                self.spatial_proj.load_state_dict(diffae.spatial_proj.state_dict())
            else:
                self.encoder.load_state_dict(diffae.encoder.state_dict())
            if self.training_stage == 3:
                self.dynamics.load_state_dict(diffae.dynamics.state_dict())
            self.decoder.load_state_dict(diffae.decoder.state_dict())

        self.validation_fid_model = (
            FrechetInceptionDistance(feature=64) if "fid" in self.metrics else None
        )
        self.validation_lpips_model = (
            LearnedPerceptualImagePatchSimilarity() if "lpips" in self.metrics else None
        )
        self.validation_fvd_model: FrechetVideoDistance = (
            FrechetVideoDistance() if "fvd" in self.metrics else None
        )

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Set the normalizer for the model"""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def configure_optimizers(self):
        """Configure the optimizer for the model"""
        if self.training_stage == 1:
            if self.use_vit_encoder:
                encoder_params = list(self.vit_encoder.parameters()) + list(self.spatial_proj.parameters())
            elif hasattr(self, "encoder"):
                encoder_params = list(self.encoder.parameters())
            else:
                encoder_params = list(self.resnet_encoder.parameters()) + list(self.spatial_proj.parameters())
            # emb_film: the appearance head produces part of the reconstruction
            # conditioning, so it warms up with the encoder (group 1).
            if self.use_emb_film:
                encoder_params = encoder_params + list(self.emb_head.parameters())
            param_groups = [
                {"params": self.decoder.parameters(), "lr": self.cfg.lr},
                {"params": encoder_params, "lr": self.cfg.lr},
            ]
            if self.use_emb_film:
                # Detached diagnostic probes: constant LR (aux group, idx>=2).
                param_groups.append({
                    "params": list(self.probe_scene.parameters())
                              + list(self.probe_emb.parameters()),
                    "lr": self.cfg.lr,
                })
            if getattr(self, "use_mask_subtract", False):
                # embodiment path (detached from E via stop_grad): own group, main LR
                param_groups.append({
                    "params": list(self.mask_subtract.parameters())
                              + list(self.agent_recon.parameters()),
                    "lr": self.cfg.lr,
                })
                # detached probes: aux group, constant LR (idx>=2)
                param_groups.append({
                    "params": list(self.probe_full.parameters())
                              + list(self.probe_scene.parameters())
                              + list(self.probe_emb.parameters()),
                    "lr": self.cfg.lr,
                })
            if self.use_latent_decompose and self.decompose_method == "channel_split":
                param_groups.append({
                    "params": list(self.clf_emb.parameters())
                              + list(self.clf_adv.parameters()),
                    "lr": float(self.cfg.latent_decompose.lr_classifiers),
                })
            elif self.use_dual_head:
                param_groups.append({
                    "params": list(self.dual_head.parameters())
                              + list(self.clf_emb.parameters()),
                    "lr": float(self.cfg.latent_decompose.lr_classifiers),
                })
            if self.use_dynamo_ssl:
                ssl_cfg = self.cfg.dynamo_ssl
                param_groups.extend([
                    {"params": self.vector_head.parameters(), "lr": ssl_cfg.lr},
                    {"params": self.dynamo_ssl.projector.parameters(), "lr": ssl_cfg.lr},
                    {"params": self.dynamo_ssl.forward_dynamics.parameters(), "lr": ssl_cfg.lr},
                ])
                if self.dynamo_ssl.use_sparse_idm:
                    param_groups.append(
                        {"params": self.dynamo_ssl.sparse_mask.parameters(), "lr": ssl_cfg.lr},
                    )
        elif self.training_stage == 2:
            param_groups = [
                {"params": self.dynamics.parameters(), "lr": self.cfg.lr},
            ]
        elif self.training_stage == 3:
            param_groups = [
                {"params": self.decoder.parameters(), "lr": self.cfg.lr * 0.1},
            ]
        optimizer = torch.optim.AdamW(
            params=param_groups,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.optimizer_beta,
        )
        if self.lr_scheduler == "linear":
            # Per-group warmup is needed whenever there are auxiliary param
            # groups (SSL heads or latent_decompose classifiers) that must
            # NOT warm up alongside the main encoder/decoder. Without this
            # the auxiliary classifier lr ramps from start_factor·lr up over
            # warmup_steps, under-driving the discriminator early in training
            # (Phase 0 spec requires lr_classifiers to be constant).
            if (self.use_dynamo_ssl or self.use_latent_decompose or self.use_emb_film
                    or getattr(self, "use_mask_subtract", False)):
                warmup_steps = self.cfg.warmup_steps
                start_factor = 1e-4

                def lr_lambda_fn(group_idx):
                    if group_idx < 2:
                        # decoder, encoder(+emb_head): linear warmup
                        def fn(step):
                            if step >= warmup_steps:
                                return 1.0
                            return start_factor + (1.0 - start_factor) * step / warmup_steps
                        return fn
                    else:
                        # auxiliary groups (probes, classifiers, SSL heads): constant LR
                        return lambda step: 1.0

                lr_scheduler = LambdaLR(
                    optimizer,
                    lr_lambda=[lr_lambda_fn(i) for i in range(len(param_groups))],
                )
            else:
                lr_scheduler = LinearLR(
                    optimizer,
                    start_factor=1e-4,
                    end_factor=1.0,
                    total_iters=self.cfg.warmup_steps,
                )
        elif self.lr_scheduler == "plateau":
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=50000,
                verbose=True,
                threshold=1e-3,
                threshold_mode="rel",
            )
        else:
            raise NotImplementedError(f"LR scheduler {self.lr_scheduler} not included")

        # dual_head + CLUB is a two-player game: a SECOND optimiser owns the
        # variational q-network, stepped every batch in the manual training
        # loop. Lightning sees both optimisers; the scheduler tracks the main.
        if self.use_dual_head:
            club_lr = float(self.cfg.latent_decompose.get("club_lr", 1e-4))
            opt_club = torch.optim.AdamW(
                self.club.parameters(), lr=club_lr,
                weight_decay=self.cfg.weight_decay,
                betas=self.cfg.optimizer_beta,
            )
            return (
                {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": lr_scheduler,
                        "interval": "step",
                        "frequency": 1,
                        "name": "lr_scheduler",
                    },
                },
                {"optimizer": opt_club},
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "training/loss",
                "strict": True,
                "name": "lr_scheduler",
            },
        }

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        """For emb_film, clip ONLY the model param groups (decoder=0,
        encoder+emb_head=1), NOT the detached diagnostic probes (group 2).

        The probes are added to the optimised loss so their params train, but
        their gradients must not inflate the global grad-norm — otherwise the
        single global clip would dilute the model's clipped gradient budget.
        """
        if (self.use_emb_film or getattr(self, "use_mask_subtract", False)) and gradient_clip_val:
            model_params = [
                p
                for g in optimizer.param_groups[:2]
                for p in g["params"]
                if p.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(model_params, float(gradient_clip_val))
            return
        super().configure_gradient_clipping(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    # ------------------------------------------------------------------
    # Latent-decompose helpers (Phase 0 align module)
    # ------------------------------------------------------------------

    def _lambda_adv_now(self) -> float:
        """Current value of the gradient-reversal scale, per cfg schedule."""
        from interactive_world_sim.algorithms.latent_decompose.align_losses import (
            linear_ramp,
        )
        sched = self.cfg.latent_decompose.lambda_adv_schedule
        return linear_ramp(
            self.global_step,
            int(sched.start_step), int(sched.end_step),
            float(sched.start_value), float(sched.end_value),
        )

    def _apply_decompose_losses(
        self,
        batch: dict,
        rec_loss: torch.Tensor,
        z_task_list: list[torch.Tensor] | None,
        z_emb_list:  list[torch.Tensor] | None,
    ) -> torch.Tensor:
        """Compute and log L_dom + L_adv from the per-view split tensors
        produced by the (single) encoder pass in training_step. Adds them
        to rec_loss and returns the total.

        G1 short-circuit: when all loss weights are zero we return rec_loss
        directly and ignore the split lists, so the RNG state (and therefore
        rec_loss) matches the enabled=False baseline. Trade-off: classifier
        heads receive no gradient on zero-weight steps, which is fine since
        the loss contribution is zero anyway.

        Caller contract: when latent_decompose.enabled is true, the caller
        must obtain (z_task_list, z_emb_list) from encoder_forward(...,
        return_split=True) on the same `xs` that produced rec_loss — this
        guarantees one ViT pass per step (no duplicate work)."""
        sched = self.cfg.latent_decompose.lambda_adv_schedule
        if (
            float(self.cfg.latent_decompose.lambda_dom) == 0.0
            and float(sched.start_value) == 0.0
            and float(sched.end_value) == 0.0
        ):
            return rec_loss

        from interactive_world_sim.algorithms.latent_decompose.align_losses import (
            compute_L_adv,
            compute_L_dom,
        )

        assert z_task_list is not None and z_emb_list is not None, (
            "latent_decompose.enabled but split tensors were not threaded "
            "from encoder_forward — caller bug."
        )

        T = self.cfg.n_frames
        V = len(self.obs_keys)
        dl = batch["domain_label"]                        # (B,) long

        z_task = torch.cat(z_task_list, dim=0)            # (V*B*T, d_task, gh, gw)
        z_emb  = torch.cat(z_emb_list,  dim=0)            # (V*B*T, d_emb,  gh, gw)
        # Order: v slowest, b middle, t fastest — matches the cat above
        # because z_task_list[v] has shape (B*T, ...) in (b slow, t fast).
        dl_rep = dl.repeat_interleave(T).repeat(V).to(z_task.device)

        L_dom = compute_L_dom(z_emb, dl_rep, self.clf_emb)
        lam = self._lambda_adv_now()
        L_adv = compute_L_adv(z_task, dl_rep, self.clf_adv, lam)

        lambda_dom = float(self.cfg.latent_decompose.lambda_dom)
        total = rec_loss + lambda_dom * L_dom + L_adv

        self.log("training/L_dom", L_dom)
        self.log("training/L_adv", L_adv)
        self.log("training/lambda_adv", lam)
        return total

    def _dual_head_manual_step(
        self,
        batch: dict,
        rec_loss: torch.Tensor,
        z_task_list: list[torch.Tensor],
        z_emb_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Manual two-optimiser CLUB update for the dual_head decompose path.

        Player 1 (q-network): minimise NLL of q(z_emb|z_task) every step.
        Player 2 (encoder + decoder + clf_emb): minimise
            rec_loss + lambda_dom * L_dom [+ lambda_club * CLUB_MI every N steps].
        The decoder path is untouched by the decompose machinery — z_task/z_emb
        are pooled SIDE outputs from dual_head, so reconstruction quality is not
        traded away. Returns total_loss for logging only."""
        import torch.nn.functional as F

        T = self.cfg.n_frames
        V = len(self.obs_keys)
        dl = batch["domain_label"]
        z_task = torch.cat(z_task_list, dim=0)      # (V*B*T, d_task)
        z_emb = torch.cat(z_emb_list, dim=0)        # (V*B*T, d_emb)
        dl_rep = dl.repeat_interleave(T).repeat(V).to(z_task.device)

        opt_main, opt_club = self.optimizers()
        lr_sched = self.lr_schedulers()

        # --- Player 1: q-network NLL (every step), detached from encoder ---
        opt_club.zero_grad()
        club_ll = self.club.learning_loss(z_task.detach(), z_emb.detach())
        self.manual_backward(club_ll)
        opt_club.step()

        # --- Player 2: encoder/decoder/clf_emb update ---
        ld = self.cfg.latent_decompose
        lambda_dom = float(ld.lambda_dom)
        lambda_club = float(ld.get("lambda_club", 1.0))
        club_every = int(ld.get("club_every_n_steps", 10))

        L_dom = F.cross_entropy(self.clf_emb(z_emb), dl_rep)
        total_loss = rec_loss + lambda_dom * L_dom
        apply_club = (self.global_step % club_every == 0)
        if apply_club:
            mi = self.club.mi_est(z_task, z_emb)
            total_loss = total_loss + lambda_club * mi

        opt_main.zero_grad()
        opt_club.zero_grad()  # discard q-grads so the MI term cannot leak into q
        self.manual_backward(total_loss)
        # Use torch's clip directly: self.clip_gradients() validates against the
        # Trainer's gradient_clip_val (which must be 0 under manual optimisation),
        # so it would raise on a non-zero value here.
        clip_val = self.cfg.get("gradient_clip_val", None)
        if clip_val:
            params = [p for g in opt_main.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(params, float(clip_val))
        opt_main.step()
        if lr_sched is not None:
            sched = lr_sched[0] if isinstance(lr_sched, (list, tuple)) else lr_sched
            sched.step()

        self.log("training/rec_loss", rec_loss)
        self.log("training/L_dom", L_dom)
        self.log("training/club_learning_loss", club_ll)
        if apply_club:
            self.log("training/club_mi_est", mi)
        self.log("training/loss", total_loss)
        return total_loss

    def _emb_film_probe_loss(
        self,
        batch: dict,
        z: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """emb_film (measure-first): DETACHED domain probes on the scene latent
        and the appearance code; logs their accuracy. Inputs are detached so the
        probe gradients NEVER reach the encoder/decoder — this only *measures*
        domain-separability, applying no alignment pressure.

        Returns ONLY the probe CE (kept OUT of the reported model loss, and
        excluded from the model's gradient clipping — see
        configure_gradient_clipping). z is the augmented latent
        (B*T, C_scene + c_emb_total, H, W).
        """
        if "domain_label" not in batch:
            return z.new_zeros(())
        c_scene = int(self.cfg.num_latent_channel)
        z_scene = z[:, :c_scene].mean(dim=(2, 3))          # (B*T, C_scene)
        z_emb_vec = z[:, c_scene:].mean(dim=(2, 3))         # (B*T, c_emb_total)
        dl = batch["domain_label"].to(z.device).long()      # (B,)
        dl_rep = dl.repeat_interleave(seq_len)               # (B*T,) b-slow t-fast

        logits_scene = self.probe_scene(z_scene.detach())
        logits_emb = self.probe_emb(z_emb_vec.detach())
        loss_scene = F.cross_entropy(logits_scene, dl_rep)
        loss_emb = F.cross_entropy(logits_emb, dl_rep)

        with torch.no_grad():
            acc_scene = (logits_scene.argmax(-1) == dl_rep).float().mean()
            acc_emb = (logits_emb.argmax(-1) == dl_rep).float().mean()
        # We WANT acc_scene -> chance (~0.5, scene becomes embodiment-agnostic)
        # and acc_emb -> high (appearance code carries embodiment).
        self.log("training/probe_scene_acc", acc_scene)
        self.log("training/probe_emb_acc", acc_emb)
        self.log("training/probe_scene_loss", loss_scene)
        self.log("training/probe_emb_loss", loss_emb)
        return loss_scene + loss_emb

    def _mask_subtract_losses(self, batch: dict, z: torch.Tensor, seq_len: int):
        """mask_subtract: derive z_scene from the full latent F, compute the
        embodiment-path losses (detached from the encoder via MaskSubtractHead)
        plus detached domain probes.

        Returns (z_scene (B*T,C,Hl,Wl), aux_loss). aux_loss = agent-recon +
        anchor + detached-probe CE. The mask-exterior scene-reconstruction term is
        added in training_step (it needs a decoder forward). z is F of shape
        (B*T, C_scene, Hl, Wl); batch["agent_mask"] is (B,T,1,H,W).
        """
        Hl = z.shape[-1]
        m = batch["agent_mask"].to(z.device).float()        # (B,T,1,H,W)
        B, T = m.shape[0], m.shape[1]
        m = m.reshape(B * T, 1, m.shape[-2], m.shape[-1])    # (B*T,1,H,W)
        mask_lat = F.interpolate(m, size=(Hl, Hl), mode="area")
        mask_lat = (mask_lat > 0.5).float()                  # (B*T,1,Hl,Wl)

        out = self.mask_subtract(z, mask_lat)
        z_scene = out["z_scene"]

        # (1) agent reconstruction (mask interior) against the input frames
        xs = batch["obs"][self.obs_keys[0]].to(z.device).float()   # (B,T,3,H,W)
        xs = xs.reshape(B * T, *xs.shape[2:])                # (B*T,3,H,W)
        agent_rgb = self.agent_recon(out["z_emb_spatial"])   # (B*T,3,H,W)
        m_img = (F.interpolate(m, size=agent_rgb.shape[-2:], mode="area") > 0.5).float()
        denom = (m_img.sum() * 3.0).clamp_min(1.0)
        L_agent = (((agent_rgb - xs) ** 2) * m_img).sum() / denom

        # (2) supervised-contrastive domain anchor on the pooled embedding
        from interactive_world_sim.algorithms.latent_decompose.mask_subtract import (
            supcon_anchor,
        )
        dl = batch["domain_label"].to(z.device).long()       # (B,)
        dl_rep = dl.repeat_interleave(T)                      # (B*T,)
        L_anchor = supcon_anchor(
            out["e"], dl_rep,
            temperature=float(self.cfg.latent_decompose.get("anchor_temperature", 0.1)),
        )

        # (3) detached probes (measure only; never reach the encoder)
        zf = z.mean(dim=(2, 3)).detach()
        zs = z_scene.mean(dim=(2, 3)).detach()
        ze = out["e"].detach()
        lf, ls, le = self.probe_full(zf), self.probe_scene(zs), self.probe_emb(ze)
        probe_ce = (F.cross_entropy(lf, dl_rep)
                    + F.cross_entropy(ls, dl_rep)
                    + F.cross_entropy(le, dl_rep))
        with torch.no_grad():
            self.log("training/probe_full_acc", (lf.argmax(-1) == dl_rep).float().mean())
            self.log("training/probe_scene_acc", (ls.argmax(-1) == dl_rep).float().mean())
            self.log("training/probe_emb_acc", (le.argmax(-1) == dl_rep).float().mean())
        self.log("training/L_agent_rec", L_agent)
        self.log("training/L_anchor", L_anchor)

        ld = self.cfg.latent_decompose
        aux = (float(ld.get("lambda_agent_rec", 1.0)) * L_agent
               + float(ld.get("lambda_anchor", 0.5)) * L_anchor
               + probe_ce)
        return z_scene, aux

    def _ot_alpha_now(self) -> float:
        """Warmup ramp for the OT loss weight (0 -> alpha over [start, end])."""
        oc = self.ot_cfg
        a = float(oc.get("alpha", 0.1))
        s0 = int(oc.get("warmup_start", 5000))
        s1 = int(oc.get("warmup_end", 20000))
        step = int(self.global_step)
        if step <= s0:
            return 0.0
        if step >= s1:
            return a
        return a * (step - s0) / max(1, s1 - s0)

    def _get_T_cam_world(self, device):
        """Cached robot world->cam_high extrinsic (4,4) on the given device."""
        if self._T_cam_world_t is None:
            from interactive_world_sim.algorithms.latent_decompose.ot_align import (
                robot_world_to_cam,
            )
            urdf = self.ot_cfg.get("urdf_path", None) if self.ot_cfg is not None else None
            T = robot_world_to_cam(urdf) if urdf else robot_world_to_cam()
            self._T_cam_world_t = torch.as_tensor(T, dtype=torch.float32)
        return self._T_cam_world_t.to(device)

    def _ot_bank_push(self, d: int, f, traj, a, cap: int) -> None:
        """FIFO push of detached opposite-domain features into the memory bank."""
        cur = self._ot_bank[d]
        if cur is None:
            new = (f, traj, a)
        else:
            new = (
                torch.cat([cur[0], f]),
                torch.cat([cur[1], traj]),
                torch.cat([cur[2], a]),
            )
        if new[0].shape[0] > cap:
            new = tuple(x[-cap:] for x in new)
        self._ot_bank[d] = new

    def _apply_ot_align(self, z, action, batch, seq_len: int):
        """EgoBridge-style OT alignment of z_scene across domains (Phase 0b).

        Aligns ONLY the scene channels z[:, :C_scene] (z_emb is left untouched so
        the decoder keeps its embodiment-specific route). Current-batch features
        (with grad) are transported toward the OPPOSITE-domain memory bank
        (detached), so the encoder is pulled to overlap the two domains' scene
        distributions while DTW pseudo-pairs keep the coupling behaviour-aware.
        """
        if "domain_label" not in batch:
            return None
        from interactive_world_sim.algorithms.latent_decompose.ot_align import (
            ot_align_loss,
        )
        oc = self.ot_cfg
        c_scene = int(self.cfg.num_latent_channel)
        f = z[:, :c_scene].flatten(1)                     # (B*T, F) grad
        B = int(batch["domain_label"].shape[0])
        T = int(seq_len)
        dl = batch["domain_label"].to(z.device).long()
        dl_rep = dl.repeat_interleave(T)                  # (B*T,) b-slow t-fast

        # Action behaviour = EEF POSITION trajectory, unified to the cam_high
        # frame. Robot EEF (domain_label==1) is world-frame -> transform with the
        # static extrinsic; human EEF (==0) is already cam-frame. Position-only
        # (no rotation) keeps the transform robust and is the dominant behaviour
        # signal for DTW pseudo-pairing.
        raw_act = batch["action"].to(z.device).float()   # (B, T, 8) raw, native frame
        pos = raw_act[..., :3].clone()                    # (B, T, 3)
        robot_seq = dl == 1                               # (B,)
        if bool(robot_seq.any()):
            Tcw = self._get_T_cam_world(z.device)         # (4,4)
            Rt, tt = Tcw[:3, :3], Tcw[:3, 3]
            pos[robot_seq] = pos[robot_seq] @ Rt.t() + tt
        # per-frame trajectory = its sequence's position trajectory; mean pos.
        traj = pos[:, None].expand(B, T, T, 3).reshape(B * T, T, 3)
        a_mean = pos.mean(dim=1).repeat_interleave(T, dim=0)  # (B*T, 3)

        use_dtw = bool(oc.get("use_dtw", True))
        min_bank = int(oc.get("min_bank", 16))
        total = None
        n_terms = 0
        for d in (0, 1):
            m = dl_rep == d
            if int(m.sum()) < 1:
                continue
            bank = self._ot_bank[1 - d]
            if bank is None or bank[0].shape[0] < min_bank:
                continue
            loss_d = ot_align_loss(
                f[m], bank[0],
                traj_cur=traj[m] if use_dtw else None,
                traj_bank=bank[1] if use_dtw else None,
                a_cur=a_mean[m], a_bank=bank[2],
                eps=float(oc.get("eps", 0.1)),
                n_iters=int(oc.get("n_iters", 50)),
                lam=float(oc.get("lam", 0.1)),
                action_weight=float(oc.get("action_weight", 1.0)),
            )
            total = loss_d if total is None else total + loss_d
            n_terms += 1

        # Update banks with detached current features (after computing the loss).
        cap = int(oc.get("bank_cap", 128))
        for d in (0, 1):
            m = dl_rep == d
            if int(m.sum()) > 0:
                self._ot_bank_push(
                    d, f[m].detach(), traj[m].detach(), a_mean[m].detach(), cap,
                )

        alpha = self._ot_alpha_now()
        self.log("training/ot_alpha", torch.tensor(float(alpha), device=z.device))
        if total is None or n_terms == 0:
            return None
        ot_mean = total / n_terms
        self.log("training/ot_loss", ot_mean)
        return alpha * ot_mean

    def encoder_forward(
        self,
        obs: torch.Tensor,
        return_split: bool = False,
    ):
        """Forward pass of the encoder.

        Args:
            obs: (B, C, H, W) where C = 3 * num_views.
            return_split: when True AND latent_decompose is active, also
                returns (z_task_list, z_emb_list) — lists of V tensors of
                shape (B, d_task, gh, gw) / (B, d_emb, gh, gw) in
                view-outer order. When False (default) the signature is
                unchanged: returns the normalised spatial latent z only.

        Returns:
            z: (B, C_latent, H_latent, W_latent)
            (z_task_list, z_emb_list): only if `return_split=True` and
                                       `self.use_latent_decompose` is True.
        """
        assert (
            len(obs.shape) == 4
        ), f"Expected obs to have shape (B, C, H, W) but got {obs.shape}"

        z_task_list: list[torch.Tensor] = []
        z_emb_list:  list[torch.Tensor] = []
        emit_split = bool(return_split and self.use_latent_decompose)
        emb_vecs: list[torch.Tensor] = []  # emb_film: per-view appearance codes

        if self.use_resnet_encoder:
            num_views = len(self.obs_keys)
            z_views = []
            for v in range(num_views):
                view_obs = obs[:, v * 3 : (v + 1) * 3]  # (B, 3, H, W)
                if self.use_vit_encoder:
                    if self.use_latent_decompose and self.decompose_method == "channel_split":
                        z_task, z_emb, cls_token = self.split_encoder(view_obs)
                        spatial_feat = (
                            self.split_encoder.concat(z_task, z_emb)
                        )
                        if emit_split:
                            z_task_list.append(z_task)
                            z_emb_list.append(z_emb)
                    elif self.use_dual_head:
                        # Decoder path is unchanged (uses the raw ViT feature);
                        # dual_head produces pooled z_task/z_emb as SIDE outputs
                        # consumed only by the alignment losses.
                        spatial_feat, cls_token = self.vit_encoder(view_obs)
                        if emit_split:
                            z_task, z_emb = self.dual_head(spatial_feat)
                            z_task_list.append(z_task)
                            z_emb_list.append(z_emb)
                    else:
                        spatial_feat, cls_token = self.vit_encoder(view_obs)
                    if self.use_dynamo_ssl and self.detach_rec_from_encoder:
                        spatial = self.spatial_proj(
                            spatial_feat.detach(), cls_token.detach()
                        )
                    else:
                        spatial = self.spatial_proj(spatial_feat, cls_token)
                    if self.use_emb_film:
                        emb_vecs.append(self.emb_head(cls_token))  # (B, c_emb)
                else:
                    resnet_feat = self.resnet_encoder(view_obs)
                    if self.use_dynamo_ssl and self.detach_rec_from_encoder:
                        spatial = self.spatial_proj(resnet_feat.detach())
                    else:
                        spatial = self.spatial_proj(resnet_feat)
                z_views.append(spatial)
            z = torch.cat(z_views, dim=1)
        else:
            z = self.encoder(obs)

        num_views = len(self.obs_keys)
        c_per_v = z.shape[1] // num_views
        for i in range(num_views):
            z_chunk = z[:, i * c_per_v : (i + 1) * c_per_v].clone()
            z[:, i * c_per_v : (i + 1) * c_per_v] = z_chunk / (
                torch.norm(z_chunk, dim=(1), keepdim=True) + 1e-8
            )

        # emb_film: append the (un-normalised) global appearance code as constant
        # spatial channels AFTER per-view L2 normalisation of the scene latent,
        # so the appended code is not folded into the scene norm. The decoder's
        # control_net was built with num_cond_channel = C_scene + c_emb_total.
        if self.use_emb_film:
            from interactive_world_sim.algorithms.latent_decompose.emb_film import (
                broadcast_emb,
            )
            z_emb = torch.cat(emb_vecs, dim=1)  # (B, c_emb_total)
            z = torch.cat([z, broadcast_emb(z_emb, z.shape[2], z.shape[3])], dim=1)

        if emit_split:
            return z, z_task_list, z_emb_list
        return z

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Save EMA beta state for resume support."""
        if self.use_dynamo_ssl and self.dynamo_ssl.ema_beta is not None:
            checkpoint["dynamo_ssl_ema_beta_current"] = (
                self.dynamo_ssl.ema_encoder.beta
            )

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Restore EMA beta state on resume, with backward compatibility."""
        if self.use_dynamo_ssl:
            # Restore EMA beta
            if (
                self.dynamo_ssl.ema_beta is not None
                and checkpoint.get("dynamo_ssl_ema_beta_current") is not None
            ):
                self.dynamo_ssl.ema_encoder.beta = checkpoint[
                    "dynamo_ssl_ema_beta_current"
                ]
                self.dynamo_ssl.ema_vector_head.beta = checkpoint[
                    "dynamo_ssl_ema_beta_current"
                ]

            # Handle optimizer state mismatch from old ckpts (e.g. 3 optimizers → 1)
            if self.use_dynamo_ssl:
                expected_groups = 6 if self.dynamo_ssl.use_sparse_idm else 5
            else:
                expected_groups = 2
            if "optimizer_states" in checkpoint and len(checkpoint["optimizer_states"]) > 0:
                old_opt_states = checkpoint["optimizer_states"]

                if len(old_opt_states) != 1:
                    # Old ckpt had multiple optimizers → merge into one
                    # Keep the first optimizer's state (main), discard others
                    print(
                        f"[Resume] Migrating from {len(old_opt_states)} optimizer(s) to 1. "
                        f"Keeping main optimizer state, SSL optimizers reset."
                    )
                    merged = old_opt_states[0]
                    old_n = len(merged["param_groups"])
                    # Add empty param groups for new SSL groups
                    for _ in range(expected_groups - old_n):
                        new_group = dict(merged["param_groups"][0])
                        new_group["params"] = []
                        merged["param_groups"].append(new_group)
                    checkpoint["optimizer_states"] = [merged]

                elif len(old_opt_states[0]["param_groups"]) != expected_groups:
                    # Same single optimizer but different param group count
                    old_n = len(old_opt_states[0]["param_groups"])
                    print(
                        f"[Resume] Migrating param groups: {old_n} → {expected_groups}. "
                        f"Existing groups preserved, new groups initialized fresh."
                    )
                    merged = old_opt_states[0]
                    if old_n < expected_groups:
                        for _ in range(expected_groups - old_n):
                            new_group = dict(merged["param_groups"][0])
                            new_group["params"] = []
                            merged["param_groups"].append(new_group)
                    else:
                        merged["param_groups"] = merged["param_groups"][:expected_groups]
                    checkpoint["optimizer_states"] = [merged]

            # LR scheduler: always reset to avoid mismatch
            if "lr_schedulers" in checkpoint:
                checkpoint["lr_schedulers"] = []

    def on_train_epoch_start(self) -> None:
        """Called at the beginning of each training epoch."""
        pass

    def optimizer_step(
        self,
        epoch: dict,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Callable,
    ) -> None:
        """Override the optimizer step to manually warm up the learning rate"""
        if self.use_dynamo_ssl or self.use_dual_head:
            # Manual optimization handles its own optimizer steps in training_step
            return
        # update params
        optimizer.step(closure=optimizer_closure)
        if self.training_stage == 2:
            for name, param in self.dynamics.named_parameters():
                if (
                    param.requires_grad
                    and (param.grad is not None)
                    and torch.isnan(param.grad).any()
                ):
                    print(f"NaN in gradient of {name}")
                    print(f"Parameter: {name}, Gradient: {param.grad}")
                    print(f"Parameter: {name}, Value: {param.data}")
                    print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
                    print(f"Parameter: {name}, Shape: {param.shape}")
                    print(f"Parameter: {name}, Device: {param.device}")
                    print(f"Parameter: {name}, Type: {param.dtype}")
                    print(f"Parameter: {name}, Isnan: {torch.isnan(param).any()}")
                    print(f"Parameter: {name}, Isinf: {torch.isinf(param).any()}")
                    exit()

    # ========= forward  ============
    def _forward(
        self,
        model: Any,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        stop_time: torch.Tensor,
        external_cond: Any = None,
        clamp: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the model"""
        assert (timestep >= stop_time).all()
        assert (timestep[-1] > stop_time[-1]).all()
        denoise = lambda x, t, s: model(x, t, s, external_cond=external_cond)
        return self.noise_scheduler.CTM_calc_out(
            denoise, sample, timestep, stop_time, clamp=clamp
        )

    # ========= inference  ============
    @torch.no_grad()
    def dynamics_forward(self, z_0: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """dynamics forward pass"""
        z_0 = rearrange(z_0, "b t c h w -> t b c h w")  # (T_hist, B, C, H, W)
        action = rearrange(action, "b t c -> t b c")  # (T_hist + T_act, B, A)
        T_hist = z_0.shape[0]
        T_act = action.shape[0] - T_hist
        chunk_size = 1
        curr_end = T_hist + chunk_size
        total_frames = T_hist + T_act
        xs_pred = z_0.clone()
        batch_size = z_0.shape[1]

        # pbar = tqdm(total=total_frames, initial=curr_end, desc="Sampling")
        while curr_end <= total_frames:
            horizon = chunk_size

            chunk = torch.randn(
                (horizon, batch_size, *z_0.shape[2:]),
                device=self.device,
                dtype=self.dtype,
            )
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)
            xs_pred = torch.cat([xs_pred, chunk], 0)

            # sliding window: only input the last n_tokens frames
            curr_start = max(0, curr_end - self.n_tokens)

            # pbar.set_postfix(
            #     {
            #         "start": curr_start,
            #         "end": curr_end,
            #     }
            # )

            clean_t = (
                torch.ones((xs_pred[curr_start:].shape[0] - 1,), device=self.device)
                * self.noise_scheduler.stabilization_level
            )
            timesteps = torch.linspace(
                self.noise_scheduler.timesteps - 1,
                0,
                self.dyn_infer_steps + 1,
                device=z_0.device,
            )
            action_chunk = action[curr_start:curr_end]
            if self.mask_prev_action:
                action_chunk[:-1] = 0

            for step_i in range(self.dyn_infer_steps):
                t = timesteps[step_i].unsqueeze(0)
                s = timesteps[step_i + 1].unsqueeze(0)
                t = torch.cat([clean_t, t], 0)
                t = torch.tile(t[:, None], (1, xs_pred.shape[1]))
                s = torch.cat([clean_t, s], 0)
                s = torch.tile(s[:, None], (1, xs_pred.shape[1]))
                t = t.long()
                s = s.long()
                xs_pred_updated = self._forward(
                    self.dynamics,
                    xs_pred[curr_start:],
                    t,
                    s,
                    external_cond=action_chunk,
                )  # clamp at inference time
                if self.last_frame_loss_only:
                    xs_pred[-1:] = xs_pred_updated[-1:]
                else:
                    xs_pred[curr_start:] = xs_pred_updated

            curr_end += horizon
            # pbar.update(horizon)

        # normalization
        num_views = len(self.obs_keys)
        c_per_v = xs_pred.shape[2] // num_views
        for i in range(num_views):
            xs_pred_chunk = xs_pred[:, :, i * c_per_v : (i + 1) * c_per_v].clone()
            xs_pred[:, :, i * c_per_v : (i + 1) * c_per_v] = xs_pred_chunk / (
                torch.norm(xs_pred_chunk, dim=(2), keepdim=True) + 1e-8
            )
        xs_pred = rearrange(xs_pred[T_hist:], "t b c h w -> b t c h w")
        return xs_pred

    def validation_step(
        self, batch: dict, batch_idx: int, namespace: str = "validation"
    ) -> STEP_OUTPUT:
        """Validation step of the model"""
        # compute diffusion loss
        # (B, T, C, H, W)
        obs_ls = [self.normalizer[k].normalize(batch["obs"][k]) for k in self.obs_keys]
        obs = torch.cat(obs_ls, dim=2)
        action = self.normalizer["action"].normalize(batch["action"])  # (B, T, A)

        obs = obs.float()
        action = action.float()

        # compute gt latent
        xs = obs
        xs = rearrange(xs, "b t c h w -> (b t) c h w")
        z_gt = self.encoder_forward(xs)
        z_gt = rearrange(z_gt, "(b t) c h w -> b t c h w", b=obs.shape[0])

        if self.training_stage in [1]:
            # compute predicted latent
            z_seq = z_gt
        elif self.training_stage in [2]:
            # compute predicted latent
            z_0 = z_gt[:, 0]
            z_seq_ls = []
            z_last = z_0.clone()
            horizon = z_gt.shape[1]

            for i in range(1, action.shape[1], horizon):
                action_chunk = action[:, i : i + horizon]  # (B, horizon, A)
                init_action_size = action_chunk.shape[1]
                if init_action_size < horizon:
                    # pad the last action to match the horizon
                    action_chunk = F.pad(
                        action_chunk,
                        (0, 0, 0, horizon - action_chunk.shape[1]),
                        mode="replicate",
                    )
                z_seq = self.dynamics_forward(
                    z_last[:, None],
                    action_chunk,
                )  # (B, T, latent_dim)
                z_seq = z_seq[:, :init_action_size]
                z_seq_ls.append(z_seq)
                z_last = z_seq[:, -1].clone()
            z_seq = torch.cat(z_seq_ls, 1)
            z_seq = torch.cat([z_0.unsqueeze(1), z_seq], 1)  # (B, T, latent_dim)
            val_loss = F.mse_loss(z_seq, z_gt, reduction="none")  # (B, T, latent_dim)
            if torch.isnan(val_loss).any():
                print("NaN in val_loss")
            val_loss = val_loss[:, 1:].mean()
            self.log(f"{namespace}/dyn_loss", val_loss)
            if "dyn_loss" not in self.validation_metrics:
                self.validation_metrics["dyn_loss"] = []
            self.validation_metrics["dyn_loss"].append(val_loss)
        else:
            z_seq = z_gt
        z_seq = rearrange(z_seq, "b t c h w -> (b t) c h w")

        # render images
        if self.val_render:
            xs_pred = render_img_cm(
                self, z_seq, xs.shape[-1], self.normalizer, num_views=self.num_views
            )
            xs_pred = rearrange(xs_pred, "(b t) c h w -> t b c h w", b=obs.shape[0])
            xs = torch.cat([batch["obs"][k] for k in self.obs_keys], dim=2)
            xs = rearrange(xs, "b t c h w -> t b c h w", b=obs.shape[0])
            xs_pred = xs_pred.detach().cpu()
            xs = xs.detach().cpu()
            if len(self.validation_step_outputs) < self.max_val_render_batches:
                self.validation_step_outputs.append((xs_pred, xs))
        return

    # ========= training  ============
    def _generate_ctm_noise_levels(self, xs: torch.Tensor) -> torch.Tensor:
        """Generate noise levels for training."""
        num_frames, batch_size, *_ = xs.shape
        min_t = self.noise_scheduler.stabilization_level + 1
        last_t = torch.randint(100 + min_t + 2, self.timesteps, (batch_size,))
        last_s = torch.cat(
            [torch.randint(min_t, int(t_i.item()) - 1, (1,)) for t_i in last_t]
        )
        last_u_ls = []
        for s_i, t_i in zip(last_s, last_t, strict=False):
            min_u = max(s_i.item() + 1, int(t_i.item() - 100))
            last_u_ls.append(torch.randint(min_u, int(t_i.item()), (1,)))
        last_u = torch.cat(last_u_ls)
        last_t = last_t.unsqueeze(0).to(xs.device)
        last_s = last_s.unsqueeze(0).to(xs.device)
        last_u = last_u.unsqueeze(0).to(xs.device)

        prev_noise_levels = torch.randint(
            min_t,
            int(self.timesteps * 0.1),
            (num_frames - 1, batch_size),
            device=xs.device,
        )
        t = torch.cat([prev_noise_levels, last_t], 0)
        s = torch.cat([prev_noise_levels, last_s], 0)
        u = torch.cat([prev_noise_levels, last_u], 0)

        return t, s, u

    def _generate_noise_levels(
        self, xs: torch.Tensor, cm_steps: int = -1
    ) -> torch.Tensor:
        """Generate noise levels for training."""
        num_frames, batch_size, *_ = xs.shape

        if self.sampling_strategy == "uniform":
            last_t = torch.randint(2, self.timesteps, (batch_size,))
            last_s = torch.cat(
                [torch.randint(1, int(t_i.item()), (1,)) for t_i in last_t]
            )
            last_t = last_t.unsqueeze(0).to(xs.device)
            last_s = last_s.unsqueeze(0).to(xs.device)
        elif self.sampling_strategy == "terminal_only":
            last_t = torch.ones((batch_size,)) * (self.timesteps - 1)
            last_t = last_t.unsqueeze(0).to(xs.device)
            if cm_steps == 1:
                last_s = torch.zeros((batch_size,))
                last_s = last_s.unsqueeze(0).to(xs.device)
            else:
                intermediate_s = np.linspace(
                    0, self.timesteps - 1, cm_steps + 1, dtype=int
                )
                s_val = np.random.choice(intermediate_s[1:-1], size=(batch_size,))
                last_s = torch.ones((batch_size,)) * s_val
                last_s = last_s.unsqueeze(0).to(xs.device)

        prev_noise_levels = torch.randint(
            1,
            int(self.timesteps * self.prev_frame_noise_scale),
            (num_frames - 1, batch_size),
            device=xs.device,
        )
        t = torch.cat([prev_noise_levels, last_t], 0)
        s = torch.cat([prev_noise_levels, last_s], 0)

        return t.long(), s.long()

    def training_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        """Training step of the model"""
        if batch["obs"][self.obs_keys[0]].shape[0] == 0:
            return None
        # normalize input
        if batch_idx % 1000 == 0:
            current_snapshot = tracemalloc.take_snapshot()
            top_stats = current_snapshot.compare_to(self.tracemalloc_snapshot, "lineno")

            print(f"\n[ Top 10 memory diff from start to step {batch_idx} ]")
            for stat in top_stats[:10]:
                print(stat)
        assert "valid_mask" not in batch
        obs_ls = [self.normalizer[k].normalize(batch["obs"][k]) for k in self.obs_keys]
        obs = torch.cat(obs_ls, dim=2)
        action = self.normalizer["action"].normalize(batch["action"])  # (B, T, A)

        obs = obs.float()
        action = action.float()

        xs = obs  # (B, T, C, H, W)
        xs = rearrange(xs, "b t c h w -> (b t) c h w")

        output_dict = {}

        # generate impainting mask

        if self.training_stage == 1:
            # stage 1: train encoder and decoder
            # When using DynaMo SSL, run encoder once and cache features
            if self.use_dynamo_ssl:
                batch_size = obs.shape[0]
                seq_len = obs.shape[1]
                z_views = []
                obs_vectors = []
                obs_per_view_imgs = []
                for v in range(self.num_views):
                    view_obs = xs[:, v * 3 : (v + 1) * 3]  # (B*T, 3, H, W)
                    obs_per_view_imgs.append(view_obs)
                    if self.use_vit_encoder:
                        spatial_feat, cls_token = self.vit_encoder(view_obs)
                        vec = self.vector_head(cls_token)
                        vec = rearrange(vec, "(b t) d -> b t d", b=batch_size)
                        obs_vectors.append(vec)
                        if self.detach_rec_from_encoder:
                            spatial = self.spatial_proj(
                                spatial_feat.detach(), cls_token.detach()
                            )
                        else:
                            spatial = self.spatial_proj(spatial_feat, cls_token)
                    else:
                        resnet_feat = self.resnet_encoder(view_obs)
                        vec = self.vector_head(resnet_feat)
                        vec = rearrange(vec, "(b t) d -> b t d", b=batch_size)
                        obs_vectors.append(vec)
                        feat_for_decoder = resnet_feat.detach() if self.detach_rec_from_encoder else resnet_feat
                        spatial = self.spatial_proj(feat_for_decoder)
                    z_views.append(spatial)
                z = torch.cat(z_views, dim=1)  # (B*T, C_latent, H_lat, W_lat)
                # Per-view normalization
                num_views = len(self.obs_keys)
                c_per_v = z.shape[1] // num_views
                for i in range(num_views):
                    z_chunk = z[:, i * c_per_v : (i + 1) * c_per_v].clone()
                    z[:, i * c_per_v : (i + 1) * c_per_v] = z_chunk / (
                        torch.norm(z_chunk, dim=(1), keepdim=True) + 1e-8
                    )
                obs_enc = torch.stack(obs_vectors, dim=2)  # (B, T, V, feature_dim)
            else:
                # Single ViT pass: when latent_decompose is on, ask for the
                # per-view split tensors here so we don't run the encoder
                # twice per step (the rec_loss path AND the L_dom/L_adv path
                # used to do their own encoder forwards — that's a ~25%
                # throughput hit on a 200k-step Stage 1 run).
                if self.use_latent_decompose:
                    z, z_task_list, z_emb_list = self.encoder_forward(
                        xs, return_split=True,
                    )
                else:
                    z = self.encoder_forward(xs)  # (B*T, C, H, W)
                    z_task_list = z_emb_list = None

            if self.robust_latent:
                z += torch.randn_like(z) * 0.02

            t, s = self._generate_noise_levels(xs[None], self.dec_infer_steps)  # (1, B)
            weights_t = self.noise_scheduler.get_weights(t)[0]  # (1, B)
            weights_s = self.noise_scheduler.get_weights(s)[0]  # (1, B)
            noisy_xs_t, noisy_xs_s = self.noise_scheduler.add_noise_to_t_s(
                xs[None], t, s
            )  # (1, B, C, H, W)
            noisy_xs_t = noisy_xs_t.squeeze(0)  # (B, C, H, W)
            noisy_xs_s = noisy_xs_s.squeeze(0)  # (B, C, H, W)
            t = t.squeeze(0)  # (B)
            s = s.squeeze(0)  # (B)

            u = torch.zeros_like(t).to(self.device)
            pred_s = self._forward(
                self.decoder,
                noisy_xs_t,
                t,
                s,
                external_cond=z,
            )
            if self.dec_infer_steps > 1:
                pred_u = self._forward(
                    self.decoder,
                    noisy_xs_s,
                    s,
                    u,
                    external_cond=z,
                )

            if self.last_frame_loss_only:
                loss_s = F.mse_loss(
                    pred_s[-1:], noisy_xs_s[-1:].detach(), reduction="none"
                )
                weights_t = weights_t.view(
                    *weights_t.shape, *((1,) * (loss_s.ndim - 2))
                )[-1:]
                loss_s = loss_s * weights_t
                if self.dec_infer_steps > 1:
                    loss_u = F.mse_loss(pred_u[-1:], xs[-1:].detach(), reduction="none")
                    weights_s = weights_s.view(
                        *weights_s.shape, *((1,) * (loss_u.ndim - 2))
                    )[-1:]
                    loss_u = loss_u * weights_s
                    loss = loss_s + loss_u
                else:
                    loss = loss_s
                loss = loss.mean()
            else:
                loss_s = F.mse_loss(pred_s, noisy_xs_s.detach(), reduction="none")
                weights_t = weights_t.view(
                    *weights_t.shape, *((1,) * (loss_s.ndim - 1))
                )
                loss_s = loss_s * weights_t
                if self.dec_infer_steps > 1:
                    loss_u = F.mse_loss(pred_u, xs.detach(), reduction="none")
                    weights_s = weights_s.view(
                        *weights_s.shape, *((1,) * (loss_s.ndim - 1))
                    )
                    loss_u = loss_u * weights_s
                    loss = loss_s + loss_u
                else:
                    loss = loss_s
                loss = loss.mean()

            rec_loss = loss

            if self.use_dynamo_ssl:
                online_encoder = self.vit_encoder if self.use_vit_encoder else self.resnet_encoder
                obs_target = self.dynamo_ssl.get_ema_target(
                    obs_per_view_imgs,
                    online_encoder,
                    self.vector_head,
                    batch_size,
                    seq_len,
                )

                # Compute SSL loss
                ssl_loss, ssl_components = self.dynamo_ssl.compute_ssl_loss(
                    obs_enc, obs_target
                )

                total_loss = rec_loss + self.ssl_loss_coef * ssl_loss

                # Manual optimization with single optimizer
                opt = self.optimizers()
                lr_sched = self.lr_schedulers()

                opt.zero_grad()
                self.manual_backward(total_loss)

                if self.cfg.get("gradient_clip_val", None):
                    self.clip_gradients(
                        opt, gradient_clip_val=self.cfg.gradient_clip_val
                    )

                opt.step()

                if lr_sched is not None:
                    lr_sched.step()

                # Adjust EMA beta by step-based cosine schedule, then update EMA
                self.dynamo_ssl.adjust_beta(
                    self.global_step, self.trainer.max_steps
                )
                ema_encoder = self.vit_encoder if self.use_vit_encoder else self.resnet_encoder
                self.dynamo_ssl.update_ema(ema_encoder, self.vector_head)

                # Logging
                self.log("training/rec_loss", rec_loss)
                self.log("training/ssl_loss", ssl_loss)
                self.log("training/loss", total_loss)
                for k, v in ssl_components.items():
                    self.log(f"training/ssl_{k}", v)
                return None  # manual optimization
            elif self.use_dual_head:
                total_loss = self._dual_head_manual_step(
                    batch, rec_loss, z_task_list, z_emb_list,
                )
                return None  # manual optimization
            else:
                if self.use_latent_decompose:
                    total_loss = self._apply_decompose_losses(
                        batch, rec_loss, z_task_list, z_emb_list,
                    )
                    log_loss = total_loss
                elif self.use_emb_film:
                    # Model objective = reconstruction (+ OT alignment in 0b).
                    model_loss = rec_loss
                    if self.use_ot_align:
                        ot = self._apply_ot_align(z, action, batch, obs.shape[1])
                        if ot is not None:
                            model_loss = model_loss + ot
                    # Diagnostic probes are added to the optimised loss so the
                    # probe params train, but they are EXCLUDED from the reported
                    # loss and from the model's gradient clip (see
                    # configure_gradient_clipping) so they don't dilute training.
                    probe_loss = self._emb_film_probe_loss(batch, z, obs.shape[1])
                    total_loss = model_loss + probe_loss
                    log_loss = model_loss   # report the clean model objective
                elif getattr(self, "use_mask_subtract", False):
                    z_scene, aux = self._mask_subtract_losses(batch, z, obs.shape[1])
                    model_loss = rec_loss
                    # mask-exterior scene reconstruction from z_scene
                    # (sufficiency / anti-collapse). Second decoder forward.
                    lam_sr = float(self.cfg.latent_decompose.get("lambda_scene_rec", 1.0))
                    if lam_sr > 0:
                        pred_scene = self._forward(
                            self.decoder, noisy_xs_t, t, s, external_cond=z_scene,
                        )
                        m = batch["agent_mask"].to(z.device).float()
                        Bb, Tt = m.shape[0], m.shape[1]
                        m = m.reshape(Bb * Tt, 1, m.shape[-2], m.shape[-1])
                        m_img = F.interpolate(m, size=pred_scene.shape[-2:], mode="area")
                        ext = (m_img <= 0.5).float()             # mask EXTERIOR
                        denom = (ext.sum() * pred_scene.shape[1]).clamp_min(1.0)
                        L_scene = ((((pred_scene - noisy_xs_s.detach()) ** 2) * ext).sum()
                                   / denom)
                        self.log("training/L_scene_rec", L_scene)
                        model_loss = model_loss + lam_sr * L_scene
                    total_loss = model_loss + aux
                    log_loss = model_loss
                else:
                    total_loss = rec_loss
                    log_loss = total_loss
                self.log("training/rec_loss", rec_loss)
                self.log("training/loss", log_loss)
                return {"loss": total_loss}
        elif self.training_stage == 2:
            # stage 2: train dynamics
            with torch.no_grad():
                z = self.encoder_forward(xs)  # (B*T, C, H, W)
            z = rearrange(z, "(b t) c h w -> t b c h w", b=obs.shape[0])
            action = rearrange(action, "b t a -> t b a")

            t, s = self._generate_noise_levels(z, self.dyn_infer_steps)
            weights_t = self.noise_scheduler.get_weights(t)
            weights_s = self.noise_scheduler.get_weights(s)
            noisy_z_t, noisy_z_s = self.noise_scheduler.add_noise_to_t_s(z, t, s)

            u = torch.zeros_like(t).to(self.device)
            if self.mask_prev_action:
                action[:-1] = 0
            pred_s = self._forward(
                self.dynamics,
                noisy_z_t,
                t,
                s,
                external_cond=action,
            )
            if self.dyn_infer_steps > 1:
                pred_u = self._forward(
                    self.dynamics,
                    noisy_z_s,
                    s,
                    u,
                    external_cond=action,
                )

            if self.last_frame_loss_only:
                loss_s = F.mse_loss(
                    pred_s[-1:], noisy_z_s[-1:].detach(), reduction="none"
                )
                weights_t = weights_t.view(
                    *weights_t.shape, *((1,) * (loss_s.ndim - 2))
                )[-1:]
                loss_s = loss_s * weights_t
                if self.dyn_infer_steps > 1:
                    loss_u = F.mse_loss(pred_u[-1:], z[-1:].detach(), reduction="none")
                    weights_s = weights_s.view(
                        *weights_s.shape, *((1,) * (loss_u.ndim - 2))
                    )[-1:]
                    loss_u = loss_u * weights_s
                    loss = loss_s + loss_u
                else:
                    loss = loss_s
                loss = loss.mean()
            else:
                loss_s = F.mse_loss(pred_s, noisy_z_s.detach(), reduction="none")
                weights_t = weights_t.view(
                    *weights_t.shape, *((1,) * (loss_s.ndim - 2))
                )
                loss_s = loss_s * weights_t
                if self.dyn_infer_steps > 1:
                    loss_u = F.mse_loss(pred_u, z.detach(), reduction="none")
                    weights_s = weights_s.view(
                        *weights_s.shape, *((1,) * (loss_s.ndim - 2))
                    )
                    loss_u = loss_u * weights_s
                    loss = loss_s + loss_u
                else:
                    loss = loss_s
                loss = loss.mean()

            output_dict["loss"] = loss

            self.log("training/loss", output_dict["loss"])
            for key in output_dict.keys():
                self.log(f"training/{key}", output_dict[key])
        elif self.training_stage == 3:
            with torch.no_grad():
                z = self.encoder_forward(xs)  # (B*T, C, H, W)
                z += torch.randn_like(z) * 0.02

            t, s = self._generate_noise_levels(xs[None], self.dec_infer_steps)  # (1, B)
            weights_t = self.noise_scheduler.get_weights(t)[0]  # (1, B)
            weights_s = self.noise_scheduler.get_weights(s)[0]  # (1, B)
            noisy_xs_t, noisy_xs_s = self.noise_scheduler.add_noise_to_t_s(
                xs[None], t, s
            )  # (1, B, C, H, W)
            noisy_xs_t = noisy_xs_t.squeeze(0)  # (B, C, H, W)
            noisy_xs_s = noisy_xs_s.squeeze(0)  # (B, C, H, W)
            t = t.squeeze(0)  # (B)
            s = s.squeeze(0)  # (B)

            u = torch.zeros_like(t).to(self.device)
            pred_s = self._forward(
                self.decoder,
                noisy_xs_t,
                t,
                s,
                external_cond=z,
            )
            if self.dec_infer_steps > 1:
                pred_u = self._forward(
                    self.decoder,
                    noisy_xs_s,
                    s,
                    u,
                    external_cond=z,
                )

            if self.last_frame_loss_only:
                loss_s = F.mse_loss(
                    pred_s[-1:], noisy_xs_s[-1:].detach(), reduction="none"
                )
                weights_t = weights_t.view(
                    *weights_t.shape, *((1,) * (loss_s.ndim - 2))
                )[-1:]
                loss_s = loss_s * weights_t
                if self.dec_infer_steps > 1:
                    loss_u = F.mse_loss(pred_u[-1:], xs[-1:].detach(), reduction="none")
                    weights_s = weights_s.view(
                        *weights_s.shape, *((1,) * (loss_u.ndim - 2))
                    )[-1:]
                    loss_u = loss_u * weights_s
                    loss = loss_s + loss_u
                else:
                    loss = loss_s
                loss = loss.mean()
            else:
                loss_s = F.mse_loss(pred_s, noisy_xs_s.detach(), reduction="none")
                weights_t = weights_t.view(
                    *weights_t.shape, *((1,) * (loss_s.ndim - 1))
                )
                loss_s = loss_s * weights_t
                if self.dec_infer_steps > 1:
                    loss_u = F.mse_loss(pred_u, xs.detach(), reduction="none")
                    weights_s = weights_s.view(
                        *weights_s.shape, *((1,) * (loss_s.ndim - 1))
                    )
                    loss_u = loss_u * weights_s
                    loss = loss_s + loss_u
                else:
                    loss = loss_s
                loss = loss.mean()

            self.log("training/rec_loss", loss)
            output_dict = {
                "loss": loss,
            }
            return output_dict
        return output_dict

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """Test step of the model"""
        return self.validation_step(*args, **kwargs, namespace="test")  # type: ignore

    def on_test_epoch_end(self) -> None:
        """Operations when the test epoch ends"""
        self.on_validation_epoch_end(namespace="test")

    def on_validation_epoch_end(self, namespace: str = "validation") -> None:
        """Operations when the validation epoch ends"""
        if not self.validation_step_outputs:
            return
        xs_pred_ls = []
        xs_ls = []
        for pred, gt in self.validation_step_outputs:
            xs_pred_ls.append(pred)
            xs_ls.append(gt)
        xs_pred = torch.cat(xs_pred_ls, 1)
        xs = torch.cat(xs_ls, 1)

        if self.logger:
            log_video(
                xs_pred,
                xs.clone(),
                step=None if namespace == "test" else self.global_step,
                namespace=namespace + "_vis",
                context_frames=0,
                logger=self.logger.experiment,
            )

        metric_dict = get_validation_metrics_for_videos(
            xs_pred,
            xs,
            lpips_model=self.validation_lpips_model,
            fid_model=self.validation_fid_model,
            fvd_model=self.validation_fvd_model,
        )
        self.log_dict(
            {f"{namespace}/{k}": v for k, v in metric_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.validation_step_outputs.clear()

    def on_train_start(self) -> None:
        """Start tracing memory allocations"""
        tracemalloc.start()
        self.tracemalloc_snapshot = tracemalloc.take_snapshot()
