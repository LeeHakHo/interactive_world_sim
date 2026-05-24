"""LatentWorldModel constructs cleanly when latent_decompose.enabled is true
or false. Does not run a training step.

Skipped if the LWM module cannot be imported (e.g. hydra/numba missing on
this host's default python). Run in the `iws` conda env for full coverage.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf, DictConfig


def _import_lwm():
    try:
        from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
            LatentWorldModel,
        )
    except ImportError as e:
        pytest.skip(f"LatentWorldModel import failed: {e}")
    return LatentWorldModel


def _make_cfg(enabled: bool) -> DictConfig:
    """Build a minimal OmegaConf config sufficient for LatentWorldModel.__init__
    + _build_model without running a full Hydra compose (which requires
    dataset/experiment context). All interpolations are replaced with literals.
    """
    raw = {
        # --- base_algo / base_pytorch_algo ---
        "debug": False,
        "lr": 1e-4,
        # --- latent world model hyperparams ---
        "weight_decay": 1e-4,
        "warmup_steps": 10000,
        "lr_scheduler": "linear",
        "optimizer_beta": [0.9, 0.999],
        "latent_dim": 512,
        "action_dim": 10,
        "enc_dim": 64,
        "num_components": 1,
        "obs_keys": ["camera_0_color"],
        "x_shape": [3, 128, 128],
        "num_latent_downsample": 2,
        "num_views": 1,
        "num_latent_channel": 4,
        "latent_resolution": 32,
        "training_stage": 1,
        "load_ae": None,
        # dtype intentionally omitted: base_pytorch_algo defaults to torch.float32
        # when the key is absent (OmegaConf cannot hold a torch.dtype object).
        "mask_prev_action": False,
        "device": "cpu",
        "noise_level": "log_normal",
        "val_render": False,
        "max_val_render_batches": 4,
        "scheduling_matrix": "autoregressive",
        "uncertainty_scale": 1.0,
        "guidance_scale": 1.0,
        "n_frames": 4,
        "dyn_infer_steps": 1,
        "dec_infer_steps": 1,
        "last_frame_loss_only": False,
        "prev_frame_noise_scale": 0.1,
        "robust_latent": False,
        "delta": 0.01,
        "sampling_strategy": "uniform",
        "sampling_strategy_params": [],
        "norm_scale": 6.0,
        # --- metrics ---
        "metrics": [],
        # --- diffusion ---
        "diffusion": {
            "beta_schedule": "sigmoid",
            "objective": "pred_v",
            "use_fused_snr": True,
            "cum_snr_decay": 0.96,
            "clip_noise": 6.0,
            "schedule_fn_kwargs": {},
            "timesteps": 1000,
            "sampling_timesteps": 50,
            "ddim_sampling_eta": 0.0,
            "snr_clip": 5.0,
            "model_channels": 64,
            "num_latent_downsample": 2,
            "num_latent_channel": 4,
            "num_res_blocks": 2,
            "attention_resolutions": [2, 4, 8],
            "dropout": 0.1,
            "channel_mult": [1, 2, 3],
            "num_head_channels": 64,
            "resblock_updown": True,
            "use_scale_shift_norm": True,
            "num_components": 1,
            "image_size": 128,
            "stabilization_level": 15,
        },
        # --- noise_scheduler ---
        "noise_scheduler": {
            "_target_": "interactive_world_sim.utils.cm_utils.DDPMScheduler",
            "x_shape": [3, 128, 128],
            "timesteps": 1000,
            "sampling_timesteps": 50,
            "beta_schedule": "sigmoid",
            "schedule_fn_kwargs": {},
            "objective": "pred_v",
            "loss_weighting": "fused_snr",
            "snr_clip": 5.0,
            "cum_snr_decay": 0.96,
            "ddim_sampling_eta": 0.0,
            "clip_noise": 6.0,
            "stabilization_level": 15,
            # dtype omitted — DDPMScheduler defaults to torch.float32
        },
        # --- dynamics ---
        "dynamics": {
            "_target_": "interactive_world_sim.algorithms.latent_dynamics.models.cm_latent_dynamics.CMLatentDynamics",
            "action_dim": 10,
            "latent_dim": 4,
            "dim": 64,
            "action_emb_dim": 512,
            "resnet_block_groups": 8,
            "dim_mults": [1, 2],
            "attn_dim_head": 128,
            "attn_heads": 4,
            "use_linear_attn": True,
            "use_init_temporal_attn": True,
            "init_kernel_size": 5,
            "is_causal": True,
            "time_emb_type": "rotary",
            # dtype omitted — CMLatentDynamics defaults to torch.float32
        },
        # --- dynamo_ssl (encoder_backbone=vit, enabled=false) ---
        "dynamo_ssl": {
            "enabled": False,
            "encoder_backbone": "vit",
            "loss_coef": 1.0,
            "pretrained_encoder": False,
            "feature_dim": 512,
            "projection_dim": 32,
            "window_size": 4,
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 120,
            "dropout": 0.0,
            "covariance_reg_coef": 0.04,
            "dynamics_loss_coef": 1.0,
            "ema_beta": None,
            "beta_scheduling": True,
            "lr": 1e-4,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "separate_single_views": True,
            "detach_rec_from_encoder": True,
            "spatial_proj_mode": "plain",
            "use_sigreg": False,
            "sigreg_weight": 0.09,
            "sigreg_knots": 17,
            "sigreg_num_proj": 1024,
            "use_sparse_idm": False,
            "sparse_lambda": 0.01,
            "sparse_mask_init": 0.0,
            "vit": {
                "img_size": 128,
                "patch_size": 8,
                "embed_dim": 384,
                "depth": 12,
                "num_heads": 6,
                "mlp_ratio": 4.0,
                "drop_rate": 0.0,
            },
        },
        # --- latent_decompose ---
        "latent_decompose": {
            "enabled": enabled,
            "d_task": 288,
            "d_emb": 96,
            "lambda_dom": 0.1,
            "lambda_adv_schedule": {
                "type": "linear_ramp",
                "start_step": 2000,
                "end_step": 10000,
                "start_value": 0.0,
                "end_value": 0.3,
            },
            "lr_classifiers": 3e-4,
        },
    }
    return OmegaConf.create(raw)


def test_construct_disabled():
    LatentWorldModel = _import_lwm()
    cfg = _make_cfg(enabled=False)
    m = LatentWorldModel(cfg)
    assert m.use_latent_decompose is False
    assert not hasattr(m, "split_encoder")


def test_construct_enabled():
    LatentWorldModel = _import_lwm()
    cfg = _make_cfg(enabled=True)
    m = LatentWorldModel(cfg)
    assert m.use_latent_decompose is True
    assert hasattr(m, "split_encoder")
    assert hasattr(m, "clf_emb")
    assert hasattr(m, "clf_adv")
