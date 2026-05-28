"""Few-step smoke for the mask_subtract method: build LWM, run 3 training steps on
the mixed dataset, verify the losses/probes log and nothing NaNs. Run on 1 GPU
(env iws), e.g. `CUDA_VISIBLE_DEVICES=2 python smoke_mask_subtract.py`.
"""
import subprocess
import sys

CMD = [
    sys.executable, "main.py",
    "+name=smoke_mask_subtract",
    "algorithm=latent_world_model", "experiment=exp_latent_dyn",
    "dataset=play_mixed_eef",
    "dataset.horizon=4", "dataset.val_horizon=16",
    "dataset.obs_keys=[camera_0_color]", "dataset.sample_ratio=0.5",
    "experiment.training.batch_size=2",
    "experiment.training.optim.accumulate_grad_batches=1",
    "experiment.training.max_steps=3",
    "experiment.training.log_every_n_steps=1",
    "experiment.validation.val_every_n_step=1000000",
    "algorithm.latent_dim=512", "algorithm.action_dim=8",
    "algorithm.training_stage=1",
    "algorithm.dynamo_ssl.enabled=false",
    "algorithm.dynamo_ssl.encoder_backbone=vit",
    "algorithm.latent_decompose.enabled=true",
    "algorithm.latent_decompose.method=mask_subtract",
]
sys.exit(subprocess.call(CMD))
