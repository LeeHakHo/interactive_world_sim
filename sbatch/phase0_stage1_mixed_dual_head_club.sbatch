#!/bin/bash
#SBATCH --job-name=phase0_dualhead_club
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/phase0_stage1_mixed_dual_head_club.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=128G

# Phase 0 v4 (dual_head + CLUB): Stage 1 on mixed robot+human EEF data.
# Replaces v3 channel-split (which collapsed reconstruction, PSNR 23.8 vs 40).
#   - Two independent projection heads (z_task / z_emb) on the shared ViT-S
#   - CLUB MI penalty pushes them apart (smooth, not GRL adversary)
#   - Decoder UNTOUCHED -> reconstruction not sacrificed
#   - balanced 50/50 robot/human (dataset.sample_ratio=0.5)
#   - DynaMo SSL OFF

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws

cd /scr2/yusenluo/interactive_world_sim

JOBTAG="${SLURM_JOB_ID:-$(date +%s)}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  hydra.run.dir="outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}_job${JOBTAG}_phase0_dualhead_club" \
  +name=phase0_dualhead_club \
  algorithm=latent_world_model \
  experiment=exp_latent_dyn \
  dataset=play_mixed_eef \
  dataset.horizon=4 dataset.val_horizon=16 \
  dataset.obs_keys=[camera_0_color] \
  dataset.sample_ratio=0.5 \
  experiment.training.batch_size=4 \
  experiment.training.optim.accumulate_grad_batches=1 \
  experiment.training.optim.gradient_clip_val=0 \
  +algorithm.gradient_clip_val=1.0 \
  experiment.training.max_steps=300005 \
  experiment.training.log_every_n_steps=100 \
  experiment.training.checkpointing.every_n_train_steps=20000 \
  experiment.training.data.num_workers=4 \
  experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=10000 \
  experiment.validation.data.num_workers=2 \
  algorithm.latent_dim=512 algorithm.action_dim=8 \
  algorithm.training_stage=1 \
  algorithm.dynamo_ssl.enabled=false \
  algorithm.dynamo_ssl.encoder_backbone=vit \
  algorithm.latent_decompose.enabled=true \
  algorithm.latent_decompose.method=dual_head
