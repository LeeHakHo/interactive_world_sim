#!/bin/bash
#SBATCH --job-name=phase0_stage1_mixed_vit
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/phase0_stage1_mixed_vit.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=128G

# Phase 0 v3 Step 1: Stage 1 with SplitEncoder + L_dom + L_adv on mixed
# robot+human EEF data. See docs/superpowers/specs/2026-05-24-phase0-latent-decompose-design.md
# and docs/superpowers/plans/2026-05-24-phase0-latent-decompose.md.
#   - ViT-S patch=8 encoder, channel-split into z_task(288) + z_emb(96)
#   - L_dom (CE on z_emb) + L_adv (GRL + CE on z_task) added to rec loss
#   - balanced 50/50 robot/human batch (dataset.sample_ratio=0.5)
#   - DynaMo SSL OFF (replaced by latent_decompose)
#   - 12 robot + 3 human episodes, cam_high only
#   - max 300k steps (same as robot-only baseline for fair G3 comparison)

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws

cd /scr2/yusenluo/interactive_world_sim

JOBTAG="${SLURM_JOB_ID:-$(date +%s)}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  hydra.run.dir="outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}_job${JOBTAG}_phase0_stage1_mixed_vit" \
  +name=phase0_stage1_mixed_vit \
  algorithm=latent_world_model \
  experiment=phase0_step1 \
  dataset=play_mixed_eef \
  dataset.horizon=4 dataset.val_horizon=16 \
  dataset.obs_keys=[camera_0_color] \
  dataset.sample_ratio=0.5 \
  experiment.training.batch_size=2 \
  experiment.training.optim.accumulate_grad_batches=4 \
  experiment.training.max_steps=300005 \
  experiment.training.log_every_n_steps=100 \
  experiment.training.checkpointing.every_n_train_steps=20000 \
  +experiment.training.checkpointing.save_top_k=1 \
  experiment.training.data.num_workers=4 \
  experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=10000 \
  experiment.validation.data.num_workers=2 \
  algorithm.latent_dim=512 algorithm.action_dim=8 \
  algorithm.training_stage=1 \
  algorithm.dynamo_ssl.enabled=false \
  algorithm.dynamo_ssl.encoder_backbone=vit \
  algorithm.latent_decompose.enabled=true
