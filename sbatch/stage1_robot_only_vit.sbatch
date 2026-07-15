#!/bin/bash
#SBATCH --job-name=stage1_robot_only_vit
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/stage1_robot_only_vit.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=128G

# Phase 0 task 1.6: Robot-only Stage 1 floor baseline.
#   - ViT-S patch=8 encoder (new default)
#   - DynaMo SSL + sparse mask OFF (per Plan-2 §3)
#   - 12 play_robot_{1..12}_eef episodes, cam_high only
#   - Pixel-reconstruction loss only

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws

cd /scr2/yusenluo/interactive_world_sim

# Force a unique hydra output dir per SLURM job so concurrent jobs don't
# collide on `outputs/<date>/<HH-MM-SS>/`.
JOBTAG="${SLURM_JOB_ID:-$(date +%s)}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# batch_size=2 keeps memory under 48GiB even with precision=32-true (diffusion
# needs fp32 stability) + ViT-S encoder + CMDecoder. Effective batch via
# accumulate_grad_batches if you need to bump.
python main.py \
  hydra.run.dir="outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}_job${JOBTAG}_stage1_robot_only_vit" \
  +name=stage1_robot_only_vit \
  algorithm=latent_world_model \
  experiment=exp_latent_dyn \
  dataset=play_robot_eef \
  dataset.horizon=4 dataset.val_horizon=16 \
  dataset.obs_keys=[camera_0_color] \
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
  algorithm.dynamo_ssl.encoder_backbone=vit
