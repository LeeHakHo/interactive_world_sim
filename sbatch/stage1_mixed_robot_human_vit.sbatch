#!/bin/bash
#SBATCH --job-name=stage1_mixed_robot_human_vit
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/stage1_mixed_robot_human_vit.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=128G

# Phase 0 task 1.4 (mixed baseline) — HUMAN_ROBOT_ALIGN_PLAN-2 §3:
#   - IWS Stage 1 trained on robot + human EEF play data, naive co-training.
#   - Pixel reconstruction loss only.
#   - Expected outcome: reproduce the encoder-disjoint failure (Plan-2 §1.4).
#   - Comparison baseline for robot-only Stage 1 (job 45678).

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws

cd /scr2/yusenluo/interactive_world_sim

JOBTAG="${SLURM_JOB_ID:-$(date +%s)}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  hydra.run.dir="outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}_job${JOBTAG}_stage1_mixed_robot_human_vit" \
  +name=stage1_mixed_robot_human_vit \
  algorithm=latent_world_model \
  experiment=exp_latent_dyn \
  dataset=play_mixed_eef \
  dataset.horizon=4 dataset.val_horizon=4 \
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
