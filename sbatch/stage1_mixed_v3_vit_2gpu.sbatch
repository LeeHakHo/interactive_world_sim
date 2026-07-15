#!/bin/bash
#SBATCH --job-name=stage1_mixed_v3_vit_2gpu
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/stage1_mixed_v3_vit_2gpu.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=40:00:00
#SBATCH --mem=192G

# Plain IWS Stage 1 (NO latent_decompose) on the v3 pinch-aligned mixed dataset.
#   - ViT-S patch=8 encoder (encoder_backbone=vit, embed_dim=384), DynaMo SSL OFF.
#   - latent_decompose stays at its default (enabled=false) => this is the vanilla
#     IWS Stage 1, naive robot+human co-training, pixel-reconstruction loss only.
#   - dataset = play_mixed_v3_eef (18 robot + 12 human, unified 8-dim eef action,
#     robot-world frame, z-locked, gripper [0,1]). sample_ratio = config default
#     "auto" (time-proportional, ~0.6 robot). action_dim=8.
#   - 2 GPUs (DDP). Effective batch = 2 GPU x bs2 x accum4 = 16.
#
# Why srun + ntasks-per-node=2: Lightning detects SLURM and expects one task per
# device, so DDP must be launched by srun (a bare `python` would hang waiting for
# the 2nd rank). RUNDIR is pinned to ${JOBTAG}/${DATESTR} (NOT ${now:%H-%M-%S}) so
# both ranks resolve the SAME hydra run dir.

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws

cd /scr2/yusenluo/interactive_world_sim

JOBTAG="${SLURM_JOB_ID:-$(date +%s)}"
DATESTR="$(date +%Y-%m-%d)"
RUNDIR="outputs/${DATESTR}/job${JOBTAG}_stage1_mixed_v3_vit_2gpu"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python main.py \
  hydra.run.dir="${RUNDIR}" \
  +name=stage1_mixed_v3_vit_2gpu \
  algorithm=latent_world_model \
  experiment=exp_latent_dyn \
  experiment.num_devices=2 \
  dataset=play_mixed_v3_eef \
  dataset.horizon=4 dataset.val_horizon=4 \
  dataset.obs_keys=[camera_0_color] \
  experiment.training.batch_size=2 \
  experiment.training.optim.accumulate_grad_batches=4 \
  experiment.training.max_steps=300005 \
  experiment.training.log_every_n_steps=100 \
  experiment.training.checkpointing.every_n_train_steps=4000 \
  +experiment.training.checkpointing.save_top_k=2 \
  +experiment.training.checkpointing.monitor=ckpt_step \
  +experiment.training.checkpointing.mode=max \
  experiment.training.data.num_workers=4 \
  experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=10000 \
  experiment.validation.data.num_workers=2 \
  algorithm.latent_dim=512 algorithm.action_dim=8 \
  algorithm.training_stage=1 \
  algorithm.dynamo_ssl.enabled=false \
  algorithm.dynamo_ssl.encoder_backbone=vit
