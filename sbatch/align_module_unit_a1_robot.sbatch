#!/bin/bash
#SBATCH --job-name=align_module_unit_a1_robot
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/align_module_unit_a1_robot.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --mem=64G

# Phase A1: UniT-style align module training, robot-only sanity.
#   - Frozen DINOv2 ViT-S/14 patch features (input 224×224, 256 patches × 384)
#   - 3 branches: vision (DINOv2 patches), action (8-dim EEF chunk), fusion
#   - Shared RQ-VAE codebook: K=256, depth=2, EMA + dead-code reinit
#   - Cross-recon losses on all 3 branches × {vision, action}

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws

cd /scr2/yusenluo/interactive_world_sim

# Make sure DINOv2 weights are pre-downloaded (one-time torch.hub).
mkdir -p ~/.cache/torch/hub

# Force a unique hydra output dir per SLURM job so concurrent jobs don't
# collide on `outputs/<date>/<HH-MM-SS>/`.
JOBTAG="${SLURM_JOB_ID:-$(date +%s)}"

python main.py \
  hydra.run.dir="outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}_job${JOBTAG}_align_a1_robot" \
  +name=align_module_unit_a1_robot \
  algorithm=align_module_unit \
  experiment=exp_latent_dyn \
  dataset=play_robot_eef \
  dataset.horizon=4 dataset.val_horizon=4 \
  dataset.obs_keys=[camera_0_color] \
  experiment.training.batch_size=16 \
  experiment.training.max_steps=200005 \
  experiment.training.log_every_n_steps=50 \
  experiment.training.data.num_workers=4 \
  experiment.validation.batch_size=8 \
  experiment.validation.val_every_n_step=5000 \
  experiment.validation.data.num_workers=2 \
  algorithm.action_dim=8 \
  algorithm.training_stage=1 \
  algorithm.lambda_vision=1.0 \
  algorithm.lambda_action=1.0 \
  algorithm.num_codes=256 \
  algorithm.rqvae_depth=2 \
  algorithm.default_embodiment=robot
