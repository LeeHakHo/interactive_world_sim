#!/bin/bash
#SBATCH --job-name=precompute_dinov2_robot
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/precompute_dinov2_robot.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --mem=32G

# Offline precomputation of frozen DINOv2 ViT-S/14 patch features for all
# 12 robot play_robot_*_eef episodes. Output: float16 memmap (.npy + .shape.json)
# under each <dataset_dir>/_dinov2_cache/.
# OPTIONAL: align module training will work without this (it falls back to
# online DINOv2 forward passes). Run this to speed up training ~3-5x.

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws

cd /scr2/yusenluo/interactive_world_sim

python scripts/precompute_dinov2_features.py --dataset play_robot_eef --batch_size 64
