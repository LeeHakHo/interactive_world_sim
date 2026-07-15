#!/bin/bash
#SBATCH --job-name=tracks3d_can
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/tracks3d_can_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=64G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
python augment_clips_tracks3d.py \
  outputs/flow_render_dataset_can_grip/clips_robot.npz \
  outputs/flow_render_dataset_can/clips_robot.npz \
  outputs/flow_render_dataset_can/clips_human.npz
