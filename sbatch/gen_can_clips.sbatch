#!/bin/bash
#SBATCH --job-name=gen_can_clips
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/gen_can_clips_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=128G

# CoTracker clips for the can datasets (robot side; human joins after eef build).
# SMOKE=1 -> first episode, 8 clips (QC pass before the full run).
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
ROBOT_ONLY=${ROBOT_ONLY:-1} SMOKE=${SMOKE:-0} OUTDIR=${OUTDIR:-outputs/flow_render_dataset_can} \
  python gen_flow_render_dataset_caneef.py
