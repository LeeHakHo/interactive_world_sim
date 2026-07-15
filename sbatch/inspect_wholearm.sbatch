#!/bin/bash
#SBATCH --job-name=inspect_wholearm
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/inspect_wholearm.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --mem=32G
set -e
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python inspect_wholearm.py
