#!/bin/bash
#SBATCH --job-name=eval2d3d_low
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/eval2d3d_low_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=64G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
python eval_2d3d_camlow.py
