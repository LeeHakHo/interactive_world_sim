#!/bin/bash
#SBATCH --job-name=flow3d_wm_can
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/flow3d_wm_can_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=128G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
python exp_scel_flow3d_wm.py
