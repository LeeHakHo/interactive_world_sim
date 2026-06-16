#!/bin/bash
#SBATCH --job-name=scel_m1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=outputs/cross_embodiment_wm/scel_m1_agentframe/slurm_%j.out
mkdir -p outputs/cross_embodiment_wm/scel_m1_agentframe
source ~/.bashrc
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
python exp_scel_agentframe.py
