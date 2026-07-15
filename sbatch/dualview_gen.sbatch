#!/bin/bash
#SBATCH --job-name=dualview_gen
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/dualview_gen_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=128G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
python gen_dualview_aligned.py
