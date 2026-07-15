#!/bin/bash
#SBATCH --job-name=maskgen_can
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/maskgen_can_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=64G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate phantom
cd /scr2/yusenluo/interactive_world_sim
python train_maskgen_caneef.py
