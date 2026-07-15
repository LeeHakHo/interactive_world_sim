#!/bin/bash
#SBATCH --job-name=inpaint_gen_smoke
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/inpaint_gen_smoke.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=48G
set -e
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate phantom
cd /scr2/yusenluo/interactive_world_sim
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python inpaint_gen_smoke.py
