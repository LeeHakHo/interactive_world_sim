#!/bin/bash
#SBATCH --job-name=derisk_warp
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/derisk_warp_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=64G

# Eyeball: what does a SIMPLE pixel warp by GT object flow look like? (warp ceiling)
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES job $SLURM_JOB_ID ==="
python derisk_warp.py
echo "=== DONE ==="
