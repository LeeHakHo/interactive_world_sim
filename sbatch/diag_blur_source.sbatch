#!/bin/bash
#SBATCH --job-name=diag_blur
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/diag_blur_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:40:00
#SBATCH --mem=64G

# Separate cube/agent blur into latent-bottleneck (ceiling) vs conditioning (fixable).
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES job $SLURM_JOB_ID ==="
python diag_blur_source.py
echo "=== DONE ==="
