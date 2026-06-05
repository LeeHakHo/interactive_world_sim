#!/bin/bash
#SBATCH --job-name=arm_mask_g
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/arm_mask_g_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --mem=48G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate phantom
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES job $SLURM_JOB_ID ==="
python precompute_robot_arm_mask_g.py
echo "=== DONE ==="
