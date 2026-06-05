#!/bin/bash
#SBATCH --job-name=maskgen_v3
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/maskgen_v3_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=48G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
echo "=== node $(hostname) job $SLURM_JOB_ID ==="
python train_maskgen_joint2mask_v3.py
echo "=== DONE ==="
