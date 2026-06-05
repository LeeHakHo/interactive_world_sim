#!/bin/bash
#SBATCH --job-name=wm_sens
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/wm_sens_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --mem=48G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
echo "=== node $(hostname) job $SLURM_JOB_ID ==="
python eval_wm_sensitivity.py
echo "=== DONE ==="
