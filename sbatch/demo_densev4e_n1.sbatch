#!/bin/bash
#SBATCH --job-name=demo_v4e
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/demo_v4e_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --mem=64G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
echo "=== node $(hostname) job $SLURM_JOB_ID ==="
python demo_densev4e_n1.py
echo "=== DONE ==="
