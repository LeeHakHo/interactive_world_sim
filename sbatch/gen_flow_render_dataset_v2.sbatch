#!/bin/bash
#SBATCH --job-name=gen_clips2
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/gen_clips2_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=96G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES job $SLURM_JOB_ID ==="
python gen_flow_render_dataset_v2.py
echo "=== DONE ==="
