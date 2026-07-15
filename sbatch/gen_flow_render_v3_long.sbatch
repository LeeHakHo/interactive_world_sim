#!/bin/bash
#SBATCH --job-name=v3long
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/v3long_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=64G
# Long-horizon ROBOT eval clips: L=48 (S=4 -> spans 192 raw frames) supports rollout H up to ~44.
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
echo "=== node $(hostname) job $SLURM_JOB_ID | long robot clips L=$CLIP_L OUT=$OUTDIR ==="
python gen_flow_render_dataset_v3eef.py
echo "=== DONE ==="
