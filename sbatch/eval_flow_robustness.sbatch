#!/bin/bash
#SBATCH --job-name=eval_flow_robustness
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/eval_flow_robustness_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=96G

# Eval-only: scan ③ renderer robustness to PREDICTED-flow error (point-position noise),
# reusing the densev3 decoder ckpt. Answers "不确定 pred flow 会怎么样" before training ②.
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES job $SLURM_JOB_ID ==="
python eval_flow_robustness.py
echo "=== DONE ==="
