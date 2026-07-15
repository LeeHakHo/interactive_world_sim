#!/bin/bash
#SBATCH --job-name=gen_flowrender_ds
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/gen_flowrender_ds_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=96G

# Build the proper (full-data, big-time-step) flow-cond-decoder training set:
# per-clip frames@128 + object tracks + 3 EEF points. -> outputs/flow_render_dataset/
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES job $SLURM_JOB_ID ==="
python gen_flow_render_dataset.py
echo "=== DONE ==="
