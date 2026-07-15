#!/bin/bash
#SBATCH --job-name=train_flowcond_dec_v2
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/train_flowcond_dec_v2_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=96G

# PROPER (full-data, big-time-step) training of the flow-conditioned decoder on the
# job45895 AE, with held-out GT/shuffle/zero-flow eval. Needs outputs/flow_render_dataset/.
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES job $SLURM_JOB_ID ==="
python train_flow_cond_decoder_densev2.py
echo "=== DONE ==="
