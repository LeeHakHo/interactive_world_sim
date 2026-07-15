#!/bin/bash
#SBATCH --job-name=train_flowcond_v4a
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/train_flowcond_v4a_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=96G

# densev4a: ORACLE agent-mask conditioning (robot-only) on top of the flow-cond decoder.
# Answers R1's basic question: does an agent silhouette channel fix the blurry gripper?
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES job $SLURM_JOB_ID ==="
python train_flow_cond_decoder_densev4a.py
echo "=== DONE ==="
