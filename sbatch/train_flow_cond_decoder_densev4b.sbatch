#!/bin/bash
#SBATCH --job-name=train_flowcond_v4b
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/train_flowcond_v4b_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=96G
#SBATCH --array=0-1

# densev4b: DEPLOYABLE 3-point gripper silhouette (robot-only), train/test same source.
# array task 0 = noarm, 1 = arm  -> data says whether drawing a crude arm helps.
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VARIANT=$([ "$SLURM_ARRAY_TASK_ID" = "1" ] && echo arm || echo noarm)
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES job $SLURM_JOB_ID variant=$VARIANT ==="
python train_flow_cond_decoder_densev4b.py $VARIANT
echo "=== DONE ==="
