#!/bin/bash
#SBATCH --job-name=flow_wm_v3
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/flow_wm_v3_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64G

# v3: build flow_ds_v3 (object tracks + 3 EEF points center/base+2tips) then re-run
# the data-scarcity human-helps validation with the 3-point action.
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES job $SLURM_JOB_ID ==="
echo "--- gen flow_ds_v3 ---"
python gen_flow_dataset_v3.py
echo "--- scarcity re-validation (3 EEF pts) ---"
python train_flow_wm_scarcity_v3.py
echo "=== DONE ==="
