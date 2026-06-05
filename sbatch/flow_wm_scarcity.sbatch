#!/bin/bash
#SBATCH --job-name=flow_wm_scarcity
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/flow_wm_scarcity_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64G

# Step 3 data-scarcity: does human help MORE when robot data is scarce?
# action-conditioned flow WM, robot-train subsampled to {1400,400,200,100,50},
# robot-only vs robot+human on held-out robot. Uses existing flow_ds_v2.npz.
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES job $SLURM_JOB_ID ==="
python train_flow_wm_scarcity.py
echo "=== DONE ==="
