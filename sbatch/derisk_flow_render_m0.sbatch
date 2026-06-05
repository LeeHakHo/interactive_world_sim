#!/bin/bash
#SBATCH --job-name=flow_render_m0
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/flow_render_m0_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=64G

# Milestone 0 (kill switch): can a flow-conditioned decoder draw the moved
# object + moved gripper from real I0 + GT flow (teacher-forced overfit)?
# Includes the mandatory shuffle-flow ablation. out: outputs/flow_render_m0.png
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES job $SLURM_JOB_ID ==="
python derisk_flow_render_m0.py
echo "=== DONE ==="
