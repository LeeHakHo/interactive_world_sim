#!/bin/bash
#SBATCH --job-name=e2e_flow_wm_render
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/e2e_flow_wm_render_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --mem=96G

# END-TO-END demo: ② FlowWM (robot-only vs robot+human, trained on clips) predicts object
# flow -> ③ densev3 renderer draws pixels. Shows human-helps IN PIXEL SPACE on held-out robot.
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES job $SLURM_JOB_ID ==="
python e2e_flow_wm_render.py
echo "=== DONE ==="
