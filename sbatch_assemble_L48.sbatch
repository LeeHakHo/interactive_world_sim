#!/bin/bash
#SBATCH --job-name=asmL48
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/flow_render_dataset_can_dual_L48/logs/asmL48_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=shard:8
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=64G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
mkdir -p outputs/flow_render_dataset_can_dual_L48/logs
export BASE=${BASE:-outputs/flow_render_dataset_can_dual_L48}
export HCLIP=${HCLIP:-clips_human_L48}
export DOMAINS=${DOMAINS:-human}
echo "=== assemble retrack L48 BASE=$BASE HCLIP=$HCLIP DOMAINS=$DOMAINS ==="
python assemble_retrack_clips.py
echo "=== DONE $(ls -la $BASE/${HCLIP}_retrack.npz 2>/dev/null) ==="
