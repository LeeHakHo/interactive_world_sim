#!/bin/bash
#SBATCH --job-name=genhL48
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/flow_render_dataset_can_dual_L48/logs/genhL48_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=shard:28
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=64G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
mkdir -p outputs/flow_render_dataset_can_dual_L48/logs
export CLIP_L=48 HUMAN_ONLY=1 ROBOT_ONLY=0
export OUTDIR=outputs/flow_render_dataset_can_dual_L48
echo "=== gen human base L48 (re-track CoTracker) SMOKE=${SMOKE:-0} CLIP_L=$CLIP_L OUTDIR=$OUTDIR ==="
python gen_flow_render_dataset_caneef.py
echo "=== DONE $(ls -la $OUTDIR/clips_human.npz 2>/dev/null) ==="
