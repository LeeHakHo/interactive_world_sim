#!/bin/bash
#SBATCH --job-name=dualvL48
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/flow_render_dataset_can_dual_L48/logs/dualvL48_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=48G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
mkdir -p outputs/flow_render_dataset_can_dual_L48/logs
# SRC = stage1 base human high npz; OUT = L48 dualview clips (adds tracks_low/eef_low/vis_low)
export SRC=${SRC:-outputs/flow_render_dataset_can_dual_L48/clips_human.npz}
export OUT=${OUT:-outputs/flow_render_dataset_can_dual_L48/clips_human_L48.npz}
echo "=== gen dualview cam_low L48 SMOKE=${SMOKE:-0} SRC=$SRC OUT=$OUT ==="
python gen_dualview_aligned.py
echo "=== DONE $(ls -la $OUT 2>/dev/null) ==="
