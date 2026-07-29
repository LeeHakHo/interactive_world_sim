#!/bin/bash
#SBATCH --job-name=rwL48
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/flow_render_dataset_can_dual_L48/logs/rwL48_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=shard:8
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
mkdir -p outputs/flow_render_dataset_can_dual_L48/logs
export BASE=${BASE:-outputs/flow_render_dataset_can_dual_L48}
export HCLIP=${HCLIP:-clips_human_L48}
# 1) wrist sidecar low (真腕 world3D -> high/low 投影); reads {BASE}/{HCLIP}.npz
echo "=== [5a] build wrist sidecar low BASE=$BASE HCLIP=$HCLIP ==="
.venv_wan/bin/python build_human_wrist_low.py || { echo "WRIST_LOW FAILED"; exit 1; }
# 2) patch real wrist into eef; reads {BASE}/{HCLIP}_retrack.npz + wrist_sidecar_human_low.npz
echo "=== [5b] build realwrist clips ==="
.venv_wan/bin/python build_human_realwrist_clips.py || { echo "REALWRIST FAILED"; exit 1; }
echo "=== DONE $(ls -la $BASE/${HCLIP}_retrack_realwrist.npz 2>/dev/null) ==="
