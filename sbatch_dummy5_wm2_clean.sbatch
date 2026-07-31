#!/bin/bash
#SBATCH --job-name=d5wm2
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/logs/d5wm2_%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=48G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
mkdir -p outputs/cross_embodiment_wm/logs "$OUT_DIR"
# ★dummy5 ② 干净 L48: retrack 新track + episode-split(HELDOUT_VIDS内部export避SLURM逗号截断)
export ACTION=dummy5 SPLIT=okfirst
export HELDOUT_VIDS="100,102"          # ← 内部export, comma安全
export DS=outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz
export DS_H=${DS_H:-outputs/flow_render_dataset_can_dual_L48/clips_human_L48_retrack_realwrist.npz}   # ★L48对齐mp(原L24是confound)
# MIX / OUT_DIR 由 --export 传(无逗号)
echo "=== d5wm2 ACTION=$ACTION MIX=$MIX OUT=$OUT_DIR DS=$DS HELDOUT_VIDS=$HELDOUT_VIDS ==="
python exp_scel_dualview_wm.py
echo "=== DONE: $OUT_DIR/wm_dual.pt ==="
