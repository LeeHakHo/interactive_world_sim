#!/bin/bash
#SBATCH --job-name=aug_h3d
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/flow_render_dataset_can_dual_L48/aug_h3d_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export VIEW=high
python augment_clips_tracks3d.py outputs/flow_render_dataset_can_dual_L48/clips_human_L48_retrack_realwrist_3d.npz
echo "=== DONE ==="
