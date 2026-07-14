#!/bin/bash
#SBATCH --job-name=dv_warp
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/dualview_gmask/%x_%a.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --array=0-1

# arm0: warp-as-condition (rigid Umeyama RGB preview) on top of g1low (60ep, controlled)
# arm1: definitive 100ep of current best (g1low) for the formal quality number
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  GMASK=1 GMASK_LOW=1 WARP=1 EPOCHS=60 OUT=outputs/cross_embodiment_wm/dualview_gmask/g1low_warp \
    python exp_scel_dualview_gmaskcond.py
else
  GMASK=1 GMASK_LOW=1 EPOCHS=100 OUT=outputs/cross_embodiment_wm/dualview_gmask/g1low_100ep \
    python exp_scel_dualview_gmaskcond.py
fi
