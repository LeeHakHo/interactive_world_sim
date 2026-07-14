#!/bin/bash
#SBATCH --job-name=dv_gm_v1
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/dualview_gmask/%x_%a.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --array=0-1

# cam_low uplift + capacity ablation (single-variable chain vs g1-60ep):
#   arm0: + GMASK_LOW (v1 real silhouette, maskgen_caneef_low IoU0.836)
#   arm1: + GMASK_LOW + capacity (DIM 384->512, depth 8->12)
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  GMASK=1 GMASK_LOW=1 EPOCHS=60 OUT=outputs/cross_embodiment_wm/dualview_gmask/g1low \
    python exp_scel_dualview_gmaskcond.py
else
  GMASK=1 GMASK_LOW=1 DIM_G=512 DEPTH_G=12 EPOCHS=60 OUT=outputs/cross_embodiment_wm/dualview_gmask/g1low_big \
    python exp_scel_dualview_gmaskcond.py
fi
