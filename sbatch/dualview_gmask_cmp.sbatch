#!/bin/bash
#SBATCH --job-name=dv_gmask_cmp
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/dualview_gmask/compare.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --dependency=afterok:53256_0:53256_1
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
COMPARE=outputs/cross_embodiment_wm/dualview_gmask/g0,outputs/cross_embodiment_wm/dualview_gmask/g1 OUT=outputs/cross_embodiment_wm/dualview_gmask/compare python exp_scel_dualview_gmaskcond.py
