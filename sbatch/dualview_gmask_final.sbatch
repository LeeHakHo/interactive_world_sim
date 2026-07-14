#!/bin/bash
#SBATCH --job-name=dv_final
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/dualview_gmask/dv_final.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem=64G

# definitive III: dual-view silhouettes + rigid warp preview, 100ep.
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
GMASK=1 GMASK_LOW=1 WARP=1 EPOCHS=100 \
OUT=outputs/cross_embodiment_wm/dualview_gmask/final_warp_100ep \
python exp_scel_dualview_gmaskcond.py
