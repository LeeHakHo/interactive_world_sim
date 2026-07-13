#!/bin/bash
#SBATCH --job-name=dv_gm100
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/dualview_gmask/g1_100ep.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --mem=64G

# quality push: gmask agent-cond III at detmem-level training budget (100ep).
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
GMASK=1 EPOCHS=100 OUT=outputs/cross_embodiment_wm/dualview_gmask/g1_100ep \
python exp_scel_dualview_gmaskcond.py
