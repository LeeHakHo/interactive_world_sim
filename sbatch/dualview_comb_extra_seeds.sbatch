#!/bin/bash
#SBATCH --job-name=dcomb_extra
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/dualview_comb/extra400_s%a/slurm.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --array=2-4

# dual-view combine: N=400 extra seeds (seed2 cell lost to disk-full crash;
# seeds 3-4 extend to 5 for variance-first significance on the rh improvement).
# Runs dummy5/skel anchors + align_wm; saves paired rh ckpts for e2e.

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
mkdir -p outputs/cross_embodiment_wm/dualview_comb/extra400_s${SLURM_ARRAY_TASK_ID}

SEEDS=${SLURM_ARRAY_TASK_ID} N_ROB_LIST=400 COMBINES=align_wm \
OUT=outputs/cross_embodiment_wm/dualview_comb/extra400_s${SLURM_ARRAY_TASK_ID} \
python exp_scel_dualview_comb.py
