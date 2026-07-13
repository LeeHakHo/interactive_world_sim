#!/bin/bash
#SBATCH --job-name=comb_extra
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/combine_action/extra100_s%a/slurm.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --array=3-4

# v3 combine: extra seeds at the money cell N=100 (variance-first: the 3-seed
# BREAK for align_wm was suggestive but not >2x std; extend to 5 seeds).
# Runs world/skel anchors + align_wm only.

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
mkdir -p outputs/cross_embodiment_wm/combine_action/extra100_s${SLURM_ARRAY_TASK_ID}

SEEDS=${SLURM_ARRAY_TASK_ID} N_ROB_LIST=100 COMBINES=align_wm \
OUT=outputs/cross_embodiment_wm/combine_action/extra100_s${SLURM_ARRAY_TASK_ID} \
python exp_scel_combine_action.py
