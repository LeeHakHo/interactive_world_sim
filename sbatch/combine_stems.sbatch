#!/bin/bash
#SBATCH --job-name=comb_stems
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/combine_action/stems_s%a/slurm.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --array=0-4

# EgoWAM-inspired 'stems' mechanism (per-embodiment action encoders, NO explicit
# alignment; shared trunk + shared flow objective aligns implicitly). Money cells
# only; anchors skipped (deterministic split -> merge with existing cells by seed).
# Question: is the frozen teacher (align_wm) even needed, or does the shared
# objective suffice (EgoWAM's claim)?
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
S=${SLURM_ARRAY_TASK_ID}
mkdir -p outputs/cross_embodiment_wm/combine_action/stems_s${S} outputs/cross_embodiment_wm/dualview_comb/stems_s${S}

SEEDS=${S} ANCHORS= COMBINES=stems N_ROB_LIST=100 \
OUT=outputs/cross_embodiment_wm/combine_action/stems_s${S} python exp_scel_combine_action.py

SEEDS=${S} ANCHORS= COMBINES=stems N_ROB_LIST=400 \
OUT=outputs/cross_embodiment_wm/dualview_comb/stems_s${S} python exp_scel_dualview_comb.py
