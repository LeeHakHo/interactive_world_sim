#!/bin/bash
#SBATCH --job-name=stems_v3
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/combine_action/stems_s%a/slurm_fix.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --array=0-4

# rerun of the v3 half of stems money cells (empty-regs stack bug fixed).
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
SEEDS=${SLURM_ARRAY_TASK_ID} ANCHORS= COMBINES=stems N_ROB_LIST=100 \
OUT=outputs/cross_embodiment_wm/combine_action/stems_s${SLURM_ARRAY_TASK_ID} \
python exp_scel_combine_action.py
