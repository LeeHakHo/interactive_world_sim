#!/bin/bash
#SBATCH --job-name=data_equiv
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/data_equiv/%x_%a.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --array=0-2

# Human-vs-robot data exchange rate: extend the robot-only (ro) scaling curve to
# large N (rh(N) points already measured). alpha = M/1800 where ro(N+M)=rh(N).
# v3: world anchor only; dual: dummy5 anchor only. seed = array idx.

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
S=${SLURM_ARRAY_TASK_ID}
mkdir -p outputs/cross_embodiment_wm/data_equiv/v3_s${S} outputs/cross_embodiment_wm/data_equiv/dual_s${S}

SEEDS=${S} ANCHORS=world COMBINES= N_ROB_LIST=200,800,1600,2550 \
OUT=outputs/cross_embodiment_wm/data_equiv/v3_s${S} python exp_scel_combine_action.py

SEEDS=${S} ANCHORS=dummy5 COMBINES= N_ROB_LIST=800,1600,2460 \
OUT=outputs/cross_embodiment_wm/data_equiv/dual_s${S} python exp_scel_dualview_comb.py
