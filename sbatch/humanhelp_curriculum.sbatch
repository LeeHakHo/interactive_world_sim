#!/bin/bash
#SBATCH --job-name=hh_curr
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/hh_curr_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --mem=200G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
ARCH=${ARCH:-lwc} ACT=${ACT:-dummy5} N_ROB_LIST=${N_ROB_LIST:-20,100} SEEDS=${SEEDS:-0,1,2} \
OUT=${OUT:-outputs/cross_embodiment_wm/humanhelp_${ARCH}_${ACT}} \
  python exp_scel_humanhelp_curriculum.py
