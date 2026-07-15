#!/bin/bash
#SBATCH --job-name=grip_wm_can
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/grip_wm_can_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=128G

# ② FlowWM (base + grip-aware) on the CAN L=48 clips.
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
DS=${DS:-outputs/flow_render_dataset_can_grip} \
OUT_DIR=${OUT_DIR:-outputs/cross_embodiment_wm/grip_wm_can} \
SMOKE=${SMOKE:-0} \
  python exp_scel_grip_wm.py
