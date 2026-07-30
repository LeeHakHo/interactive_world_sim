#!/bin/bash
#SBATCH --job-name=buildgripcond
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/logs/buildgripcond_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=48G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HUB_OFFLINE=1
export MASKGEN=outputs/flow_wm/maskgen_caneef/maskgen_caneef.pt GMASK=1 GMASK_LOW=1 WARP=1 RES=128
set -e
COND=outputs/video_arch_wm/cond_can_dual/cond_skel_grip_retrack.npz
if [ ! -f "$COND" ]; then
  echo "=== build retrack + grip-aware + warp 7ch cond ==="
  SKEL=1 SKELF=outputs/flow_render_dataset_can_dual/skel_sidecar_robot_gripaware.npz \
    DS=outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz \
    COND_NAME=cond_skel_grip_retrack .venv_wan/bin/python build_can_cond.py
else echo "== 已存在 =="; fi
echo "=== DONE: $COND ==="
