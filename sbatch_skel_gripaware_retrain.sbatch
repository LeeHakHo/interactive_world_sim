#!/bin/bash
#SBATCH --job-name=skelgrip
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/logs/skelgrip_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=48G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HUB_OFFLINE=1
export MASKGEN=outputs/flow_wm/maskgen_caneef/maskgen_caneef.pt GMASK=1 GMASK_LOW=1 WARP=1 RES=128
set -e
SIDE=outputs/flow_render_dataset_can_dual/skel_sidecar_robot_gripaware.npz
COND=outputs/video_arch_wm/cond_can_dual/cond_skel_grip_all.npz
echo "=== 1/3: grip-aware sidecar (phantom FK, ~2700 clips) ==="
conda run -n phantom python gen_gripaware_sidecar.py
echo "=== 2/3: build grip-aware skel cond ==="
SKEL=1 SKELF=$SIDE COND_NAME=cond_skel_grip_all .venv_wan/bin/python build_can_cond.py
echo "=== 3/3: train skel③ grip-aware (60000 步) ==="
LAT=outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz \
COND=$COND OUT=outputs/video_arch_wm/m4_skel_gripaware STEPS=60000 \
.venv_wan/bin/python train_video_dit.py
echo "=== ALL DONE: m4_skel_gripaware/video_dit_ema.pt ==="
