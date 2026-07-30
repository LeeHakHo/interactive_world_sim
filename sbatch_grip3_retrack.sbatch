#!/bin/bash
#SBATCH --job-name=grip3rt
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/logs/grip3rt_%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --mem=48G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HUB_OFFLINE=1
export MASKGEN=outputs/flow_wm/maskgen_caneef/maskgen_caneef.pt GMASK=1 GMASK_LOW=1 WARP=1 RES=128
mkdir -p outputs/video_arch_wm/logs "$OUT"
set -e
# ★retrack + grip-aware + warp 的 7ch cond(老 cond_skel_grip_all 是老track, 按用户要求用retrack重建)
SIDE=outputs/flow_render_dataset_can_dual/skel_sidecar_robot_gripaware.npz
COND=outputs/video_arch_wm/cond_can_dual/cond_skel_grip_retrack.npz
if [ ! -f "$COND" ]; then
  echo "=== build retrack grip cond ==="
  SKEL=1 SKELF=$SIDE DS=outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz \
    COND_NAME=cond_skel_grip_retrack .venv_wan/bin/python build_can_cond.py
else echo "== retrack grip cond 已存在 =="; fi
# 训练: VideoDiT 7ch(matches旧m4_skel_gripaware + keyboard-eval按VideoDiT走7ch); MIX/OUT/LAT_H/COND_H由--export传
echo "=== grip3rt MIX=${MIX:-robot-only} OUT=$OUT STEPS=${STEPS:-60000} COND=$COND ==="
LAT=outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz \
COND=$COND OUT=$OUT STEPS=${STEPS:-60000} TLCAP=6 \
MIX=${MIX:-} LAT_H=${LAT_H:-outputs/video_arch_wm/wan_latents_can_dual_L48/latents_human_all.npz} \
COND_H=${COND_H:-outputs/video_arch_wm/cond_can_dual/cond_human_eef3_L48.npz} \
.venv_wan/bin/python train_video_dit.py
echo "=== DONE: $OUT/video_dit_ema.pt ==="
