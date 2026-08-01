#!/bin/bash
#SBATCH --job-name=sk_dec
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/logs/sk_dec_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --mem=48G
# 交互草图 decoder(12ch=9草图+warp3: flow_img3+world_dz+skel+grip+trace+attach+contact). 主头robot-only+aux(objheat)robot+human.
cd /scr2/yusenluo/interactive_world_sim
export HF_HUB_OFFLINE=1
export LAT=outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz
export COND=outputs/video_arch_wm/sketch_can_dual/sketch_robot_warp.npz
export LAT_H=outputs/video_arch_wm/wan_latents_can_dual_L48/latents_human_all.npz
export COND_H=outputs/video_arch_wm/sketch_can_dual/sketch_human_warp.npz
export CCOND=12 TLCAP=0 PRED=abs HELDOUT_VIDS=100,102 STEPS=60000 BS=16 EVAL_EVERY=4000
mkdir -p outputs/video_arch_wm/logs
echo "=== sketch decoder MIX=${MIX:-robot-only} AUX=${AUX:-none} OUT=$OUT ==="
.venv_wan/bin/python train_sketch_decoder.py
echo "=== DONE $OUT ==="
