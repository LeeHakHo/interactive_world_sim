#!/bin/bash
#SBATCH --job-name=mhL48
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/logs/mhL48_%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HUB_OFFLINE=1
# ROBOT latents/cond = existing L48 (unchanged); HUMAN = new L48 rebuild
export LAT=${LAT:-outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz}
export COND=${COND:-outputs/video_arch_wm/cond_can_dual/cond_skel_all_retrack.npz}
export LAT_H=${LAT_H:-outputs/video_arch_wm/wan_latents_can_dual_L48/latents_human_all.npz}
export COND_H=${COND_H:-outputs/video_arch_wm/cond_can_dual/cond_human_eef3_L48.npz}
export CCOND=4 TLCAP=6 HFRAC=0.5 PRED=abs
export OVERFIT=${OVERFIT:-0}
# ★episode(vid)分组 heldout (修帧泄漏); robot demo seqs 属 vid100/102
export HELDOUT_VIDS=${HELDOUT_VIDS:-100,102}
export STEPS=${STEPS:-40000} BS=${BS:-16} EVAL_EVERY=${EVAL_EVERY:-4000}
mkdir -p "$OUT" outputs/video_arch_wm/logs
echo "=== mh-L48 OUT=$OUT MIX=${MIX:-robot-only} HELDOUT_VIDS=$HELDOUT_VIDS LAT_H=$LAT_H COND_H=$COND_H STEPS=$STEPS ==="
.venv_wan/bin/python train_multihead_wm.py
echo "=== DONE: $OUT/mh_ema.pt ==="
