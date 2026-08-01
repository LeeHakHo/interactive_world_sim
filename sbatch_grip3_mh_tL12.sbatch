#!/bin/bash
#SBATCH --job-name=grip3mh
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/logs/grip3mh_tL12_%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem=64G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HUB_OFFLINE=1
export MASKGEN=outputs/flow_wm/maskgen_caneef/maskgen_caneef.pt GMASK=1 GMASK_LOW=1 WARP=1 RES=128
mkdir -p outputs/video_arch_wm/logs "$OUT"
set -e
# ★grip-aware+warp ③ 用【多头WM】(AUX空=plain多头, 与ro_80k同架构, 只差cond: 7ch flow+grip+warp vs 4ch flow+skel)
# retrack grip cond(cond-build阶段已出); CCOND=7全通道; episode-split干净heldout(多头支持)
export CCOND=7 TLCAP=0 HFRAC=0.5 PRED=abs OVERFIT=0
export HELDOUT_VIDS="100,102"          # ← 内部export, comma安全; 干净episode-split
export LAT=outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz
export COND=${COND:-outputs/video_arch_wm/cond_can_dual/cond_skel_grip_retrack.npz}   # 默认grip; 外部可传非grip(warp隔离)
export LAT_H=${LAT_H:-outputs/video_arch_wm/wan_latents_can_dual_L48/latents_human_all.npz}
export COND_H=${COND_H:-outputs/video_arch_wm/cond_can_dual/cond_human_eef3_L48.npz}
export STEPS=${STEPS:-80000} BS=16 EVAL_EVERY=4000
export RESUME=${RESUME:-auto}          # ★重启防护: 重投自动从 $OUT/resume.pt 续训(不在则从头), 无损
echo "=== grip3mh(多头) MIX=${MIX:-robot-only} OUT=$OUT CCOND=7 grip+warp COND=$COND STEPS=$STEPS HELDOUT_VIDS=$HELDOUT_VIDS ==="
.venv_wan/bin/python train_multihead_wm.py
echo "=== DONE: $OUT/mh_ema.pt ==="
