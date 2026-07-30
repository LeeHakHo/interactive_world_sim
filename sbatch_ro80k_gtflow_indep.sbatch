#!/bin/bash
#SBATCH --job-name=ro80kgt
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/logs/ro80kgt_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=48G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HUB_OFFLINE=1
mkdir -p outputs/video_arch_wm/logs
set -e
# 独立核对: ro_80k 的 GT-flow(retrack) render-LPIPS + cube_px, 对另session的0.0723/10.40/11%
echo "=== 独立 ro_80k GT-flow eval (retrack flow, 非老) ==="
BDIR=outputs/video_arch_wm/epsplit_L48_mh \
  RUNS=ro_80k \
  CONDP=outputs/video_arch_wm/cond_can_dual/cond_skel_all_retrack.npz \
  SEQS=332,59,418,442 \
  OUT=outputs/video_arch_wm/mh_eval_ro80k_gtflow_indep \
  .venv_wan/bin/python eval_mh_render.py
echo "=== DONE ==="
