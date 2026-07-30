#!/bin/bash
#SBATCH --job-name=mh3_80keval
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/logs/mh3_80keval_%x_%j.txt
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
echo "=== 全量 ③ 80k(resume) render-LPIPS + cube_px, 对照 40k ==="
BDIR=outputs/video_arch_wm/epsplit_L48_mh \
  RUNS=ro_80k,rh_80k \
  CONDP=outputs/video_arch_wm/cond_can_dual/cond_skel_all_retrack.npz \
  SEQS=332,59,418,442 \
  OUT=outputs/video_arch_wm/mh_eval_L48_80k \
  .venv_wan/bin/python eval_mh_render.py
echo "=== DONE: outputs/video_arch_wm/mh_eval_L48_80k/render_summary.txt ==="
