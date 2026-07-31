#!/bin/bash
#SBATCH --job-name=grip3eval
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/logs/grip3eval_%j.txt
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
# grip③(CCOND=7 grip+warp)GT-flow render-LPIPS + cube_px; grip_ro vs grip_rh = ③human-helps; vs ro_80k = grip+warp效果
echo "=== grip③ GT-flow eval (grip cond, CCOND=7) ==="
BDIR=outputs/video_arch_wm/epsplit_L48_mh \
  RUNS=grip_ro,grip_rh \
  CCOND=7 \
  CONDP=outputs/video_arch_wm/cond_can_dual/cond_skel_grip_retrack.npz \
  SEQS=332,59,418,442 \
  OUT=outputs/video_arch_wm/mh_eval_grip3 \
  .venv_wan/bin/python eval_mh_render.py
echo "=== DONE ==="
