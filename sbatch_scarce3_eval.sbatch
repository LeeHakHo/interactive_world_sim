#!/bin/bash
#SBATCH --job-name=mh3scarceeval
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/logs/mh3scarceeval_%x_%j.txt
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
echo "=== scarce ③ human-helps: robot heldout render-LPIPS + cube_px (clean L48) ==="
echo "=== ro_nN(只N robot) vs rh_nN(N robot+human), N=100/300; ② 对应 n100 +4.9 / n300 +2.2 ==="
BDIR=outputs/video_arch_wm/epsplit_L48_mh \
  RUNS=ro_n100,rh_n100,ro_n300,rh_n300 \
  CONDP=outputs/video_arch_wm/cond_can_dual/cond_skel_all_retrack.npz \
  SEQS=332,59,418,442 \
  OUT=outputs/video_arch_wm/mh_eval_L48_scarce3 \
  .venv_wan/bin/python eval_mh_render.py
echo "=== DONE: outputs/video_arch_wm/mh_eval_L48_scarce3/render_summary.txt ==="
