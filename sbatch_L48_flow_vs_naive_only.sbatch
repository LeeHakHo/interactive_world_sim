#!/bin/bash
#SBATCH --job-name=fvnaive2
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/logs/fvnaive2_%x_%j.txt
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
echo "=== flow vs naive-cond L48: RUNS=ro,ro_noflow (job 55079 ro_noflow ckpt ready) ==="
BDIR=outputs/video_arch_wm/epsplit_L48_mh \
  RUNS=ro,ro_noflow \
  CONDP=outputs/video_arch_wm/cond_can_dual/cond_skel_all_retrack.npz \
  SEQS=332,59,418,442 \
  OUT=outputs/video_arch_wm/mh_eval_L48_flowcond_vs_naive/flow_vs_naive \
  .venv_wan/bin/python eval_mh_render.py
echo "=== DONE: outputs/video_arch_wm/mh_eval_L48_flowcond_vs_naive/flow_vs_naive/render_summary.txt ==="
