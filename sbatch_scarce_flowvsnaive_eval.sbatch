#!/bin/bash
#SBATCH --job-name=scfvn_eval
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/logs/scfvn_%x_%j.txt
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
# scarce flow-cond vs naive(NOFLOW): 同③backbone/同N robot, 只换cond有无object-flow(FlowWAM式)
# GT-flow 供给, naive 名字含noflow→eval清零flow通道; 出 render-LPIPS + cube_px + det/vanish + 自动flow-vs-naive行
echo "=== scarce flow-cond vs naive (GT-flow, 只换cond): n100 + n300 ==="
BDIR=outputs/video_arch_wm/epsplit_L48_mh \
  RUNS=ro_n100,ro_n100_noflow,ro_n300,ro_n300_noflow \
  CONDP=outputs/video_arch_wm/cond_can_dual/cond_skel_all_retrack.npz \
  SEQS=332,59,418,442 \
  OUT=outputs/video_arch_wm/mh_eval_L48_scarce_flowvsnaive \
  .venv_wan/bin/python eval_mh_render.py
echo "=== DONE: outputs/video_arch_wm/mh_eval_L48_scarce_flowvsnaive/render_summary.txt ==="
