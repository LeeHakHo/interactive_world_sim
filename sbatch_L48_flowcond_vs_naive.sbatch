#!/bin/bash
#SBATCH --job-name=fcvnaive
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/logs/fcvnaive_%x_%j.txt
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
echo "=== L48 flow-cond vs naive-cond 干净对比(FlowWAM Fig.4式): render-LPIPS + cube_px ==="
echo "=== Part A: 已训好 flow ③ (ro,rh) L48 render-LPIPS 基线 ==="
BDIR=outputs/video_arch_wm/epsplit_L48_mh \
  RUNS=ro,rh \
  CONDP=outputs/video_arch_wm/cond_can_dual/cond_skel_all_retrack.npz \
  SEQS=332,59,418,442 \
  OUT=outputs/video_arch_wm/mh_eval_L48_flowcond_vs_naive/flow_ro_rh \
  .venv_wan/bin/python eval_mh_render.py
echo "=== Part B: naive(NOFLOW) ro_noflow, 若 ckpt 已出(job 55079)一并跑 flow vs naive 对比 ==="
if [ -f outputs/video_arch_wm/epsplit_L48_mh/ro_noflow/mh_ema.pt ]; then
  BDIR=outputs/video_arch_wm/epsplit_L48_mh \
    RUNS=ro,ro_noflow \
    CONDP=outputs/video_arch_wm/cond_can_dual/cond_skel_all_retrack.npz \
    SEQS=332,59,418,442 \
    OUT=outputs/video_arch_wm/mh_eval_L48_flowcond_vs_naive/flow_vs_naive \
    .venv_wan/bin/python eval_mh_render.py
else
  echo "=== ro_noflow/mh_ema.pt 尚未就绪(job 55079 未完), 跳过 Part B, 之后单独重跑 ==="
fi
echo "=== DONE: outputs/video_arch_wm/mh_eval_L48_flowcond_vs_naive/{flow_ro_rh,flow_vs_naive}/render_summary.txt ==="
