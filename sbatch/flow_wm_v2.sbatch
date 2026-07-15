#!/bin/bash
#SBATCH --job-name=flow_wm_v2
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/flow_wm_v2_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64G

# Object-flow WM v2 = EEF-action-conditioned (cross-embodiment, 2D, no depth):
#   gen_flow_dataset_v2.py : object tracks + agent EEF 2D trajectory (action)
#                            -> outputs/flow_dataset/flow_ds_v2.npz  (+ QC png)
#   train_flow_wm_v2.py    : object-history + EEF-action -> object-future WM.
#       ablation use_action False(baseline)/True x configs robot-only/human-only/
#       robot+human on held-out robot. Answers (a) does action help, (b) does human help.
# See memory project_object_flow_shared_step0 for the route.

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES job $SLURM_JOB_ID ==="

if [ ! -f outputs/flow_dataset/flow_ds_v2.npz ]; then
  echo "=== gen v2 dataset (object tracks + EEF action) ==="
  python gen_flow_dataset_v2.py
else
  echo "=== flow_ds_v2.npz exists, skipping ==="
fi

echo "=== train v2 (action ablation x human-help) ==="
python train_flow_wm_v2.py
echo "=== DONE ==="
