#!/bin/bash
#SBATCH --job-name=flow_wm_pipeline
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/flow_wm_pipeline_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64G

# Object-flow WM pipeline (cross-embodiment, 2D point tracks, no depth):
#   Step 2a  gen_flow_dataset.py : CoTracker3 2D object-flow tracks for human+robot
#            -> outputs/flow_dataset/flow_ds.npz   (skipped if it already exists)
#   Step 2b  train_flow_wm.py    : history(4)->future(12) trajectory WM, 3 configs
#            (robot-only / human-only / robot+human) on a held-out robot video
#            -> ADE/FDE table answering "does human data help robot?" (Step 3)
# See memory project_object_flow_shared_step0 for the route + Step0/0b green light.

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim

export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== node: $(hostname)  gpu: $CUDA_VISIBLE_DEVICES  job: $SLURM_JOB_ID ==="

if [ ! -f outputs/flow_dataset/flow_ds.npz ]; then
  echo "=== Step 2a: generating object-flow dataset ==="
  python gen_flow_dataset.py
else
  echo "=== Step 2a: flow_ds.npz exists, skipping generation ==="
fi

echo "=== Step 2b/3: training flow WM + does-human-help eval ==="
python train_flow_wm.py

echo "=== DONE ==="
