#!/bin/bash
#SBATCH --job-name=idmxf
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/idm_derisk/logs/idmxf_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --mem=48G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
cd /scr2/yusenluo/interactive_world_sim
export HF_HUB_OFFLINE=1 ACTION=mp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p outputs/idm_derisk/logs
set -e
for ARM in ${ARMS:-A1 C1}; do
  conda activate iws
  echo "--- train ${ARM}xf ---"
  ARM=$ARM EPOCHS=${EPOCHS:-300} OUT_DIR=outputs/idm_derisk/${ARM}xf python idm_train_xf.py
  conda activate phantom
  echo "--- eval ${ARM}xf ---"
  ARM=$ARM OUT_DIR=outputs/idm_derisk/${ARM}xf python idm_eval.py
done
echo DONE_XF
