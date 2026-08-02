#!/bin/bash
#SBATCH --job-name=idm_abl
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/idm_derisk/logs/idm_%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=shard:22        # ★分片(小任务),整卡计48挤QoS;按峰值显存调
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=48G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
cd /scr2/yusenluo/interactive_world_sim
mkdir -p outputs/idm_derisk/logs
export HF_HUB_OFFLINE=1
set -e
ARMS=${ARMS:-"A0 A1 A3"}          # 核心臂; A2(+grip)可选加
EPOCHS=${EPOCHS:-200}
DO_TRAIN=${DO_TRAIN:-1}
DO_EVAL=${DO_EVAL:-1}
echo "=== IDM 消融 ARMS=[$ARMS] EPOCHS=$EPOCHS train=$DO_TRAIN eval=$DO_EVAL ==="

if [ "$DO_TRAIN" = "1" ]; then
  conda activate iws
  for ARM in $ARMS; do
    echo "--- train $ARM ---"
    ARM=$ARM EPOCHS=$EPOCHS OUT_DIR=outputs/idm_derisk/$ARM${OUT_SUF} python idm_train.py
  done
fi

if [ "$DO_EVAL" = "1" ]; then
  conda activate phantom
  for ARM in $ARMS; do
    echo "--- eval $ARM ---"
    ARM=$ARM OUT_DIR=outputs/idm_derisk/$ARM${OUT_SUF} python idm_eval.py
  done
fi

conda activate iws
ARMS="$ARMS" ROOT=outputs/idm_derisk python idm_ablation_table.py
echo "=== DONE: outputs/idm_derisk/ABLATION.txt ==="
