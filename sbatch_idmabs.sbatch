#!/bin/bash
#SBATCH --job-name=idmabs
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/idm_derisk/logs/idmabs_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=shard:22        # ★分片(小任务),整卡计48挤QoS;按峰值显存调
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=48G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
cd /scr2/yusenluo/interactive_world_sim
export HF_HUB_OFFLINE=1 YMODE=abs
mkdir -p outputs/idm_derisk/logs
set -e
conda activate iws
ARM=A1 EPOCHS=200 OUT_DIR=outputs/idm_derisk/A1abs python idm_train.py
conda activate phantom
ARM=A1 OUT_DIR=outputs/idm_derisk/A1abs python idm_eval.py
echo DONE_ABS
