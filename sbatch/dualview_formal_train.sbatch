#!/bin/bash
#SBATCH --job-name=dvf
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
# sbatch/dualview_formal_train.sbatch — 通用单卡模板. 用 --export 传 SCRIPT + 实验 env
# (COND/CROSSVIEW/SEED/MIX/EPOCHS/SMOKE/MODE...), --job-name/--output 由提交方覆盖.
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python "${SCRIPT:-exp_scel_dualview_dit_formal.py}"
