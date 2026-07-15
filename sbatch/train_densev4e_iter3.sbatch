#!/bin/bash
#SBATCH --job-name=v4e_iter3
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/v4e_iter3_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=64G
# iter3: native-128 flow+mask injected straight into decoder conv_in (bypass 32-res control_net).
# Config via exported env: OUTDIR HIRES_FLOW OBJ_WEIGHT COND_INIT EPOCHS (+ STAGE1/DS/GCK).
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
echo "=== node $(hostname) job $SLURM_JOB_ID | OUTDIR=$OUTDIR HIRES=$HIRES_FLOW OBJW=$OBJ_WEIGHT COND_INIT=$COND_INIT EPOCHS=$EPOCHS ==="
mkdir -p "$OUTDIR"
python train_flow_cond_decoder_densev4e.py
echo "=== DONE ==="
