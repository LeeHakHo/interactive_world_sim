#!/bin/bash
# run_dualview_formal_sweep.sh — STAGE=A: variance-first (flow/eeffilm cv1 rh x seed0-2, 6 runs)
#                                STAGE=B: 其余 controlled 格子 (eefsp cv1 + 三臂 cv0, 12 runs)
#                                STAGE=C: M1c human-helps (flow/eeffilm cv1 robot-only x seed0-2, 6 runs)
# 格子编码 cond:cv:seed:mix   用法: STAGE=A bash run_dualview_formal_sweep.sh
set -e
cd /scr2/yusenluo/interactive_world_sim
STAGE=${STAGE:-A}
if [ "$STAGE" = "A" ]; then
  GRID_RUNS=(flow:1:0:rh flow:1:1:rh flow:1:2:rh eeffilm:1:0:rh eeffilm:1:1:rh eeffilm:1:2:rh)
elif [ "$STAGE" = "B" ]; then
  GRID_RUNS=(eefsp:1:0:rh eefsp:1:1:rh eefsp:1:2:rh flow:0:0:rh flow:0:1:rh flow:0:2:rh \
             eefsp:0:0:rh eefsp:0:1:rh eefsp:0:2:rh eeffilm:0:0:rh eeffilm:0:1:rh eeffilm:0:2:rh)
else  # C: M1c human-helps 消融 (robot-only 对照, cv1)
  GRID_RUNS=(flow:1:0:r flow:1:1:r flow:1:2:r eeffilm:1:0:r eeffilm:1:1:r eeffilm:1:2:r)
fi
LOGD=outputs/cross_embodiment_wm/dualview_dit_formal/logs
mkdir -p "$LOGD"
for r in "${GRID_RUNS[@]}"; do
  IFS=: read -r cond cv seed mix <<< "$r"
  suf=$([ "$mix" = "r" ] && echo "_ronly" || echo "")
  name=dvf_${cond}_cv${cv}_s${seed}${suf}
  sbatch --job-name="$name" --output="$LOGD/${name}_%j.log" \
    --export=ALL,SCRIPT=exp_scel_dualview_dit_formal.py,COND=$cond,CROSSVIEW=$cv,SEED=$seed,MIX=$mix \
    sbatch/dualview_formal_train.sbatch
done
squeue -u "$USER" -o "%.10i %.24j %.2t %.10M" | tail -n +1
echo "STAGE $STAGE submitted"
