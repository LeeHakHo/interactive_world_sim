#!/bin/bash
# scarce ③ 渲染器 human-helps 实验(clean L48 episode-split)
# 问题: ② 稀缺 robot 时 human 真帮(n100 +4.9 / n300 +2.2), ③ 渲染层帮不帮?
# 设计: OVERFIT=N 取前 N robot clip(ro/rh 同批保证公平) + MIX=human HFRAC=0.5 加全量 human.
#   ro_nN = 只 N robot; rh_nN = N robot + human. 比 robot heldout 的 render-LPIPS + cube_px.
# heldout 走 sbatch 内默认 HELDOUT_VIDS=100,102(无逗号传参→避 SLURM 截断 bug).
set -e
cd /scr2/yusenluo/interactive_world_sim
BASE=outputs/video_arch_wm/epsplit_L48_mh
SB=sbatch_mh_train_L48.sbatch

declare -A JID
for N in 100 300; do
  # robot-only 稀缺
  OUT_RO=$BASE/ro_n${N}
  JID[ro$N]=$(sbatch --parsable --job-name=mh3ro$N \
    --export=ALL,OUT=$OUT_RO,MIX=robot-only,OVERFIT=$N,STEPS=40000 $SB)
  # robot+human 稀缺
  OUT_RH=$BASE/rh_n${N}
  JID[rh$N]=$(sbatch --parsable --job-name=mh3rh$N \
    --export=ALL,OUT=$OUT_RH,MIX=human,OVERFIT=$N,STEPS=40000 $SB)
  echo "N=$N: ro=${JID[ro$N]} rh=${JID[rh$N]}"
done

# eval 依赖 4 个训练全 ok
DEP="afterok:${JID[ro100]}:${JID[rh100]}:${JID[ro300]}:${JID[rh300]}"
EJID=$(sbatch --parsable --job-name=mh3scarceeval --dependency=$DEP \
  --export=ALL sbatch_scarce3_eval.sbatch)
echo "eval(dep $DEP) = $EJID"
