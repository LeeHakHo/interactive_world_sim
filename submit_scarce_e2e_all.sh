#!/bin/bash
# 全量 4 对匹配 scarce e2e: ③(scarce mh) ← 匹配的 scarce ②(mp) 预测 flow
# 每对整链同 N、同 with/without-human; 输出 gtflow天花板 vs e2e LPIPS + flow叠加gif
set -e
cd /scr2/yusenluo/interactive_world_sim
MH=outputs/video_arch_wm/epsplit_L48_mh
WM=outputs/cross_embodiment_wm/epsplit_L48
for N in 100 300; do
  for mix in r rh; do
    ro=$( [ "$mix" = "r" ] && echo ro || echo rh )
    CKPT=$MH/${ro}_n${N}/mh_ema.pt
    WMCK=$WM/mp_${mix}_n${N}/wm_dual.pt
    OUT=outputs/video_arch_wm/scarce_e2e/${ro}_n${N}
    # SEQS 不传(用 eval_video_e2e 默认 332,59,418,442)避 SLURM 逗号截断
    J=$(sbatch --parsable --job-name=e2e_${ro}${N} \
      --export=ALL,ACTION=mp,CKPT=$CKPT,WM_CKPT=$WMCK,OUT=$OUT \
      sbatch_scarce_e2e.sbatch)
    echo "${ro}_n${N}: ③=$CKPT ②=$WMCK -> job $J"
  done
done