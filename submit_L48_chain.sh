#!/bin/bash
# L48 human rebuild + clean episode-split retrain — full SLURM dependency chain.
# afterok gating: 上游失败则下游自动不跑, 无需守。逗号 env 先 export 再 --export=ALL。
cd /scr2/yusenluo/interactive_world_sim
L48=outputs/flow_render_dataset_can_dual_L48
LATL48=outputs/video_arch_wm/wan_latents_can_dual_L48
RW=$L48/clips_human_L48_retrack_realwrist.npz
CONDH=outputs/video_arch_wm/cond_can_dual/cond_human_eef3_L48.npz
LATH=$LATL48/latents_human_all.npz
DSROB=outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz
E=outputs/cross_embodiment_wm/epsplit_L48
M=outputs/video_arch_wm/epsplit_L48_mh

# ---- Stage 1: base human L48 (full 12 ep) ----
jid1=$(sbatch --parsable --export=ALL sbatch_gen_human_L48.sbatch)
echo "jid1_gen=$jid1"

# ---- Stage 2: dualview cam_low ----
export SRC=$L48/clips_human.npz OUT=$L48/clips_human_L48.npz
jid2=$(sbatch --parsable -d afterok:$jid1 --export=ALL sbatch_gen_dualview_L48.sbatch)
echo "jid2_dualview=$jid2"; unset SRC OUT

# ---- Stage 3: retrack GDINO+SAM2 (high + low) ----
export BASE=$L48 HCLIP=human_L48 DOMAIN=human
export VIEW=high; jid3a=$(sbatch --parsable -d afterok:$jid2 --export=ALL sbatch_retrack.sbatch)
export VIEW=low;  jid3b=$(sbatch --parsable -d afterok:$jid2 --export=ALL sbatch_retrack.sbatch)
echo "jid3a_retrack_high=$jid3a jid3b_retrack_low=$jid3b"; unset BASE HCLIP DOMAIN VIEW

# ---- Stage 4: assemble ----
export BASE=$L48 HCLIP=clips_human_L48 DOMAINS=human
jid4=$(sbatch --parsable -d afterok:$jid3a:$jid3b --export=ALL sbatch_assemble_L48.sbatch)
echo "jid4_assemble=$jid4"; unset BASE HCLIP DOMAINS

# ---- Stage 5: realwrist + wrist sidecar ----
export BASE=$L48 HCLIP=clips_human_L48
jid5=$(sbatch --parsable -d afterok:$jid4 --export=ALL sbatch_realwrist_L48.sbatch)
echo "jid5_realwrist=$jid5"; unset BASE HCLIP

# ---- Stage 6: wan latents L48 (.venv_wan, isolated OUTDIR) ----
export SRC=human DS=$RW OUTDIR=$LATL48
jid6=$(sbatch --parsable -d afterok:$jid5 --export=ALL sbatch_human_latents.sbatch)
echo "jid6_latents=$jid6"; unset SRC DS OUTDIR

# ---- Stage 7: cond agent(eef3) L48 ----
export DS=$RW AGENT=eef3 COND_NAME=cond_human_eef3_L48
jid7=$(sbatch --parsable -d afterok:$jid6 --export=ALL sbatch_cond_human_L48.sbatch)
echo "jid7_cond=$jid7"; unset DS AGENT COND_NAME

# ================= Stage 8a: ② retrain (dep jid5, clips ready) =================
export DS=$DSROB DS_H=$RW ACTION=mp
export HELDOUT_VIDS="100,102" HELDOUT_VIDS_H="0,1"
# ro full
export MIX=r  HEAD_MODE=single NROB="" OUT_DIR=$E/mp_r_all
jid_ro=$(sbatch --parsable -d afterok:$jid5 --export=ALL sbatch_wm2_shard.sbatch)
# rh full
export MIX=rh HEAD_MODE=single NROB="" OUT_DIR=$E/mp_rh_all
jid_rh=$(sbatch --parsable -d afterok:$jid5 --export=ALL sbatch_wm2_shard.sbatch)
# rh_two full  ★双头判决
export MIX=rh HEAD_MODE=two    NROB="" OUT_DIR=$E/mp_rh_all_two
jid_rht=$(sbatch --parsable -d afterok:$jid5 --export=ALL sbatch_wm2_shard.sbatch)
echo "jid8_ro=$jid_ro jid8_rh=$jid_rh jid8_rh_two=$jid_rht"
# scarcity ro/rh n100,n300 (human-helps 曲线)
for N in 100 300; do
  export MIX=r  HEAD_MODE=single NROB=$N OUT_DIR=$E/mp_r_n$N
  j=$(sbatch --parsable -d afterok:$jid5 --export=ALL sbatch_wm2_shard.sbatch); echo "jid8_ro_n$N=$j"
  export MIX=rh HEAD_MODE=single NROB=$N OUT_DIR=$E/mp_rh_n$N
  j=$(sbatch --parsable -d afterok:$jid5 --export=ALL sbatch_wm2_shard.sbatch); echo "jid8_rh_n$N=$j"
done
unset DS DS_H ACTION HELDOUT_VIDS HELDOUT_VIDS_H MIX HEAD_MODE NROB OUT_DIR

# ================= Stage 8b: ③ retrain (dep jid7, latents+cond ready) =================
export LAT_H=$LATH COND_H=$CONDH HELDOUT_VIDS="100,102" STEPS=40000
# ro (robot-only)
export MIX="" OUT=$M/ro
jid_mh_ro=$(sbatch --parsable -d afterok:$jid7 --export=ALL sbatch_mh_train_L48.sbatch)
# rh (cotrain human)
export MIX=human OUT=$M/rh
jid_mh_rh=$(sbatch --parsable -d afterok:$jid7 --export=ALL sbatch_mh_train_L48.sbatch)
echo "jid9_mh_ro=$jid_mh_ro jid9_mh_rh=$jid_mh_rh"
echo "=== CHAIN SUBMITTED ==="
