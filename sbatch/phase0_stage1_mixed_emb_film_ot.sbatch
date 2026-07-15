#!/bin/bash
#SBATCH --job-name=phase0_emb_film_ot
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/phase0_stage1_mixed_emb_film_ot.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=128G

# Phase 0b (emb_film + EgoBridge-style OT): Stage 1 on mixed robot+human EEF.
# Builds on emb_film (appearance routed to z_emb so the scene latent is free of
# embodiment pixels) and ADDS an Optimal-Transport alignment loss on z_scene to
# pull human/robot scene-latent distributions into the same subspace (attacks
# the UMAP-disjoint-cluster problem).
#   - OT only touches z_scene; z_emb (appearance) is untouched -> decode stays
#     intact (the GRL collapse mode is avoided by construction).
#   - Joint (latent, EEF-position) cost + DTW pseudo-pairs (EgoBridge 2509.19626).
#   - EEF positions unified to the cam_high frame: robot world-frame EEF is
#     transformed by the static URDF extrinsic; human EEF is already cam-frame.
#   - alpha ramps in over [warmup_start, warmup_end]; PSNR is the hard guard
#     (if reconstruction degrades, alpha is too high).
#   - per-domain memory bank decouples OT from the small decoder batch size.
#   - measure-first probes still logged (probe_scene_acc should now DROP).

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws

cd /scr2/yusenluo/interactive_world_sim

JOBTAG="${SLURM_JOB_ID:-$(date +%s)}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

python main.py \
  hydra.run.dir="outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}_job${JOBTAG}_phase0_emb_film_ot" \
  +name=phase0_emb_film_ot \
  algorithm=latent_world_model \
  experiment=exp_latent_dyn \
  dataset=play_mixed_eef \
  dataset.horizon=4 dataset.val_horizon=16 \
  dataset.obs_keys=[camera_0_color] \
  dataset.sample_ratio=0.5 \
  experiment.training.batch_size=2 \
  experiment.training.optim.accumulate_grad_batches=4 \
  experiment.training.max_steps=300005 \
  experiment.training.log_every_n_steps=100 \
  experiment.training.checkpointing.every_n_train_steps=10000 \
  +experiment.training.checkpointing.save_last=true \
  experiment.training.data.num_workers=4 \
  experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=10000 \
  experiment.validation.data.num_workers=2 \
  algorithm.latent_dim=512 algorithm.action_dim=8 \
  algorithm.training_stage=1 \
  algorithm.dynamo_ssl.enabled=false \
  algorithm.dynamo_ssl.encoder_backbone=vit \
  algorithm.latent_decompose.enabled=true \
  algorithm.latent_decompose.method=emb_film \
  algorithm.latent_decompose.c_emb=16 \
  algorithm.latent_decompose.ot_align.enabled=true \
  algorithm.latent_decompose.ot_align.alpha=0.1 \
  algorithm.latent_decompose.ot_align.warmup_start=5000 \
  algorithm.latent_decompose.ot_align.warmup_end=20000
