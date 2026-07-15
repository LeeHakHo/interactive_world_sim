#!/bin/bash
#SBATCH --job-name=phase0_emb_film
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/phase0_stage1_mixed_emb_film.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=128G

# Phase 0 v5 (emb_film): Stage 1 observation decomposition on mixed robot+human
# EEF data. Attacks the stage-1 subspace-separation problem (UMAP completely
# splits human vs robot) by giving APPEARANCE a dedicated route, so the scene
# latent is free to drop embodiment-specific pixels.
#   - z_scene: existing spatial latent (4ch), feeds decoder + dynamics (UNCHANGED)
#   - z_emb:   global appearance code (c_emb=16) from the [CLS] token, broadcast
#              + appended to the decoder conditioning (control_net cond ch 4->20)
#   - reconstruction depends on z_emb -> it cannot collapse like dual_head's did
#   - AUTOMATIC optimisation (no CLUB / no manual opt)
#   - measure-first: NO alignment loss; detached domain probes log how separable
#     z_scene vs z_emb are (training/probe_scene_acc, training/probe_emb_acc).
#     We WANT probe_scene_acc -> ~0.5 (chance) and probe_emb_acc -> high.
#   - balanced 50/50 robot/human (dataset.sample_ratio=0.5)

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws

cd /scr2/yusenluo/interactive_world_sim

JOBTAG="${SLURM_JOB_ID:-$(date +%s)}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

python main.py \
  hydra.run.dir="outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}_job${JOBTAG}_phase0_emb_film" \
  +name=phase0_emb_film \
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
  algorithm.latent_decompose.c_emb=16
