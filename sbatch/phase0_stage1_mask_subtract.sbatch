#!/bin/bash
#SBATCH --job-name=phase0_stage1_mask_subtract
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/phase0_stage1_mask_subtract.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=40:00:00
#SBATCH --mem=128G

# Phase 0 mask_subtract: Stage 1 on mixed robot+human EEF data.
# See docs/superpowers/specs/2026-05-28-mask-subtract-residual-decompose-design.md
# and docs/superpowers/plans/2026-05-28-mask-subtract-residual-decompose.md.
#   - agent-mask-grounded z_emb (whole-arm mask) with agent-recon + domain anchor
#   - z_scene = F with the anchored embodiment direction projected out (ortho_proj)
#   - embodiment path DETACHED from encoder-decoder; decoder unchanged (4ch latent)
#   - NO CLUB, NO adversary
#   - lambda_scene_rec=0 for this v1 (the mask-exterior scene-rec sufficiency term
#     needs a 2nd decoder forward → OOM at bs=2 on 49GiB; anti-collapse is still
#     structural [E trained only by full recon + bounded rank-k projection] and
#     sufficiency is MEASURED by probe_full vs probe_scene). Enable later with
#     grad-checkpointing.
#   - balanced 50/50 robot/human (sample_ratio=0.5), cam_high only, ViT-S patch=8
#   - max 300k steps (matches the mixed-vit baseline for fair comparison)

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws

cd /scr2/yusenluo/interactive_world_sim

JOBTAG="${SLURM_JOB_ID:-$(date +%s)}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  hydra.run.dir="outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}_job${JOBTAG}_phase0_stage1_mask_subtract" \
  +name=phase0_stage1_mask_subtract \
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
  experiment.training.checkpointing.every_n_train_steps=5000 \
  +experiment.training.checkpointing.save_top_k=1 \
  experiment.training.data.num_workers=4 \
  experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=10000 \
  experiment.validation.data.num_workers=2 \
  algorithm.latent_dim=512 algorithm.action_dim=8 \
  algorithm.training_stage=1 \
  algorithm.dynamo_ssl.enabled=false \
  algorithm.dynamo_ssl.encoder_backbone=vit \
  algorithm.latent_decompose.enabled=true \
  algorithm.latent_decompose.method=mask_subtract \
  algorithm.latent_decompose.lambda_scene_rec=0
