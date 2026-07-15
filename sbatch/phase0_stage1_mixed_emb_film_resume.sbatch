#!/bin/bash
#SBATCH --job-name=phase0_emb_film_rs
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/phase0_stage1_mixed_emb_film_resume.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=128G

# RESUME of the emb_film (no-OT) Stage-1 run (was job 45723, hit 24h time limit
# at step 20000). Local resume via `load=<last.ckpt>` -> trainer.fit(ckpt_path=)
# restores optimizer/step/scheduler and continues toward max_steps=300005.

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws

cd /scr2/yusenluo/interactive_world_sim

JOBTAG="${SLURM_JOB_ID:-$(date +%s)}"
RESUME_CKPT="/scr2/yusenluo/interactive_world_sim/outputs/2026-05-25/10-26-17_job45723_phase0_emb_film/checkpoints/last.ckpt"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

python main.py \
  hydra.run.dir="outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}_job${JOBTAG}_phase0_emb_film_resume" \
  +name=phase0_emb_film_resume \
  load="${RESUME_CKPT}" \
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
  experiment.training.checkpointing.every_n_train_steps=2000 \
  +experiment.training.checkpointing.save_last=true \
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
  algorithm.latent_decompose.method=emb_film \
  algorithm.latent_decompose.c_emb=16
