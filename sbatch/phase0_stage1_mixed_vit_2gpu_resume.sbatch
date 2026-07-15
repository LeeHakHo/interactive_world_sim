#!/bin/bash
#SBATCH --job-name=p0_mixed_vit_2gpu
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/phase0_stage1_mixed_vit_2gpu_resume.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=40:00:00
#SBATCH --mem=192G

# Resume of job 45714 (phase0_stage1_mixed_vit) on 2 GPUs (DDP).
#   - SAME config as 45714: latent_decompose channel_split (v3, GRL), ViT-S,
#     mixed robot+human EEF, sample_ratio=0.5, max_steps=300005.
#   - Resumes full trainer state (global_step/optim/scheduler) from the
#     step-20000 rolling ckpt via `load=` → trainer.fit(ckpt_path=...).
#   - channel_split uses automatic optimization, so accumulate_grad_batches and
#     DDP work cleanly (find_unused_parameters=True is already set in exp_base).
#
# Why srun + ntasks-per-node=2: Lightning detects SLURM and expects one task
# per device, so DDP must be launched by srun (a bare `python` would hang
# waiting for the 2nd rank). Output dir is pinned to ${JOBTAG}/${DATESTR}
# (NOT ${now}) so both ranks resolve the SAME hydra run dir.

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws

cd /scr2/yusenluo/interactive_world_sim

JOBTAG="${SLURM_JOB_ID:-$(date +%s)}"
DATESTR="$(date +%Y-%m-%d)"
RUNDIR="outputs/${DATESTR}/job${JOBTAG}_phase0_stage1_mixed_vit_2gpu_resume"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python main.py \
  hydra.run.dir="${RUNDIR}" \
  +name=phase0_stage1_mixed_vit_2gpu_resume \
  algorithm=latent_world_model \
  experiment=exp_latent_dyn \
  experiment.num_devices=2 \
  load=outputs/_resume_ckpts/mixed_vit_step20000.ckpt \
  dataset=play_mixed_eef \
  dataset.horizon=4 dataset.val_horizon=16 \
  dataset.obs_keys=[camera_0_color] \
  dataset.sample_ratio=0.5 \
  experiment.training.batch_size=2 \
  experiment.training.optim.accumulate_grad_batches=4 \
  experiment.training.max_steps=300005 \
  experiment.training.log_every_n_steps=100 \
  experiment.training.checkpointing.every_n_train_steps=20000 \
  +experiment.training.checkpointing.save_top_k=1 \
  experiment.training.data.num_workers=4 \
  experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=10000 \
  experiment.validation.data.num_workers=2 \
  algorithm.latent_dim=512 algorithm.action_dim=8 \
  algorithm.training_stage=1 \
  algorithm.dynamo_ssl.enabled=false \
  algorithm.dynamo_ssl.encoder_backbone=vit \
  algorithm.latent_decompose.enabled=true
