#!/usr/bin/env bash
set -euo pipefail

: "${student:?Must export student=<model yaml>}"
: "${exp_name:?Must export exp_name=<experiment name>}"
: "${env_name:?Must export env_name}"
: "${job_name:?Must export job_name}"
: "${save_dir:?Must export save_dir=<output directory>}"

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
config_dir="$project_dir/config/experiments/$exp_name"

hpc="scripts/bash/triton/hpc.sh"
gpu_big="--gpus=1 --gres=min-vram:35g"

log_folder="--chdir=$project_dir --output=$project_dir/logs/$exp_name/%A_%a.log"

# Datasets
train_para=config/data/falcon/train_para.yaml
train_code=config/data/falcon/train_code.yaml
test_para=config/data/falcon/test_para.yaml
test_code=config/data/falcon/test_code.yaml
# Training tasks
train_sft=config/task/train/sft/train_sft.yaml
train_dpo_v1=config/task/train/dpo/train_dpo_v1.yaml
train_grpo_v1=config/task/train/grpo/train_grpo_v1.yaml
# Evaluation
eval_rollout=config/task/evaluate/eval_rollout.yaml

rm -rf "$config_dir"
mkdir -p "$config_dir"

# Preprocessing
python3 scripts/generate_config.py \
  --name "$exp_name" \
  --dataset "$train_para" \
  --task config/task/preprocess/preprocess.yaml \
  --save_dir "$save_dir"

python3 scripts/generate_config.py \
  --name "$exp_name" \
  --dataset "$test_para" \
  --task config/task/preprocess/preprocess.yaml \
  --save_dir "$save_dir"

python3 scripts/generate_config.py \
  --name "$exp_name" \
  --dataset "$train_code" \
  --task config/task/preprocess/preprocess.yaml \
  --save_dir "$save_dir"

python3 scripts/generate_config.py \
  --name "$exp_name" \
  --dataset "$test_code" \
  --task config/task/preprocess/preprocess.yaml \
  --save_dir "$save_dir"

# Training

# PARA — SFT
python3 scripts/generate_config.py \
  --name "$exp_name" \
  --model "$student" \
  --dataset "$config_dir/0.json" \
  --task "$train_sft" \
  --save_dir "$save_dir"

python3 scripts/generate_config.py \
  --name "$exp_name" \
  --model "$config_dir/4.json" \
  --dataset "$config_dir/1.json" \
  --task "$eval_rollout" \
  --save_dir "$save_dir"

# CODE — SFT
python3 scripts/generate_config.py \
  --name "$exp_name" \
  --model "$student" \
  --dataset "$config_dir/2.json" \
  --task "$train_sft" \
  --save_dir "$save_dir"

python3 scripts/generate_config.py \
  --name "$exp_name" \
  --model "$config_dir/6.json" \
  --dataset "$config_dir/3.json" \
  --task "$eval_rollout" \
  --save_dir "$save_dir"

# DPO
python3 scripts/generate_config.py \
  --name "$exp_name" \
  --model "$config_dir/6.json" \
  --dataset "$config_dir/2.json" \
  --task "$train_dpo_v1" \
  --save_dir "$save_dir"

python3 scripts/generate_config.py \
  --name "$exp_name" \
  --model "$config_dir/8.json" \
  --dataset "$config_dir/3.json" \
  --task "$eval_rollout" \
  --save_dir "$save_dir"

# GRPO (from SFT checkpoint)
python3 scripts/generate_config.py \
  --name "$exp_name" \
  --model "$config_dir/6.json" \
  --dataset "$config_dir/2.json" \
  --task "$train_grpo_v1" \
  --save_dir "$save_dir"

python3 scripts/generate_config.py \
  --name "$exp_name" \
  --model "$config_dir/10.json" \
  --dataset "$config_dir/3.json" \
  --task "$eval_rollout" \
  --save_dir "$save_dir"

# GRPO (from base)
python3 scripts/generate_config.py \
  --name "$exp_name" \
  --model "$student" \
  --dataset "$config_dir/2.json" \
  --task "$train_grpo_v1" \
  --save_dir "$save_dir"

python3 scripts/generate_config.py \
  --name "$exp_name" \
  --model "$config_dir/12.json" \
  --dataset "$config_dir/3.json" \
  --task "$eval_rollout" \
  --save_dir "$save_dir"

# BASE (no training)
python3 scripts/generate_config.py \
  --name "$exp_name" \
  --model "$student" \
  --dataset "$config_dir/0.json" \
  --task "$eval_rollout" \
  --save_dir "$save_dir"

# Submission

# Preprocessing (configs 0–3)
sbatch \
  --wait \
  --export=env_name="$env_name",folder="$exp_name" \
  --array=0-3 \
  --job-name="${job_name}_PREPROCESS" \
  $log_folder \
  $hpc

# PARA SFT train + eval
sbatch \
  --export=env_name="$env_name",folder="$exp_name" \
  --array=4,5%1 \
  --job-name="${job_name}_SFT_PARA" \
  $log_folder \
  $gpu_big \
  $hpc

# CODE SFT train
sbatch \
  --wait \
  --export=env_name="$env_name",folder="$exp_name" \
  --array=6 \
  --job-name="${job_name}_SFT_CODE" \
  $log_folder \
  $gpu_big \
  $hpc

# CODE SFT eval
sbatch \
  --export=env_name="$env_name",folder="$exp_name" \
  --array=7 \
  --job-name="${job_name}_EVAL" \
  $log_folder \
  $gpu_big \
  $hpc

# DPO train + eval
sbatch \
  --export=env_name="$env_name",folder="$exp_name" \
  --array=8,9%1 \
  --job-name="${job_name}_DPO" \
  $log_folder \
  $gpu_big \
  $hpc

# GRPO (from SFT) train + eval
sbatch \
  --export=env_name="$env_name",folder="$exp_name" \
  --array=10,11%1 \
  --job-name="${job_name}_GRPO_SFT" \
  $log_folder \
  $gpu_big \
  $hpc

# GRPO (from base) eval
sbatch \
  --export=env_name="$env_name",folder="$exp_name" \
  --array=14 \
  --job-name="${job_name}_GRPO_BASE" \
  $log_folder \
  $gpu_big \
  $hpc
