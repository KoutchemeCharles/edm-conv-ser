#!/usr/bin/env bash
set -euo pipefail

: "${model:?Must export model=<model yaml>}"
: "${exp_name:?Must export exp_name=<experiment name>}"
: "${env_name:?Must export env_name}"
: "${job_name:?Must export job_name}"
: "${save_dir:?Must export save_dir=<output directory>}"

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
config_dir="$project_dir/config/experiments/$exp_name"

hpc="scripts/bash/triton/hpc.sh"

log_folder="--chdir=$project_dir --output=$project_dir/logs/$exp_name/%A_%a.log"

# Datasets
test_code=config/data/falcon/test_code.yaml
# Evaluation
eval_rollout=config/task/evaluate/eval_rollout_llm.yaml

rm -rf "$config_dir"
mkdir -p "$config_dir"

# Preprocessing
python3 scripts/generate_config.py \
  --name "$exp_name" \
  --dataset "$test_code" \
  --task config/task/preprocess/preprocess.yaml \
  --save_dir "$save_dir"

# Evaluation
python3 scripts/generate_config.py \
  --name "$exp_name" \
  --model "$model" \
  --dataset "$config_dir/0.json" \
  --task "$eval_rollout" \
  --save_dir "$save_dir"

sbatch \
  --export=env_name="$env_name",folder="$exp_name" \
  --array=1 \
  --job-name="${job_name}_EVAL" \
  $log_folder \
  $hpc
