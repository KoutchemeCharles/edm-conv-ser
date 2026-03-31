#!/bin/bash -l
#SBATCH --job-name=gpu_short
#SBATCH --mem=64GB
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=2

export PYTHONPATH="$(pwd):$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

# Per-job HF cache to avoid collisions between array tasks
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$HF_HOME"
export VLLM_CACHE_ROOT=$HF_HOME

# API keys — set these before submitting, or export them in your environment
export HF_ACCESS_TOKEN="<your_huggingface_token>"
export HF_API_TOKEN="<your_huggingface_token>"
export OPENAI_API_KEY="<your_openai_key>"  # only needed for GPT-based experiments

# Load your cluster's modules here, e.g.:
#   module load gcc cuda
#   conda activate "$env_name"
module load gcc cuda
conda activate "$env_name"

current_dir=$(pwd)
config_path="$current_dir/config/experiments/$folder/${SLURM_ARRAY_TASK_ID}.json"
python3 scripts/run.py --config "$config_path"
