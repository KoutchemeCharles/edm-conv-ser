#!/bin/bash -l
#SBATCH --job-name=run
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2

export PYTHONPATH="$(pwd):$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

# Per-job HF cache to avoid collisions between array tasks
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$HF_HOME"

export HF_HUB_DOWNLOAD_TIMEOUT=240
export HF_HUB_ETAG_TIMEOUT=240
export HF_HUB_METADATA_TIMEOUT=240
export VLLM_CACHE_ROOT=$HF_HOME

export WANDB_START_METHOD="thread"
export WANDB_INIT_TIMEOUT=300

# API keys — set these before submitting, or export them in your environment
export HF_ACCESS_TOKEN="<your_huggingface_token>"
export HF_API_TOKEN="<your_huggingface_token>"
export OPENAI_API_KEY="<your_openai_key>"  # only needed for GPT-based experiments

# Load your cluster's modules here, e.g.:
#   module load gcc cuda
#   conda activate "$env_name"
# The lines below are examples from the original cluster and will need adapting:
module load gcc cuda
conda activate "$env_name"

hf auth login --token "$HF_ACCESS_TOKEN"

current_dir=$(pwd)
config_path="$current_dir/config/experiments/$folder/${SLURM_ARRAY_TASK_ID}.json"
python3 scripts/run.py --config "$config_path"
