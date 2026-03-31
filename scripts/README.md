# `scripts/` — Entry Points

## Main entry point

```bash
python scripts/run.py --config <path/to/experiment.json> [--test_run]
```

`--test_run` caps training to 10 steps for quick validation.

The stage is determined by `task.name` in the config:

| `task.name` | What runs |
|-------------|-----------|
| `preprocess` | `Preprocess.py` — conversationalize trajectories → `dataframe.csv` |
| `sft` | `trl/SFT.py` — SFT training |
| `dpo` | `trl/DPO.py` — DPO training |
| `grpo` | `trl/GRPO.py` — GRPO training |
| `eval` | `Evaluation.py` — multi-step rollout evaluation |
| `one_step` | `OneStepEvaluation.py` — single-step CodeBLEU evaluation |

## Other scripts

```bash
# Compose a new experiment config
python scripts/generate_config.py \
  --name <name> \
  --model config/model/local/unsloth/qwen3-8b.yaml \
  --dataset config/data/falcon/train_code.yaml \
  --task config/task/train/sft/train_sft.yaml

# Aggregate results and generate plots
python scripts/results.py

# Reproduce dataset statistics table (LaTeX)
python scripts/dataset_stats.py \
  --train config/data/falcon/train_code.yaml \
  --test  config/data/falcon/test_code.yaml \
  --output outputs/tables/dataset_stats_table.tex
```

## Bash launchers

### Local / background

```bash
export save_dir="/path/to/output"

# Full pipeline for a single model (preprocess → SFT → DPO → GRPO → eval)
bash scripts/bash/new/models/qwenb_falcon.sh

# Run all model variants in parallel
bash scripts/bash/new/run_all_falcon.sh
```

Model-specific variables (model name, HF repo, config paths) are set in `scripts/bash/new/models/`.

### SLURM cluster

```bash
export folder="paper_llama"   # maps to config/experiments/paper_llama/
export env_name="rl"

# Submit configs 0–8 as parallel array tasks
sbatch --array=0-8 scripts/bash/triton/hpc.sh
```

**Adapting to a different cluster:** replace `module load` commands in `hpc.sh` / `gpu_short.sh` with your cluster's equivalents, update `conda activate`, and adjust `--mem` / `--time` / `--cpus-per-task` to your limits.
