# `config/` — Experiment Configuration

Each experiment is a single JSON file that composes three YAML components.

## Structure

```
config/
├── model/
│   ├── local/unsloth/    # Llama 3.1 8B, Qwen 3 8B, SmolLM 1.7B, …
│   └── remote/           # GPT-4o, Kimi-K2, …
├── data/
│   └── falcon/           # train_code.yaml, test_code.yaml, train_para.yaml, …
├── task/
│   ├── train/
│   │   ├── sft/          # SFT hyperparameters
│   │   ├── dpo/          # DPO hyperparameters + sampling strategy
│   │   └── grpo/         # GRPO hyperparameters + reward config
│   └── evaluate/         # eval_rollout.yaml, eval_one_step.yaml
└── experiments/          # Composed JSON files (auto-generated)
    ├── paper_llama/      # 0.json … N.json
    ├── paper_qwen/
    └── …
```

## Composing a Config

```bash
python scripts/generate_config.py \
  --name my_experiment \
  --model config/model/local/unsloth/qwen3-8b.yaml \
  --dataset config/data/falcon/train_code.yaml \
  --task config/task/train/sft/train_sft.yaml
# → writes config/experiments/my_experiment/0.json
```

The resulting JSON merges `model`, `data`, and `task` sections with a `save_dir` path derived from the experiment name.

## Key Config Fields

### Data config (`config/data/falcon/`)

| Field | Values | Description |
|-------|--------|-------------|
| `format` | `code`, `dual` | Student code representation format |
| `feedback_level` | `full`, `none` | Whether to include unit-test details in grader feedback |
| `split` | `train`, `test` | Dataset split |

### Task config — training (`config/task/train/`)

| Field | Description |
|-------|-------------|
| `name` | `sft` / `dpo` / `grpo` |
| `args.max_length` | Max token budget per training example |
| `lora` | LoRA hyperparameters (`r`, `lora_alpha`, …) |
| `sampling` | DPO only: `next` / `random` / `grade` |
| `max_pairs_per_traj` | DPO only: cap on pairs per trajectory |
| `args.max_prompt_length` | GRPO only: prompt token budget |
| `args.max_completion_length` | GRPO only: generation token budget |
| `k` | GRPO only: examples sampled per trajectory position |

### Task config — evaluation (`config/task/evaluate/`)

| Field | Description |
|-------|-------------|
| `name` | `eval` (rollout) or `one_step` |
| `gen_kwargs` | Generation parameters (temperature, max_tokens, …) |
| `min_k` | Starting context size (number of seen submissions) |
