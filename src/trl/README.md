# `src/trl/` — Training Methods

Three training paradigms, all using Unsloth LoRA and uploading merged models to the HuggingFace Hub.

## Inheritance

```
Experiment  (src/Experiment.py)
    └── TRL         — shared LoRA setup, training args, Hub upload
            ├── SFT — next-submission prediction
            ├── DPO — preference pairs from temporal order
            └── GRPO — execution-based reward RL
```

## SFT (`SFT.py`)

Trains the model to predict each student's next code submission.

**Dataset construction:** each trajectory with *N* assistant turns is expanded into *N* independent training examples (one per turn to predict). Loss is computed on assistant turns only (`train_on_responses_only`).

**Context window:** left-truncated to `max_length` tokens. The system prompt and first user turn (problem description) are **never** dropped — only intermediate exchange pairs are trimmed.

## DPO (`DPO.py`)

Trains via preference pairs derived from the temporal ordering of student submissions: an earlier submission is *chosen* over a later one (modeling the trajectory direction a student takes).

Three sampling strategies (set by `config.task.sampling`):

| Strategy | `chosen` | `rejected` |
|----------|----------|------------|
| `next` | submission *t* | submission *t+2* |
| `random` | submission *t* | random future submission ≥ *t+3* |
| `grade` | submission *t* | first future submission with a different unit-test grade |

Each pair is truncated independently to fit `max_length` (preamble preserved).

## GRPO (`GRPO.py`)

Reward-based RL. The model generates code, which is executed against unit tests, and the reward drives policy updates.

**Reward function:**

| Outcome | Score |
|---------|-------|
| Exact code match (normalized AST) | +2.0 |
| Same unit-test grade, different code | +1.0 |
| No match | 0.0 |
| Syntax error | −0.5 |
| No fenced code block | −1.0 |
| + Length penalty (weight 0.5) | variable |
| Floor | −0.75 |

## Shared Mechanics

- **LoRA:** rank/alpha configurable per experiment via `config.task.lora`.
- **Checkpointing:** LoRA weights saved to `save_dir`; merged 16-bit model pushed to HF Hub.
- **WandB:** all runs logged to project `EDM2026SM`.
- **Test run:** `--test_run` limits training to 10 steps for quick validation.
