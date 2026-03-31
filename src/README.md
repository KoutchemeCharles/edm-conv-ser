# `src/` — Source Library

Core Python library for the student simulation pipeline.

## Paper-to-Code Mapping

| Paper concept | Module | Key symbol |
|---------------|--------|------------|
| §3 Dataset loading & preprocessing | `data/falcon/FalconCode.py` | `FalconCode._preprocess` |
| §4.1 Conversationalization | `data/Student.py` | `Student.conversationalize` |
| §4.1 Student representation formats (`code` / `dual`) | `data/serialization.py` | `format_code_as_assistant_payload` |
| §4.1 Novice learner system prompt | `data/Student.py` | `Student.form_system_prompt` |
| §4.1 Preamble-preserving left truncation | `trl/SFT.py` | `left_truncate_by_assistant_turns_to_fit` |
| §4.2 SFT — trajectory expansion | `trl/SFT.py` | `expand_trajectory_to_sft_rows` |
| §4.3 DPO — next-step pairs | `trl/DPO.py` | `process_dpo_next_preferences` |
| §4.3 DPO — temporal pairs | `trl/DPO.py` | `process_dpo_temporal_preferences` |
| §4.3 DPO — grade-based pairs | `trl/DPO.py` | `process_dpo_grades` |
| §4.4 GRPO — execution reward | `trl/GRPO.py` | `GRPO.train` → inner `reward` |
| §5 Multi-step rollout evaluation | `Evaluation.py` | `Evaluation`, `EvaluationTask` |
| §5 Code execution / grading | `data/falcon/execution.py` | `grade_fn` |
| Code normalization (duplicate detection) | `utils/normalization.py` | `robust_normalize` |

## Module Overview

```
src/
├── Experiment.py       # Base class: config loading, dataset loading, agent init
├── Preprocess.py       # Stage 1: conversationalize and cache trajectory dataframe
├── Evaluation.py       # Stage 5: multi-step rollout evaluator
│
├── data/               # Data pipeline — see data/README.md
├── trl/                # Training methods — see trl/README.md
├── model/              # Model wrappers — see model/README.md
└── utils/              # Shared utilities (seeding, files, code analysis, distances)
```
