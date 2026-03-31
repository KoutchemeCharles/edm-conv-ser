# `src/data/` — Data Pipeline

Converts raw FalconCode execution logs into multi-turn conversations for training and evaluation.

## Pipeline

```
HuggingFace Hub (koutch/falcon_code)
        │
        ▼
FalconCode._preprocess()          # filter, deduplicate, attach problem descriptions
        │
        ▼
build_conversations_dataframe()   # group by student → Student.conversationalize()
        │
        ▼
DataFrame: (user_id, problem_id, messages)
```

## Conversation Format

Each trajectory becomes a multi-turn conversation:

| Role | Content |
|------|---------|
| `system` | Novice learner persona + submission format instructions |
| `user` | Problem description (first turn) |
| `assistant` | Student's code submission |
| `user` | Grader feedback (unit test results + grade) |
| `assistant` | Next submission … |

The first `user` turn (problem description) is **always preserved** during context-window truncation.

## Student Representation Formats

Set via `format` in the data config:

- **`code`** — every submission is a full code rewrite in a ` ```python ``` ` block.
- **`dual`** — unified diff (` ```diff ``` `) for small edits (single hunk, <50 % of lines changed); full rewrite otherwise.

The `<actual_submission>` tag is embedded in each assistant turn during preprocessing so evaluators can reconstruct the ground truth. It is stripped before the model sees input.

## Key Files

| File | Description |
|------|-------------|
| `falcon/FalconCode.py` | Loads and preprocesses `koutch/falcon_code`; provides `extract_grade`, `execute` |
| `falcon/execution.py` | Sandboxed single-subprocess Python execution engine |
| `Student.py` | Builds one conversation per (student, problem) pair |
| `serialization.py` | Diff generation, format dispatch, `conversationalize_improvements` |
| `Dataset.py` | Abstract base class + parallel conversation builder |
