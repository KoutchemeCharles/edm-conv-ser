# `src/model/` — Model Wrappers

Uniform `batch_query(messages, gen_kwargs)` interface over local and remote models.

## Models

### `UnslothModel` — local models (training + inference)

Wraps Unsloth's `FastLanguageModel` for efficient 4-bit quantized LoRA training. After training the LoRA adapter is merged to 16-bit, handed off to a `VLLM` engine for inference, then the merged checkpoint is deleted to recover disk space.

Three lifecycle modes controlled by constructor flags:

| `is_training` | `was_trained` | Behaviour |
|---------------|---------------|-----------|
| `True` | — | Load model + tokenizer for LoRA training |
| `False` | `False` | Skip model load; init vLLM directly from Hub checkpoint |
| `False` | `True` | Merge LoRA → save → init vLLM → delete merged copy |

### `VLLM` — batched inference engine

Thin wrapper around `vllm.LLM`. Automatically detects assistant-prefill mode from message roles (`continue_final_message`).

### `RemoteModel` — cloud APIs

Supports OpenAI (default), HuggingFace Inference API, Groq, Google Generative Language, and Anthropic. For native OpenAI models, requests are submitted via the **Batch API** (50 % cost reduction) and polled until completion. All other providers fall back to sequential calls.

## Config Keys (model config YAML)

| Key | Description |
|-----|-------------|
| `name` | HuggingFace model ID or local path |
| `source` | `openai` / `huggingface` / `groq` / `google` / `anthropic` |
| `from_pretrained_kwargs` | Forwarded to `FastLanguageModel.from_pretrained` (e.g. `max_seq_length`, `load_in_4bit`) |
| `gen_kwargs` | Default sampling params (temperature, max_tokens, …) |
| `chat_kwargs` | Extra kwargs forwarded to `llm.chat` |
| `vllm` | kwargs for the vLLM engine (e.g. `tensor_parallel_size`) |
| `chat_template` | Unsloth chat template name (or `null` to use tokenizer default) |
