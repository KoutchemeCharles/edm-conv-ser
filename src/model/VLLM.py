"""
vLLM inference engine wrapper.

Wraps the ``vllm.LLM`` engine with a consistent ``batch_query`` interface used
across all inference stages (evaluation, GRPO rollout).  Generation parameters
and chat template options are read from the experiment config.
"""

import os
from vllm import (LLM, SamplingParams)
from huggingface_hub import scan_cache_dir
from transformers import AutoTokenizer

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)


class VLLM():
    """
    Batched inference engine backed by vLLM.

    Attributes:
        config: Model config DotMap (``name``, ``gen_kwargs``, ``chat_kwargs``).
        llm: The underlying ``vllm.LLM`` instance.
        tokenizer: HuggingFace tokenizer for the model (used by callers for
            chat-template rendering).
    """

    def __init__(self, config, **kwargs):
        """
        Initialize the vLLM engine.

        Args:
            config: Model config DotMap with at minimum:
                - ``name``: HuggingFace model identifier or local path.
                - ``gen_kwargs``: Default sampling parameters (temperature, etc.).
                - ``chat_kwargs`` (optional): Extra kwargs forwarded to
                  ``llm.chat`` (e.g. ``chat_template``).
            **kwargs: Additional keyword arguments forwarded to ``vllm.LLM``
                (e.g. ``tensor_parallel_size``, ``gpu_memory_utilization``).
        """
        self.config = config
        self.llm = LLM(model=self.config.name,
                       **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.name)

    def batch_query(self, messages, gen_kwargs=None, cfm=False):
        """
        Run batched chat completions.

        Automatically detects whether the last message in the first conversation
        is an assistant prefill (``continue_final_message=True``) by inspecting
        the role of ``messages[0][-1]``.

        Args:
            messages (list[list[dict]]): Batch of conversations; each inner list
                is a sequence of ``{"role": ..., "content": ...}`` dicts.
            gen_kwargs (dict, optional): Sampling parameters (temperature,
                max_tokens, etc.).  If ``None``, falls back to
                ``config.gen_kwargs``.
            cfm (bool): Ignored — the flag is re-derived from the message roles.

        Returns:
            list[str]: One generated text string per conversation in the batch.

        Raises:
            ValueError: If neither ``gen_kwargs`` nor ``config.gen_kwargs`` is
                available.
        """
        if not gen_kwargs and self.config.gen_kwargs:
            gen_kwargs = self.config.gen_kwargs.toDict()
            logger.info(f"Using model gen kwargs {gen_kwargs}")
        elif gen_kwargs:
            logger.info(f"Using set argument config {gen_kwargs}")
        else:
             raise ValueError("Unknown generation config to use")

        other_kwargs = {}
        if self.config.chat_kwargs:
            other_kwargs = self.config.chat_kwargs.toDict()
            logger.info(f"Adding other kwargs to chat {other_kwargs}")
            
        cfm = messages[0][-1]["role"] == "assistant"
        params = SamplingParams(**gen_kwargs)
        outputs = self.llm.chat(messages=messages, 
                                sampling_params=params, 
                                continue_final_message=cfm, 
                                add_generation_prompt = not cfm,
                                use_tqdm=True,
                                **other_kwargs)

        return [output.outputs[0].text for output in outputs]
    