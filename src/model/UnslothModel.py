"""
Unsloth-based local model wrapper.

Loads a HuggingFace model using Unsloth's ``FastLanguageModel`` for efficient
4-bit quantized training and inference.  After training the LoRA adapter is
merged into the base weights and saved to disk; inference is then delegated
to a :class:`~src.model.VLLM.VLLM` engine pointed at the merged checkpoint.

Lifecycle:
  - ``is_training=True``: loads model + tokenizer with Unsloth, exposes
    ``model`` and ``tokenizer`` for the TRL trainer.
  - ``is_training=False, was_trained=False``: skips model loading entirely and
    initialises a vLLM engine directly (pure inference from a Hub checkpoint).
  - ``is_training=False, was_trained=True``: merges the trained LoRA weights to
    16-bit, saves to ``<config.name>/vllm``, initialises a vLLM engine, then
    deletes the merged checkpoint to recover disk space.
"""

import os
import gc
import torch
import shutil
import unsloth
from copy import deepcopy
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from src.model.VLLM import VLLM

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)


class UnslothModel():

    """
    Unsloth local model that supports both training (LoRA) and inference (vLLM).

    During training the raw Unsloth model and tokenizer are available as
    ``self.model`` and ``self.tokenizer``.  After training (or when only
    inference is needed) ``self.vllm`` is populated and used for generation.

    Attributes:
        config: Model config DotMap (``name``, ``from_pretrained_kwargs``,
            ``chat_template``, ``vllm``).
        is_training (bool): Whether the model was initialised for training.
        was_trained (bool): Whether a LoRA adapter has already been fitted and
            should be merged before inference.
        vllm (VLLM or None): vLLM inference engine (set during inference init).
        model: Unsloth ``FastLanguageModel`` (set during training init).
        tokenizer: HuggingFace tokenizer (set during training init).
    """

    def __init__(self, config, is_training=False, was_trained=False) -> None:
        """
        Initialize the Unsloth model.

        Args:
            config: Model config DotMap with at minimum:
                - ``name``: HuggingFace model identifier or local path.
                - ``from_pretrained_kwargs``: kwargs forwarded to
                  ``FastLanguageModel.from_pretrained`` (e.g. ``max_seq_length``,
                  ``load_in_4bit``).
                - ``chat_template`` (str or None): Unsloth chat template name;
                  if None the tokenizer's default template is used.
                - ``vllm``: kwargs forwarded to the ``VLLM`` engine for inference.
            is_training (bool): If True, load the model in training mode (keeps
                LoRA-compatible weights in memory).
            was_trained (bool): If True, merge LoRA weights into base model
                before launching vLLM for inference.
        """
        
        self.config = config 
        self.is_training = is_training
        self.was_trained = was_trained
        self.vllm = None 

        if self.is_training or was_trained:
            self.load_model_and_tokenizer()

        if not self.is_training:
            
            path = os.path.join(self.config.name, "vllm")
            if was_trained:
                self.config.name = path
                self.model.save_pretrained_merged(path, self.tokenizer, save_method = "merged_16bit")
                del self.model
                torch.cuda.empty_cache()
                gc.collect()
                
            kwargs = self.config.vllm
            self.vllm = VLLM(self.config, **kwargs)

            if was_trained:
                try:
                    shutil.rmtree(path)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))
            
        
    def batch_query(self, batch, gen_kwargs, cfm=False):
        """
        Batch query the model for a list of messages.

        Args:
            batch (list of dict): A list of messages in the format of 
                [{"role": "user", "content": "message"}, ...].
            gen_kwargs (dict): Generation arguments.

        Returns:
            list of str: The generated responses.
        """

        return self.vllm.batch_query(messages=batch, gen_kwargs=gen_kwargs, cfm=cfm)


    def query(self, messages, gen_kwargs):
        return self.vllm.batch_query([messages])
    

    def load_model_and_tokenizer(self):
        """
        Load the model and tokenizer using Unsloth's ``FastLanguageModel``.

        Applies the Unsloth chat template if ``config.chat_template`` is set.
        Switches to inference mode (``FastLanguageModel.for_inference``) when
        ``is_training`` is False.
        """

        logger.info(f"Loading model with unsloth from {self.config.name}")

        from_pretrained_kwargs = self.config.from_pretrained_kwargs.toDict()
        logger.info(f"Config for from_pretrained {from_pretrained_kwargs}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.config.name,
            **from_pretrained_kwargs
        )
        
        if not self.is_training:
            logger.info("Preparing model for inference")
            self.model = FastLanguageModel.for_inference(self.model)

        if self.config.chat_template:
            logger.info(f"Using unsloth {self.config.chat_template} tokenizer")
            self.tokenizer = get_chat_template(
                self.tokenizer,
                self.config.chat_template,
            )
        else:
            logger.info(f"Using default tokenizer {self.tokenizer} tokenizer with template {self.tokenizer.chat_template}")

        logger.info("Finnished initialization")

        