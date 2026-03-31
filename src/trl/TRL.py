"""
Base class for all TRL-based training stages (SFT, DPO, GRPO).

Inherits dataset loading and model instantiation from
:class:`~src.Experiment.Experiment` and adds:
  - Dataset preparation (delegated to subclass ``prepare_dataset``).
  - LoRA adapter configuration via Unsloth's ``get_peft_model``.
  - Training argument construction (shared defaults + config overrides).
  - Post-training model merging and upload to the HuggingFace Hub.

Concrete subclasses (``SFT``, ``DPO``, ``GRPO``) must set:
  - ``self.Trainer`` — the TRL trainer class (e.g. ``SFTTrainer``).
  - ``self.TrainerArgs`` — the corresponding config class (e.g. ``SFTConfig``).
  - Override ``prepare_dataset`` to produce a ``datasets.Dataset`` or
    ``DatasetDict``.
"""

import os
import weave  # Needed by WANDB apparently
import torch
import unsloth
from warnings import warn
from src.Experiment import Experiment
from src.utils.files import create_dir
from datasets import DatasetDict
from peft import PeftModel, LoraConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import concatenate_datasets


class TRL(Experiment):
    """
    Base training stage: wraps TRL trainers with Unsloth LoRA and Hub upload.

    Attributes:
        dataset_save_path (str): Directory where the HuggingFace Dataset is
            cached to disk before training.
        other_args (dict): Extra keyword arguments merged into training args
            (set by subclasses, e.g. ``dataset_text_field`` for SFT).
        response_only (bool): If True, applies ``train_on_responses_only``
            to mask the loss on prompt tokens (assistant-turn loss only).
        instruction_part (str): Token/string marking the start of the
            instruction in the chat template (for response-only masking).
        response_part (str): Token/string marking the start of the assistant
            response (for response-only masking).
        Trainer: TRL trainer class set by the subclass.
        TrainerArgs: TRL trainer config class set by the subclass.
    """

    def __init__(self, config, test_run, lazy_load=False) -> None:
        """
        Initialize the training stage.

        Args:
            config: Experiment config DotMap.
            test_run (bool): If True, limit training to 10 steps and skip eval.
            lazy_load (bool): If True, skip dataset and model loading
                (inherited from Experiment; unused in practice for TRL stages).
        """

        super().__init__(config, test_run, is_training=True, lazy_load=False)
        self.dataset_save_path = os.path.join(self.save_dir, "dataset")
        create_dir(self.dataset_save_path)
        self.other_args = {}
        self.response_only = False 

        self.instruction_part = self.config.model.response_only.instruction_part
        self.response_part = self.config.model.response_only.response_part
        os.environ["WANDB_PROJECT"] = "EDM2026SM"

        self.config.task
        

    def run(self):
        """
        Execute the full training pipeline.

        Steps:
          1. ``prepare_dataset`` — build and cache the HuggingFace Dataset.
          2. ``prepare_training`` — construct trainer arguments.
          3. ``train`` — run the trainer, save LoRA weights, and push the
             merged model to the HuggingFace Hub.
        """
        dataset = self.prepare_dataset(self.dataframe)
        dataset.save_to_disk(self.dataset_save_path)
        train_args = self.prepare_training()
        
        self.train(dataset, train_args)


    def train(self, dataset, train_args, **other_trainer_args):
        """
        Run the TRL trainer and upload the merged model to the Hub.

        If the model has not been LoRA-adapted before (``config.model.model``
        is falsy), applies ``FastLanguageModel.get_peft_model`` with the LoRA
        config from ``prepare_peft_config``.

        When ``response_only`` is True and the training args request it, applies
        ``train_on_responses_only`` so loss is computed only on assistant turns.

        After training:
          - LoRA weights are saved to ``model_save_path``.
          - The merged 16-bit model is pushed to
            ``koutch/<family>_<experiment_name>`` on the HuggingFace Hub.
          - Model and trainer are deleted and GPU memory is cleared.

        Args:
            dataset: HuggingFace ``Dataset`` or ``DatasetDict``.
            train_args (dict): Training argument dict from ``prepare_training``.
            **other_trainer_args: Extra kwargs forwarded to the Trainer
                constructor (e.g. ``reward_funcs`` for GRPO).
        """
        model = self.agent.model
        processing_class = self.agent.tokenizer
        if not self.config.model.model: # if the model was never trained before
            # https://docs.unsloth.ai/basics/continued-pretraining
            lora_config = self.prepare_peft_config()
            if lora_config is None:
                raise ValueError("Training a model (assuming always with LORA) but no config provided")

            ugc = False
            if train_args["gradient_checkpointing"]: ugc = "unsloth"
            print("Unsloth gradient chekpointing", ugc)

            model = FastLanguageModel.get_peft_model(
                model,
                use_gradient_checkpointing = ugc,
                random_state = 3407,
                **lora_config,
            )
        else:
            print("Model had already lora paramters")
            
        args = self.TrainerArgs(**train_args)

        print("Dataset", dataset)
        print("Arguments", args)
        # print("Model", model)

        if isinstance(dataset, DatasetDict):
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]
        elif train_args["eval_strategy"] == "no" and isinstance(dataset, DatasetDict):
            train_dataset = concatenate_datasets(list(dataset.values()))
        else:
            train_dataset = dataset
            eval_dataset = None

        # Merge any trainer_kwargs from the subclass
        if hasattr(self, 'trainer_kwargs'):
            other_trainer_args = {**self.trainer_kwargs, **other_trainer_args}

        trainer = self.Trainer(
            model=model,
            processing_class=processing_class,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=args,
            **other_trainer_args
        )

        if self.response_only and (
            train_args["assistant_only_loss"] 
            or train_args["completion_only_loss"]):
            print("Training only on assistant loss")
            print("Instruction part: ", self.instruction_part)
            print("Response part", self.response_part)
            trainer = train_on_responses_only(
                trainer,
                instruction_part = self.instruction_part,
                response_part = self.response_part
            )

        trainer_stats = trainer.train()
        print(trainer_stats)
        
        # Save the LoRa weights on the disk 
        
        model.save_pretrained(train_args["output_dir"])
        processing_class.save_pretrained(train_args["output_dir"])
        cwd = os.getcwd()
        os.chdir(train_args["output_dir"])
        model.push_to_hub_merged(self.hub_save_path, 
                                 processing_class, 
                                 save_method = "merged_16bit",
                                 use_temp_dir=train_args["output_dir"]),
                                 #token=os.environ.get("HF_ACCESS_TOKEN"))
        os.chdir(cwd)

        del self.agent.model
        del trainer
        torch.cuda.empty_cache()


    def prepare_dataset(self, df):
        raise NotImplementedError()
        
    def prepare_training(self):
        """
        Prepare a dictionary of training arguments for the Trainer.

        This is a simplified version of the arguments, which will be
        updated with the experiment specific arguments.

        Args:
            None

        Returns:
            base_training_args: A dictionary of training arguments
        """

        base_training_args = {}
        base_training_args["output_dir"] = self.model_save_path
        base_training_args["overwrite_output_dir"] = True

        ## Efficient training
        if supports_flash_attention():
            base_training_args["fp16"] = False
            base_training_args["bf16"] = True
            base_training_args["bf16_full_eval"] = True

        else:
            base_training_args["fp16"] = True
            base_training_args["bf16"] = False
            base_training_args["fp16_full_eval"] = True


        base_training_args["gradient_accumulation_steps"] = 16
        base_training_args["per_device_train_batch_size"] = 1
        base_training_args["per_device_eval_batch_size"] = 1
        base_training_args["gradient_checkpointing"] = True
        base_training_args["use_liger_kernel"] = False  # https://github.com/huggingface/trl/issues/3480
        ## Base training arguments
        base_training_args["num_train_epochs"] = 3
        base_training_args["lr_scheduler_type"] = "cosine"
        base_training_args["max_grad_norm"] = 1.0 # careful, changed 
        base_training_args["warmup_ratio"] = 0.1
        base_training_args["eval_strategy"] = "steps"
        base_training_args["eval_steps"] = 0.1
        base_training_args["save_strategy"] = "steps"
        base_training_args["save_steps"] = 0.1
        base_training_args["logging_strategy"] = "steps"
        base_training_args["logging_steps"] = 10
        base_training_args["save_total_limit"] = 2
        base_training_args["load_best_model_at_end"] = True 

        #base_training_args["gradient_checkpointing_kwargs"] = {'use_reentrant': True}
        ## Bonus 
        base_training_args["report_to"] = "wandb"

        if self.test_run:
            base_training_args["max_steps"] = 10
            base_training_args["eval_strategy"] = "no" 
            base_training_args["load_best_model_at_end"] = False  
        
        if self.other_args:
            base_training_args.update(**self.other_args)
            
        if self.config.task.args:
            base_training_args.update(**self.config.task.args.toDict())

        return base_training_args


    def prepare_peft_config(self):
        """
        Build the LoRA configuration dict from the experiment config.

        Reads ``config.task.lora`` (a dict of LoRA hyperparameters) and merges
        it on top of safe defaults.  Returns ``None`` if LoRA is not configured,
        which causes ``train`` to raise a ``ValueError`` (training always uses
        LoRA in this pipeline).

        Returns:
            dict or None: LoRA config kwargs for ``FastLanguageModel.get_peft_model``,
            or ``None`` if ``config.task.lora`` is falsy.
        """
        peft_config = None
        if self.config.task.lora:
            lora_config = {
                "r": 32, 
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", 
                                   "gate_proj", "up_proj", "down_proj"]
                #"task_type": "CAUSAL_LM",
            }
            lora_config.update(self.config.task.lora)
            return lora_config

        return None  
    
    
def supports_flash_attention():
    """Check if a GPU supports FlashAttention."""

    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    major, minor = torch.cuda.get_device_capability(DEVICE)
    
    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0
    
    return is_sm8x or is_sm90
