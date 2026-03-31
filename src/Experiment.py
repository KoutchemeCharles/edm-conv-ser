"""
Base experiment class for all pipeline stages.

An Experiment encapsulates the three resources every stage needs:
  - a configuration (DotMap loaded from a JSON experiment file),
  - a dataframe (the preprocessed student trajectories), and
  - an agent (the language model, either a local vLLM instance or a remote API).

Concrete stages (Preprocessing, SFT, DPO, GRPO, Evaluation) inherit from this
class and override run().
"""

import os
import pandas as pd
from src.model.VLLM import VLLM
from src.model.RemoteModel import RemoteModel
from src.utils.files import create_dir, save_json
from src.data.falcon.FalconCode import FalconCode


class Experiment():
    """
    Base class for all pipeline stages.

    Handles directory setup, dataset loading, and model instantiation so that
    every concrete stage starts with a consistent working environment.

    A config JSON may reference another experiment config in its ``model`` field
    to chain stages (e.g. evaluate the model trained in a previous SFT run).
    In that case ``find_original_model_config`` walks the chain to locate the
    base model configuration.

    Attributes:
        config: DotMap of the experiment configuration.
        test_run (bool): When True the dataframe is limited to 10 rows and
            training uses at most 10 steps — useful for smoke tests.
        is_training (bool): Controls whether the model is loaded with LoRA
            adapter support (UnslothModel) or as a pure inference engine (VLLM).
        save_dir (str): Directory where all outputs are written.
        dataframe (pd.DataFrame): Preprocessed student trajectory dataframe.
        agent: The language model (VLLM, UnslothModel, or RemoteModel).
        ds_handler: Dataset handler instance used during evaluation.
    """

    def __init__(self, config, test_run, lazy_load=False, is_training=False) -> None:
        """
        Initialize the experiment.

        Args:
            config: DotMap of the experiment configuration.
            test_run (bool): Limit data and training to a small subset for
                quick debugging.
            lazy_load (bool): When True, skip dataset and model loading.
                Useful for constructing an Experiment solely to resolve
                path information from a chained config.
            is_training (bool): When True, load the model in training mode
                (UnslothModel with LoRA support) rather than inference mode.
        """
        self.config = config
        self.test_run = test_run
        self.is_training = is_training

        self.__init_directories()
        if not lazy_load:
            self.dataframe = self.__load_dataframe()
            self.dataframe.to_csv(self.dataframe_save_path)
            if self.config.model:
                self.agent = self.__load_agent()

    def run(self):
        """Execute the stage. Must be overridden by subclasses."""
        return

    def __init_directories(self):
        """
        Create the experiment output directory and resolve all output paths.

        The directory is named ``<save_dir>/<experiment_name>`` and contains:
        - ``experiment_configuration.json`` — a snapshot of the config used
        - ``dataframe.csv``                 — the preprocessed input data
        - ``generations.csv``               — evaluation/training outputs
        """
        if self.test_run:
            self.config.name = self.config.name + "_test_run"
        self.save_dir = os.path.join(self.config.save_dir, self.config.name)
        create_dir(self.save_dir)
        save_json(self.config, os.path.join(self.save_dir, "experiment_configuration.json"))
        self.results_save_path = os.path.join(self.save_dir, "generations.csv")
        self.model_save_path = self.save_dir
        sub_name = self.config.save_dir.split("/")[-1]
        # Hub path: "koutch/<experiment-family>_<experiment-name>"
        self.hub_save_path = f"koutch/{sub_name}_{self.config.name}"
        print("Hub save path", self.hub_save_path)
        self.dataframe_save_path = os.path.join(self.save_dir, "dataframe.csv")

    def __load_dataframe(self):
        """
        Load and return the student trajectory dataframe.

        Dataset entries in the config can be of two kinds:

        1. **Raw split** (``ds.name`` starts with ``"falcon"``): loads directly
           from the FalconCode dataset handler.
        2. **Chained experiment** (any other name): reads the ``generations.csv``
           written by a prior preprocessing stage. This is how evaluation stages
           reuse already-preprocessed data.

        Returns:
            pd.DataFrame: Concatenated dataframe with one row per
            (student, problem) trajectory.
        """
        dataframe = []
        for ds in self.config.dataset:
            if ds.name.startswith("falcon"):
                self.ds_handler = FalconCode(ds)
                df = self.ds_handler.get_split()
            else:
                # Load from a previously saved preprocessing run
                exp = Experiment(ds, test_run=self.test_run, lazy_load=True)
                df = pd.read_csv(exp.results_save_path)
                df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
                df = df.dropna(axis=1, how="all")
                if "messages" in df:
                    df.messages = df.messages.apply(
                        lambda m: eval(m) if type(m) == str else m
                    )

                self.ds_handler = FalconCode(ds.dataset[0])
                self.ds_handler.dataframe = df
                print(df, df.columns)

            if self.test_run:
                df = df.iloc[:10]
            dataframe.append(df)

        df = pd.concat(dataframe, axis=0, ignore_index=True)
        df.to_csv(self.dataframe_save_path)
        return df

    def __load_agent(self):
        """
        Instantiate and return the language model agent.

        Supports three model sources defined in ``config.model.source``:

        - ``"openai"`` / ``"huggingface"`` / ``"groq"`` / ``"google"``:
          returns a :class:`~src.model.RemoteModel.RemoteModel` that calls the
          respective external API.
        - ``"local"``: returns either an
          :class:`~src.model.UnslothModel.UnslothModel` (training mode) or a
          :class:`~src.model.VLLM.VLLM` (inference mode).
        - **Chained config** (``"model"`` key present in model config): resolves
          the checkpoint path from the prior experiment's output directory and
          loads it accordingly.

        Returns:
            The agent object with a ``batch_query()`` method.

        Raises:
            ValueError: If the model source is unrecognized.
        """
        if self.config.model.source in ("openai", "huggingface", "groq", "google"):
            return RemoteModel(self.config.model)

        if self.config.model.source == "local":
            if self.is_training:
                from src.model.UnslothModel import UnslothModel

                agent_config = self.config.model
                if self.config.task.need_fast_inference:
                    print("Auto-setting fast inference for GRPO")
                    agent_config.from_pretrained_kwargs.fast_inference = True

                return UnslothModel(agent_config,
                                    is_training=self.is_training,
                                    was_trained=False)
            else:
                vllm_kwargs = self.config.model.vllm_kwargs.toDict()
                return VLLM(self.config.model, **vllm_kwargs)

        # Load a model that was trained in a prior experiment stage.
        # The config's ``model`` field is itself another experiment config,
        # so we resolve the save path from that experiment.
        elif "model" in self.config.model:
            exp = Experiment(self.config.model,
                             test_run=self.test_run,
                             lazy_load=True)

            agent_config = find_original_model_config(self.config)

            if self.is_training:
                from src.model.UnslothModel import UnslothModel
                agent_config.name = exp.model_save_path

                if self.config.task.need_fast_inference:
                    print("Auto-setting fast inference for GRPO")
                    agent_config.from_pretrained_kwargs.fast_inference = True

                return UnslothModel(agent_config,
                                    is_training=True,
                                    was_trained=True)
            else:
                # Inference always reads the merged model uploaded to the Hub
                agent_config.name = exp.hub_save_path
                return VLLM(agent_config, **agent_config.vllm_kwargs.toDict())
        else:
            raise ValueError(f"Unrecognized model source in config: {self.config.model}")


def find_original_model_config(config):
    """
    Walk a chain of experiment configs to find the base model configuration.

    Experiments can be chained: an evaluation config's ``model`` field may
    point to a training config, whose ``model`` field may in turn point to
    a preprocessing config.  This function traverses the chain until it
    reaches a leaf node (a config with no further ``model`` nesting) and
    returns that leaf, which holds the original HuggingFace model name and
    inference parameters.

    Args:
        config: The top-level experiment DotMap.

    Returns:
        DotMap: The innermost model configuration.
    """
    agent_config = config.model
    while True:
        if not agent_config.model:
            break
        agent_config = agent_config.model
    return agent_config
