"""
Preprocessing stage: build and persist the student trajectory dataframe.

This is the first stage in the pipeline. It loads raw execution logs from the
FalconCode dataset, filters and conversationalizes student trajectories, and
writes the resulting dataframe to disk so that training and evaluation stages
can reuse it without re-running the (slow) preprocessing step.
"""

import os
import pandas as pd
from copy import deepcopy
from src.model.VLLM import VLLM
from src.model.RemoteModel import RemoteModel
from src.utils.files import create_dir, save_json
from src.data.falcon.FalconCode import FalconCode
from src.Experiment import Experiment


class Preprocessing(Experiment):
    """
    Preprocessing stage: build and save the student trajectory dataframe.

    Inherits dataset loading from :class:`~src.Experiment.Experiment`.
    The ``run()`` method simply persists the already-loaded dataframe to
    ``<save_dir>/generations.csv`` so that downstream stages (SFT, DPO,
    evaluation) can reference it via a chained experiment config instead
    of re-running preprocessing.
    """

    def __init__(self, config, test_run, lazy_load=False, is_training=False) -> None:
        """
        Initialize the preprocessing stage.

        Args:
            config: DotMap of the experiment configuration.
            test_run (bool): Limit data to 10 rows for quick smoke tests.
            lazy_load (bool): Unused here; always False for preprocessing.
            is_training (bool): Unused here; preprocessing never loads a model.
        """
        super().__init__(config, test_run, is_training=False, lazy_load=False)

    def run(self):
        """
        Save the preprocessed dataframe to disk.

        Writes the dataframe to ``results_save_path`` (``generations.csv``).
        Downstream stages that reference this experiment in their config will
        read from this file instead of re-running the FalconCode pipeline.
        """
        self.dataframe.to_csv(self.results_save_path)
        print('Saved dataframe to path', self.results_save_path)
        print("Columns in df", self.dataframe, self.dataframe.columns)
