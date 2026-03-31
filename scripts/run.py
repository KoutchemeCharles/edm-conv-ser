"""
Run the annotation process
"""
import os
os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"

from argparse import ArgumentParser
from src.utils.core import set_seed
from src.utils.files import read_config

def parse_args():
    parser = ArgumentParser(description="Running experiments")
    parser.add_argument("--config", required=True,
                        help="Path towards the configuration file")
    parser.add_argument('--test_run',
                        help="Whether to do a test run to ensure the pipeline works without issues",
                        action="store_true")

    return parser.parse_args()


def load_experiment(config):
    if "preprocess" in config.name:
        from src.Preprocess import Preprocessing
        experiment = Preprocessing 
    elif "sft" in config.name:
        import unsloth
        from src.trl.SFT import SFT
        experiment = SFT   
    elif "dpo" in config.name:
        import unsloth
        from src.trl.DPO import DPO 
        experiment = DPO 
    elif "grpo" in config.name:
        import unsloth
        from src.trl.GRPO import GRPO
        experiment = GRPO
    elif "eval" in config.task.name:
        from src.Evaluation import Evaluation      
        experiment = Evaluation
    else:
        raise ValueError(f"Unknown experiment for config {config.name}")
        
    return experiment

def main():
    args = parse_args()
    config = read_config(args.config)
    set_seed(config.seed)

    EXP_CLASS = load_experiment(config)
    experiment = EXP_CLASS(config, test_run=args.test_run)
    experiment.run()


if __name__ == "__main__":
    main()
