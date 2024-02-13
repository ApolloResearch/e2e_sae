"""Run the training script with different config params.

TODO: Replace this with wandb sweeps.
Usage:
    python run_sweep.py <path/to/base/config.yaml>
"""

import yaml
from fire import Fire

from sparsify.scripts.train_tlens_saes.run_train_tlens_saes import Config
from sparsify.scripts.train_tlens_saes.run_train_tlens_saes import main as run_train
from sparsify.utils import replace_pydantic_model


def main(config_path_str: str) -> None:
    """Run the training script with different sae_position_name values."""
    values = [f"blocks.{i}.hook_resid_post" for i in [0, 2, 4, 5]]

    with open(config_path_str) as f:
        base_config = Config(**yaml.safe_load(f))

    for value in values:
        update_dict = {"saes": {"sae_position_name": value}}
        new_config = replace_pydantic_model(base_config, update_dict)
        print(new_config)
        run_train(new_config)


if __name__ == "__main__":
    Fire(main)
