"""Run the training script with different config params.

TODO: Replace this with wandb sweeps.
Usage:
    python run_sweep.py <path/to/base/config.yaml>
"""

import yaml
from fire import Fire

from e2e_sae.scripts.train_tlens_saes.run_train_tlens_saes import Config
from e2e_sae.scripts.train_tlens_saes.run_train_tlens_saes import main as run_train
from e2e_sae.utils import replace_pydantic_model


def main(config_path_str: str) -> None:
    """Run the training script with different sae_position values."""
    sweep_name = "tinystories-1m_sparsity-coeff"
    values = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    with open(config_path_str) as f:
        base_config = Config(**yaml.safe_load(f))

    for value in values:
        update_dict = {
            "train": {"loss": {"sparsity": {"coeff": value}}},
            "wandb_project": sweep_name,
        }
        new_config = replace_pydantic_model(base_config, update_dict)
        print(new_config)
        run_train(new_config)


if __name__ == "__main__":
    Fire(main)
