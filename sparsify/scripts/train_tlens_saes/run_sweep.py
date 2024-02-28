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
    sweep_name = "gpt2-small_compare-with-joseph_layerwise_2024_02_27"
    update_dicts =  [
            {   
                "train":{
                    "lr": 0.0004,
                    "loss_configs":{
                        "sparsity": {
                            "coeff": 0.00008
                        }
                    },
                },
                "saes": {
                    "sae_position_names": "blocks.2.hook_resid_post"
                },
                "wandb_project": sweep_name,
            },
            #########################################
            {   
                "train":{
                    "lr": 0.004,
                    "loss_configs":{
                        "sparsity": {
                            "coeff": 0.00008
                        }
                    },
                },
                "saes": {
                    "sae_position_names": "blocks.2.hook_resid_post"
                },
                "wandb_project": sweep_name,
            },
            #########################################
            {   
                "train":{
                    "lr": 0.0004,
                    "loss_configs":{
                        "sparsity": {
                            "coeff": 0.0008
                        }
                    },
                },
                "saes": {
                    "sae_position_names": "blocks.2.hook_resid_post"
                },
                "wandb_project": sweep_name,
            },
    ]

    with open(config_path_str) as f:
        base_config = Config(**yaml.safe_load(f))

    for update_dict in update_dicts:
        new_config = replace_pydantic_model(base_config, update_dict)
        print(new_config)
        run_train(new_config)

if __name__ == "__main__":
    Fire(main)
