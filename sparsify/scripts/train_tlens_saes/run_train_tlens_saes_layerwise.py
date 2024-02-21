"""Script for training multiple SAEs one layer at a time.

Simple wrapper around `run_train_tlens_saes.py` that creates a new config for each SAE position in
the given config and trains the SAEs one at a time.

Usage:
    python run_train_tlens_saes_layerwise.py <path/to/config.yaml>
"""
from pathlib import Path

import fire
import torch

from sparsify.data import create_data_loader
from sparsify.log import logger
from sparsify.models.transformers import SAETransformer
from sparsify.scripts.train_tlens_saes.run_train_tlens_saes import Config, load_tlens_model, train
from sparsify.utils import filter_names, load_config, replace_pydantic_model, set_seed


def main(config_path_or_obj: Path | str | Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_config = load_config(config_path_or_obj, config_model=Config)
    set_seed(raw_config.seed)

    data_loader, _ = create_data_loader(raw_config.data, batch_size=raw_config.train.batch_size)
    tlens_model = load_tlens_model(raw_config)

    raw_sae_position_names = filter_names(
        list(tlens_model.hook_dict.keys()), raw_config.saes.sae_position_names
    )

    logger.info(f"Training SAEs layer-wise at positions: {raw_sae_position_names}")
    # Train only one sae_position at a time
    for sae_position_name in raw_sae_position_names:
        logger.info(f"Training SAE at position: {sae_position_name}")
        # Create a new config for each sae_position
        config = replace_pydantic_model(
            raw_config, {"saes": {"sae_position_names": sae_position_name}}
        )
        model = SAETransformer(
            config=config, tlens_model=tlens_model, raw_sae_position_names=[sae_position_name]
        ).to(device=device)

        trainable_param_names = [name for name, _ in model.saes.named_parameters()]
        train(
            config=config,
            model=model,
            data_loader=data_loader,
            trainable_param_names=trainable_param_names,
            device=device,
        )


if __name__ == "__main__":
    fire.Fire(main)
