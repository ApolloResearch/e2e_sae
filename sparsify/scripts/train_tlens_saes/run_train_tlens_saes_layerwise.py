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
from sparsify.loader import load_tlens_model
from sparsify.log import logger
from sparsify.models.transformers import SAETransformer
from sparsify.scripts.train_tlens_saes.run_train_tlens_saes import Config, train
from sparsify.utils import filter_names, load_config, replace_pydantic_model, set_seed


def main(config_path_or_obj: Path | str | Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_config = load_config(config_path_or_obj, config_model=Config)
    set_seed(raw_config.seed)

    data_loader = create_data_loader(
        raw_config.train_data, batch_size=raw_config.batch_size, global_seed=raw_config.seed
    )[0]
    tlens_model = load_tlens_model(raw_config.tlens_model_name, raw_config.tlens_model_path)

    raw_sae_positions = filter_names(
        list(tlens_model.hook_dict.keys()), raw_config.saes.sae_positions
    )

    logger.info(f"Training SAEs layer-wise at positions: {raw_sae_positions}")
    # Train only one sae_position at a time
    for sae_position in raw_sae_positions:
        logger.info(f"Training SAE at position: {sae_position}")
        # Create a new config for each sae_position
        config = replace_pydantic_model(raw_config, {"saes": {"sae_positions": sae_position}})
        model = SAETransformer(
            config=config, tlens_model=tlens_model, raw_sae_positions=[sae_position]
        ).to(device=device)

        trainable_param_names = [name for name, _ in model.saes.named_parameters()]
        train(
            config=config,
            model=model,
            train_loader=data_loader,
            trainable_param_names=trainable_param_names,
            device=device,
        )


if __name__ == "__main__":
    fire.Fire(main)
