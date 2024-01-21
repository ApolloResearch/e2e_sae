from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch import nn

from sparsify.log import logger


def save_trainable_params(config_dict: Dict[str, Any], save_dir: Path, model: nn.Module, epoch: int, sparse: bool) -> None:
    # If the save_dir doesn't exist, create it and save the config
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        logger.info("Saving config to %s", save_dir)
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f)
    logger.info("Saving model to %s", save_dir)
    # Get name of trainable parameters
    
    if not sparse:
        torch.save(model.state_dict(), save_dir / f"model_epoch_{epoch + 1}.pt")
    else:
        torch.save(model.state_dict(), save_dir / f"sparse_model_epoch_{epoch + 1}.pt")
