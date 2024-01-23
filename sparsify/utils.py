from pathlib import Path
from typing import Any, Dict, Type, TypeVar, Union

import torch
import yaml
from pydantic import BaseModel
from torch import nn

from sparsify.log import logger

T = TypeVar("T", bound=BaseModel)


def save_model(
    config_dict: Dict[str, Any], save_dir: Path, model: nn.Module, epoch: int, sparse: bool
) -> None:
    # If the save_dir doesn't exist, create it and save the config
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        logger.info("Saving config to %s", save_dir)
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f)
    logger.info("Saving model to %s", save_dir)
    if not sparse:
        torch.save(model.state_dict(), save_dir / f"model_epoch_{epoch + 1}.pt")
    else:
        torch.save(model.state_dict(), save_dir / f"sparse_model_epoch_{epoch + 1}.pt")


def get_model_attr(model: torch.nn.Module, attr_path: str) -> torch.nn.Module:
    """Retrieve a nested attribute of a PyTorch module by a string of attribute names.

    Each attribute name in the path is separated by a period ('.').

    Since models often have lists of modules, the attribute path can also include an index.

    Args:
        model (torch.nn.Module): The PyTorch model.
        attr_path (str): A string representing the path to the attribute.

    Returns:
        torch.nn.Module: The attribute of the model.

    Example:
        >>> mlp_config = MLPConfig(hidden_sizes=[5], input_size=2, output_size=3)
        >>> mlp = MLP(mlp_config)
        >>> mlp
        MLP(
            (layers): ModuleList(
                (0): Layer(
                    (linear): Linear(in_features=2, out_features=5, bias=True)
                    (activation): ReLU()
                )
                (1): Layer(
                    (linear): Linear(in_features=5, out_features=3, bias=True)
                )
            )
        )
        - get_model_attr(model, "layers") -> ModuleList(...)
        - get_model_attr(model, "layers.0") -> Layer(...)
        - get_model_attr(model, "layers.0.activation") -> ReLU()
        - get_model_attr(model, "layers.1.linear") -> LinearFoldedBias(...)
    """
    attr_names = attr_path.split(".")
    attr = model

    for name in attr_names:
        try:
            if isinstance(attr, torch.nn.ModuleList) and name.isdigit():
                attr = attr[int(name)]
            else:
                attr = getattr(attr, name)
        except AttributeError:
            logger.error(f"Attribute '{name}' not found in the path '{attr_path}'.")
            raise
    return attr


def load_config(config_path_or_obj: Union[Path, str, T], config_model: Type[T]) -> T:
    """Load the config of class `config_model`, either from YAML file or existing config object.

    Args:
        config_path_or_obj (Union[Path, str, `config_model`]): if config object, must be instance
            of `config_model`. If str or Path, this must be the path to a .yaml.
        config_model: the class of the config that we are loading
    """
    if isinstance(config_path_or_obj, config_model):
        return config_path_or_obj

    if isinstance(config_path_or_obj, str):
        config_path_or_obj = Path(config_path_or_obj)

    assert isinstance(
        config_path_or_obj, Path
    ), f"passed config is of invalid type {type(config_path_or_obj)}"
    assert (
        config_path_or_obj.suffix == ".yaml"
    ), f"Config file {config_path_or_obj} must be a YAML file."
    assert Path(config_path_or_obj).exists(), f"Config file {config_path_or_obj} does not exist."
    with open(config_path_or_obj, "r") as f:
        config_dict = yaml.safe_load(f)
    return config_model(**config_dict)
