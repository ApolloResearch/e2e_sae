import random
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
import yaml
from pydantic import BaseModel
from pydantic.v1.utils import deep_update
from torch import nn

from sparsify.log import logger
from sparsify.settings import REPO_ROOT

T = TypeVar("T", bound=BaseModel)


def to_root_path(path: str | Path):
    """Converts relative paths to absolute ones, assuming they are relative to the rib root."""
    return Path(path) if Path(path).is_absolute() else Path(REPO_ROOT / path)


def save_model(
    config_dict: dict[str, Any],
    save_dir: Path,
    model: nn.Module,
    model_filename: str,
) -> None:
    """Save the model and config to the save_dir.

    The config will only be saved if the save_dir doesn't exist (i.e. the first time the model is
    saved assuming the save_dir is unique to the model).

    Args:
        config_dict: Dictionary representation of the config to save.
        save_dir: Directory to save the model and config to.
        model: The model to save.
        model_filename: The name of the file to save the model to.

    """
    # If the save_dir doesn't exist, create it and save the config
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        filename = save_dir / "config.yaml"
        logger.info("Saving config to %s", filename)
        with open(filename, "w") as f:
            yaml.dump(config_dict, f)

    torch.save(model.state_dict(), save_dir / model_filename)
    logger.info("Saved model to %s", save_dir / model_filename)


def load_config(config_path_or_obj: Path | str | T, config_model: type[T]) -> T:
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
    with open(config_path_or_obj) as f:
        config_dict = yaml.safe_load(f)
    return config_model(**config_dict)


def set_seed(seed: int | None) -> None:
    """Set the random seed for random, PyTorch and NumPy"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


BaseModelType = TypeVar("BaseModelType", bound=BaseModel)


def replace_pydantic_model(model: BaseModelType, *updates: dict[str, Any]) -> BaseModelType:
    """Create a new model with (potentially nested) updates in the form of dictionaries.

    Args:
        model: The model to update.
        updates: The zero or more dictionaries of updates that will be applied sequentially.

    Returns:
        A replica of the model with the updates applied.

    Examples:
        >>> class Foo(BaseModel):
        ...     a: int
        ...     b: int
        >>> foo = Foo(a=1, b=2)
        >>> foo2 = replace_pydantic_model(foo, {"a": 3})
        >>> foo2
        Foo(a=3, b=2)
        >>> class Bar(BaseModel):
        ...     foo: Foo
        >>> bar = Bar(foo={"a": 1, "b": 2})
        >>> bar2 = replace_pydantic_model(bar, {"foo": {"a": 3}})
        >>> bar2
        Bar(foo=Foo(a=3, b=2))
    """
    return model.__class__(**deep_update(model.model_dump(), *updates))
