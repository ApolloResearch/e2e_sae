import random
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
import yaml
from pydantic import BaseModel
from pydantic.v1.utils import deep_update
from torch import nn
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from sparsify.log import logger
from sparsify.settings import REPO_ROOT

T = TypeVar("T", bound=BaseModel)


def to_root_path(path: str | Path):
    """Converts relative paths to absolute ones, assuming they are relative to the rib root."""
    return Path(path) if Path(path).is_absolute() else Path(REPO_ROOT / path)


def save_module(
    config_dict: dict[str, Any],
    save_dir: Path,
    module: nn.Module,
    model_filename: str,
) -> None:
    """Save the pytorch module and config to the save_dir.

    The config will only be saved if the save_dir doesn't exist (i.e. the first time the module is
    saved assuming the save_dir is unique to the module).

    Args:
        config_dict: Dictionary representation of the config to save.
        save_dir: Directory to save the module.
        module: The module to save.
        model_filename: The name of the file to save the module to.

    """
    # If the save_dir doesn't exist, create it and save the config
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        filename = save_dir / "config.yaml"
        logger.info("Saving config to %s", filename)
        with open(filename, "w") as f:
            yaml.dump(config_dict, f)

    torch.save(module.state_dict(), save_dir / model_filename)
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


def filter_names(all_names: list[str], filter_names: list[str]) -> list[str]:
    """Use filter_names to filter `all_names` by partial match.

    The filtering is done by checking if any of the filter_names are in the all_names. Partial
    matches are allowed. E.g. "hook_resid_pre" matches ["blocks.0.hook_resid_pre",
    "blocks.1.hook_resid_pre", ...].

    Args:
        all_names: The names to filter.
        filter_names: The names to use to filter all_names by partial match.
    Returns:
        The filtered names.
    """
    return [name for name in all_names if any(filter_name in name for filter_name in filter_names)]


def get_hook_shapes(tlens_model: HookedTransformer, hook_names: list[str]) -> dict[str, list[int]]:
    """Get the shapes of activations at the hook points labelled by hook_names"""
    # Sadly I can't see any way to easily get the shapes of activations at hook_points without
    # actually running the model.
    hook_shapes = {}

    def get_activation_shape_hook_function(activation: torch.Tensor, hook: HookPoint) -> None:
        hook_shapes[hook.name] = activation.shape

    def hook_names_filter(name: str) -> bool:
        return name in hook_names

    test_prompt = torch.tensor([0])
    tlens_model.run_with_hooks(
        test_prompt,
        return_type=None,
        fwd_hooks=[(hook_names_filter, get_activation_shape_hook_function)],
    )
    return hook_shapes
