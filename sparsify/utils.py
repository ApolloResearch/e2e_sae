import math
import random
from collections.abc import Callable
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


def get_linear_lr_schedule(
    warmup_samples: int, cooldown_samples: int, n_samples: int | None, effective_batch_size: int
) -> Callable[[int], float]:
    """
    Generates a linear learning rate schedule function that incorporates warmup and cooldown phases.

    If warmup_samples and cooldown_samples are both 0, the learning rate will be constant at 1.0
    throughout training.

    Args:
        warmup_samples: The number of samples to use for warmup.
        cooldown_samples: The number of samples to use for cooldown.
        effective_batch_size: The effective batch size used during training.

    Returns:
        A function that takes a training step as input and returns the corresponding learning rate.

    Raises:
        ValueError: If the cooldown period starts before the warmup period ends.
        AssertionError: If a cooldown is requested but the total number of samples is not provided.
    """
    warmup_steps = warmup_samples // effective_batch_size
    cooldown_steps = cooldown_samples // effective_batch_size

    if n_samples is None:
        assert cooldown_samples == 0, "Cooldown requested but total number of samples not provided."
        cooldown_start = float("inf")
    else:
        # NOTE: There may be 1 fewer steps if batch_size < effective_batch_size, but this won't
        # make a big difference for most learning setups.
        total_steps = math.ceil(n_samples / effective_batch_size)
        # Calculate the start step for cooldown
        cooldown_start = total_steps - cooldown_steps

        # Check for overlap between warmup and cooldown
        assert (
            cooldown_start > warmup_steps
        ), "Cooldown starts before warmup ends. Adjust your parameters."

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            # Warmup phase: linearly increase learning rate
            return (step + 1) / warmup_steps
        elif step >= cooldown_start:
            # Cooldown phase: linearly decrease learning rate
            # Calculate how many steps have been taken in the cooldown phase
            steps_into_cooldown = step - cooldown_start
            # Linearly decrease the learning rate
            return max(0.0, 1 - (steps_into_cooldown / cooldown_steps))
        else:
            # Maintain maximum learning rate after warmup and before cooldown
            return 1.0

    return lr_schedule
