from typing import Any, Union

from jaxtyping import Float
from torch import Tensor, nn

from sparsify.models.sparsifiers import SAE

InputActType = Union[
    tuple[Float[Tensor, "batch emb_in"]],
    tuple[Float[Tensor, "batch pos emb_in"]],
    tuple[Float[Tensor, "batch pos _"], ...],
]


def sae_acts_pre_forward_hook_fn(
    module: nn.Module,
    inputs: InputActType,
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    sae: SAE,
) -> None:
    """Hook function for storing the output activations.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module (not used).
        output: Output of the module. Handles modules with one or two outputs of varying sizes
            and with or without positional indices. If no positional indices, assumes one output.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
    """
    assert len(inputs) == 1, "SAE hook only supports one input."
    assert isinstance(data_key, list) and len(data_key) == 2, "SAE should return an output and a c."
    output, c = sae(inputs[0])
    # Store the output and c
    hooked_data[hook_name] = {data_key[0]: output, data_key[1]: c}
