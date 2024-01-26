
import torch
from jaxtyping import Float
from torch import Tensor

from sparsify.configs import Config


def calc_loss(
    config: Config,
    orig_acts: dict[str, Tensor],
    sae_acts: dict[str, dict[str, Tensor]],
) -> Float[Tensor, ""]:
    """Compute loss between orig_acts and sae_acts.

    Args:
        orig_acts: Dictionary of original activations, keyed by tlens attribute
        sae_acts: Dictionary of SAE activations. First level keys should match orig_acts.
            Second level keys are "output" and "c".
        sae_orig: Whether to use original activations in loss.
        sae_sparsity: Whether to use sparsity in loss.

    Returns:
        loss: Scalar tensor representing the loss.
    """
    assert set(orig_acts.keys()) == set(sae_acts.keys()), (
        f"Keys of orig_acts and sae_acts must match, got {orig_acts.keys()} and "
        f"{sae_acts.keys()}"
    )
    loss: Float[Tensor, ""] = 0.0
    for name, orig_act in orig_acts.items():  # TODO make losses similar to MLP version
        # Convert from inference tensor. TODO: Make more memory efficient
        orig_act = orig_act.clone()
        sae_act = sae_acts[name]
        if config.train.loss_include_sae_orig:
            loss += torch.nn.functional.mse_loss(orig_act, sae_act["output"])
        if config.train.loss_include_sae_sparsity:
            loss += torch.norm(sae_act["c"], p=0.6, dim=-1).mean()
    return loss
