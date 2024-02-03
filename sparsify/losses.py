import torch
from jaxtyping import Float
from torch import Tensor


def calc_loss(
    orig_acts: dict[str, Tensor],
    sae_acts: dict[str, dict[str, Tensor]],
    orig_logits: Float[Tensor, "batch pos vocab"],
    new_logits: Float[Tensor, "batch pos vocab"],
    sae_inp_orig: bool = False,
    sae_out_orig: bool = False,
    sae_inp_sae_out: bool = False,
    sae_sparsity: bool = False,
    sparsity_p_norm: float = 1.0,
    act_sparsity_lambda: float | None = 1.0,
) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
    """Compute loss between orig_acts and sae_acts.

    TODO: Pass in loss config for this

    Args:
        orig_acts: Dictionary of original activations, keyed by tlens attribute
        sae_acts: Dictionary of SAE activations. First level keys should match orig_acts.
            Second level keys are "output" and "c".
        orig_logits: Original logits
        new_logits: New logits

    Returns:
        loss: Scalar tensor representing the loss.
        loss_dict: Dictionary of losses, keyed by loss name.
    """
    assert set(orig_acts.keys()) == set(sae_acts.keys()), (
        f"Keys of orig_acts and sae_acts must match, got {orig_acts.keys()} and "
        f"{sae_acts.keys()}"
    )
    loss: Float[Tensor, ""] = torch.zeros(1, device=orig_logits.device, dtype=orig_logits.dtype)
    loss_dict = {}

    # TODO Future: Maintain a record of batch-element-wise losses

    # Calculate difference of logits
    loss_logits = torch.nn.functional.mse_loss(
        new_logits, orig_logits.clone().detach()
    )  # TODO explore KL and other losses
    loss_dict["loss/logits"] = loss_logits
    loss = loss + loss_logits

    for name, orig_act in orig_acts.items():
        # Convert from inference tensor. TODO: Make more memory efficient
        orig_act = orig_act.clone()
        sae_act = sae_acts[name]

        if sae_inp_orig:
            loss_sae_inp_orig = torch.nn.functional.mse_loss(sae_act["input"], orig_act.detach())
            loss_dict[f"loss/sae_inp_to_orig/{name}"] = loss_sae_inp_orig
            loss = loss + loss_sae_inp_orig
        if sae_out_orig:
            loss_sae_out_orig = torch.nn.functional.mse_loss(sae_act["output"], orig_act.detach())
            loss_dict[f"loss/sae_out_to_orig/{name}"] = loss_sae_out_orig
            loss = loss + loss_sae_out_orig
        if sae_inp_sae_out:
            loss_sae_inp_sae_out = torch.nn.functional.mse_loss(
                sae_act["output"], sae_act["input"].detach()
            )
            loss_dict[f"loss/sae_inp_to_sae_out/{name}"] = loss_sae_inp_sae_out
            loss = loss + loss_sae_inp_sae_out
        if sae_sparsity:
            assert act_sparsity_lambda is not None, "act_sparsity_lambda must be provided"
            loss_sparsity = torch.norm(sae_act["c"], p=sparsity_p_norm, dim=-1).mean()
            loss_dict[f"loss/sparsity/pre-scaling/{name}"] = loss_sparsity.clone()
            loss_sparsity *= act_sparsity_lambda
            loss_dict[f"loss/sparsity/post-scaling/{name}"] = loss_sparsity
            loss = loss + loss_sparsity

        # Record L_0 norm of the cs
        l_0_norm = torch.norm(sae_act["c"], p=0, dim=-1).mean()
        loss_dict[f"sparsity/L_0/{name}"] = l_0_norm

        # Record fraction of zeros in the cs
        frac_zeros = (sae_act["c"] == 0).sum() / sae_act["c"].numel()
        loss_dict[f"sparsity/frac_zeros/{name}"] = frac_zeros

    return loss, loss_dict
