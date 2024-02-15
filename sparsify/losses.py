from typing import Any

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor


class BaseLossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    coeff: float

    def calc_loss(self, *args: Any, **kwargs: Any) -> Float[Tensor, ""]:
        raise NotImplementedError


class SparsityLossConfig(BaseLossConfig):
    p_norm: float = 1.0

    def calc_loss(self, *args: Any, c: Float[Tensor, "... c"], **kwargs: Any) -> Float[Tensor, ""]:
        """Calculate the sparsity loss.

        Args:
            c: The activations after the non-linearity in the SAE.
        """
        return torch.norm(c, p=self.p_norm, dim=-1).mean()


class InpToOrigLossConfig(BaseLossConfig):
    def calc_loss(
        self,
        *args: Any,
        input: Float[Tensor, "... dim"],
        orig: Float[Tensor, "... dim"],
        **kwargs: Any,
    ) -> Float[Tensor, ""]:
        """Calculate loss between the input of the SAE and the non-SAE-augmented activations."""
        return F.mse_loss(input, orig)


class OutToOrigLossConfig(BaseLossConfig):
    def calc_loss(
        self,
        *args: Any,
        output: Float[Tensor, "... dim"],
        orig: Float[Tensor, "... dim"],
        **kwargs: Any,
    ) -> Float[Tensor, ""]:
        """Calculate loss between the output of the SAE and the non-SAE-augmented activations."""
        return F.mse_loss(output, orig)


class InpToOutLossConfig(BaseLossConfig):
    def calc_loss(
        self,
        *args: Any,
        input: Float[Tensor, "... dim"],
        output: Float[Tensor, "... dim"],
        **kwargs: Any,
    ) -> Float[Tensor, ""]:
        """Calculate loss between the input and output of the SAE."""
        return F.mse_loss(input, output)


class LogitsKLLossConfig(BaseLossConfig):
    def calc_loss(
        self,
        *args: Any,
        new_logits: Float[Tensor, "... dim"],
        orig_logits: Float[Tensor, "... dim"],
        **kwargs: Any,
    ) -> Float[Tensor, ""]:
        """Calculate KL divergence between SAE-augmented and non-SAE-augmented logits.

        Important: new_logits should be passed first as we want the relative entropy from
        new_logits to orig_logits - KL(new_logits || orig_logits).

        We flatten all but the last dimensions and take the mean over this new dimension.
        """
        new_logits_flat = einops.rearrange(new_logits, "... dim -> (...) dim")
        orig_logits_flat = einops.rearrange(orig_logits, "... dim -> (...) dim")

        return F.kl_div(
            F.log_softmax(new_logits_flat, dim=-1),
            F.log_softmax(orig_logits_flat, dim=-1),
            log_target=True,
            reduction="batchmean",
        )


class LossConfigs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sparsity: SparsityLossConfig
    inp_to_orig: InpToOrigLossConfig | None
    out_to_orig: OutToOrigLossConfig | None
    inp_to_out: InpToOutLossConfig | None
    logits_kl: LogitsKLLossConfig | None


def calc_loss(
    orig_acts: dict[str, Tensor],
    sae_acts: dict[str, dict[str, Float[Tensor, "... dim"]]],
    orig_logits: Float[Tensor, "batch pos vocab"],
    new_logits: Float[Tensor, "batch pos vocab"],
    loss_configs: LossConfigs,
) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
    """Compute losses.

    Note that some losses may be computed on the final logits, while others may be computed on
    intermediate activations.

    Args:
        orig_acts: Dictionary of original activations, keyed by tlens attribute.
        sae_acts: Dictionary of SAE activations. First level keys should match orig_acts.
            Second level keys are "output" and "c".
        orig_logits: Logits from non-SAE-augmented model.
        new_logits: Logits from SAE-augmented model.
        loss_configs: Config for the losses to be computed.

    Returns:
        loss: Scalar tensor representing the loss.
        loss_dict: Dictionary of losses, keyed by loss type and name.
    """
    assert set(orig_acts.keys()) == set(sae_acts.keys()), (
        f"Keys of orig_acts and sae_acts must match, got {orig_acts.keys()} and "
        f"{sae_acts.keys()}"
    )

    loss: Float[Tensor, ""] = torch.zeros(1, device=orig_logits.device, dtype=orig_logits.dtype)
    loss_dict = {}

    if loss_configs.logits_kl:
        loss_dict["loss/logits_kl"] = loss_configs.logits_kl.calc_loss(
            new_logits=new_logits, orig_logits=orig_logits.detach().clone()
        )
        loss = loss + loss_configs.logits_kl.coeff * loss_dict["loss/logits_kl"]
    # TODO Future: Maintain a record of batch-element-wise losses

    for name, orig_act in orig_acts.items():
        # Convert from inference tensor.
        orig_act = orig_act.detach().clone()
        sae_act = sae_acts[name]

        for config_type in ["inp_to_orig", "out_to_orig", "inp_to_out", "sparsity"]:
            loss_config: BaseLossConfig | None = getattr(loss_configs, config_type)
            if loss_config:
                loss_dict[f"loss/{config_type}/{name}"] = loss_config.calc_loss(
                    input=sae_act["input"],
                    output=sae_act["output"],
                    orig=orig_act,
                    c=sae_act["c"],
                )
                loss = loss + loss_config.coeff * loss_dict[f"loss/{config_type}/{name}"]

    return loss, loss_dict
