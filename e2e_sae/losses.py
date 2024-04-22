from typing import Annotated

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from torch import Tensor

from e2e_sae.hooks import CacheActs, SAEActs


def _layer_norm_pre(x: Float[Tensor, "... dim"], eps: float = 1e-5) -> Float[Tensor, "... dim"]:
    """Layernorm without the affine transformation."""
    x = x - x.mean(dim=-1, keepdim=True)
    scale = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return x / scale


def calc_explained_variance(
    pred: Float[Tensor, "... dim"], target: Float[Tensor, "... dim"], layer_norm: bool = False
) -> Float[Tensor, "..."]:
    """Calculate the explained variance of the pred and target.

    Args:
        pred: The prediction to compare to the target.
        target: The target to compare the prediction to.
        layer_norm: Whether to apply layer norm to the pred and target before calculating the loss.

    Returns:
        The explained variance between the prediction and target for each batch and sequence pos.
    """
    if layer_norm:
        pred = _layer_norm_pre(pred)
        target = _layer_norm_pre(target)
    sample_dims = tuple(range(pred.ndim - 1))
    per_token_l2_loss = (pred - target).pow(2).sum(dim=-1)
    total_variance = (target - target.mean(dim=sample_dims)).pow(2).sum(dim=-1)
    return 1 - per_token_l2_loss / total_variance


class SparsityLoss(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    coeff: float
    p_norm: float = 1.0

    def calc_loss(self, c: Float[Tensor, "... c"], dense_dim: int) -> Float[Tensor, ""]:
        """Calculate the sparsity loss.
        Note that we divide by the dimension of the input to the SAE. This helps with using the same
        hyperparameters across different model sizes (input dimension is more relevant than the c
        dimension for Lp loss).
        Args:
            c: The activations after the non-linearity in the SAE.
            dense_dim: The dimension of the input to the SAE. Used to normalize the loss.
        Returns:
            The L_p norm of the activations.
        """
        return torch.norm(c, p=self.p_norm, dim=-1).mean() / dense_dim


class InToOrigLoss(BaseModel):
    """Config for the loss between the input and original activations.

    The input activations may come from the input to an SAE or the activations at a cache_hook.

    Note that `run_train_tlens_saes.evaluate` will automatically log the in_to_orig loss for all
    residual stream positions, so you do not need to set values here with coeff=0.0 for logging.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    total_coeff: float = Field(
        ..., description="The sum of coefficients equally weighted across all hook_positions."
    )
    hook_positions: Annotated[
        list[str], BeforeValidator(lambda x: [x] if isinstance(x, str) else x)
    ] = Field(
        ...,
        description="The exact hook positions at which to compare raw and SAE-augmented "
        "activations. E.g. 'blocks.3.hook_resid_post' or "
        "['blocks.3.hook_resid_post', 'blocks.5.hook_resid_post'].",
    )

    @property
    def coeff(self) -> float:
        """The coefficient for the loss of each hook position."""
        return self.total_coeff / len(self.hook_positions)

    def calc_loss(
        self, input: Float[Tensor, "... dim"], orig: Float[Tensor, "... dim"]
    ) -> Float[Tensor, ""]:
        """Calculate the MSE between the input and orig."""
        return F.mse_loss(input, orig)


class OutToOrigLoss(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    coeff: float

    def calc_loss(
        self, output: Float[Tensor, "... dim"], orig: Float[Tensor, "... dim"]
    ) -> Float[Tensor, ""]:
        """Calculate loss between the output of the SAE and the non-SAE-augmented activations."""
        return F.mse_loss(output, orig)


class OutToInLoss(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    coeff: float

    def calc_loss(
        self, input: Float[Tensor, "... dim"], output: Float[Tensor, "... dim"]
    ) -> Float[Tensor, ""]:
        """Calculate loss between the input and output of the SAE."""
        return F.mse_loss(input, output)


class LogitsKLLoss(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    coeff: float

    def calc_loss(
        self, new_logits: Float[Tensor, "... vocab"], orig_logits: Float[Tensor, "... vocab"]
    ) -> Float[Tensor, ""]:
        """Calculate KL divergence between SAE-augmented and non-SAE-augmented logits.

        Important: new_logits should be passed first as we want the relative entropy from
        new_logits to orig_logits - KL(new_logits || orig_logits).

        We flatten all but the last dimensions and take the mean over this new dimension.
        """
        new_logits_flat = einops.rearrange(new_logits, "... vocab -> (...) vocab")
        orig_logits_flat = einops.rearrange(orig_logits, "... vocab -> (...) vocab")

        return F.kl_div(
            F.log_softmax(new_logits_flat, dim=-1),
            F.log_softmax(orig_logits_flat, dim=-1),
            log_target=True,
            reduction="batchmean",
        )


class LossConfigs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sparsity: SparsityLoss
    in_to_orig: InToOrigLoss | None
    out_to_orig: OutToOrigLoss | None
    out_to_in: OutToInLoss | None
    logits_kl: LogitsKLLoss | None

    @property
    def activation_loss_configs(
        self,
    ) -> dict[str, SparsityLoss | InToOrigLoss | OutToOrigLoss | OutToInLoss | None]:
        return {
            "sparsity": self.sparsity,
            "in_to_orig": self.in_to_orig,
            "out_to_orig": self.out_to_orig,
            "out_to_in": self.out_to_in,
        }


def calc_loss(
    orig_acts: dict[str, Tensor],
    new_acts: dict[str, SAEActs | CacheActs],
    orig_logits: Float[Tensor, "batch pos vocab"] | None,
    new_logits: Float[Tensor, "batch pos vocab"] | None,
    loss_configs: LossConfigs,
    is_log_step: bool = False,
    train: bool = True,
) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
    """Compute losses.

    Note that some losses may be computed on the final logits, while others may be computed on
    intermediate activations.

    Additionally, for cache activations, only the in_to_orig loss is computed.

    Args:
        orig_acts: Dictionary of original activations, keyed by tlens attribute.
        new_acts: Dictionary of SAE or cache activations. Keys should match orig_acts.
        orig_logits: Logits from non-SAE-augmented model.
        new_logits: Logits from SAE-augmented model.
        loss_configs: Config for the losses to be computed.
        is_log_step: Whether to store additional loss information for logging.
        train: Whether in train or evaluation mode. Only affects the keys of the loss_dict.

    Returns:
        loss: Scalar tensor representing the loss.
        loss_dict: Dictionary of losses, keyed by loss type and name.
    """
    assert set(orig_acts.keys()) == set(new_acts.keys()), (
        f"Keys of orig_acts and new_acts must match, got {orig_acts.keys()} and "
        f"{new_acts.keys()}"
    )

    prefix = "loss/train" if train else "loss/eval"

    loss: Float[Tensor, ""] = torch.zeros(
        1, device=next(iter(orig_acts.values())).device, dtype=next(iter(orig_acts.values())).dtype
    )
    loss_dict = {}

    if loss_configs.logits_kl and orig_logits is not None and new_logits is not None:
        loss_dict[f"{prefix}/logits_kl"] = loss_configs.logits_kl.calc_loss(
            new_logits=new_logits, orig_logits=orig_logits
        )
        loss = loss + loss_configs.logits_kl.coeff * loss_dict[f"{prefix}/logits_kl"]

    for name, orig_act in orig_acts.items():
        # Convert from inference tensor.
        orig_act = orig_act.detach().clone()
        new_act = new_acts[name]

        for config_type, loss_config in loss_configs.activation_loss_configs.items():
            if isinstance(new_act, CacheActs) and not isinstance(loss_config, InToOrigLoss):
                # Cache acts are only used for in_to_orig loss
                continue

            var: Float[Tensor, "batch_token"] | None = None  # noqa: F821
            var_ln: Float[Tensor, "batch_token"] | None = None  # noqa: F821
            if isinstance(loss_config, InToOrigLoss) and name in loss_config.hook_positions:
                # Note that out_to_in can calculate losses using CacheActs or SAEActs.
                loss_val = loss_config.calc_loss(new_act.input, orig_act)
                var = calc_explained_variance(
                    new_act.input.detach().clone(), orig_act, layer_norm=False
                )
                var_ln = calc_explained_variance(
                    new_act.input.detach().clone(), orig_act, layer_norm=True
                )
            elif isinstance(loss_config, OutToOrigLoss):
                assert isinstance(new_act, SAEActs)
                loss_val = loss_config.calc_loss(new_act.output, orig_act)
                var = calc_explained_variance(
                    new_act.output.detach().clone(), orig_act, layer_norm=False
                )
                var_ln = calc_explained_variance(
                    new_act.output.detach().clone(), orig_act, layer_norm=True
                )
            elif isinstance(loss_config, OutToInLoss):
                assert isinstance(new_act, SAEActs)
                loss_val = loss_config.calc_loss(new_act.input, new_act.output)
                var = calc_explained_variance(
                    new_act.input.detach().clone(), new_act.output, layer_norm=False
                )
                var_ln = calc_explained_variance(
                    new_act.input.detach().clone(), new_act.output, layer_norm=True
                )
            elif isinstance(loss_config, SparsityLoss):
                assert isinstance(new_act, SAEActs)
                loss_val = loss_config.calc_loss(new_act.c, dense_dim=new_act.input.shape[-1])
            else:
                assert loss_config is None or (
                    isinstance(loss_config, InToOrigLoss) and name not in loss_config.hook_positions
                ), f"Unexpected loss_config {loss_config} for name {name}"
                continue

            loss = loss + loss_config.coeff * loss_val
            loss_dict[f"{prefix}/{config_type}/{name}"] = loss_val.detach().clone()

            if (
                var is not None
                and var_ln is not None
                and is_log_step
                and isinstance(loss_config, InToOrigLoss | OutToOrigLoss | OutToInLoss)
            ):
                loss_dict[f"{prefix}/{config_type}/explained_variance/{name}"] = var.mean()
                loss_dict[f"{prefix}/{config_type}/explained_variance_std/{name}"] = var.std()
                loss_dict[f"{prefix}/{config_type}/explained_variance_ln/{name}"] = var_ln.mean()
                loss_dict[f"{prefix}/{config_type}/explained_variance_ln_std/{name}"] = var_ln.std()

    return loss, loss_dict
