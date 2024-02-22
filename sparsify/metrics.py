import torch
import wandb
from jaxtyping import Float
from torch import Tensor
from transformer_lens.utils import lm_cross_entropy_loss


class DiscreteMetrics:
    """Manages metrics such as dict activation frequencies and alive dictionary elements."""

    def __init__(self, dict_sizes: dict[str, int], has_pos_dim: bool, device: torch.device) -> None:
        """Initialize the DiscreteMetrics object.

        Args:
            dict_sizes: Sizes of the dictionaries for each sae position.
            has_pos_dim: Whether the sae activations have a position dimension.
            device: Device to store the dictionary element frequencies on.
        """
        self.has_pos_dim = has_pos_dim
        self.tokens_used = 0  # Number of tokens used in dict_el_frequencies
        self.dict_el_frequencies: dict[str, Float[Tensor, "dims"]] = {  # noqa: F821
            sae_pos: torch.zeros(dict_size, device=device)
            for sae_pos, dict_size in dict_sizes.items()
        }

    def update_dict_el_frequencies(
        self, sae_acts: dict[str, dict[str, Float[Tensor, "... dim"]]], batch_tokens: int
    ) -> None:
        """Update the dictionary element frequencies with the new batch frequencies.

        Args:
            sae_acts: Dictionary of activations for each SAE position.
            batch_tokens: Number of tokens used to produce the sae acts.
        """
        sum_dims = (0, 1) if self.has_pos_dim else (0,)
        for sae_pos in self.dict_el_frequencies:
            self.dict_el_frequencies[sae_pos] += (sae_acts[sae_pos]["c"] != 0).sum(dim=sum_dims)
        self.tokens_used += batch_tokens

    def collect_for_logging(self, log_wandb_histogram: bool = True) -> dict[str, list[float] | int]:
        """Collect the discrete metrics for logging.

        Currently collects:
        - The number of alive dictionary elements for each hook.
        - The histogram of dictionary element activation frequencies for each hook (if
          log_wandb_histogram is True).

        Note that the dictionary element frequencies are divided by the number of tokens used to
        calculate them.

        Args:
            log_wandb_histogram: Whether to log the dictionary element activation frequency
                histograms to wandb.
        """
        log_dict = {}
        for sae_pos in self.dict_el_frequencies:
            self.dict_el_frequencies[sae_pos] /= self.tokens_used

            log_dict[f"sparsity/alive_dict_elements/{sae_pos}"] = (
                self.dict_el_frequencies[sae_pos].gt(0).sum().item()
            )

            if log_wandb_histogram:
                data = [[s] for s in self.dict_el_frequencies[sae_pos]]
                table = wandb.Table(data=data, columns=["dict element activation frequency"])
                plot = wandb.plot.histogram(
                    table,
                    "dict element activation frequency",
                    title=f"{sae_pos} (most_recent_n_tokens={self.tokens_used} "
                    f"dict_size={self.dict_el_frequencies[sae_pos].shape[0]})",
                )
                log_dict[f"sparsity/dict_el_frequencies_hist/{sae_pos}"] = plot

        return log_dict


def collect_wandb_metrics(
    loss: float,
    grad_updates: int,
    sae_acts: dict[str, dict[str, Float[Tensor, "... dim"]]],
    loss_dict: dict[str, Float[Tensor, ""]],
    grad_norm: float | None,
    orig_logits: Float[Tensor, "... dim"] | None,
    new_logits: Float[Tensor, "... dim"] | None,
    tokens: Float[Tensor, "... dim"],
    lr: float,
) -> dict[str, int | float]:
    """Collect metrics for logging to wandb.

    Args:
        loss: The final loss value.
        grad_updates: The number of gradient updates performed.
        sae_acts: Dictionary of activations for each SAE position.
        loss_dict: Dictionary of loss values that make up the final loss.
        grad_norm: The norm of the gradients.
        orig_logits: The logits produced by the original model.
        new_logits: The logits produced by the SAE model.
        tokens: The tokens used to produce the logits and activations.
        lr: The learning rate used for the current update.

    Returns:
        Dictionary of metrics to log to wandb.
    """
    wandb_log_info = {"loss": loss, "grad_updates": grad_updates, "lr": lr}
    for name, sae_act in sae_acts.items():
        # Record L_0 norm of the cs
        l_0_norm = torch.norm(sae_act["c"], p=0, dim=-1).mean().item()
        wandb_log_info[f"sparsity/L_0/{name}"] = l_0_norm

        # Record fraction of zeros in the cs
        frac_zeros = ((sae_act["c"] == 0).sum() / sae_act["c"].numel()).item()
        wandb_log_info[f"sparsity/frac_zeros/{name}"] = frac_zeros

    for loss_name, loss_value in loss_dict.items():
        wandb_log_info[loss_name] = loss_value.item()

    if grad_norm is not None:
        wandb_log_info["grad_norm"] = grad_norm

    if new_logits is not None and orig_logits is not None:
        orig_model_performance_loss = lm_cross_entropy_loss(orig_logits, tokens, per_token=False)
        sae_model_performance_loss = lm_cross_entropy_loss(new_logits, tokens, per_token=False)

        wandb_log_info.update(
            {
                "performance/orig_model_ce_loss": orig_model_performance_loss.item(),
                "performance/sae_model_ce_loss": sae_model_performance_loss.item(),
                "performance/difference_loss": (
                    orig_model_performance_loss - sae_model_performance_loss
                ).item(),
            },
        )
    return wandb_log_info
