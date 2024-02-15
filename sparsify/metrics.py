import einops
import torch
import wandb
from jaxtyping import Float
from torch import Tensor


@torch.inference_mode
def calc_batch_dict_el_frequencies(
    sae_acts: dict[str, dict[str, Float[Tensor, "... dim"]]],
) -> dict[str, list[float]]:
    """Calculate the frequency of dictionary elements in the batch.

    Args:
        sae_acts: Dictionary of SAE activations. First level keys are hook names. Second level keys
            are "input", "output", and "c".

    Returns:
        frequencies: Dictionary of frequencies, keyed by hook name. Each value is a list of floats
            representing the frequency of each dictionary element for this batch.
    """
    frequencies: dict[str, list[float]] = {}
    for hook_name, hook_acts in sae_acts.items():
        # Flatten the batch and pos dimensions
        flat_cs = einops.rearrange(hook_acts["c"], "... dim -> (...) dim")
        frequencies[hook_name] = (flat_cs != 0).float().mean(dim=0).tolist()
    return frequencies


def update_dict_el_frequencies(
    dict_el_frequencies: dict[str, list[float]],
    batch_dict_el_frequencies: dict[str, list[float]],
    tokens_since_last_freq_save: int,
    batch_tokens: int,
) -> None:
    """Update the dictionary element frequencies with the new batch frequencies.

    Takes a weighted average of the old and new frequencies.

    Args:
        dict_el_frequencies: Dictionary of frequencies, keyed by hook name. Each value is a list of
            floats representing the frequency of each dictionary element.
        batch_dict_el_frequencies: Dictionary of frequencies, keyed by hook name. Each value is a
            list of floats representing the frequency of each dictionary element for this batch.
        tokens_since_last_freq_save: Number of tokens used to calculate dict_el_frequencies.
        batch_tokens: Number of tokens in this batch.
    """
    for hook_name, batch_freqs in batch_dict_el_frequencies.items():
        if hook_name not in dict_el_frequencies:
            dict_el_frequencies[hook_name] = [0.0] * len(batch_freqs)
        for i, batch_freq in enumerate(batch_freqs):
            dict_el_frequencies[hook_name][i] = (
                dict_el_frequencies[hook_name][i] * tokens_since_last_freq_save
                + batch_freq * batch_tokens
            ) / (tokens_since_last_freq_save + batch_tokens)


@torch.inference_mode
def calc_sparsity_metrics(
    sae_acts: dict[str, dict[str, Float[Tensor, "... dim"]]],
    dict_el_frequencies: dict[str, list[float]],
    tokens_since_discrete_metrics_save: int,
    discrete_metrics_n_tokens: int,
    batch_tokens: int,
    create_wandb_hist: bool = False,
) -> tuple[dict[str, float | list[float]], int]:
    """Calculate sparsity metrics for the SAE activations.

    NOTE: dict_el_frequencies is updated in place with the new batch frequencies.

    Discrete metrics such as dictionary element activation frequencies and dead neurons are saved
    if there are enough tokens since the last save.

    Args:
        sae_acts: Dictionary of SAE activations. First level keys are hook names. Second level keys
            are "input", "output" and "c".
        dict_el_frequencies: Dictionary of activations frequencies for each dictionary element.
        tokens_since_discrete_metrics_save: Number of tokens since the last discrete metric save.
        discrete_metrics_n_tokens: Save discrete metrics every this many tokens.
        batch_tokens: Number of tokens in the current batch.
        create_wandb_hist: Whether to create a histogram of the activation frequencies for logging
            to Weights & Biases.

    Returns:
        - Dictionary of sparsity metrics, keyed by metric name.
        - Number of tokens since the last discrete metric save.
    """
    batch_dict_el_frequencies = calc_batch_dict_el_frequencies(sae_acts)
    update_dict_el_frequencies(
        dict_el_frequencies=dict_el_frequencies,
        batch_dict_el_frequencies=batch_dict_el_frequencies,
        tokens_since_last_freq_save=tokens_since_discrete_metrics_save,
        batch_tokens=batch_tokens,
    )
    tokens_in_dict: int = tokens_since_discrete_metrics_save + batch_tokens
    save_discrete_metrics: bool = tokens_in_dict >= discrete_metrics_n_tokens

    sparsity_metrics: dict[str, float | list[float]] = {}
    for name, sae_act in sae_acts.items():
        dict_size = sae_act["c"].shape[-1]
        # Record L_0 norm of the cs
        l_0_norm = torch.norm(sae_act["c"], p=0, dim=-1).mean()
        sparsity_metrics[f"sparsity/L_0/{name}"] = l_0_norm

        # Record fraction of zeros in the cs
        frac_zeros = (sae_act["c"] == 0).sum() / sae_act["c"].numel()
        sparsity_metrics[f"sparsity/frac_zeros/{name}"] = frac_zeros

        if save_discrete_metrics:
            # Record the dictionary element frequencies and alive dict elements
            sparsity_metrics[f"sparsity/dict_el_frequencies/{name}"] = dict_el_frequencies[
                name
            ].copy()
            sparsity_metrics[f"sparsity/alive_dict_elements/{name}"] = sum(
                el > 0 for el in dict_el_frequencies[name]
            )

            if create_wandb_hist:
                data = [[s] for s in sparsity_metrics[f"sparsity/dict_el_frequencies/{name}"]]
                table = wandb.Table(data=data, columns=["dict element activation frequency"])
                plot = wandb.plot.histogram(
                    table,
                    "dict element activation frequency",
                    title=f"{name} (most_recent_n_tokens={tokens_in_dict} dict_size={dict_size})",
                )
                sparsity_metrics[f"sparsity/dict_el_frequencies_hist/{name}"] = plot

    if save_discrete_metrics:
        # Since we saved, reset the frequency dictionary and the tokens_in_dict counter
        dict_el_frequencies = {}
        tokens_in_dict = 0

    return sparsity_metrics, tokens_in_dict
