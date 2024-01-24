from functools import partial
from typing import Any, Literal, Optional

import fire
import torch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from torch import Tensor
from transformer_lens import HookedTransformer, HookedTransformerConfig, evals
from transformer_lens.hook_points import HookPoint

from sparsify.models.sparsifiers import SAE
from sparsify.models.transformers import SAETransformer
from sparsify.utils import load_config

StrDtype = Literal["float32", "float64", "bfloat16"]
TORCH_DTYPES: dict[StrDtype, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


class HookedTransformerPreConfig(BaseModel):
    """Pydantic model whose arguments will be passed to a HookedTransformerConfig."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)
    d_model: int
    n_layers: int
    n_ctx: int
    d_head: int
    d_vocab: int
    act_fn: str
    dtype: Optional[torch.dtype]

    @field_validator("dtype", mode="before")
    @classmethod
    def dtype_to_torch_dtype(cls, v: Optional[StrDtype]) -> Optional[torch.dtype]:
        if v is None:
            return None
        return TORCH_DTYPES[v]


class TrainConfig(BaseModel):
    batch_size: int
    lr: float


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tlens_model_name: Optional[str] = None
    tlens_config: Optional[HookedTransformerPreConfig] = None
    sae_position_name: str
    input_size: int
    n_dict_components: int
    sae_orig: bool
    sae_sparsity: bool
    train: TrainConfig

    @model_validator(mode="before")
    @classmethod
    def check_only_one_model_definition(cls, values: dict[str, Any]) -> dict[str, Any]:
        assert (values.get("tlens_model_name") is not None) + (
            values.get("tlens_config") is not None
        ) == 1, "Must specify exactly one of tlens_model_name or tlens_config."
        return values


def sae_hook(
    value: Float[torch.Tensor, "... d_head"], hook: HookPoint, sae: SAE, hook_acts: dict[str, Any]
) -> Float[torch.Tensor, "... d_head"]:
    """Runs the SAE on the input and stores the output and c in hook_acts."""
    output, c = sae(value)
    hook_acts["output"] = output
    hook_acts["c"] = c
    return output


def calc_loss(
    orig_acts: dict[str, Tensor],
    sae_acts: dict[str, dict[str, Tensor]],
    sae_orig: bool,
    sae_sparsity: bool,
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
    for name, orig_act in orig_acts.items():
        # Convert from inference tensor. TODO: Make more memory efficient
        orig_act = orig_act.clone()
        sae_act = sae_acts[name]
        if sae_orig:
            loss += torch.nn.functional.mse_loss(orig_act, sae_act["output"])
        if sae_sparsity:
            loss += torch.norm(sae_act["c"], p=0.6, dim=-1).mean()
    return loss


def train(config: Config, model: SAETransformer, device: torch.device) -> None:
    tokenizer = model.tlens_model.tokenizer
    assert tokenizer is not None, "Tokenizer must be defined for training."
    train_loader = evals.make_pile_data_loader(tokenizer, batch_size=config.train.batch_size)

    optimizer = torch.optim.Adam(model.saes.parameters(), lr=config.train.lr)
    sae_acts = {layer: {} for layer in model.saes.keys()}

    orig_resid_names = lambda name: model.sae_position_name in name
    for batch in train_loader:
        tokens = batch["tokens"].to(device=device)
        # Run model without SAEs
        with torch.inference_mode():
            _, orig_acts = model.tlens_model.run_with_cache(tokens, names_filter=orig_resid_names)

        # Run model with SAEs
        sae_acts = {hook_name: {} for hook_name in orig_acts}
        fwd_hooks = [
            (hook_name, partial(sae_hook, sae=model.saes[str(i)], hook_acts=sae_acts[hook_name]))
            for i, hook_name in enumerate(orig_acts)
        ]
        model.tlens_model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        loss = calc_loss(
            orig_acts=orig_acts,
            sae_acts=sae_acts,
            sae_orig=config.sae_orig,
            sae_sparsity=config.sae_sparsity,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main(config_path_str: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path_str, config_model=Config)

    if config.tlens_model_name is not None:
        tlens_model = HookedTransformer.from_pretrained(config.tlens_model_name)
    else:
        hooked_transformer_config = HookedTransformerConfig(**config.tlens_config.model_dump())
        tlens_model = HookedTransformer(hooked_transformer_config)

    model = SAETransformer(tlens_model, config).to(device=device)
    train(config, model, device=device)


if __name__ == "__main__":
    fire.Fire(main)
