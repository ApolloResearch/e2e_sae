from functools import partial
from typing import Any

import fire
import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer, HookedTransformerConfig, evals
from transformer_lens.hook_points import HookPoint

from sparsify.configs import Config
from sparsify.losses import calc_loss
from sparsify.models.sparsifiers import SAE
from sparsify.models.transformers import SAETransformer
from sparsify.utils import load_config


def sae_hook(
    value: Float[torch.Tensor, "... d_head"], hook: HookPoint, sae: SAE, hook_acts: dict[str, Any]
) -> Float[torch.Tensor, "... d_head"]:
    """Runs the SAE on the input and stores the output and c in hook_acts."""
    output, c = sae(value)
    hook_acts["output"] = output
    hook_acts["c"] = c
    return output


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
            config=config,
            orig_acts=orig_acts,
            sae_acts=sae_acts,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)


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
