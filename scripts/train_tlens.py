import fire
import torch
from jaxtyping import Float
from pydantic import BaseModel
from torch import Tensor, nn
from transformer_lens import HookedTransformer, evals

from sparsify.hook_fns import sae_acts_pre_forward_hook_fn
from sparsify.hook_manager import Hook, HookedModel
from sparsify.models.sparsifiers import SAE
from sparsify.utils import load_config


class TrainConfig(BaseModel):
    batch_size: int
    lr: float


class Config(BaseModel):
    tlens_model_name: str
    input_size: int
    n_dict_components: int
    sae_orig: bool
    sae_sparsity: bool
    train: TrainConfig


class SAETransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.tlens_model = HookedTransformer.from_pretrained(config.tlens_model_name)

        self.saes = nn.ModuleDict()
        for i in range(self.tlens_model.cfg.n_layers):
            self.saes[str(i)] = SAE(
                input_size=config.input_size, n_dict_components=config.n_dict_components
            )
        # TODO: find a better way to specify positions
        self.sae_position_name = "hook_resid_post"

    def forward(self, x: Tensor) -> Tensor:
        return self.tlens_model(x)

    def to(self, *args, **kwargs) -> "SAETransformer":
        """TODO: Fix this. Tlens implementation of to makes this annoying"""

        if len(args) == 1:
            self.tlens_model.to(device_or_dtype=args[0])
        elif len(args) == 2:
            self.tlens_model.to(device_or_dtype=args[0])
            self.tlens_model.to(device_or_dtype=args[1])
        elif len(kwargs) == 1:
            if "device" or "dtype" in kwargs:
                arg = kwargs["device"] if "device" in kwargs else kwargs["dtype"]
                self.tlens_model.to(device_or_dtype=arg)
            else:
                raise ValueError("Invalid keyword argument.")
        elif len(kwargs) == 2:
            assert "device" in kwargs and "dtype" in kwargs, "Invalid keyword arguments."
            self.tlens_model.to(device_or_dtype=kwargs["device"])
            self.tlens_model.to(device_or_dtype=kwargs["dtype"])
        else:
            raise ValueError("Invalid arguments.")

        self.saes.to(*args, **kwargs)
        return self


def get_loss(
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


def train(config: Config, hooked_model: HookedModel, device: torch.device) -> None:
    assert isinstance(hooked_model.model, SAETransformer)
    tokenizer = hooked_model.model.tlens_model.tokenizer
    train_loader = evals.make_pile_data_loader(tokenizer, batch_size=config.train.batch_size)

    optimizer = torch.optim.Adam(hooked_model.model.saes.parameters(), lr=config.train.lr)

    orig_resid_names = lambda name: hooked_model.model.sae_position_name in name
    for batch in train_loader:
        tokens = batch["tokens"].to(device=device)
        # Run model without SAEs
        with torch.inference_mode():
            _, orig_acts = hooked_model.model.tlens_model.run_with_cache(
                tokens, names_filter=orig_resid_names
            )

        # Run model with SAEs
        hooks = [
            Hook(
                name=f"blocks.{i}.{hooked_model.model.sae_position_name}",
                data_key=["output", "c"],
                fn=sae_acts_pre_forward_hook_fn,
                module_name=f"tlens_model.blocks.{i}.{hooked_model.model.sae_position_name}",
                fn_kwargs={"sae": hooked_model.model.saes[f"{i}"]},
            )
            for i in range(hooked_model.model.tlens_model.cfg.n_layers)
        ]
        hooked_model(tokens, hooks=hooks)

        loss = get_loss(
            orig_acts=orig_acts,
            sae_acts=hooked_model.hooked_data,
            sae_orig=config.sae_orig,
            sae_sparsity=config.sae_sparsity,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main(config_path_str: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path_str, config_model=Config)

    model = SAETransformer(config).to(device=device)
    hooked_model = HookedModel(model)
    train(config, hooked_model, device=device)


if __name__ == "__main__":
    fire.Fire(main)
