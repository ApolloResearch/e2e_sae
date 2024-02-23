from pathlib import Path

import torch
import yaml
from transformer_lens import HookedTransformer, HookedTransformerConfig

from sparsify.scripts.train_tlens.run_train_tlens import HookedTransformerPreConfig
from sparsify.types import RootPath


def load_tlens_model(
    tlens_model_name: str | None, tlens_model_path: RootPath | None
) -> HookedTransformer:
    """Load transformerlens model from either HuggingFace or local path."""
    if tlens_model_name is not None:
        tlens_model = HookedTransformer.from_pretrained(tlens_model_name)
    else:
        assert tlens_model_path is not None, "tlens_model_path is None."
        # Load the tlens_config
        with open(tlens_model_path / "config.yaml") as f:
            tlens_config = HookedTransformerPreConfig(**yaml.safe_load(f)["tlens_config"])
        hooked_transformer_config = HookedTransformerConfig(**tlens_config.model_dump())

        # Load the model
        tlens_model = HookedTransformer(hooked_transformer_config)
        latest_model_path = max(
            tlens_model_path.glob("*.pt"), key=lambda x: int(x.stem.split("_")[-1])
        )
        tlens_model.load_state_dict(torch.load(latest_model_path))

    assert tlens_model.tokenizer is not None
    return tlens_model


def load_pretrained_saes(
    model_saes: torch.nn.ModuleDict,
    pretrained_sae_paths: list[Path],
    all_param_names: list[str],
    retrain_saes: bool,
) -> list[str]:
    """Load in the pretrained SAEs to model_saes (in place) and return the trainable param names.

    Args:
        model_saes: The SAE model to load the pretrained SAEs into. Updated in place.
        pretrained_sae_paths: List of paths to the pretrained SAEs.
        all_param_names: List of all the parameter names in model_saes.
        retrain_saes: Whether to retrain the pretrained SAEs.

    Returns:
        The updated all_param_names.
    """
    pretrained_sae_params = {}
    for pretrained_sae_path in pretrained_sae_paths:
        # Add new sae params (note that this will overwrite existing SAEs with the same name)
        pretrained_sae_params = {**pretrained_sae_params, **torch.load(pretrained_sae_path)}
    sae_state_dict = {**dict(model_saes.named_parameters()), **pretrained_sae_params}

    model_saes.load_state_dict(sae_state_dict)
    if not retrain_saes:
        # Don't retrain the pretrained SAEs
        trainable_param_names = [
            name for name in all_param_names if name not in pretrained_sae_params
        ]
    else:
        trainable_param_names = all_param_names
    assert len(trainable_param_names) > 0, "No trainable parameters found."

    return trainable_param_names
