import pytest
import torch

from e2e_sae.loader import load_tlens_model
from e2e_sae.models.transformers import SAETransformer
from tests.utils import get_tinystories_config


@pytest.fixture(scope="module")
def tinystories_model() -> SAETransformer:
    tlens_model = load_tlens_model(
        tlens_model_name="roneneldan/TinyStories-1M", tlens_model_path=None
    )
    sae_positions = ["blocks.2.hook_resid_pre"]
    config = get_tinystories_config({"saes": {"sae_positions": sae_positions}})
    model = SAETransformer(
        tlens_model=tlens_model,
        raw_sae_positions=sae_positions,
        dict_size_to_input_ratio=config.saes.dict_size_to_input_ratio,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def test_generate(tinystories_model: SAETransformer, prompt: str = "One", max_new_tokens: int = 2):
    completion = tinystories_model.generate(
        input=prompt, sae_positions=None, max_new_tokens=max_new_tokens, temperature=0
    )
    assert completion == "One day,"
