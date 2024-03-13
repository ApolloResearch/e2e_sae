import pytest
import torch

from sparsify.loader import load_tlens_model
from sparsify.models.transformers import SAETransformer
from tests.utils import get_tinystories_config


@pytest.fixture(scope="module")
def tinystories_model() -> SAETransformer:
    tlens_model = load_tlens_model(
        tlens_model_name="roneneldan/TinyStories-1M", tlens_model_path=None
    )
    sae_positions = []
    config = get_tinystories_config({"saes": {"sae_positions": sae_positions}})
    model = SAETransformer(config=config, tlens_model=tlens_model, raw_sae_positions=sae_positions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def test_generate(tinystories_model: SAETransformer, prompt: str = "One", max_new_tokens: int = 12):
    completion = tinystories_model.generate(
        input=prompt, sae_positions=None, max_new_tokens=max_new_tokens, temperature=0
    )
    assert completion == "One day, a little girl named Lily was playing in the park"
