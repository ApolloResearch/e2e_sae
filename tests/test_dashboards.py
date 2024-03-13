import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from sparsify.data import DatasetConfig
from sparsify.loader import load_tlens_model
from sparsify.models.transformers import SAETransformer
from sparsify.scripts.generate_dashboards import (
    DashboardsConfig,
    compute_feature_acts,
    create_vocab_dict,
    generate_dashboards,
)
from sparsify.utils import set_seed
from tests.utils import get_tinystories_config

Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast


@pytest.fixture(scope="module")
def tinystories_model() -> SAETransformer:
    tlens_model = load_tlens_model(
        tlens_model_name="roneneldan/TinyStories-1M", tlens_model_path=None
    )
    sae_position = "blocks.2.hook_resid_post"
    config = get_tinystories_config({"saes": {"sae_positions": sae_position}})
    model = SAETransformer(config=config, tlens_model=tlens_model, raw_sae_positions=[sae_position])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


@pytest.fixture(scope="function")
def tmp_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        yield Path(tmp_dir_path)


def test_compute_feature_acts(tinystories_model: SAETransformer):
    set_seed(0)
    prompt = "Once upon a time,"
    tokenizer = tinystories_model.tlens_model.tokenizer
    assert tokenizer is not None
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
    assert isinstance(tokens, torch.Tensor)
    feature_indices = {name: list(range(7)) for name in tinystories_model.raw_sae_positions}
    feature_acts, final_resid_acts = compute_feature_acts(
        tinystories_model, tokens, feature_indices=feature_indices
    )
    for sae_name, acts in feature_acts.items():
        assert acts.shape[0] == 1  # batch size
        assert acts.shape[2] == 7  # feature_indices


def test_create_vocab_dict(tinystories_model: SAETransformer):
    tokenizer = tinystories_model.tlens_model.tokenizer
    assert tokenizer is not None
    vocab_dict = create_vocab_dict(tokenizer)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    assert len(vocab_dict) == len(tokenizer.vocab)
    for token_id, token_str in vocab_dict.items():
        assert isinstance(token_id, int)
        assert isinstance(token_str, str)


def check_valid_feature_dashboard_htmls(folder: Path):
    assert folder.exists()
    for html_file in folder.iterdir():
        assert html_file.name.endswith(".html")
        assert html_file.exists()
        with open(html_file) as f:
            html_content = f.read()
            assert isinstance(html_content, str)
            assert len(html_content) > 100
            assert "Plotly.newPlot('histogram-acts'" in html_content
            assert '<div id="histogram-acts" class="plotly-hist' in html_content
            assert '<div id="histogram-logits" class="plotly-hist' in html_content
            assert "var barHeightsLogits = [" in html_content
            assert "var barHeightsFreq = [" in html_content
            assert '</span><span class="hover-text"' in html_content
            assert '<div class="tooltip" id=' in html_content


def check_valid_prompt_dashboard_htmls(folder: Path):
    assert folder.exists()
    for html_file in folder.iterdir():
        assert html_file.name.endswith(".html")
        assert html_file.exists()
        with open(html_file) as f:
            html_content = f.read()
            assert isinstance(html_content, str)
            assert len(html_content) > 100
            assert "<div class='grid-container'>'" in html_content
            assert "Feature #" in html_content
            assert '<div id="histogram-logits" class="plotly-hist' in html_content
            assert "var barHeightsLogits = [" in html_content
            assert "var barHeightsFreq = [" in html_content
            assert '</span><span class="hover-text"' in html_content
            assert '<div class="tooltip" id=' in html_content


@pytest.mark.slow
def test_generate_dashboards(tinystories_model: SAETransformer, tmp_dir: Path):
    # This function also tests compute_feature_acts_on_distribution()
    set_seed(0)
    dashboards_config = DashboardsConfig(
        n_samples=10,
        batch_size=2,
        minibatch_size_features=5,
        save_dir=Path(tmp_dir),
        sae_positions=["blocks.2.hook_resid_post"],
        pretrained_sae_paths=None,
        feature_indices=list(range(5)),
        data=DatasetConfig(
            dataset_name="apollo-research/sae-skeskinen-TinyStories-hf-tokenizer-gpt2",
            tokenizer_name="gpt2",
            split="train",
            n_ctx=512,
        ),
    )
    generate_dashboards(tinystories_model, dashboards_config)
    check_valid_feature_dashboard_htmls(
        tmp_dir / "feature-dashboards" / "dashboards_blocks.2.hook_resid_post"
    )
