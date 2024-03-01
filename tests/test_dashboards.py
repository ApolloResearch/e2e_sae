import torch
from sae_vis.data_storing_fns import MultiFeatureData
from sae_vis.utils_fns import get_device
from transformer_lens import HookedTransformer
from utils import get_tinystories_config

from sparsify.models.transformers import SAETransformer
from sparsify.scripts.generate_dashboards import compute_feature_acts, get_feature_dashboard_data


def test_get_feature_dashboard_data(
    n_samples: int = 10, batch_size: int = 5, n_features_to_plot: int = 10
):
    # This function also uses test_compute_feature_acts and and test_compute_feature_acts_on_distribution

    # Make a dummy tlens config with two layers
    tlens_model = HookedTransformer.from_pretrained("tiny-stories-1M")
    sae_position = "blocks.0.hook_resid_post"
    config = get_tinystories_config({"saes": {"sae_position_names": sae_position}})

    model = SAETransformer(
        config=config, tlens_model=tlens_model, raw_sae_position_names=[sae_position]
    )

    tokens = torch.randint(0, model.tlens_model.W_E.shape[0], (100, 100))
    for current_tokens in (None, tokens):
        feature_dashboard_data = get_feature_dashboard_data(
            model,
            config,
            tokens=current_tokens,
            batch_size=batch_size,
            n_samples=n_samples,
            feature_indices=list(range(n_features_to_plot)),
        )

        for sae_name in model.raw_sae_position_names:
            assert isinstance(feature_dashboard_data[sae_name], MultiFeatureData)
            for feature_idx in feature_dashboard_data[sae_name].keys():
                html_str = feature_dashboard_data[sae_name][feature_idx].get_html()
                assert isinstance(html_str, str)
                assert len(html_str) > 100
                assert "Plotly.newPlot('histogram-acts'" in html_str


def test_compute_feature_acts():
    tlens_model = HookedTransformer.from_pretrained("tiny-stories-1M")
    sae_position = "blocks.2.hook_resid_post"
    config = get_tinystories_config({"saes": {"sae_position_names": sae_position}})
    model = SAETransformer(
        config=config, tlens_model=tlens_model, raw_sae_position_names=[sae_position]
    )
    device = get_device()
    model.to(device)
    tokens = torch.randint(low=0, high=500, size=(2, 5))
    feature_acts_prev = None
    for feature_indices in [{sae_position: torch.arange(10)}, {sae_position: list(range(10))}]:
        for store_features_as_sparse in [True, False]:
            feature_acts, final_resid_acts = compute_feature_acts(
                model=model,
                tokens=tokens,
                raw_sae_position_names=[sae_position],
                feature_indices=feature_indices,
                store_features_as_sparse=store_features_as_sparse,
            )
            if feature_acts_prev is not None:
                assert torch.allclose(
                    feature_acts[sae_position].to_dense(),
                    feature_acts_prev[sae_position].to_dense(),
                )
            feature_acts_prev = feature_acts


test_compute_feature_acts()
