import math

import torch
from einops import einsum, rearrange
from jaxtyping import Float, Int
from pydantic import PositiveInt
from torch import Tensor
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from sae_vis.data_fetching_fns import get_sequences_data
from sae_vis.data_storing_fns import (
    FeatureData,
    FeatureVisParams,
    HistogramData,
    MiddlePlotsData,
    MultiFeatureData,
    SequenceMultiGroupData,
)
from sae_vis.utils_fns import QuantileCalculator, TopK, get_device, process_str_tok
from sparsify.data import create_data_loader
from sparsify.models.transformers import SAETransformer
from sparsify.scripts.train_tlens_saes.run_train_tlens_saes import Config
from sparsify.utils import filter_names


def compute_feature_acts(
    model: SAETransformer,
    tokens: Int[Tensor, "batch pos"],
    raw_sae_position_names: list[str] | None = None,
    feature_indices: dict[str, Int[Tensor, "some_feats"]] | dict[str, list[int]] | None = None,
    stop_at_layer: int = -1,
    store_features_as_sparse: bool = False,
) -> tuple[dict[str, Float[Tensor, "... some_feats"]], Float[Tensor, "... dim"]]:
    """Compute the activations of the SAEs in the model given a tensor of input tokens
    RETURNS
        feature_acts:
            Feature activations for each SAE.
            feature_acts[sae_position_name] = the feature activations of that SAE
                                              shape: batch pos some_feats
        final_resid_acts:
            The residual stream activations of the model at the final layer (or at stop_at_layer)
    """
    if raw_sae_position_names is None:
        raw_sae_position_names = model.raw_sae_position_names
    # Run model without SAEs
    final_resid_acts, orig_acts = model.tlens_model.run_with_cache(
        tokens,
        names_filter=raw_sae_position_names,
        return_cache_object=False,
        stop_at_layer=stop_at_layer,
    )
    assert isinstance(final_resid_acts, Tensor)
    feature_acts: dict[str, Float[Tensor, "... some_feats"]] = {}
    # Run the activations through the SAEs
    for hook_name in orig_acts:
        sae = model.saes[hook_name.replace(".", "-")]
        output, feature_acts[hook_name] = sae(orig_acts[hook_name])
        if feature_indices is not None:
            feature_acts[hook_name] = feature_acts[hook_name][..., feature_indices[hook_name]]
        if store_features_as_sparse:
            feature_acts[hook_name] = feature_acts[hook_name].to_sparse()
    return feature_acts, final_resid_acts


def compute_feature_acts_on_distribution(
    model: SAETransformer,
    config: Config,
    n_samples: PositiveInt | None = None,
    batch_size: PositiveInt | None = None,
    raw_sae_position_names: list[str] | None = None,
    feature_indices: dict[str, Int[Tensor, "some_feats"]] | dict[str, list[int]] | None = None,
    stop_at_layer: int = -1,
    store_features_as_sparse: bool = False,
) -> tuple[
    dict[str, Float[Tensor, "... some_feats"]], Float[Tensor, "... d_resid"], Int[Tensor, "..."]
]:
    """Compute the activations of the SAEs in the model on the training distribution of input tokens
    RETURNS
        feature_acts:
            a dict of SAE inputs, activations, and outputs for each SAE.
            feature_acts[sae_position_name] = the feature activations of that SAE
                                                   shape: batch pos feats (or # feature_indices)
        final_resid_acts:
            The residual stream activations of the model at the final layer (or at stop_at_layer)

        tokens:
            The tokens used as input to the model
    """
    device = get_device()
    model.to(device)
    if n_samples is None:
        n_samples = config.train.n_samples
    if batch_size is None:
        batch_size = config.train.batch_size
    data_loader, _ = create_data_loader(config.data, batch_size=batch_size)
    if raw_sae_position_names is None:
        raw_sae_position_names = model.raw_sae_position_names

    if n_samples is None:
        # If streaming (i.e. if the dataset is an IterableDataset), we don't know the length
        n_batches = None if isinstance(data_loader.dataset, IterableDataset) else len(data_loader)
    else:
        n_batches = math.ceil(n_samples / batch_size)

    total_samples = 0
    feature_acts_lists: dict[str, list[Float[Tensor, "... some_feats"]]] = {
        hook_name: [] for hook_name in raw_sae_position_names
    }
    final_resid_acts_list: list[Float[Tensor, "... d_resid"]] = []
    tokens_list: list[Int[Tensor, "..."]] = []
    for batch_idx, batch in tqdm(enumerate(data_loader), total=n_batches, desc="Steps"):
        batch_tokens: Int[Tensor, "..."] = batch[config.data.column_name].to(device=device)
        batch_feature_acts, batch_final_resid_acts = compute_feature_acts(
            model=model,
            tokens=batch_tokens,
            raw_sae_position_names=raw_sae_position_names,
            feature_indices=feature_indices,
            stop_at_layer=stop_at_layer,
            store_features_as_sparse=store_features_as_sparse,
        )
        for hook_name in raw_sae_position_names:
            feature_acts_lists[hook_name].append(batch_feature_acts[hook_name])
        final_resid_acts_list.append(batch_final_resid_acts)
        tokens_list.append(batch_tokens)
        total_samples += batch_tokens.shape[0]
        if n_samples is not None and total_samples >= n_samples:
            break
    final_resid_acts: Float[Tensor, "... d_resid"] = torch.cat(final_resid_acts_list, dim=0)
    tokens: Int[Tensor, "..."] = torch.cat(tokens_list, dim=0)
    feature_acts: dict[str, Float[Tensor, "... some_feats"]] = {}
    for hook_name in raw_sae_position_names:
        feature_acts[hook_name] = torch.cat(feature_acts_lists[hook_name], dim=0)
    return feature_acts, final_resid_acts, tokens


def process_vocab_dict(tokenizer: PreTrainedTokenizerBase) -> dict[int, str]:
    """
    Creates a vocab dict suitable for dashboards by replacing all the special tokens with their
    HTML representations. This function is adapted from sae_vis.create_vocab_dict()
    """
    vocab_dict: dict[str, int] = tokenizer.get_vocab()
    vocab_dict_processed: dict[int, str] = {v: process_str_tok(k) for k, v in vocab_dict.items()}
    return vocab_dict_processed


def parse_activation_data(
    tokens: Int[Tensor, "batch pos"],
    feature_acts: dict[str, Float[Tensor, "... some_feats"]],
    final_resid_acts: Float[Tensor, "... d_resid"],
    feature_resid_dirs: Float[Tensor, "some_feats dim"],
    feature_indices_list: list[int],
    W_U: Float[Tensor, "dim d_vocab"],
    vocab_dict: dict[int, str],
    fvp: FeatureVisParams,
) -> MultiFeatureData:
    """Convert generic activation data into a MultiFeatureData object, which can be used to create
    the feature-centric visualisation.

    final_resid_acts + W_U are used for the logit lens.

    Args:
        tokens: The inputs to the model
        feature_acts: The activations values of the features
        final_resid_acts: The activations of the final layer of the model
        feature_resid_dirs: The directions that each feature writes to the logit output
        feature_indices_list: The indices of the features we're interested in
        W_U: The weights of the logit lens
        vocab_dict: A dictionary mapping vocab indices to strings
        fvp: FeatureVisParams, containing a bunch of settings. See the FeatureVisParams docstring in
                sae_vis for more information.
    """
    sequence_data_dict: dict[int, SequenceMultiGroupData] = {}
    middle_plots_data_dict: dict[int, MiddlePlotsData] = {}
    feature_dashboard_data: dict[int, FeatureData] = {}
    # Calculate all data for the right-hand visualisations, i.e. the sequences
    for i, feat in enumerate(feature_indices_list):
        # Add this feature's sequence data to the list
        sequence_data_dict[feat] = get_sequences_data(
            tokens=tokens,
            feat_acts=feature_acts[..., i],
            resid_post=final_resid_acts,
            feature_resid_dir=feature_resid_dirs[i],
            W_U=W_U,
            fvp=fvp,
        )

    # Get the logits of all features (i.e. the directions this feature writes to the logit output)
    logits = einsum(
        feature_resid_dirs,
        W_U,
        "feats d_model, d_model d_vocab -> feats d_vocab",
    )
    for i, (feat, logit) in enumerate(zip(feature_indices_list, logits, strict=True)):
        # Get data for logits (the histogram, and the table)
        logits_histogram_data = HistogramData(logit, n_bins=40, tickmode="5 ticks")
        top10_logits = TopK(logit, k=15, largest=True)
        bottom10_logits = TopK(logit, k=15, largest=False)

        # Get data for feature activations histogram (the title, and the histogram)
        feat_acts = feature_acts[..., i]
        nonzero_feat_acts = feat_acts[feat_acts > 0]
        frac_nonzero = nonzero_feat_acts.numel() / feat_acts.numel()
        freq_histogram_data = HistogramData(nonzero_feat_acts, n_bins=40, tickmode="ints")

        # Create a MiddlePlotsData object from this, and add it to the dict
        middle_plots_data_dict[feat] = MiddlePlotsData(
            bottom10_logits=bottom10_logits,
            top10_logits=top10_logits,
            logits_histogram_data=logits_histogram_data,
            freq_histogram_data=freq_histogram_data,
            frac_nonzero=frac_nonzero,
        )

    # Return the output, as a dict of FeatureData items
    for i, feat in enumerate(feature_indices_list):
        feature_dashboard_data[feat] = FeatureData(
            # Data-containing inputs (for the feature-centric visualisation)
            sequence_data=sequence_data_dict[feat],
            middle_plots_data=middle_plots_data_dict[feat],
            left_tables_data=None,
            # Non data-containing inputs
            feature_idx=feat,
            vocab_dict=vocab_dict,
            fvp=fvp,
        )

    # Also get the quantiles, which will be useful for the prompt-centric visualisation
    feature_act_quantiles = QuantileCalculator(
        data=rearrange(feature_acts, "... feats -> feats (...)")
    )
    return MultiFeatureData(feature_dashboard_data, feature_act_quantiles)


@torch.inference_mode()
def get_feature_dashboard_data(
    model: SAETransformer,
    config: Config | None = None,
    tokens: Int[Tensor, "batch pos"] | None = None,
    sae_position_names: list[str] | None = None,
    feature_indices: dict[str, Int[Tensor, "some_feats"]]
    | dict[str, list[int]]
    | list[int]
    | Int[Tensor, "some_feats"]
    | None = None,
    n_samples: PositiveInt | None = None,
    batch_size: PositiveInt | None = None,
    fvp: FeatureVisParams | None = None,
    vocab_dict: dict[int, str] | None = None,
) -> dict[str, MultiFeatureData]:
    """
    Gets data that will be used to create the sequences in the feature-centric HTML visualisation.
    Args:
        model:
            The model (with SAEs) we'll be using to get the feature activations.

        [optional] config:
            The training config (type Config(BaseModel)) used to train the model. Used to get the
            data loader. If None, then tokens must be supplied.

        [optional] tokens:
            The tokens we'll be using to get the feature activations. If None, then we'll use the
            training distribution from the config file.

        [optional] sae_position_names:
            The names of the SAEs we want to calculate feature dashboards for,
            eg. ['blocks.0.hook_resid_pre']. If none, then we'll do all of them.

        [optional] feature_indices:
            The features we're actually computing for each SAE. These might just be a subset of
            each SAE's full features. If None, then we'll do all of them.'

        [optional] n_samples: [Only used if tokens is None]
            The number of batches of data to use for calculating the feature dashboard data using
            config.data. If none, defaults to config.train.n_samples.

        [optional] batch_size: [Only used if tokens is None]
            The number of batches of data to use for calculating the feature dashboard data using
            config.data. If none, defaults to config.train.batch_size.

        [optional] fvp:
            Feature visualization parameters, containing a bunch of other stuff. See the
            FeatureVisParams docstring in sae_vis for more information.

    Returns:
        multi_feature_dashboard_data:
            A dict of [sae_position_name: MultiFeatureData]. Each MultiFeatureData contains data for
            creating each feature's visualization, as well as data for rank-ordering the feature
            visualizations when it comes time to make the prompt-centric view
            (the `feature_act_quantiles` attribute).
            Use multi_feature_dashboard_data[sae_name][feature_idx].get_html() to generate the HTML
            dashboard for a particular feature (returns a string of HTML)
    """
    device = get_device()
    model.to(device)
    # Get the vocab dict, which we'll use at the end
    if vocab_dict is None:
        assert (
            model.tlens_model.tokenizer is not None
        ), "If voacab_dict is not supplied, the model must have a tokenizer"
        vocab_dict = process_vocab_dict(model.tlens_model.tokenizer)

    if fvp is None:
        fvp = FeatureVisParams(include_left_tables=False)

    if sae_position_names is None:
        raw_sae_position_names: list[str] = model.raw_sae_position_names
    else:
        raw_sae_position_names: list[str] = filter_names(
            list(model.tlens_model.hook_dict.keys()), sae_position_names
        )
    # If we haven't supplied any feature indicies, assume that we want all of them
    if feature_indices is None:
        feature_indices = {}
        for name in raw_sae_position_names:
            feature_indices[name] = torch.arange(
                end=model.saes[name.replace(".", "-")].n_dict_components
            )
    # Otherwise make sure that feature_indices is a dict of Int[Tensor]
    elif not isinstance(feature_indices, dict):
        feature_indices = {
            name: Tensor(feature_indices, device="cpu").to(torch.int)
            for name in raw_sae_position_names
        }
    else:
        feature_indices = {
            name: Tensor(feature_indices[name], device="cpu").to(torch.int)
            for name in raw_sae_position_names
        }
    for name in raw_sae_position_names:
        assert (
            feature_indices[name].max().item()
            < model.saes[name.replace(".", "-")].n_dict_components
        ), "Error: Some feature indices are greater than the number of SAE features"
    # Get the SAE feature activations (as well as their corresponding resudual stream inputs and outputs)
    if tokens is None:
        assert config is not None, "If no tokens are supplied, then config must be supplied"
        feature_acts, final_resid_acts, tokens = compute_feature_acts_on_distribution(
            model=model,
            config=config,
            raw_sae_position_names=raw_sae_position_names,
            feature_indices=feature_indices,
            n_samples=n_samples,
            batch_size=batch_size,
        )
    else:
        tokens.to(device)
        feature_acts, final_resid_acts = compute_feature_acts(
            model=model,
            tokens=tokens,
            raw_sae_position_names=raw_sae_position_names,
            feature_indices=feature_indices,
        )

    # Filter out the never active features:
    for sae_name in raw_sae_position_names:
        acts_sum = einsum(feature_acts[sae_name], "... some_feats -> some_feats").to("cpu")
        feature_acts[sae_name] = feature_acts[sae_name][..., acts_sum > 0]
        feature_indices[sae_name] = feature_indices[sae_name][acts_sum > 0]
        del acts_sum

    multi_feature_dashboard_data: dict[str, MultiFeatureData] = {}

    for sae_name in raw_sae_position_names:
        sae = model.saes[sae_name.replace(".", "-")]
        W_dec: Float[Tensor, "feats dim"] = sae.decoder.weight.T
        feature_resid_dirs: Float[Tensor, "some_feats dim"] = W_dec[feature_indices[sae_name]]
        W_U = model.tlens_model.W_U

        multi_feature_dashboard_data[sae_name] = parse_activation_data(
            tokens=tokens,
            feature_acts=feature_acts[sae_name],
            final_resid_acts=final_resid_acts,
            feature_resid_dirs=feature_resid_dirs,
            feature_indices_list=feature_indices[sae_name].tolist(),
            W_U=W_U,
            vocab_dict=vocab_dict,
            fvp=fvp,
        )

    return multi_feature_dashboard_data


def generate_dashboard_html_files(
    multi_feature_dashboard_data: dict[str, MultiFeatureData], html_dir: str = ""
):
    """Generates viewable HTML dashboards from the compressed multi_feature_dashboard_data"""
    for sae_name in multi_feature_dashboard_data:
        for feature_idx in multi_feature_dashboard_data[sae_name].keys():
            filepath = html_dir + f"dashboard_{sae_name}_feature-{feature_idx}.html"
            html_str = multi_feature_dashboard_data[sae_name][feature_idx].get_html()
            with open(filepath, "w") as f:
                f.write(html_str)
