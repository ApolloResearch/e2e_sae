import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import torch
from eindex import eindex
from einops import einsum, rearrange
from jaxtyping import Float, Int
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, NonNegativeInt, PositiveInt
from sae_vis.data_fetching_fns import get_sequences_data
from sae_vis.data_storing_fns import (
    FeatureData,
    FeatureVisParams,
    HistogramData,
    MiddlePlotsData,
    MultiFeatureData,
    MultiPromptData,
    PromptData,
    SequenceData,
    SequenceMultiGroupData,
)
from sae_vis.utils_fns import QuantileCalculator, TopK, process_str_tok
from torch import Tensor
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from sparsify.data import DataConfig, create_data_loader
from sparsify.models.transformers import SAETransformer
from sparsify.types import RootPath
from sparsify.utils import filter_names, to_numpy

FeatureIndicesType = dict[str, list[int]] | dict[str, Int[Tensor, "some_feats"]]  # noqa: F821 (jaxtyping/pyright doesn't like single dimensions)
StrScoreType = Literal["act_size", "act_quantile", "loss_effect"]


class PromptDashboardsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_random_prompt_dashboards: NonNegativeInt = Field(
        default=50,
        description="The number of random prompts to generate prompt-centric feature dashboards for."
        "A feature-centric dashboard will be generated from a random token position in each prompt.",
    )
    data: DataConfig = Field(
        default=None,
        description="DataConfig for getting random prompts ",
    )
    prompts: list[str] | None = Field(
        default=None,
        description="Specific prompts on which to generate prompt-centric feature dashboards. "
        "A feature-centric dashboard will be generated for every token position in each prompt.",
    )
    str_score: StrScoreType = Field(
        default="loss_effect",
        description="The ordering metric for which features are most important in prompt-centric "
        "dashboards. Can be one of 'act_size', 'act_quantile', or 'loss_effect'",
    )
    num_top_features: PositiveInt = Field(
        default=10,
        description="How many of the most relevant features to show for each prompt,"
        " in the prompt-centric dashboards",
    )


class DashboardsConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, frozen=True
    )  # arbitrary_types_allowed=True because jaxtyping/pyright doesn't like single dimensions like "some_feats"
    n_samples: PositiveInt | None = None
    batch_size: PositiveInt
    data: DataConfig = Field(
        default=None,
        description="DataConfig for getting the data which will be used to generate the dashboards",
    )
    save_dir: RootPath | None = Field(
        default=Path(__file__).parent / "out",
        description="The directory for saving the HTML feature dashboard files",
    )
    minibatch_size_features: PositiveInt | None = Field(
        default=256,
        description="Num features in each batch of calculations (i.e. we break up the features to "
        "avoid OOM errors).",
    )
    sae_position_names: Annotated[
        list[str] | None, BeforeValidator(lambda x: [x] if isinstance(x, str) else x)
    ] = Field(
        None,
        description="The names of the SAE positions to generate dashboards for. "
        "e.g. 'blocks.2.hook_resid_post' ",
    )
    feature_indices: FeatureIndicesType | list[int] | None = Field(
        default=None,
        description="The features for which to generate dashboards on each SAE. If none, then "
        "we'll generate dashbaords for every feature.",
    )
    prompt_centric: PromptDashboardsConfig | None = Field(
        default=None,
        description="Used to generate prompt-centric (rather than feature-centric) dashboards."
        " Feature-centric dashboards will also be generated for every",
    )


def compute_feature_acts(
    model: SAETransformer,
    tokens: Int[Tensor, "batch pos"],
    raw_sae_position_names: list[str] | None = None,
    feature_indices: FeatureIndicesType | None = None,
    stop_at_layer: int = -1,
) -> tuple[dict[str, Float[Tensor, "... some_feats"]], Float[Tensor, "... dim"]]:
    """Compute the activations of the SAEs in the model given a tensor of input tokens
    Args:
        model: The SAETransformer containing the SAEs and the tlens_model
        tokens: The inputs to the tlens_model
        raw_sae_position_names: The names of the SAEs we're interested in
        feature_indices: The indices of the features we're interested in for each SAE
        stop_at_layer: Where to stop the forward pass. final_resid_acts will be returned from here

    Returns:
        feature_acts: Feature activations for each SAE.
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
        del output
        feature_acts[hook_name] = feature_acts[hook_name].to("cpu")
        if feature_indices is not None:
            feature_acts[hook_name] = feature_acts[hook_name][..., feature_indices[hook_name]]
    return feature_acts, final_resid_acts


def compute_feature_acts_on_distribution(
    model: SAETransformer,
    data_config: DataConfig,
    batch_size: PositiveInt,
    n_samples: PositiveInt | None = None,
    raw_sae_position_names: list[str] | None = None,
    feature_indices: FeatureIndicesType | None = None,
    stop_at_layer: int = -1,
) -> tuple[
    dict[str, Float[Tensor, "... some_feats"]], Float[Tensor, "... d_resid"], Int[Tensor, "..."]
]:
    """Compute the activations of the SAEs in the model on the training distribution of input tokens
    Args:
        model: The SAETransformer containing the SAEs and the tlens_model
        data_config: The DataConfig used to get the data loader for the tokens.
        batch_size: The batch size of data run through the model when calculating the feature acts
        n_samples: The number of batches of data to use for calculating the feature dashboard data
        raw_sae_position_names: The names of the SAEs we're interested in. If none, do all SAEs.
        feature_indices: The indices of the features we're interested in for each SAE. If none, do
                         all features.
        stop_at_layer: Where to stop the forward pass. final_resid_acts will be returned from here

    Returns:
        feature_acts:
            a dict of SAE inputs, activations, and outputs for each SAE.
            feature_acts[sae_position_name] = the feature activations of that SAE
                                              shape: batch pos feats (or # feature_indices)
        final_resid_acts:
            The residual stream activations of the model at the final layer (or at stop_at_layer)
        tokens:
            The tokens used as input to the model
    """
    data_loader, _ = create_data_loader(data_config, batch_size=batch_size)
    if raw_sae_position_names is None:
        raw_sae_position_names = model.raw_sae_position_names
    device = model.saes[raw_sae_position_names[0].replace(".", "-")].device
    if n_samples is None:
        # If streaming (i.e. if the dataset is an IterableDataset), we don't know the length
        n_batches = None if isinstance(data_loader.dataset, IterableDataset) else len(data_loader)
    else:
        n_batches = math.ceil(n_samples / batch_size)

    total_samples = 0
    feature_acts_lists: dict[str, list[Float[Tensor, "... some_feats"]]] = {
        sae_name: [] for sae_name in raw_sae_position_names
    }
    final_resid_acts_list: list[Float[Tensor, "... d_resid"]] = []
    tokens_list: list[Int[Tensor, "..."]] = []
    for batch_idx, batch in tqdm(
        enumerate(data_loader), total=n_batches, desc="Computing feature acts"
    ):
        batch_tokens: Int[Tensor, "..."] = batch[data_config.column_name].to(device=device)
        batch_feature_acts, batch_final_resid_acts = compute_feature_acts(
            model=model,
            tokens=batch_tokens,
            raw_sae_position_names=raw_sae_position_names,
            feature_indices=feature_indices,
            stop_at_layer=stop_at_layer,
        )
        for sae_name in raw_sae_position_names:
            feature_acts_lists[sae_name].append(batch_feature_acts[sae_name])
        final_resid_acts_list.append(batch_final_resid_acts)
        tokens_list.append(batch_tokens)
        total_samples += batch_tokens.shape[0]
        if n_samples is not None and total_samples >= n_samples:
            break
    final_resid_acts: Float[Tensor, "... d_resid"] = torch.cat(final_resid_acts_list, dim=0)
    tokens: Int[Tensor, "..."] = torch.cat(tokens_list, dim=0)
    feature_acts: dict[str, Float[Tensor, "... some_feats"]] = {}
    for sae_name in raw_sae_position_names:
        feature_acts[sae_name] = torch.cat(tensors=feature_acts_lists[sae_name], dim=0)
    return feature_acts, final_resid_acts, tokens


def create_vocab_dict(tokenizer: PreTrainedTokenizerBase) -> dict[int, str]:
    """
    Creates a vocab dict suitable for dashboards by replacing all the special tokens with their
    HTML representations. This function is adapted from sae_vis.create_vocab_dict()
    """
    vocab_dict: dict[str, int] = tokenizer.get_vocab()
    vocab_dict_processed: dict[int, str] = {v: process_str_tok(k) for k, v in vocab_dict.items()}
    return vocab_dict_processed


@torch.inference_mode()
def parse_activation_data(
    tokens: Int[Tensor, "batch pos"],
    feature_acts: Float[Tensor, "... some_feats"],
    final_resid_acts: Float[Tensor, "... d_resid"],
    feature_resid_dirs: Float[Tensor, "some_feats dim"],
    feature_indices_list: Iterable[int],
    W_U: Float[Tensor, "dim d_vocab"],
    vocab_dict: dict[int, str],
    fvp: FeatureVisParams,
) -> MultiFeatureData:
    """Convert generic activation data into a MultiFeatureData object, which can be used to create
    the feature-centric visualisation.
    Adapted from sae_vis.data_fetching_fns._get_feature_data()

    final_resid_acts + W_U are used for the logit lens.

    Args:
        tokens: The inputs to the model
        feature_acts: The activations values of the features
        final_resid_acts: The activations of the final layer of the model
        feature_resid_dirs: The directions that each feature writes to the logit output
        feature_indices_list: The indices of the features we're interested in
        W_U: The unembed weights for the logit lens
        vocab_dict: A dictionary mapping vocab indices to strings
        fvp: FeatureVisParams, containing a bunch of settings. See the FeatureVisParams docstring in
             sae_vis for more information.

    Returns:
        A MultiFeatureData containing data for creating each feature's visualization,
        as well as data for rank-ordering the feature visualizations when it comes time
        to make the prompt-centric view (the `feature_act_quantiles` attribute).
        Use MultiFeatureData[feature_idx].get_html() to generate the HTML dashboard for a
        particular feature (returns a string of HTML).

    """
    device = W_U.device
    feature_acts.to(device)
    sequence_data_dict: dict[int, SequenceMultiGroupData] = {}
    middle_plots_data_dict: dict[int, MiddlePlotsData] = {}
    features_data: dict[int, FeatureData] = {}
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
        features_data[feat] = FeatureData(
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
    return MultiFeatureData(features_data, feature_act_quantiles)


def feature_indices_to_tensordict(
    feature_indices_in: FeatureIndicesType | list[int] | None,
    raw_sae_position_names: list[str],
    model: SAETransformer,
) -> dict[str, Tensor]:
    """ "Convert feature indices to a dict of tensor indices"""
    if feature_indices_in is None:
        feature_indices = {}
        for sae_name in raw_sae_position_names:
            feature_indices[sae_name] = torch.arange(
                end=model.saes[sae_name.replace(".", "-")].n_dict_components
            )
    # Otherwise make sure that feature_indices is a dict of Int[Tensor]
    elif not isinstance(feature_indices_in, dict):
        feature_indices = {
            sae_name: Tensor(feature_indices_in).to("cpu").to(torch.int)
            for sae_name in raw_sae_position_names
        }
    else:
        feature_indices: dict[str, Tensor] = {
            sae_name: Tensor(feature_indices_in[sae_name]).to("cpu").to(torch.int)
            for sae_name in raw_sae_position_names
        }
    return feature_indices


@torch.inference_mode()
def get_dashboards_data(
    model: SAETransformer,
    data_config: DataConfig | None = None,
    tokens: Int[Tensor, "batch pos"] | None = None,
    sae_position_names: list[str] | None = None,
    feature_indices: FeatureIndicesType | list[int] | None = None,
    n_samples: PositiveInt | None = None,
    batch_size: PositiveInt | None = None,
    minibatch_size_features: PositiveInt | None = None,
    fvp: FeatureVisParams | None = None,
    vocab_dict: dict[int, str] | None = None,
) -> dict[str, MultiFeatureData]:
    """Gets data that will be used to create the sequences in the feature-centric HTML visualisation
        Adapted from sae_vis.data_fetching_fns._get_feature_data()
    Args:
        model:
            The model (with SAEs) we'll be using to get the feature activations.
        data_config: [Only used if tokens is None]
            The DataConfig which will be used to get the data loader. If None, then tokens must
            be supplied.
        tokens:
            The tokens we'll be using to get the feature activations. If None, then we'll use the
            distribution from the data_config.
        sae_position_names:
            The names of the SAEs we want to calculate feature dashboards for,
            eg. ['blocks.0.hook_resid_pre']. If none, then we'll do all of them.
        feature_indices:
            The features we're actually computing for each SAE. These might just be a subset of
            each SAE's full features. If None, then we'll do all of them.
        n_samples: [Only used if tokens is None]
            The number of batches of data to use for calculating the feature dashboard data when
            using data_config.
        batch_size: [Only used if tokens is None]
            The number of batches of data to use for calculating the feature dashboard data when
            using data_config
        minibatch_size_features:
            Num features in each batch of calculations (break up the features to avoid OOM errors).
        fvp:
            Feature visualization parameters, containing a bunch of other stuff. See the
            FeatureVisParams docstring in sae_vis for more information.
        vocab_dict:
            vocab dict suitable for dashboards with all the special tokens replaced with their
            HTML representations. If None then it will be created using create_vocab_dict(tokenizer)

    Returns:
        dashboards_data:
            A dict of [sae_position_name: MultiFeatureData]. Each MultiFeatureData contains data for
            creating each feature's visualization, as well as data for rank-ordering the feature
            visualizations when it comes time to make the prompt-centric view
            (the `feature_act_quantiles` attribute).
            Use dashboards_data[sae_name][feature_idx].get_html() to generate the HTML
            dashboard for a particular feature (returns a string of HTML)
    """
    # Get the vocab dict, which we'll use at the end
    if vocab_dict is None:
        assert (
            model.tlens_model.tokenizer is not None
        ), "If voacab_dict is not supplied, the model must have a tokenizer"
        vocab_dict = create_vocab_dict(model.tlens_model.tokenizer)

    if fvp is None:
        fvp = FeatureVisParams(include_left_tables=False)

    if sae_position_names is None:
        raw_sae_position_names: list[str] = model.raw_sae_position_names
    else:
        raw_sae_position_names: list[str] = filter_names(
            list(model.tlens_model.hook_dict.keys()), sae_position_names
        )
    # If we haven't supplied any feature indicies, assume that we want all of them
    feature_indices_tensors = feature_indices_to_tensordict(
        feature_indices_in=feature_indices,
        raw_sae_position_names=raw_sae_position_names,
        model=model,
    )
    for sae_name in raw_sae_position_names:
        assert (
            feature_indices_tensors[sae_name].max().item()
            < model.saes[sae_name.replace(".", "-")].n_dict_components
        ), "Error: Some feature indices are greater than the number of SAE features"

    device = model.saes[raw_sae_position_names[0].replace(".", "-")].device
    # Get the SAE feature activations (as well as their resudual stream inputs and outputs)
    if tokens is None:
        assert data_config is not None, "If no tokens are supplied, then config must be supplied"
        assert (
            batch_size is not None
        ), "If no tokens are supplied, then a batch_size must be supplied"
        feature_acts, final_resid_acts, tokens = compute_feature_acts_on_distribution(
            model=model,
            data_config=data_config,
            batch_size=batch_size,
            raw_sae_position_names=raw_sae_position_names,
            feature_indices=feature_indices_tensors,
            n_samples=n_samples,
        )
    else:
        tokens.to(device)
        feature_acts, final_resid_acts = compute_feature_acts(
            model=model,
            tokens=tokens,
            raw_sae_position_names=raw_sae_position_names,
            feature_indices=feature_indices_tensors,
        )

    # Filter out the never active features:
    for sae_name in raw_sae_position_names:
        acts_sum = einsum(feature_acts[sae_name], "... some_feats -> some_feats").to("cpu")
        feature_acts[sae_name] = feature_acts[sae_name][..., acts_sum > 0]
        feature_indices_tensors[sae_name] = feature_indices_tensors[sae_name][acts_sum > 0]
        del acts_sum

    dashboards_data: dict[str, MultiFeatureData] = {
        name: MultiFeatureData() for name in raw_sae_position_names
    }

    for sae_name in raw_sae_position_names:
        sae = model.saes[sae_name.replace(".", "-")]
        W_dec: Float[Tensor, "feats dim"] = sae.decoder.weight.T
        feature_resid_dirs: Float[Tensor, "some_feats dim"] = W_dec[
            feature_indices_tensors[sae_name]
        ]
        W_U = model.tlens_model.W_U

        # Break up the features into batches
        if minibatch_size_features is None:
            feature_acts_batches = [feature_acts[sae_name]]
            feature_batches = [feature_indices_tensors[sae_name].tolist()]
            feature_resid_dir_batches = [feature_resid_dirs]
        else:
            feature_acts_batches = feature_acts[sae_name].split(minibatch_size_features, dim=-1)
            feature_batches = [
                x.tolist() for x in feature_indices_tensors[sae_name].split(minibatch_size_features)
            ]
            feature_resid_dir_batches = feature_resid_dirs.split(minibatch_size_features)
        for i in tqdm(iterable=range(len(feature_batches)), desc="Parsing activation data"):
            new_feature_data = parse_activation_data(
                tokens=tokens,
                feature_acts=feature_acts_batches[i].to_dense().to(device),
                final_resid_acts=final_resid_acts,
                feature_resid_dirs=feature_resid_dir_batches[i],
                feature_indices_list=feature_batches[i],
                W_U=W_U,
                vocab_dict=vocab_dict,
                fvp=fvp,
            )
            dashboards_data[sae_name].update(new_feature_data)

    return dashboards_data


@torch.inference_mode()
def parse_prompt_data(
    tokens: Int[Tensor, "batch pos"],
    str_tokens: list[str],
    features_data: MultiFeatureData,
    feature_acts: Float[Tensor, "seq some_feats"],
    final_resid_acts: Float[Tensor, "seq d_resid"],
    feature_resid_dirs: Float[Tensor, "some_feats dim"],
    feature_indices_list: list[int],
    W_U: Float[Tensor, "dim d_vocab"],
    num_top_features: int = 10,
) -> MultiPromptData:
    """Gets data that will be used to create the sequences in the prompt-centric HTML visualisation.
       This visualization displays dashboards for the most relevant features on a prompt.
       Adapted from sae_vis.data_fetching_fns.get_prompt_data().
    Args:
        tokens: The input prompt to the model as tokens
        str_tokens: The input prompt to the model as a list of strings (one string per token)
        features_data: A MultiFeatureData containing information required to plot the features.
        feature_acts: The activations values of the features
        final_resid_acts: The activations of the final layer of the model
        feature_resid_dirs: The directions that each feature writes to the logit output
        feature_indices_list: The indices of the features we're interested in
        W_U: The unembed weights for the logit lens
        num_top_features: The number of top features to display in this view, for any given metric.
    Returns:
        A MultiPromptData object containing data for visualizing the most relevant features
        given the prompt.

        Similar to parse_feature_data, except it just gets the data relevant for a particular
        sequence (i.e. a custom one that the user inputs on their own).

    The ordering metric for relevant features is set by the str_score parameter in the
    MultiPromptData.get_html() method: it can be "act_size", "act_quantile", or "loss_effect"
    """
    torch.cuda.empty_cache()
    device = W_U.device
    n_feats = len(feature_indices_list)
    batch, seq_len = tokens.shape
    feats_contribution_to_loss = torch.empty(size=(n_feats, seq_len - 1), device=device)

    # Some logit computations which we only need to do once
    correct_token_unembeddings = W_U[:, tokens[0, 1:]]  # [d_model seq]
    orig_logits = (
        final_resid_acts / final_resid_acts.std(dim=-1, keepdim=True)
    ) @ W_U  # [seq d_vocab]

    sequence_data_dict: dict[int, SequenceData] = {}

    for i, feat in enumerate(feature_indices_list):
        # ! Calculate all data for the sequences (this is the only truly 'new' bit of calculation we need to do)

        # Get this feature's output vector, using an outer product over the feature activations for all tokens
        final_resid_acts_feature_effect = einsum(
            feature_acts[..., i].to_dense().to(device),
            feature_resid_dirs[i],
            "seq, d_model -> seq d_model",
        )

        # Ablate the output vector from the residual stream, and get logits post-ablation
        new_final_resid_acts = final_resid_acts - final_resid_acts_feature_effect
        new_logits = (new_final_resid_acts / new_final_resid_acts.std(dim=-1, keepdim=True)) @ W_U

        # Get the top5 & bottom5 changes in logits (don't bother with `efficient_topk` cause it's small)
        contribution_to_logprobs = orig_logits.log_softmax(dim=-1) - new_logits.log_softmax(dim=-1)
        top5_contribution_to_logits = TopK(contribution_to_logprobs[:-1], k=5)
        bottom5_contribution_to_logits = TopK(contribution_to_logprobs[:-1], k=5, largest=False)

        # Get the change in loss (which is negative of change of logprobs for correct token)
        contribution_to_loss = eindex(-contribution_to_logprobs[:-1], tokens[0, 1:], "seq [seq]")
        feats_contribution_to_loss[i, :] = contribution_to_loss

        # Store the sequence data
        sequence_data_dict[feat] = SequenceData(
            token_ids=tokens.squeeze(0).tolist(),
            feat_acts=feature_acts[..., i].tolist(),
            contribution_to_loss=[0.0] + contribution_to_loss.tolist(),
            top5_token_ids=top5_contribution_to_logits.indices.tolist(),
            top5_logit_contributions=top5_contribution_to_logits.values.tolist(),
            bottom5_token_ids=bottom5_contribution_to_logits.indices.tolist(),
            bottom5_logit_contributions=bottom5_contribution_to_logits.values.tolist(),
        )

        # Get the logits for the correct tokens
        logits_for_correct_tokens = einsum(
            feature_resid_dirs[i], correct_token_unembeddings, "d_model, d_model seq -> seq"
        )

        # Add the annotations data (feature activations and logit effect) to the histograms
        freq_line_posn = feature_acts[..., i].tolist()
        freq_line_text = [
            f"\\'{str_tok}\\'<br>{act:.3f}"
            for str_tok, act in zip(str_tokens[1:], freq_line_posn, strict=False)
        ]
        features_data[feat].middle_plots_data.freq_histogram_data.line_posn = freq_line_posn
        features_data[feat].middle_plots_data.freq_histogram_data.line_text = freq_line_text  # type: ignore (due to typing bug in sae_vis)
        logits_line_posn = logits_for_correct_tokens.tolist()
        logits_line_text = [
            f"\\'{str_tok}\\'<br>{logits:.3f}"
            for str_tok, logits in zip(str_tokens[1:], logits_line_posn, strict=False)
        ]
        features_data[feat].middle_plots_data.logits_histogram_data.line_posn = logits_line_posn
        features_data[feat].middle_plots_data.logits_histogram_data.line_text = logits_line_text  # type: ignore (due to typing bug in sae_vis)

    # ! Lastly, use the 3 possible criteria (act size, act quantile, loss effect) to find all the top-scoring features

    # Construct a scores dict, which maps from things like ("act_quantile", seq_pos) to a list of the top-scoring features
    scores_dict: dict[tuple[str, str], tuple[TopK, list[str]]] = {}

    for seq_pos, str_tok in enumerate(str_tokens):
        # Filter the feature activations, since we only need the ones that are non-zero
        feat_acts_nonzero_filter = to_numpy(feature_acts[seq_pos] > 0)
        feat_acts_nonzero_locations = np.nonzero(feat_acts_nonzero_filter)[0].tolist()
        _feature_acts = (
            feature_acts[seq_pos, feat_acts_nonzero_filter].to_dense().to(device)
        )  # [feats_filtered,]
        _feature_indices_list = np.array(feature_indices_list)[feat_acts_nonzero_filter]

        if feat_acts_nonzero_filter.sum() > 0:
            k = min(num_top_features, _feature_acts.numel())

            # Get the "act_size" scores (we return it as a TopK object)
            act_size_topk = TopK(_feature_acts, k=k, largest=True)
            # Replace the indices with feature indices (these are different when feature_indices_list argument is not [0, 1, 2, ...])
            act_size_topk.indices[:] = _feature_indices_list[act_size_topk.indices]
            scores_dict[("act_size", seq_pos)] = (act_size_topk, ".3f")  # type: ignore (due to typing bug in sae_vis)

            # Get the "act_quantile" scores, which is just the fraction of cached feat acts that it is larger than
            act_quantile, act_precision = features_data.feature_act_quantiles.get_quantile(
                _feature_acts, feat_acts_nonzero_locations
            )
            act_quantile_topk = TopK(act_quantile, k=k, largest=True)
            act_formatting_topk = [f".{act_precision[i]-2}%" for i in act_quantile_topk.indices]
            # Replace the indices with feature indices (these are different when feature_indices_list argument is not [0, 1, 2, ...])
            act_quantile_topk.indices[:] = _feature_indices_list[act_quantile_topk.indices]
            scores_dict[("act_quantile", seq_pos)] = (act_quantile_topk, act_formatting_topk)  # type: ignore (due to typing bug in sae_vis)

        # We don't measure loss effect on the first token
        if seq_pos == 0:
            continue

        # Filter the loss effects, since we only need the ones which have non-zero feature acts on the tokens before them
        prev_feat_acts_nonzero_filter = to_numpy(feature_acts[seq_pos - 1] > 0)
        _contribution_to_loss = feats_contribution_to_loss[
            prev_feat_acts_nonzero_filter, seq_pos - 1
        ]  # [feats_filtered,]
        _feature_indices_list_prev = np.array(feature_indices_list)[prev_feat_acts_nonzero_filter]

        if prev_feat_acts_nonzero_filter.sum() > 0:
            k = min(num_top_features, _contribution_to_loss.numel())

            # Get the "loss_effect" scores, which are just the min of features' contributions to loss (min because we're
            # looking for helpful features, not harmful ones)
            contribution_to_loss_topk = TopK(_contribution_to_loss, k=k, largest=False)
            # Replace the indices with feature indices (these are different when feature_indices_list argument is not [0, 1, 2, ...])
            contribution_to_loss_topk.indices[:] = _feature_indices_list_prev[
                contribution_to_loss_topk.indices
            ]
            scores_dict[("loss_effect", seq_pos)] = (contribution_to_loss_topk, ".3f")  # type: ignore (due to typing bug in sae_vis)

    # Get all the features which are required (i.e. all the sequence position indices)
    feature_indices_list_required = set()
    for score_topk, formatting_topk in scores_dict.values():
        feature_indices_list_required.update(set(score_topk.indices.tolist()))

    prompt_data_dict = {
        feat: PromptData(
            prompt_data=sequence_data_dict[feat],
            sequence_data=features_data[feat].sequence_data[0],
            middle_plots_data=features_data[feat].middle_plots_data,
        )
        for feat in feature_indices_list_required
    }

    return MultiPromptData(
        prompt_str_toks=str_tokens,
        prompt_data_dict=prompt_data_dict,
        scores_dict=scores_dict,
    )


@torch.inference_mode()
def get_prompt_data(
    model: SAETransformer,
    tokens: Int[Tensor, "batch pos"],
    str_tokens: list[str],
    dashboards_data: dict[str, MultiFeatureData],
    sae_position_names: list[str] | None = None,
    num_top_features: PositiveInt = 10,
) -> dict[str, MultiPromptData]:
    """Gets data that will be used to create the sequences in the prompt-centric HTML visualisation.
       This visualization displays dashboards for the most relevant features on a prompt.
       Adapted from sae_vis.data_fetching_fns.get_prompt_data()
    Args:
        model:
            The model (with SAEs) we'll be using to get the feature activations.
        tokens:
            The input prompt to the model as tokens
        str_tokens:
            The input prompt to the model as a list of strings (one string per token)
        dashboards_data:
            For each SAE, a MultiFeatureData containing information required to plot its features.
        sae_position_names:
            The names of the SAEs we want to find relevant features in.
            eg. ['blocks.0.hook_resid_pre']. If none, then we'll do all of them.
        num_top_features: int
            The number of top features to display in this view, for any given metric.

    Returns:
        prompt_data:
            A dict of [sae_position_name: MultiPromptData]. Each MultiPromptData contains data for
            visualizing the most relevant features in that SAE given the prompt.
            Similar to get_feature_data, except it just gets the data relevant for a particular
            sequence (i.e. a custom one that the user inputs on their own).

    The ordering metric for relevant features is set by the str_score parameter in the
    MultiPromptData.get_html() method: it can be "act_size", "act_quantile", or "loss_effect"
    """
    assert tokens.shape[-1] == len(
        str_tokens
    ), "Error: the number of tokens does not equal the number of str_tokens"
    if sae_position_names is None:
        raw_sae_position_names: list[str] = model.raw_sae_position_names
    else:
        raw_sae_position_names: list[str] = filter_names(
            list(model.tlens_model.hook_dict.keys()), sae_position_names
        )
    feature_indices: dict[str, list[int]] = {}
    for sae_name in raw_sae_position_names:
        feature_indices[sae_name] = list(dashboards_data[sae_name].feature_data_dict.keys())

    feature_acts, final_resid_acts = compute_feature_acts(
        model=model,
        tokens=tokens,
        raw_sae_position_names=raw_sae_position_names,
        feature_indices=feature_indices,
    )
    final_resid_acts = final_resid_acts.squeeze(dim=0)

    prompt_data: dict[str, MultiPromptData] = {}

    for sae_name in raw_sae_position_names:
        sae = model.saes[sae_name.replace(".", "-")]
        feature_act_dir: Float[Tensor, "dim some_feats"] = sae.encoder[0].weight.T[
            :, feature_indices[sae_name]
        ]  # [d_in feats]
        feature_resid_dirs: Float[Tensor, "some_feats dim"] = sae.decoder.weight.T[
            feature_indices[sae_name]
        ]  # [feats d_in]
        assert (
            feature_act_dir.T.shape
            == feature_resid_dirs.shape
            == (len(feature_indices[sae_name]), sae.input_size)
        )

        prompt_data[sae_name] = parse_prompt_data(
            tokens=tokens,
            str_tokens=str_tokens,
            features_data=dashboards_data[sae_name],
            feature_acts=feature_acts[sae_name].squeeze(dim=0),
            final_resid_acts=final_resid_acts,
            feature_resid_dirs=feature_resid_dirs,
            feature_indices_list=feature_indices[sae_name],
            W_U=model.tlens_model.W_U,
            num_top_features=num_top_features,
        )
    return prompt_data


@torch.inference_mode()
def generate_feature_dashboard_html_files(
    dashboards_data: dict[str, MultiFeatureData],
    feature_indices: FeatureIndicesType | dict[str, set[int]] | None,
    save_dir: str | Path = "",
):
    """Generates viewable HTML dashboards for every feature in every SAE in dashboards_data"""
    if feature_indices is None:
        feature_indices = {name: dashboards_data[name].keys() for name in dashboards_data}
    save_dir = Path(save_dir)
    if not save_dir.is_absolute():
        current_dir = Path(os.getcwd())
        save_dir = Path(current_dir) / save_dir
    if not save_dir.exists():
        os.makedirs(save_dir)
    for sae_name in feature_indices:
        print(f"Saving HTML feature dashboards for the SAE at {sae_name}:")
        folder: Path = save_dir / Path(f"dashboards_{sae_name}")
        if not folder.exists():
            os.makedirs(folder)
        for feature_idx in tqdm(feature_indices[sae_name], desc="Dashboard HTML files"):
            feature_idx = (
                int(feature_idx.item()) if isinstance(feature_idx, Tensor) else feature_idx
            )
            if feature_idx in dashboards_data[sae_name].keys():
                html_str = dashboards_data[sae_name][feature_idx].get_html()
                filepath = folder / Path(f"feature-{feature_idx}.html")
                with open(filepath, "w") as f:
                    f.write(html_str)
        print(f"Saved HTML feature dashboards in {folder}")


@torch.inference_mode()
def generate_prompt_dashboard_html_files(
    model: SAETransformer,
    tokens: Int[Tensor, "batch pos"],
    str_tokens: list[str],
    dashboards_data: dict[str, MultiFeatureData],
    seq_pos: int | list[int] | None = None,
    vocab_dict: dict[int, str] | None = None,
    str_score: StrScoreType = "loss_effect",
    save_dir: str | Path = "",
) -> dict[str, set[int]]:
    """Generates viewable HTML dashboards for the most relevant features (measured by str_score) for
    every SAE in dashboards_data.
    Returns the set of feature indices which were active"""
    assert tokens.shape[-1] == len(
        str_tokens
    ), "Error: the number of tokens does not equal the number of str_tokens"
    str_tokens = [s.replace("Ġ", " ") for s in str_tokens]
    if isinstance(seq_pos, int):
        seq_pos = [seq_pos]
    if seq_pos is None:  # Generate a dashboard for every position if none is specified
        seq_pos = list(range(1, len(str_tokens) - 1))
    if vocab_dict is None:
        assert (
            model.tlens_model.tokenizer is not None
        ), "If voacab_dict is not supplied, the model must have a tokenizer"
        vocab_dict = create_vocab_dict(model.tlens_model.tokenizer)
    prompt_data = get_prompt_data(
        model=model, tokens=tokens, str_tokens=str_tokens, dashboards_data=dashboards_data
    )
    prompt = "".join(str_tokens)
    # Use the beginning of the prompt for the filename, but make sure that it's safe for a filename
    str_tokens_safe_for_filenames = [
        "".join(c for c in token if c.isalpha() or c.isdigit() or c == " ")
        .rstrip()
        .replace(" ", "-")
        for token in str_tokens
    ]
    filename_from_prompt = "".join(str_tokens_safe_for_filenames)
    filename_from_prompt = filename_from_prompt[:50]
    save_dir = Path(save_dir)
    if not save_dir.is_absolute():
        current_dir = Path(os.getcwd())
        save_dir = Path(current_dir) / save_dir
    if not save_dir.exists():
        os.makedirs(save_dir)
    used_features: dict[str, set[int]] = {sae_name: set() for sae_name in dashboards_data}
    for sae_name in dashboards_data:
        for seq_pos_i in seq_pos:
            # Find the most relevant features (by {str_score}) for the token '{str_tokens[seq_pos_i]}' in the prompt '{prompt}:
            html_str = prompt_data[sae_name].get_html(
                seq_pos=seq_pos_i, str_score=str_score, vocab_dict=vocab_dict
            )
            # Insert a title
            title: str = (
                f"<h4>&nbsp&nbspThe most relevant features from {sae_name},<br/>&nbsp&nbsp"
                f"measured by {str_score} on the '{str_tokens[seq_pos_i].replace('Ġ',' ')}' "
                f"token (token number {seq_pos_i}) in the prompt '{prompt}':</h4>"
            )
            substr = "<div class='grid-container'>"
            html_str = html_str.replace(
                substr, "<div class='grid-container'>" + title + "</div>\n" + substr
            )
            filepath = save_dir / Path(
                f"prompt-{filename_from_prompt}_token-{seq_pos_i}-{str_tokens_safe_for_filenames[seq_pos_i]}_-{str_score.replace('_','-')}_sae-{sae_name}.html"
            )
            with open(filepath, "w") as f:
                f.write(html_str)
            scores = prompt_data[sae_name].scores_dict[(str_score, seq_pos_i)][0]  # type: ignore
            used_features[sae_name] = used_features[sae_name].union(set(scores.indices.tolist()))
    return used_features


@torch.inference_mode()
def generate_random_prompt_dashboards(
    model: SAETransformer,
    dashboards_data: dict[str, MultiFeatureData],
    dashboards_config: DashboardsConfig,
    use_model_tokenizer: bool = False,
    save_dir: Path | None = None,
) -> dict[str, set[int]]:
    """Generates prompt-centric HTML dashboards for prompts from the training distribution
    A data_loader is created using the dashboards_config.prompt_centric.data if it exists,
    otherwise using the dashboards_config.data config."""
    assert (
        dashboards_config.save_dir is not None
    ), "generate_random_prompt_dashboards() saves HTML files, but no save_dir was specified in the dashboards_config"
    assert (
        dashboards_config.prompt_centric is not None
    ), "generate_random_prompt_dashboards() makes prompt-centric dashboards: the dashboards_config.prompt_centric config must exist"
    data_config = (
        dashboards_config.prompt_centric.data
        if dashboards_config.prompt_centric.data
        else dashboards_config.data
    )
    data_loader, _ = create_data_loader(data_config=data_config, batch_size=1)
    assert model.tlens_model.tokenizer is not None, "The model must have a tokenizer"
    if use_model_tokenizer:
        tokenizer = model.tlens_model.tokenizer
        assert isinstance(tokenizer, PreTrainedTokenizer | PreTrainedTokenizerFast)
    else:
        tokenizer = AutoTokenizer.from_pretrained(dashboards_config.data.tokenizer_name)
    vocab_dict = create_vocab_dict(tokenizer)
    if dashboards_config.sae_position_names is None:
        raw_sae_position_names: list[str] = model.raw_sae_position_names
    else:
        raw_sae_position_names: list[str] = filter_names(
            list(model.tlens_model.hook_dict.keys()), dashboards_config.sae_position_names
        )

    used_features: dict[str, set[int]] = {sae_name: set() for sae_name in dashboards_data}
    device = model.saes[raw_sae_position_names[0].replace(".", "-")].device
    n_batches = (dashboards_config.prompt_centric.n_random_prompt_dashboards + 2) // 3
    for batch_idx, batch in tqdm(
        enumerate(data_loader),
        total=n_batches,
        desc="Random prompt dashboards",
    ):
        batch_tokens: Int[Tensor, "1 pos"] = batch[dashboards_config.data.column_name].to(
            device=device
        )
        assert len(batch_tokens.shape) == 2 and batch_tokens.shape[0] == 1
        str_tokens = tokenizer.convert_ids_to_tokens(batch_tokens.squeeze(dim=0).tolist())
        assert isinstance(str_tokens, list)
        seq_len: int = batch_tokens.shape[1]
        seq_pos_c = np.random.randint(2, seq_len - 2)
        seq_pos = [seq_pos_c - 1, seq_pos_c, seq_pos_c + 1]
        used_features_now = generate_prompt_dashboard_html_files(
            model=model,
            tokens=batch_tokens,
            str_tokens=str_tokens,
            dashboards_data=dashboards_data,
            seq_pos=seq_pos,
            vocab_dict=vocab_dict,
            str_score=dashboards_config.prompt_centric.str_score,
            save_dir=save_dir if save_dir else dashboards_config.save_dir,
        )
        for sae_name in used_features:
            used_features[sae_name] = used_features[sae_name].union(used_features_now[sae_name])

        if batch_idx > n_batches:
            break
    return used_features


@torch.inference_mode()
def generate_dashboards(model: SAETransformer, dashboards_config: DashboardsConfig) -> None:
    """Generate HTML feature dashboards for an SAETransformer and save them.
    First the data for the dashboards are crated using dashboards_data = get_dashboards_data(),
    then prompt-centric HTML dashboards are created (if dashboards_config.prompt_centric exists),
    then feature-centric HTML dashboards are created for any features in
    dashboards_config.feature_indices (all features if this is None), or any features which
    appeared in prompt-centric dashboards.
    Dashboards are saved in dashboards_config.save_dir
    """
    assert (
        dashboards_config.save_dir is not None
    ), "make_html_dashboards() saves HTML files, but no save_dir was specified in the dashboards_config"
    # Deal with the possible input typles of sae_position_names
    if dashboards_config.sae_position_names is None:
        raw_sae_position_names = model.raw_sae_position_names
    else:
        raw_sae_position_names = filter_names(
            list(model.tlens_model.hook_dict.keys()), dashboards_config.sae_position_names
        )
    # Deal with the possible input typles of feature_indices
    feature_indices = feature_indices_to_tensordict(
        dashboards_config.feature_indices, raw_sae_position_names, model
    )

    # Get the data used in the dashboards
    dashboards_data: dict[str, MultiFeatureData] = get_dashboards_data(
        model=model,
        data_config=dashboards_config.data,
        sae_position_names=raw_sae_position_names,
        # We need data for every feature if we're generating prompt-centric dashboards:
        feature_indices=None if dashboards_config.prompt_centric else feature_indices,
        n_samples=dashboards_config.n_samples,
        batch_size=dashboards_config.batch_size,
        minibatch_size_features=dashboards_config.minibatch_size_features,
    )

    # Generate the prompt-centric dashboards and record which features were active on them
    used_features: dict[str, set[int]] = {sae_name: set() for sae_name in dashboards_data}
    if dashboards_config.prompt_centric:
        prompt_dashboard_saving_folder = dashboards_config.save_dir / Path("prompt-dashboards")
        if dashboards_config.prompt_centric.n_random_prompt_dashboards > 0:
            used_features_now = generate_random_prompt_dashboards(
                model=model,
                dashboards_data=dashboards_data,
                dashboards_config=dashboards_config,
                save_dir=prompt_dashboard_saving_folder,
            )
            for sae_name in used_features:
                used_features[sae_name] = used_features[sae_name].union(used_features_now[sae_name])

        if dashboards_config.prompt_centric.prompts is not None:
            tokenizer = AutoTokenizer.from_pretrained(dashboards_config.data.tokenizer_name)
            vocab_dict = create_vocab_dict(tokenizer)
            for prompt in dashboards_config.prompt_centric.prompts:
                tokens = tokenizer(prompt)["input_ids"]
                list_tokens = tokens.tolist() if isinstance(tokens, Tensor) else tokens
                assert isinstance(list_tokens, list)
                str_tokens = tokenizer.convert_ids_to_tokens(list_tokens)
                assert isinstance(str_tokens, list)
                used_features_now = generate_prompt_dashboard_html_files(
                    model=model,
                    tokens=torch.Tensor(tokens).to(dtype=torch.int).unsqueeze(dim=0),
                    str_tokens=str_tokens,
                    dashboards_data=dashboards_data,
                    str_score=dashboards_config.prompt_centric.str_score,
                    vocab_dict=vocab_dict,
                    save_dir=prompt_dashboard_saving_folder,
                )
                for sae_name in used_features:
                    used_features[sae_name] = used_features[sae_name].union(
                        used_features_now[sae_name]
                    )

        for sae_name in raw_sae_position_names:
            used_features[sae_name] = used_features[sae_name].union(
                set(feature_indices[sae_name].tolist())
            )

    # Generate the viewable HTML feature dashboard files
    dashboard_html_saving_folder = dashboards_config.save_dir / Path("feature-dashboards")
    generate_feature_dashboard_html_files(
        dashboards_data=dashboards_data,
        feature_indices=used_features if dashboards_config.prompt_centric else feature_indices,
        save_dir=dashboard_html_saving_folder,
    )


# TODO: make a function for generating dashboards just from an saes filepath?
#       (see load_SAETransformer_from_saes_path in feature_dashboards.ipynb)
# TODO: Have an option for run_train_tlens_saes to automatically make dashboards?
#       (and optionally upload them to wandb, though the HTML files are quite big)
# TODO: make functions for saving and loading the dashboards_data
#       (as it's much smaller than the HTML files) piclke?
