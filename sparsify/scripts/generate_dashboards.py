"""Script for generating HTML feature dashboards
Usage: 
    $ python generate_dashboards.py </path/to/dashboards_config.yaml> </path/to/sae.pt>
    (Generates dashboards for the SAEs in </path/to/sae.pt>)
    or
    $ python generate_dashboards.py </path/to/dashboards_config.yaml>
    (Requires that a path to the sae.pt file is provided in dashboards_config.pretrained_sae_paths)

dashboard HTML files be saved in dashboards_config.save_dir
    
Two types of dashboards can be created:
    feature-centric: 
        These are individual dashboards for each feature specified in 
        dashboards_config.feature_indices, showing that feature's max activating examples, facts 
        about the distribution of when it is active, and what it promotes through the logit lens.
        feature-centric dashboards will also be generated for all of the top features which apppear
        in the prompt-centric dashboards. Saved in dashboards_config.save_dir/dashboards_{sae_name}
    prompt-centric:
        Given a prompt and a specific token position within it, find the most important features 
        active at that position, and make a dashboard showing where they all activate. There are 
        three ways of measuring the importance of features: "act_size" (show the features which 
        activated most strongly), "act_quantile" (show the features which activated much more than 
        they usually do), and "loss_effect" (show the features with the biggest logit-lens ablation 
        effect for predicting the correct next token - default). 
        Saved in dashboards_config.save_dir/prompt_dashboards

This script currently relies on an old commit of Callum McDouglal's sae_vis package: 
https://github.com/callummcdougall/sae_vis/commit/b28a0f7c7e936f4bea05528d952dfcd438533cce 
"""
import math
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal

import fire
import numpy as np
import torch
from einops import einsum
from jaxtyping import Float, Int
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, NonNegativeInt, PositiveInt
from sae_vis.data_config_classes import (
    ActsHistogramConfig,
    Column,
    LogitsHistogramConfig,
    LogitsTableConfig,
    PromptConfig,
    SaeVisConfig,
    SaeVisLayoutConfig,
    SequencesConfig,
)
from sae_vis.data_fetching_fns import parse_feature_data, parse_prompt_data
from sae_vis.data_storing_fns import SaeVisData
from sae_vis.html_fns import HTML
from sae_vis.utils_fns import get_decode_html_safe_fn
from torch import Tensor
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from sparsify.data import DatasetConfig, create_data_loader
from sparsify.loader import load_pretrained_saes, load_tlens_model
from sparsify.log import logger
from sparsify.models.transformers import SAETransformer
from sparsify.scripts.train_tlens_saes.run_train_tlens_saes import Config
from sparsify.types import RootPath
from sparsify.utils import filter_names, load_config

LAYOUT_FEATURE_VIS = SaeVisLayoutConfig(
    columns=[
        Column(ActsHistogramConfig(), LogitsTableConfig(), LogitsHistogramConfig()),
        Column(SequencesConfig(stack_mode="stack-none")),
    ],
    height=750,
)

LAYOUT_PROMPT_VIS = SaeVisLayoutConfig(
    columns=[
        Column(
            PromptConfig(),
            ActsHistogramConfig(),
            LogitsTableConfig(n_rows=5),
            SequencesConfig(top_acts_group_size=10, n_quantiles=0),
            width=450,
        ),
    ],
    height=1000,
)

FeatureIndicesType = dict[str, list[int]] | dict[str, Int[Tensor, "some_feats"]]  # noqa: F821 (jaxtyping/pyright doesn't like single dimensions)
StrScoreType = Literal["act_size", "act_quantile", "loss_effect"]


class PromptDashboardsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_random_prompt_dashboards: NonNegativeInt = Field(
        default=10,
        description="The number of random prompts to generate prompt-centric dashboards for."
        "Feature-centric dashboards will be generated for each prompt.",
    )
    data: DatasetConfig | None = Field(
        default=None,
        description="DatasetConfig for getting random prompts."
        "If None, then DashboardsConfig.data will be used",
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
        description="How many of the most relevant features to show for each prompt"
        " in the prompt-centric dashboards",
    )


class DashboardsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)
    pretrained_sae_paths: Annotated[
        list[RootPath] | None, BeforeValidator(lambda x: [x] if isinstance(x, str | Path) else x)
    ] = Field(None, description="Paths of the pretrained SAEs to load")
    sae_config_path: RootPath | None = Field(
        default=None,
        description="Path to the config file used to train the SAEs"
        " (if null, we'll assume it's at pretrained_sae_paths[0].parent / 'config.yaml')",
    )
    n_samples: PositiveInt | None = None
    batch_size: PositiveInt
    minibatch_size_features: PositiveInt | None = Field(
        default=256,
        description="Num features in each batch of calculations (i.e. we break up the features to "
        "avoid OOM errors).",
    )
    data: DatasetConfig = Field(
        description="DatasetConfig for the data which will be used to generate the dashboards",
    )
    save_dir: RootPath | None = Field(
        default=None,
        description="The directory for saving the HTML feature dashboard files",
    )
    save_json_data: bool = Field(
        default=False,
        description="Whether to save JSON data which can be used to re-generate the HTML dashboards",
    )
    sae_positions: Annotated[
        list[str] | None, BeforeValidator(lambda x: [x] if isinstance(x, str) else x)
    ] = Field(
        None,
        description="The names of the SAE positions to generate dashboards for. "
        "e.g. 'blocks.2.hook_resid_post'. If None, then all positions will be generated",
    )
    feature_indices: FeatureIndicesType | list[int] | None = Field(
        default=None,
        description="The features for which to generate dashboards on each SAE. If none, then "
        "we'll generate dashbaords for every feature.",
    )
    prompt_centric: PromptDashboardsConfig | None = Field(
        default=None,
        description="Used to generate prompt-centric (rather than feature-centric) dashboards."
        " Feature-centric dashboards will also be generated for every feature appaearing in these",
    )
    seed: NonNegativeInt = 0


def compute_feature_acts(
    model: SAETransformer,
    tokens: Int[Tensor, "batch pos"],
    raw_sae_positions: list[str] | None = None,
    feature_indices: FeatureIndicesType | None = None,
    stop_at_layer: int = -1,
) -> tuple[dict[str, Float[Tensor, "... some_feats"]], Float[Tensor, "... dim"]]:
    """Compute the activations of the SAEs in the model given a tensor of input tokens

    Args:
        model: The SAETransformer containing the SAEs and the tlens_model
        tokens: The inputs to the tlens_model
        raw_sae_positions: The names of the SAEs we're interested in
        feature_indices: The indices of the features we're interested in for each SAE
        stop_at_layer: Where to stop the forward pass. final_resid_acts will be returned from here

    Returns:
        - A dict of feature activations for each SAE.
          feature_acts[sae_position_name] = the feature activations of that SAE
                                                shape: batch pos some_feats
        - The residual stream activations of the model at the final layer (or at stop_at_layer)
    """
    if raw_sae_positions is None:
        raw_sae_positions = model.raw_sae_positions
    # Run model without SAEs
    final_resid_acts, orig_acts = model.tlens_model.run_with_cache(
        tokens,
        names_filter=raw_sae_positions,
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
    return feature_acts, final_resid_acts.to("cpu")


def compute_feature_acts_on_distribution(
    model: SAETransformer,
    dataset_config: DatasetConfig,
    batch_size: PositiveInt,
    n_samples: PositiveInt | None = None,
    raw_sae_positions: list[str] | None = None,
    feature_indices: FeatureIndicesType | None = None,
    stop_at_layer: int = -1,
) -> tuple[
    dict[str, Float[Tensor, "... some_feats"]], Float[Tensor, "... d_resid"], Int[Tensor, "..."]
]:
    """Compute the activations of the SAEs in the model on a dataset of input tokens

    Args:
        model: The SAETransformer containing the SAEs and the tlens_model
        dataset_config: The DatasetConfig used to get the data loader for the tokens.
        batch_size: The batch size of data run through the model when calculating the feature acts
        n_samples: The number of batches of data to use for calculating the feature dashboard data
        raw_sae_positions: The names of the SAEs we're interested in. If none, do all SAEs.
        feature_indices: The indices of the features we're interested in for each SAE. If none, do
                         all features.
        stop_at_layer: Where to stop the forward pass. final_resid_acts will be returned from here

    Returns:
        - a dict of SAE inputs, activations, and outputs for each SAE.
          feature_acts[sae_position_name] = the feature activations of that SAE
                                              shape: batch pos feats (or # feature_indices)
        - The residual stream activations of the model at the final layer (or at stop_at_layer)
        - The tokens used as input to the model
    """
    data_loader, _ = create_data_loader(dataset_config, batch_size, buffer_size=batch_size)
    if raw_sae_positions is None:
        raw_sae_positions = model.raw_sae_positions
        assert raw_sae_positions is not None
    device = model.saes[raw_sae_positions[0].replace(".", "-")].device
    if n_samples is None:
        # If streaming (i.e. if the dataset is an IterableDataset), we don't know the length
        n_batches = (
            float("inf") if isinstance(data_loader.dataset, IterableDataset) else len(data_loader)
        )
    else:
        n_batches = math.ceil(n_samples / batch_size)
        if not isinstance(data_loader.dataset, IterableDataset):
            n_batches = min(n_batches, len(data_loader))

    total_samples = 0
    feature_acts_lists: dict[str, list[Float[Tensor, "... some_feats"]]] = {
        sae_name: [] for sae_name in raw_sae_positions
    }
    final_resid_acts_list: list[Float[Tensor, "... d_resid"]] = []
    tokens_list: list[Int[Tensor, "..."]] = []
    for batch in tqdm(data_loader, total=n_batches, desc="Computing feature acts"):
        batch_tokens: Int[Tensor, "..."] = batch[dataset_config.column_name].to(device=device)
        batch_feature_acts, batch_final_resid_acts = compute_feature_acts(
            model,
            batch_tokens,
            raw_sae_positions=raw_sae_positions,
            feature_indices=feature_indices,
            stop_at_layer=stop_at_layer,
        )
        for sae_name in raw_sae_positions:
            feature_acts_lists[sae_name].append(batch_feature_acts[sae_name])
        final_resid_acts_list.append(batch_final_resid_acts)
        tokens_list.append(batch_tokens)
        if n_samples is not None and total_samples > n_samples:
            break
        total_samples += batch_tokens.shape[0]
    final_resid_acts: Float[Tensor, "... d_resid"] = torch.cat(final_resid_acts_list, dim=0)
    tokens: Int[Tensor, "..."] = torch.cat(tokens_list, dim=0)
    feature_acts: dict[str, Float[Tensor, "... some_feats"]] = {}
    for sae_name in raw_sae_positions:
        feature_acts[sae_name] = torch.cat(feature_acts_lists[sae_name], dim=0)
    return feature_acts, final_resid_acts, tokens


def feature_indices_to_tensordict(
    feature_indices_in: FeatureIndicesType | list[int] | None,
    raw_sae_positions: list[str],
    model: SAETransformer,
) -> dict[str, Tensor]:
    """ "Convert feature indices to a dict of tensor indices"""
    if feature_indices_in is None:
        feature_indices = {}
        for sae_name in raw_sae_positions:
            feature_indices[sae_name] = torch.arange(
                end=model.saes[sae_name.replace(".", "-")].n_dict_components
            )
    # Otherwise make sure that feature_indices is a dict of Int[Tensor]
    elif not isinstance(feature_indices_in, dict):
        feature_indices = {
            sae_name: Tensor(feature_indices_in).to("cpu").to(torch.int)
            for sae_name in raw_sae_positions
        }
    else:
        feature_indices: dict[str, Tensor] = {
            sae_name: Tensor(feature_indices_in[sae_name]).to("cpu").to(torch.int)
            for sae_name in raw_sae_positions
        }
    return feature_indices


@torch.inference_mode()
def get_dashboards_data(
    model: SAETransformer,
    dataset_config: DatasetConfig | None = None,
    tokens: Int[Tensor, "batch pos"] | None = None,
    sae_positions: list[str] | None = None,
    feature_indices: FeatureIndicesType | list[int] | None = None,
    n_samples: PositiveInt | None = None,
    batch_size: PositiveInt | None = None,
    minibatch_size_features: PositiveInt | None = None,
    cfg: SaeVisConfig | None = None,
) -> dict[str, SaeVisData]:
    """Gets data that needed to create the sequences in the feature-centric HTML visualisation

        Adapted from sae_vis.data_fetching_fns._get_feature_data()

    Args:
        model:
            The model (with SAEs) we'll be using to get the feature activations.
        dataset_config: [Only used if tokens is None]
            The DatasetConfig which will be used to get the data loader. If None, then tokens must
            be supplied.
        tokens:
            The tokens we'll be using to get the feature activations. If None, then we'll use the
            distribution from the dataset_config.
        sae_positions:
            The names of the SAEs we want to calculate feature dashboards for,
            eg. ['blocks.0.hook_resid_pre']. If none, then we'll do all of them.
        feature_indices:
            The features we're actually computing for each SAE. These might just be a subset of
            each SAE's full features. If None, then we'll do all of them.
        n_samples: [Only used if tokens is None]
            The number of batches of data to use for calculating the feature dashboard data when
            using dataset_config.
        batch_size: [Only used if tokens is None]
            The number of batches of data to use for calculating the feature dashboard data when
            using dataset_config
        minibatch_size_features:
            Num features in each batch of calculations (break up the features to avoid OOM errors).
        cfg:
            Feature visualization parameters, containing a bunch of other stuff. See the
            SaeVisConfig docstring in sae_vis for more information.

    Returns:
        A dict of [sae_position_name: SaeVisData]. Each SaeVisData contains data for
        creating each feature's visualization, as well as data for rank-ordering the feature
        visualizations when it comes time to make the prompt-centric view
        (the `feature_act_quantiles` attribute).
        Use dashboards_data[sae_name][feature_idx].get_html() to generate the HTML
        dashboard for a particular feature (returns a string of HTML)
    """

    if sae_positions is None:
        raw_sae_positions: list[str] = model.raw_sae_positions
    else:
        raw_sae_positions: list[str] = filter_names(
            list(model.tlens_model.hook_dict.keys()), sae_positions
        )
    # If we haven't supplied any feature indicies, assume that we want all of them
    feature_indices_tensors = feature_indices_to_tensordict(
        feature_indices,
        raw_sae_positions,
        model,
    )
    for sae_name in raw_sae_positions:
        assert (
            feature_indices_tensors[sae_name].max().item()
            < model.saes[sae_name.replace(".", "-")].n_dict_components
        ), "Error: Some feature indices are greater than the number of SAE features"

    device = model.saes[raw_sae_positions[0].replace(".", "-")].device
    # Get the SAE feature activations (as well as their resudual stream inputs and outputs)
    if tokens is None:
        assert dataset_config is not None, "If no tokens are supplied, then config must be supplied"
        assert (
            batch_size is not None
        ), "If no tokens are supplied, then a batch_size must be supplied"
        feature_acts, final_resid_acts, tokens = compute_feature_acts_on_distribution(
            model,
            dataset_config,
            batch_size,
            raw_sae_positions=raw_sae_positions,
            feature_indices=feature_indices_tensors,
            n_samples=n_samples,
        )
    else:
        tokens.to(device)
        feature_acts, final_resid_acts = compute_feature_acts(
            model,
            tokens,
            raw_sae_positions=raw_sae_positions,
            feature_indices=feature_indices_tensors,
        )

    # Filter out the never active features:
    for sae_name in raw_sae_positions:
        acts_sum = einsum(feature_acts[sae_name], "... some_feats -> some_feats").to("cpu")
        feature_acts[sae_name] = feature_acts[sae_name][..., acts_sum > 0]
        feature_indices_tensors[sae_name] = feature_indices_tensors[sae_name][acts_sum > 0]
        del acts_sum

    dashboards_data: dict[str, SaeVisData] = {name: SaeVisData() for name in raw_sae_positions}

    for sae_name in raw_sae_positions:
        if cfg is None:
            cfg = SaeVisConfig(
                hook_point=sae_name,
                features=feature_indices_tensors[sae_name].tolist(),
                feature_centric_layout=LAYOUT_FEATURE_VIS,
            )
        dashboards_data[sae_name].cfg = cfg
        dashboards_data[sae_name].model = model.tlens_model

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
            new_feature_data, _ = parse_feature_data(
                tokens,
                feature_batches[i],
                feature_acts_batches[i].to_dense().to(device),
                feature_resid_dir_batches[i].to(device),
                final_resid_acts.to(device),
                W_U.to(device),
                cfg,
            )
            dashboards_data[sae_name].update(new_feature_data)

    return dashboards_data


@torch.inference_mode()
def get_prompt_data(
    model: SAETransformer,
    tokens: Int[Tensor, "batch pos"],
    str_tokens: list[str],
    dashboards_data: dict[str, SaeVisData],
    sae_positions: list[str] | None = None,
    num_top_features: PositiveInt = 10,
) -> tuple[dict[str, SaeVisData], dict[str, dict[str, tuple[list[int], list[str]]]]]:
    """Gets data needed to create the sequences in the prompt-centric HTML visualisation.

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
            For each SAE, a SaeVisData containing information required to plot its features.
        sae_positions:
            The names of the SAEs we want to find relevant features in.
            eg. ['blocks.0.hook_resid_pre']. If none, then we'll do all of them.
        num_top_features: int
            The number of top features to display in this view, for any given metric.

    Returns:
        A dict of [sae_position_name: SaeVisData]. Each SaeVisData contains data for
        visualizing the most relevant features in that SAE given the prompt.
        Similar to get_feature_data, except it just gets the data relevant for a particular
        sequence (i.e. a custom one that the user inputs on their own).

    The ordering metric for relevant features is set by the str_score parameter in the
    SaeVisData.get_html() method: it can be "act_size", "act_quantile", or "loss_effect"
    """
    assert tokens.shape[-1] == len(
        str_tokens
    ), "Error: the number of tokens does not equal the number of str_tokens"
    if sae_positions is None:
        raw_sae_positions: list[str] = model.raw_sae_positions
    else:
        raw_sae_positions: list[str] = filter_names(
            list(model.tlens_model.hook_dict.keys()), sae_positions
        )
    device = model.saes[raw_sae_positions[0].replace(".", "-")].device
    tokens = tokens.to(device)
    feature_indices: dict[str, list[int]] = {}
    for sae_name in raw_sae_positions:
        feature_indices[sae_name] = list(dashboards_data[sae_name].feature_data_dict.keys())

    feature_acts, final_resid_acts = compute_feature_acts(
        model,
        tokens,
        raw_sae_positions=raw_sae_positions,
        feature_indices=feature_indices,
    )
    final_resid_acts = final_resid_acts.squeeze(dim=0)

    scores_dicts: dict[str, dict[str, tuple[list[int], list[str]]]] = {}

    for sae_name in raw_sae_positions:
        dashboards_data[sae_name].model = model.tlens_model
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

        scores_dicts[sae_name] = parse_prompt_data(
            tokens,
            str_tokens,
            dashboards_data[sae_name],
            feature_acts[sae_name].squeeze(dim=0).to(device),
            feature_resid_dirs.to(device),
            final_resid_acts.to(device),
            model.tlens_model.W_U.to(device),
            feature_idx=feature_indices[sae_name],
            num_top_features=num_top_features,
        )
    return dashboards_data, scores_dicts


@torch.inference_mode()
def generate_feature_dashboard_html_files(
    dashboards_data: dict[str, SaeVisData],
    minibatch_size_features: PositiveInt | None,
    save_dir: str | Path = "",
):
    """Generates viewable HTML dashboards for every feature in every SAE in dashboards_data"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for sae_name in dashboards_data:
        logger.info(f"Saving HTML feature dashboards for the SAE at {sae_name}:")
        folder = save_dir / Path(f"dashboards_{sae_name}")
        folder.mkdir(parents=True, exist_ok=True)
        model = dashboards_data[sae_name].model
        assert model is not None
        feature_ids = sorted(list(dashboards_data[sae_name].feature_data_dict.keys()))
        if minibatch_size_features is None:
            feature_ids_split = [feature_ids]
        else:

            def split_list(lst: list[Any], chunk_size: int) -> list[list[Any]]:
                chunks = [[] for _ in range((len(lst) + chunk_size - 1) // chunk_size)]
                for i, item in enumerate(lst):
                    chunks[i // chunk_size].append(item)
                return chunks

            feature_ids_split = split_list(feature_ids, minibatch_size_features)
        for batch_feature_ids in tqdm(feature_ids_split, desc="Dashboard HTML files"):
            batch_feature_data_dict = {
                i: dashboards_data[sae_name].feature_data_dict[i] for i in batch_feature_ids
            }
            batch_dashboards_data = SaeVisData(
                feature_data_dict=batch_feature_data_dict,
                cfg=dashboards_data[sae_name].cfg,
                model=dashboards_data[sae_name].model,
            )
            batch_dashboards_data.save_feature_centric_vis(
                filename=folder / f"features-{batch_feature_ids[0]}-to-{batch_feature_ids[-1]}.html"
            )


@torch.inference_mode()
def generate_prompt_dashboard_html_files(
    model: SAETransformer,
    tokens: Int[Tensor, "batch pos"],
    str_tokens: list[str],
    dashboards_data: dict[str, SaeVisData],
    seq_pos: list[int] | int | None = None,
    save_dir: str | Path = "",
) -> dict[str, set[int]]:
    """Generates viewable HTML dashboards for the most relevant features (measured by str_score) for
    every SAE in dashboards_data.

    This function is adapted from sae_vis.data_storing_functions.SaeVisData.save_prompt_centric_vis

    Returns the set of feature indices which were active"""

    assert tokens.shape[-1] == len(
        str_tokens
    ), "Error: the number of tokens does not equal the number of str_tokens"

    if isinstance(seq_pos, int):
        seq_pos = [seq_pos]
    if seq_pos is None:  # Generate a dashboard for every position if none is specified
        seq_pos = list(range(2, len(str_tokens) - 2))

    str_toks = [t.replace("|", "│") for t in str_tokens]  # vertical line -> pipe
    str_toks_list = [f"{t!r} ({i})" for i, t in enumerate(str_toks)]
    metric_list = ["act_quantile", "act_size", "loss_effect"]

    # Get default values for dropdowns
    first_metric = "act_quantile"
    first_seq_pos = str_toks_list[seq_pos[0]]
    first_key = f"{first_metric}|{first_seq_pos}"

    # Get tokenize function (we only need to define it once)
    decode_fn = get_decode_html_safe_fn(model.tlens_model.tokenizer)

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
    save_dir.mkdir(parents=True, exist_ok=True)

    METRIC_TITLES = {
        "act_size": "Activation Size",
        "act_quantile": "Activation Quantile",
        "loss_effect": "Loss Effect",
    }

    # Run forward passes on our prompt, and store the data within each FeatureData object
    # as `self.prompt_data` as well as returning the scores_dict (which maps from score hash to a
    # list of feature indices & formatted scores)
    dashboards_data, scores_dicts = get_prompt_data(
        model,
        tokens,
        str_tokens,
        dashboards_data=dashboards_data,
    )
    used_features: dict[str, set[int]] = {sae_name: set() for sae_name in dashboards_data}

    for sae_name in dashboards_data:
        for _metric in metric_list:
            # Initialize the object we'll eventually get_html from
            HTML_OBJ = HTML()

            # For each (metric, seqpos) object, we merge the prompt-centric views of each of the top
            # features, then we merge
            # these all together into our HTML_OBJ
            for _seq_pos in seq_pos:
                # Create the key for this given combination of metric & seqpos, and get our
                # top features & scores
                key = f"{_metric}|{str_toks_list[_seq_pos]}"
                if key not in scores_dicts[sae_name]:
                    continue
                feature_idx_list, scores_formatted = scores_dicts[sae_name][key]
                used_features[sae_name] = used_features[sae_name].union(feature_idx_list)

                # Create HTML object, to store each feature column for all the top features for
                # this particular key
                html_obj = HTML()

                for i, (feature_idx, score_formatted) in enumerate(
                    zip(feature_idx_list, scores_formatted, strict=True)
                ):
                    # Get HTML object at this column (which includes JavaScript to set the title)
                    html_obj += (
                        dashboards_data[sae_name]
                        .feature_data_dict[feature_idx]
                        ._get_html_data_prompt_centric(
                            layout=LAYOUT_PROMPT_VIS,
                            decode_fn=decode_fn,
                            column_idx=i,
                            bold_idx=_seq_pos,
                            title=f"<h3>#{feature_idx}<br>{METRIC_TITLES[_metric]}"
                            f" = {score_formatted}</h3><hr>",
                        )
                    )

                # Add the JavaScript (which includes the titles for each column)
                HTML_OBJ.js_data[key] = deepcopy(html_obj.js_data)

                # Set the HTML data to be the one with the most columns
                if len(HTML_OBJ.html_data) < len(html_obj.html_data):
                    HTML_OBJ.html_data = deepcopy(html_obj.html_data)

            # Check our first key is in the scores_dict (if not, we should pick a different key)
            assert first_key in scores_dicts[sae_name], (
                f"Key {first_key} not found in "
                "{scores_dicts[sae_name].keys()=}. Have you tried "
                "computing your initial data with more features and/or tokens, "
                "to make sure you have enough positive examples?"
            )

            filename = save_dir / Path(
                f"prompt-{filename_from_prompt}_{_metric}_sae-{sae_name}.html"
            )

            # Save our full HTML
            HTML_OBJ.get_html(
                LAYOUT_PROMPT_VIS,
                filename,
                first_key,
            )
    return used_features


@torch.inference_mode()
def generate_random_prompt_dashboards(
    model: SAETransformer,
    dashboards_data: dict[str, SaeVisData],
    dashboards_config: DashboardsConfig,
    use_model_tokenizer: bool = False,
    save_dir: RootPath | None = None,
) -> dict[str, set[int]]:
    """Generates prompt-centric HTML dashboards for prompts from the training distribution.

    A data_loader is created using the dashboards_config.prompt_centric.data if it exists,
    otherwise using the dashboards_config.data config.
    For each random prompt, dashboards are generated for three consecutive sequence positions.

    Returns the set of feature indices which were active"""
    np.random.seed(dashboards_config.seed)
    if save_dir is None:
        save_dir = dashboards_config.save_dir
        assert save_dir is not None, (
            "generate_random_prompt_dashboards() saves HTML files, but no save_dir was specified in"
            + " the dashboards_config or given as input"
        )
    assert dashboards_config.prompt_centric is not None, (
        "generate_random_prompt_dashboards() makes prompt-centric dashboards: "
        + "the dashboards_config.prompt_centric config must exist"
    )
    dataset_config = (
        dashboards_config.prompt_centric.data
        if dashboards_config.prompt_centric.data
        else dashboards_config.data
    )
    data_loader, _ = create_data_loader(dataset_config=dataset_config, batch_size=1, buffer_size=1)
    assert model.tlens_model.tokenizer is not None, "The model must have a tokenizer"
    if use_model_tokenizer:
        tokenizer = model.tlens_model.tokenizer
        assert isinstance(tokenizer, PreTrainedTokenizer | PreTrainedTokenizerFast)
    else:
        tokenizer = AutoTokenizer.from_pretrained(dashboards_config.data.tokenizer_name)
    if dashboards_config.sae_positions is None:
        raw_sae_positions: list[str] = model.raw_sae_positions
    else:
        raw_sae_positions: list[str] = filter_names(
            list(model.tlens_model.hook_dict.keys()), dashboards_config.sae_positions
        )

    used_features: dict[str, set[int]] = {sae_name: set() for sae_name in dashboards_data}
    device = model.saes[raw_sae_positions[0].replace(".", "-")].device
    n_prompts = dashboards_config.prompt_centric.n_random_prompt_dashboards
    for prompt_idx, batch in tqdm(
        enumerate(data_loader),
        total=n_prompts,
        desc="Random prompt dashboards",
    ):
        batch_tokens: Int[Tensor, "1 pos"] = batch[dashboards_config.data.column_name].to(
            device=device
        )
        assert len(batch_tokens.shape) == 2 and batch_tokens.shape[0] == 1
        # Use the tokens from the first <|endoftext|> token to the next
        bos_inds = torch.argwhere(batch_tokens == tokenizer.bos_token_id)[:, 1]
        if len(bos_inds) > 1:
            batch_tokens = batch_tokens[:, bos_inds[0] : bos_inds[1]]
        if batch_tokens.shape[1] > 50:
            batch_tokens = batch_tokens[:, :50]
        str_tokens = tokenizer.convert_ids_to_tokens(batch_tokens.squeeze(dim=0).tolist())
        assert isinstance(str_tokens, list)
        seq_len: int = batch_tokens.shape[1]
        # Generate dashboards for three consecutive positions in the prompt, chosen randomly
        seq_pos = None
        if seq_len > 4:  # Ensure the prompt is long enough for three positions + next token effect
            seq_pos_c = np.random.randint(1, seq_len - 3)
            seq_pos = [seq_pos_c - 1, seq_pos_c, seq_pos_c + 1]
        # Generate dashboards for three consecutive positions in the prompt, chosen randomly
        used_features_now = generate_prompt_dashboard_html_files(
            model,
            batch_tokens,
            str_tokens,
            dashboards_data,
            seq_pos=seq_pos,
            save_dir=save_dir,
        )
        for sae_name in used_features:
            used_features[sae_name] = used_features[sae_name].union(used_features_now[sae_name])

        if prompt_idx > n_prompts:
            break
    return used_features


@torch.inference_mode()
def generate_dashboards(
    model: SAETransformer, dashboards_config: DashboardsConfig, save_dir: RootPath | None = None
) -> None:
    """Generate HTML feature dashboards for an SAETransformer and save them.

    First the data for the dashboards are crated using dashboards_data = get_dashboards_data(),
    then prompt-centric HTML dashboards are created (if dashboards_config.prompt_centric exists),
    then feature-centric HTML dashboards are created for any features in
    dashboards_config.feature_indices (all features if this is None), or any features which
    appeared in prompt-centric dashboards.
    Dashboards are saved in dashboards_config.save_dir
    """
    if save_dir is None:
        save_dir = dashboards_config.save_dir
        assert save_dir is not None, (
            "generate_dashboards() saves HTML files, but no save_dir was specified in the"
            + " dashboards_config or given as input"
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    # Deal with the possible input typles of sae_positions
    if dashboards_config.sae_positions is None:
        raw_sae_positions = model.raw_sae_positions
    else:
        raw_sae_positions = filter_names(
            list(model.tlens_model.hook_dict.keys()), dashboards_config.sae_positions
        )
    # Deal with the possible input typles of feature_indices
    feature_indices = feature_indices_to_tensordict(
        dashboards_config.feature_indices, raw_sae_positions, model
    )

    # Get the data used in the dashboards
    dashboards_data: dict[str, SaeVisData] = get_dashboards_data(
        model,
        dataset_config=dashboards_config.data,
        sae_positions=raw_sae_positions,
        # We need data for every feature if we're generating prompt-centric dashboards:
        feature_indices=None if dashboards_config.prompt_centric else feature_indices,
        n_samples=dashboards_config.n_samples,
        batch_size=dashboards_config.batch_size,
        minibatch_size_features=dashboards_config.minibatch_size_features,
    )

    if dashboards_config.save_json_data:
        logger.info(f"Saving dashboards data json files to {save_dir}:")
        for sae_name in dashboards_data:
            dashboards_data[sae_name].save_json(
                str(save_dir / Path(f"dashboards_data_{sae_name.replace('.','-')}.json"))
            )
        logger.info("Saved.")

    # Generate the viewable HTML feature dashboard files
    dashboard_html_saving_folder = save_dir / Path("feature-dashboards")
    dashboard_html_saving_folder.mkdir(parents=True, exist_ok=True)
    generate_feature_dashboard_html_files(
        dashboards_data,
        minibatch_size_features=dashboards_config.minibatch_size_features,
        save_dir=dashboard_html_saving_folder,
    )

    # Generate the prompt-centric dashboards and record which features were active on them
    used_features: dict[str, set[int]] = {sae_name: set() for sae_name in dashboards_data}
    if dashboards_config.prompt_centric:
        prompt_dashboard_saving_folder = save_dir / Path("prompt-dashboards")
        prompt_dashboard_saving_folder.mkdir(parents=True, exist_ok=True)
        # Generate random prompt-centric dashboards
        if dashboards_config.prompt_centric.n_random_prompt_dashboards > 0:
            used_features_now = generate_random_prompt_dashboards(
                model,
                dashboards_data,
                dashboards_config,
                save_dir=prompt_dashboard_saving_folder,
            )
            for sae_name in used_features:
                used_features[sae_name] = used_features[sae_name].union(used_features_now[sae_name])

        # Generate dashboards for specific prompts
        if dashboards_config.prompt_centric.prompts is not None:
            tokenizer = AutoTokenizer.from_pretrained(dashboards_config.data.tokenizer_name)
            for prompt in dashboards_config.prompt_centric.prompts:
                tokens = tokenizer(prompt)["input_ids"]
                list_tokens = tokens.tolist() if isinstance(tokens, Tensor) else tokens
                assert isinstance(list_tokens, list)
                str_tokens = tokenizer.convert_ids_to_tokens(list_tokens)
                assert isinstance(str_tokens, list)
                used_features_now = generate_prompt_dashboard_html_files(
                    model,
                    torch.Tensor(tokens).to(dtype=torch.int).unsqueeze(dim=0),
                    str_tokens,
                    dashboards_data,
                    save_dir=prompt_dashboard_saving_folder,
                )
                for sae_name in used_features:
                    used_features[sae_name] = used_features[sae_name].union(
                        used_features_now[sae_name]
                    )

        for sae_name in raw_sae_positions:
            used_features[sae_name] = used_features[sae_name].union(
                set(feature_indices[sae_name].tolist())
            )


# Load the saved SAEs and the corresponding model
def load_SAETransformer_from_saes_paths(
    pretrained_sae_paths: list[RootPath] | list[str] | None,
    config_path: RootPath | str | None = None,
    sae_positions: list[str] | None = None,
) -> tuple[SAETransformer, Config, list[str]]:
    if pretrained_sae_paths is not None:
        pretrained_sae_paths = [Path(p) for p in pretrained_sae_paths]
        for path in pretrained_sae_paths:
            assert path.exists(), f"pretrained_sae_path: {path} does not exist"
            assert path.is_file() and (
                path.suffix == ".pt" or path.suffix == ".pth"
            ), f"pretrained_sae_path: {path} is not a .pt or .pth file"

    if config_path is None:
        assert (
            pretrained_sae_paths is not None
        ), "Either config_path or pretrained_sae_paths must be provided"
        config_path = pretrained_sae_paths[0].parent / "config.yaml"
    config_path = Path(config_path)
    assert config_path.exists(), f"config_path: {config_path} does not exist"
    assert (
        config_path.is_file() and config_path.suffix == ".yaml"
    ), f"config_path: {config_path} does not exist"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(config_path, config_model=Config)
    if pretrained_sae_paths is None:
        pretrained_sae_paths = config.saes.pretrained_sae_paths
        assert pretrained_sae_paths is not None, "pretrained_sae_paths must be given or in config"
    logger.info(config)

    tlens_model = load_tlens_model(config.tlens_model_name, config.tlens_model_path)
    assert tlens_model is not None

    if sae_positions is None:
        sae_positions = config.saes.sae_positions

    raw_sae_positions = filter_names(list(tlens_model.hook_dict.keys()), sae_positions)
    model = SAETransformer(
        tlens_model,
        raw_sae_positions,
        config.saes.dict_size_to_input_ratio,
    ).to(device=device)

    all_param_names = [name for name, _ in model.saes.named_parameters()]
    trainable_param_names = load_pretrained_saes(
        model.saes,
        pretrained_sae_paths,
        all_param_names,
        config.saes.retrain_saes,
    )
    return model, config, trainable_param_names


def main(
    config_path_or_obj: Path | str | DashboardsConfig,
    pretrained_sae_paths: Path | str | list[Path] | list[str] | None,
) -> None:
    dashboards_config = load_config(config_path_or_obj, DashboardsConfig)
    logger.info(dashboards_config)

    if pretrained_sae_paths is None:
        assert (
            dashboards_config.pretrained_sae_paths is not None
        ), "pretrained_sae_paths must be provided, either in the dashboards config or as an input"
        pretrained_sae_paths = dashboards_config.pretrained_sae_paths
    else:
        pretrained_sae_paths = (
            pretrained_sae_paths
            if isinstance(pretrained_sae_paths, list)
            else [Path(pretrained_sae_paths)]
        )

    logger.info("Loading the model and SAEs")
    model, _, _ = load_SAETransformer_from_saes_paths(
        pretrained_sae_paths, dashboards_config.sae_config_path
    )
    logger.info("done")

    save_dir = dashboards_config.save_dir or Path(pretrained_sae_paths[0]).parent
    logger.info(f"The HTML dashboards will be saved in {save_dir}")
    generate_dashboards(model, dashboards_config, save_dir=save_dir)
    logger.info("Finished.")


if __name__ == "__main__":
    fire.Fire(main)
