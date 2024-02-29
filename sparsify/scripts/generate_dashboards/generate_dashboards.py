#%%
import yaml
import torch
import math
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
import torch.nn.functional as F
import einops
from jaxtyping import Float, Int
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from tqdm import tqdm
from rich import print as rprint
from rich.table import Table
from collections import defaultdict
from sparsify.scripts.generate_dashboards.sae_vis.data_fetching_fns import get_feature_data
from sparsify.scripts.train_tlens_saes.run_train_tlens_saes import Config, sae_hook
from sparsify.loader import load_tlens_model, load_pretrained_saes
from sparsify.log import logger
from sparsify.utils import load_config, filter_names
from sparsify.models.sparsifiers import SAE
from sparsify.models.transformers import SAETransformer
from sparsify.data import DataConfig, create_data_loader
from pathlib import Path

from sae_vis.utils_fns import (
    k_largest_indices,
    random_range_indices,
    create_vocab_dict,
    QuantileCalculator,
    TopK,
    device,
)

from sae_vis.data_storing_fns import (
    FeatureVisParams,
    BatchedCorrCoef,
    SequenceData,
    SequenceGroupData,
    SequenceMultiGroupData,
    LeftTablesData,
    MiddlePlotsData,
    FeatureData,
    MultiFeatureData,
    HistogramData,
    PromptData,
    MultiPromptData,
)

# def load_SAETransformer_from_saes_path(saes_path: str | Path, retrain_saes: bool = False) -> SAETransformer:
#     saes_path = Path(saes_path)
#     if saes_path.suffix == '.pt':
#         config_path = saes_path.parent / 'config.yaml'
#     else:
#         saes_path = saes_path.glob("*.pt")[-1]
#         config_path = saes_path / 'config.yaml'
    
#     saes = torch.load(saes_path)
#     config = load_config(config_path, config_model=Config)
#     logger.info(config)
#     tlens_model = load_tlens_model(
#             tlens_model_name=config.tlens_model_name, tlens_model_path=config.tlens_model_path
#         )
#     model = SAETransformer(
#         config=config, tlens_model=tlens_model, raw_sae_position_names=config.saes.sae_position_names
#     )
#     trainable_param_names = load_pretrained_saes(
#         saes=model.saes,
#         pretrained_sae_paths=config.saes.pretrained_sae_paths,
#         all_param_names=[name for name, _ in model.saes.named_parameters()],
#         retrain_saes=retrain_saes,
#     )
#     return model


def load_SAETransformer_from_saes_path(saes_path: str | Path, config_path: str | Path | None = None) -> SAETransformer:
    saes_path = Path(saes_path)
    assert saes_path.suffix == '.pt' or saes_path.suffix == '.pth', "Invalid saes_path: it should point to a .pt or .pth pytorch file containing the saes moduleDict"
    assert saes_path.exists(), "saes_path does not exist"
    config_path = saes_path.parent / "config.yaml" if config_path is None else Path(config_path)
    assert config_path.exists(), "Could not find the config_path: config.yaml should be in the same folder as the saes_path"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path, config_model=Config)
    logger.info(config)
    print(config)
    print(config.tlens_model_name)
    print(config.tlens_model_path)
    tlens_model = load_tlens_model(
        tlens_model_name=config.tlens_model_name, tlens_model_path=config.tlens_model_path
    )
    raw_sae_position_names = filter_names(
        list(tlens_model.hook_dict.keys()), config.saes.sae_position_names
    )
    model = SAETransformer(
        config=config, tlens_model=tlens_model, raw_sae_position_names=raw_sae_position_names
    ).to(device=device)

    all_param_names = [name for name, _ in model.saes.named_parameters()]
    trainable_param_names = load_pretrained_saes(
        saes=model.saes,
        pretrained_sae_paths= [saes_path] if config.saes.pretrained_sae_paths is None else [saes_path] + config.saes.pretrained_sae_paths,
        all_param_names=all_param_names,
        retrain_saes=config.saes.retrain_saes,
    )
    return model, config, trainable_param_names


@torch.inference_mode()
def compute_sae_acts(model: SAETransformer,
                     tokens: Int[Tensor, "batch pos"],
                     raw_sae_position_names: list[str] | None = None,
                     feature_indices: dict[str, list[int] | Int[Tensor, "feats"] ] | None = None,
                     ) ->  dict[str, dict[str, Float[Tensor, "..."]]]:
    """ Compute the activations of the SAEs in the model given a tensor of input tokens
    Returns: sae_acts
            a dict of SAE inputs, activations, and outputs for each SAE.
            sae_acts[sae_position_name]["input"] = the residual stream input to that SAE
                                                   shape: batch pos dim
            sae_acts[sae_position_name]["c"] = the feature activations of that SAE
                                                   shape: batch pos feats (or # feature_indices)
            sae_acts[sae_position_name]["output"] = the output of that SAE (should approx be input)
                                                   shape: batch pos dim
    """
    if raw_sae_position_names is None:
        raw_sae_position_names = model.raw_sae_position_names
    stop_at_layer = max([int(name.split(".")[1]) for name in raw_sae_position_names]) + 1
    # Run model without SAEs
    orig_logits, orig_acts = model.tlens_model.run_with_cache(
        tokens,
        names_filter=raw_sae_position_names,
        return_cache_object=False,
        stop_at_layer=stop_at_layer,
    )   
    sae_acts = {hook_name: {} for hook_name in orig_acts}
    # Run the activations through the SAEs
    for hook_name in orig_acts:
        sae_hook(
            value=orig_acts[hook_name].detach().clone(),
            hook=None,
            sae=model.saes[hook_name.replace(".", "-")],
            hook_acts=sae_acts[hook_name],
        )
        if feature_indices is not None:
            sae_acts[hook_name]["c"] = sae_acts[hook_name]["c"][feature_indices[hook_name]]
        sae_acts[hook_name]["c"] = sae_acts[hook_name]["c"].to_sparse()
    return sae_acts 



@torch.inference_mode()
def compute_sae_acts_on_distribution(model: SAETransformer,
                                     config: Config, 
                                     raw_sae_position_names: list[str] | None = None,
                                     feature_indices: dict[str, list[int] | Int[Tensor, "feats"] ] | None = None,
                                     n_samples: PositiveInt | None = None, 
                                     batch_size: PositiveInt | None = None, 
                                     ) ->  dict[str, dict[str, Float[Tensor, "..."]]]:
    """ Compute the activations of the SAEs in the model on the training distribution of input tokens
    Returns: sae_acts
            a dict of SAE inputs, activations, and outputs for each SAE.
            sae_acts[sae_position_name]["input"] = the residual stream input to that SAE
                                                   shape: batch pos dim
            sae_acts[sae_position_name]["c"] = the feature activations of that SAE
                                                   shape: batch pos feats (or # feature_indices)
            sae_acts[sae_position_name]["output"] = the output of that SAE (should approx be input)
                                                   shape: batch pos dim
    """
    if n_samples is None:
        n_samples = config.train.n_samples
    if batch_size is None:
        batch_size = config.train.batch_size
    if raw_sae_position_names is None:
        raw_sae_position_names = model.raw_sae_position_names
    data_loader, _ = create_data_loader(config.data, batch_size=batch_size)

    if n_samples is None:
        # If streaming (i.e. if the dataset is an IterableDataset), we don't know the length
        n_batches = None if isinstance(data_loader.dataset, IterableDataset) else len(data_loader)
    else:
        n_batches = math.ceil(n_samples / batch_size)

    total_samples = 0
    all_sae_acts = []
    for batch_idx, batch in tqdm(enumerate(data_loader), total=n_batches, desc="Steps"):
        tokens: Int[Tensor, "batch pos"] = batch[config.data.column_name].to(device=device)
        
        all_sae_acts.append(compute_sae_acts(model, tokens, raw_sae_position_names=raw_sae_position_names))
        
        total_samples += tokens.shape[0]
        if total_samples >= n_samples:
            break
    sae_acts = {hook_name : {} for hook_name in raw_sae_position_names}
    for hook_name in raw_sae_position_names:
        sae_acts[hook_name]["input"] = torch.cat([s["input"] for s in all_sae_acts[hook_name]], dim=0)
        sae_acts[hook_name]["c"] = torch.cat([s["c"] for s in all_sae_acts[hook_name]], dim=0)
        sae_acts[hook_name]["output"] = torch.cat([s["output"] for s in all_sae_acts[hook_name]], dim=0)
    return sae_acts 






@torch.inference_mode()
def get_feature_data(
    model: SAETransformer,
    config: Config,
    tokens: Int[Tensor, "batch pos"] | None = None,
    sae_position_names: list[str] | None = None,
    feature_indices: dict[str, int | list[int] | Int[Tensor, "feats"]] | None = None,
    n_samples: PositiveInt | None = None, 
    batch_size: PositiveInt | None = None,
) -> Tuple[MultiFeatureData, Dict[str, float]]:
    '''
    Gets data that will be used to create the sequences in the feature-centric HTML visualisation.
    
    Note - this function isn't called directly by the user, it actually gets called by the `get_feature_data` function
    which does exactly the same thing except it also batches this computation by features (in accordance with the
    arguments `features` and `minibatch_size_features` from the FeatureVisParams object).

    Args:
        model: SAETransformer
            The model (with SAEs) we'll be using to get the feature activations.

        tokens: Int[Tensor, "batch pos"]
            The tokens we'll be using to get the feature activations.

        feature_indices: Union[int, list[int]]
            The features we're actually computing. These might just be a subset of the model's full features.

        fvp: FeatureVisParams
            Feature visualization parameters, containing a bunch of other stuff. See the FeatureVisParams docstring for more information.

        progress_bars: Dict[str, tqdm]
            A dictionary containing the progress bars for the forward passes and the sequence data. This is used to update the progress bars as the computation progresses.

    Returns:
        MultiFeatureData
            Containing data for creating each feature visualization, as well as data for rank-ordering the feature
            visualizations when it comes time to make the prompt-centric view (the `feature_act_quantiles` attribute).
    '''
    if sae_position_names is None:
        sae_position_names = config.saes.sae_position_names
    raw_sae_position_names = filter_names(
        list(model.tlens_model.hook_dict.keys()), sae_position_names
    )
    
    device = model.device()

    # If we haven't supplied any feature indicies, assume that we want all of them
    if feature_indices is None:
        for name in raw_sae_position_names:
            feature_indices[name] = list(range(model.saes[name.replace(".", "-")].n_dict_components))
    # Ensure that the feature_indices of each feature is a list
    for name in raw_sae_position_names:
        if isinstance(feature_indices[name], int): 
            feature_indices[name] = [feature_indices]
        feature_indices[name] = torch.Tensor(feature_indices[name],device=device).int()
        assert(max(feature_indices[name]) < model.saes[name.replace(".", "-")].n_dict_components)


    # TODO: Use sparse storage for the feature activations ?

    # Get the SAE feature activations (as well as their corresponding resudual stream inputs and outputs)
    if tokens is None:
        sae_acts = compute_sae_acts_on_distribution(model = model,
                                                    config = config, 
                                                    raw_sae_position_names = raw_sae_position_names,
                                                    feature_indices = feature_indices,
                                                    n_samples = n_samples, 
                                                    batch_size = batch_size)
    else:
        tokens.to(device)
        sae_acts = compute_sae_acts(model = model,
                                    tokens = tokens,
                                    raw_sae_position_names = raw_sae_position_names,
                                    feature_indices = feature_indices,
                                    )
    return sae_acts















    # ##################################################################################################    

    # vocab_dict = create_vocab_dict(model.tokenizer)

    # token_minibatches = (tokens,) 
    # token_minibatches = [tok.to(device) for tok in token_minibatches]

    # # ! Data setup code (defining the main objects we'll eventually return)
    # sequence_data_dict: Dict[int, SequenceMultiGroupData] = {}
    # middle_plots_data_dict: Dict[int, MiddlePlotsData] = {}

    # # ! Calculate all data for the right-hand visualisations, i.e. the sequences
    
    # for i, feat in enumerate(feature_indices):

    #     # Add this feature's sequence data to the list
    #     sequence_data_dict[feat] = get_sequences_data(
    #         tokens = tokens,
    #         feat_acts = all_feat_acts[..., i],
    #         resid_post = all_resid_post,
    #         feature_resid_dir = feature_resid_dir[i],
    #         W_U = model.W_U,
    #         fvp = fvp,
    #     )
    #     progress_bars["feats"].update(1)


    # # ! Get all data for the middle column visualisations, i.e. the two histograms & the logit table

    # # Get the logits of all features (i.e. the directions this feature writes to the logit output)
    # logits = einops.einsum(feature_resid_dir, model.W_U, "feats d_model, d_model d_vocab -> feats d_vocab")

    # for i, (feat, logit) in enumerate(zip(feature_indices, logits)):

    #     # Get data for logits (the histogram, and the table)
    #     logits_histogram_data = HistogramData(logit, n_bins=40, tickmode="5 ticks")
    #     top10_logits = TopK(logit, k=10, largest=True)
    #     bottom10_logits = TopK(logit, k=10, largest=False)

    #     # Get data for feature activations histogram (the title, and the histogram)
    #     feat_acts = all_feat_acts[..., i]
    #     nonzero_feat_acts = feat_acts[feat_acts > 0]
    #     frac_nonzero = nonzero_feat_acts.numel() / feat_acts.numel()
    #     freq_histogram_data = HistogramData(nonzero_feat_acts, n_bins=40, tickmode="ints")

    #     # Create a MiddlePlotsData object from this, and add it to the dict
    #     middle_plots_data_dict[feat] = MiddlePlotsData(
    #         bottom10_logits = bottom10_logits,
    #         top10_logits = top10_logits,
    #         logits_histogram_data = logits_histogram_data,
    #         freq_histogram_data = freq_histogram_data,
    #         frac_nonzero = frac_nonzero,
    #     )


    # # ! Return the output, as a dict of FeatureData items

    # feature_data = {
    #     feat: FeatureData(
    #         # Data-containing inputs (for the feature-centric visualisation)
    #         sequence_data = sequence_data_dict[feat],
    #         middle_plots_data = middle_plots_data_dict[feat],
    #         left_tables_data = {feat: None for feat in feature_indices},
    #         # Non data-containing inputs
    #         feature_idx = feat,
    #         vocab_dict = vocab_dict,
    #         fvp = fvp,
    #     )
    #     for feat in feature_indices
    # }

    # # Also get the quantiles, which will be useful for the prompt-centric visualisation
    # feature_act_quantiles = QuantileCalculator(data=einops.rearrange(all_feat_acts, "b s feats -> feats (b s)"))


    # return MultiFeatureData(feature_data, feature_act_quantiles)





# def get_feature_data(
#     model: SAETransformer,
#     tokens: Int[Tensor, "batch pos"],
#     fvp: FeatureVisParams,
#     encoder_B: Optional[AutoEncoder] = None,
# ) -> MultiFeatureData:
#     '''
#     This is the main function which users will run to generate the feature visualization data. It batches this
#     computation over features, in accordance with the arguments in the FeatureVisParams object (we don't want to
#     compute all the features at once, since might be too memory-intensive).

#     See the `_get_feature_data` function for an explanation of the arguments, as well as a more detailed explanation
#     of what this function is doing.
#     '''
#     # Create objects to store all the data we'll get from `_get_feature_data`
#     feature_data = MultiFeatureData()
#     time_logs = {}

#     # Get a feature list (need to deal with the case where `fvp.features` is an int, or None)
#     if fvp.features is None:
#         features_list = list(range(encoder.cfg.d_hidden))
#     elif isinstance(fvp.features, int):
#         features_list = [fvp.features]
#     else:
#         features_list = list(fvp.features)

#     # Break up the features into batches
#     feature_batches = [x.tolist() for x in torch.tensor(features_list).split(fvp.minibatch_size_features)]
#     # Calculate how many minibatches of tokens there will be (for the progress bar)
#     n_token_batches = 1 if (fvp.minibatch_size_tokens is None) else math.ceil(len(tokens) / fvp.minibatch_size_tokens)

#     # Add two progress bars (one for the forward passes, one for getting the sequence data)
#     progress_bar_tokens = tqdm(total=n_token_batches*len(feature_batches), desc="Forward passes to gather data")
#     progress_bar_feats = tqdm(total=len(features_list), desc="Getting sequence data")
#     progress_bars = {"tokens": progress_bar_tokens, "feats": progress_bar_feats}

#     # If the model is from TransformerLens, we need to apply a wrapper to it for standardization
#     assert isinstance(model, HookedTransformer), "Error: non-HookedTransformer models are not yet supported."
#     model = TransformerLensWrapper(model, fvp.hook_point)

#     # For each feat: get new data and update global data storage objects
#     for features in feature_batches:
#         new_feature_data, new_time_logs = _get_feature_data(encoder, encoder_B, model, tokens, features, fvp, progress_bars)
#         feature_data.update(new_feature_data)
#         for key, value in new_time_logs.items():
#             time_logs[key] += value

#     # If verbose, then print the output
#     if fvp.verbose:
#         total_time = sum(time_logs.values())
#         table = Table("Task", "Time", "Pct %")
#         for task, duration in time_logs.items():
#             table.add_row(task, f"{duration:.2f}s", f"{duration/total_time:.1%}")
#         rprint(table)

#     return feature_data




saes_path = "/mnt/c/Users/nadro/Desktop/SAEs-gpt2-like-joseph/layerwise_2024-02-27/out/blocks.2.hook_resid_pre_ratio-32.0_lr-0.0004_lpcoeff-8e-05_2024-02-27_20-11-33/samples_300000.pt"
model, config, _ = load_SAETransformer_from_saes_path(saes_path)
sae_acts = compute_sae_acts_on_distribution(model, config, n_samples=10, batch_size=2)
sae_acts_2 = get_feature_data(model, config, n_samples=10, batch_size=2)

# %%
load_tlens_model("gpt2-small", None)
# %%
