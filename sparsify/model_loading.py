from collections import OrderedDict
import torch
from torch import nn
from sparsify.models import MLP, SparsifiedMLP
from sparsify.models.sparsifiers import SAE, Codebook, DecomposedMatrix
from sparsify.configs import Config, get_yaml_at_path, SparsifiedModelConfig


def get_base_model(config: Config):
    """Get the base model."""

    if config.base_model.type == 'mlp':
        if config.load_base_model_path:
            base_model_config = get_yaml_at_path(config.load_base_model_path)
            assert config.base_model is None, "Cannot specify both 'load_base_model_path' and 'base_model' config in the top-level config."
            config.base_model = base_model_config

        base_model = MLP(config)
        
        if config.load_base_model_path:
            model_trained_statedict = torch.load(config.load_base_model_path)
            base_model.load_state_dict(model_trained_statedict)

        if config.load_sparsifier_path is not None or config.sparsifiers is not None:
            # TODO add hooks
            pass

    elif config.base_model.type == 'transformer':
        raise NotImplementedError
    
    return base_model

def get_sparsified_model(config: Config, base_model: nn.Module):
    """Get the sparsified model.
    
    To then get the sparsifiers separately from the sparsified_model, 
    you have to either load previously saved sparsifiers and combine them with
    the trained base_model's statedict or extract them from an
    instantiation of a sparsified model.
    
    """

    if config.load_sparsifier_path is None and config.sparsifiers is None:
        print("No sparsifiers specified. Returning None.")
        return None

    combined_statedict = base_model.state_dict()

    # Load sparsifier config from path
    if config.load_sparsifier_path:
        assert config.sparsifiers is None, "Cannot specify both 'load_sparsifier_path' and 'sparsifiers' config in the top-level config."
        sparsifier_config = get_yaml_at_path(config.load_sparsifier_path)
        config.sparsifiers = sparsifier_config
        sparsifiers_state_dict = torch.load(config.load_sparsifier_path)
        combined_statedict.update(sparsifiers_state_dict)
    
    # Instantiate sparsified model
    if config.base_model.type == 'mlp':
        sparsified_model = SparsifiedMLP(config)
    elif config.base_model.type == 'transformer':
        raise NotImplementedError
    
    # Extract only the untrained sparsifiers from the sparsified model
    if not config.load_sparsifier_path:
        # sparsifiers_state_dict = sparsified_model.sparsifiers.state_dict() # TODO check if this works instead of the below for loop
        # combined_statedict.update(sparsifiers_state_dict)
        for k, v in sparsified_model.state_dict().items():
            if k.startswith("sparsifiers"):
                combined_statedict[k] = v

    # Now you can load everything (the trained base model and the 
    # trained/untrained sparsifiers) into the sparsified model
    sparsified_model.load_state_dict(combined_statedict)
    return sparsified_model

def get_models(config: Config):
    """Get the models."""

    base_model = get_base_model(config)
    sparsified_model = get_sparsified_model(config, base_model)

    return base_model, sparsified_model


