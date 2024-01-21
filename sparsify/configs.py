"""Config loading and saving.

Configs for various model types and their training. 

Hierarchy of configs:
- SparsifierConfig
- ModelConfig
    - BaseModelConfig
        - BaseModelMLPConfig
        - BaseModelTransformerConfig
    - SparsifiedModelConfig
    - TranscoderedModelConfig
    - NeuralNetworkSkeletonConfig
- TrainConfig
- WandbConfig
- Config

"""
from pathlib import Path
from typing import List, Optional, Dict, Type
import yaml
from pydantic import BaseModel as PydanticBaseModel 
from pydantic import validator
# Atypical import: BaseModel is a pydantic class whereas 'BaseModel' in 
# e.g. 'BaseModelMLPConfig' is a neural network. They're unrelated.

class ModelConfig(PydanticBaseModel):
    pass

class BaseModelConfig(ModelConfig):
    type: str
    
class BaseModelMLPConfig(BaseModelConfig):
    type: str
    input_size: int
    hidden_sizes: Optional[List[int]]
    output_size: int

class BaseModelTransformerConfig(BaseModelConfig):
    pass

class SparsifierConfig(PydanticBaseModel):
    type: str
    dict_eles_to_input_ratio: float
    use_bias: bool
    k: Optional[int]

class SparsifiedModelConfig(ModelConfig):
    sparsifiers: SparsifierConfig
    base_model: BaseModelConfig

class TranscoderedModelConfig(ModelConfig):
    transcoders: SparsifierConfig
    sparsifiers: SparsifierConfig
    base_model: BaseModelConfig

class NeuralNetworkSkeletonConfig(ModelConfig):
    transcoders: SparsifierConfig

class TrainConfig(PydanticBaseModel):
    learning_rate: float
    batch_size: int
    epochs: int
    sparsity_p_value: Optional[float]
    feat_sparsity_lambda: Optional[float]
    weight_sparsity_lambda: Optional[float]
    sparsifier_inp_out_recon_loss_scale: Optional[float]
    save_every_n_epochs: Optional[int]
    save_dir: Optional[Path] # Recommend varying this by train_type

class WandbConfig(PydanticBaseModel):
    project: str
    entity: str

class DataConfig(PydanticBaseModel):
    dataset: str
    dataset_path: Path

class Config(PydanticBaseModel):
    seed: int
    train_type: str
    train: Optional[TrainConfig]
    wandb: Optional[WandbConfig]
    data: Optional[DataConfig]

    load_base_model_path: Optional[Path] = None
    load_sparsifier_path: Optional[Path] = None
    load_transcoder_path: Optional[Path] = None
    load_meta_sae_path: Optional[Path] = None

    base_model: Optional[BaseModelConfig] = None
    sparsifiers: Optional[SparsifierConfig] = None
    transcoders: Optional[SparsifierConfig] = None
    meta_sae: Optional[SparsifierConfig] = None
    

def load_config(config_path: Path) -> Config:
    """Load the config from a YAML file into a Pydantic model and then load additional configs from specified paths."""
    # Load the top-level Config
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path, "r") as f:
        top_level_config_dict = yaml.safe_load(f)

    # Instantiate the top-level config
    top_level_config = Config(**top_level_config_dict)

    # Use factory function to create base_model instance
    top_level_config.base_model = model_config_factory(top_level_config_dict['base_model'])

    # Ensure that the train_type has valid model configs
    validate_train_type(top_level_config)

    # Load additional configs from paths if specified else just use provided configs
    top_level_config = loading_models_from_path_or_config(top_level_config)

    return top_level_config


def assert_config_presence(config, config_attr, path_attr):
    """
    Asserts that either the config attribute or the corresponding path is not None.
    """
    check = getattr(config, config_attr) is not None or getattr(config, path_attr) is not None
    assert check, f"Must specify either '{config_attr}' configs or '{path_attr}' in the top-level config."

def validate_train_type(top_level_config):
    """
    Validates the top-level configuration based on the train type.
    """
    config_checks = {
        "base_model": [('base_model', 'load_base_model_path')],
        "sparsified_model": [('base_model', 'load_base_model_path'), ('sparsifiers', 'load_sparsifier_path')],
        "transcodered_model": [('base_model', 'load_base_model_path'), ('sparsifiers', 'load_sparsifier_path'), ('transcoders', 'load_transcoder_path')],
        "meta_sae": [('base_model', 'load_base_model_path'), ('sparsifiers', 'load_sparsifier_path'), ('transcoders', 'load_transcoder_path'), ('meta_sae', 'load_meta_sae_path')]
    }

    checks = config_checks.get(top_level_config.train_type)
    if checks is None:
        raise ValueError(f"Invalid train_type {top_level_config.train_type} in the top-level config.")

    for config_attr, path_attr in checks:
        assert_config_presence(top_level_config, config_attr, path_attr)

def loading_models_from_path_or_config(top_level_config):
    if top_level_config.load_base_model_path is not None:
        if top_level_config.base_model is not None:
            raise ValueError("Cannot specify both 'load_base_model_path' and 'base_model' in the top-level config.")
        else:
            top_level_config.base_model = load_specific_config(BaseModelConfig, top_level_config.load_base_model_path)
    if top_level_config.load_sparsifier_path is not None:
        if top_level_config.sparsifiers is not None:
            raise ValueError("Cannot specify both 'load_sparsifier_path' and 'sparsifiers' in the top-level config.")
        else:
            top_level_config.sparsifiers = load_specific_config(SparsifierConfig, top_level_config.load_sparsifier_path)
    if top_level_config.load_transcoder_path is not None:
        if top_level_config.transcoders is not None:
            raise ValueError("Cannot specify both 'load_transcoder_path' and 'transcoders' in the top-level config.")
        else:
            top_level_config.transcoders = load_specific_config(SparsifierConfig, top_level_config.load_transcoder_path)
    if top_level_config.load_meta_sae_path is not None:
        if top_level_config.meta_sae is not None:
            raise ValueError("Cannot specify both 'load_meta_sae_path' and 'meta_sae' in the top-level config.")
        else:
            top_level_config.meta_sae = load_specific_config(SparsifierConfig, top_level_config.load_meta_sae_path)

    return top_level_config

def model_config_factory(config_dict: Dict) -> BaseModelConfig:
    """
    Factory function to create an instance of the correct BaseModelConfig subclass 
    based on the 'type' field in the provided configuration dictionary.

    Args:
    - config_dict (Dict): The configuration dictionary.

    Returns:
    - An instance of the appropriate BaseModelConfig subclass.
    """

    # Mapping of type values to corresponding model config classes
    model_type_to_class: Dict[str, Type[BaseModelConfig]] = {
        'mlp': BaseModelMLPConfig,
        'transformer': BaseModelTransformerConfig
    }

    # Extract the type from the config dict
    model_type = config_dict.get('type')

    # Get the appropriate class for the given type
    model_class = model_type_to_class.get(model_type)

    if model_class is None:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create an instance of the model class with the config dictionary
    return model_class(**config_dict)

def load_specific_config(config_class: PydanticBaseModel, config_path: Path) -> PydanticBaseModel:
    """Load a specific config from a YAML file into the specified Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return config_class(**config_dict)

def get_yaml_at_path(path):
    """Read a single YAML file from a directory and return its content."""
    # Ensure the path is a directory
    if not path.is_dir():
        raise ValueError("Provided path is not a directory")

    # Find all YAML files in the directory
    yaml_files = list(path.glob('*.yaml')) + list(path.glob('*.yml'))

    # Check if there is exactly one YAML file
    if len(yaml_files) != 1:
        raise ValueError("There should be exactly one YAML file in the directory")

    # Read and parse the YAML file
    with open(yaml_files[0], 'r') as file:
        content = yaml.safe_load(file)

    return content