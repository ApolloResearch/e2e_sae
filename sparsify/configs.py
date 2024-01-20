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
    - TrainBaseModelConfig
    - TrainSparsifiedModelConfig
    - TrainTranscodersConfig
    - TrainMetaSAEsConfig
    - TrainLayerwiseSAEsConfig
- WandbConfig
- Config

"""
from pathlib import Path
from typing import List, Optional
import yaml
import pydantic.BaseModel as PydanticBaseModel 
# Atypical import: BaseModel is a pydantic class whereas 'BaseModel' in 
# e.g. 'BaseModelMLPConfig' is a neural network. They're unrelated.

class ModelConfig(PydanticBaseModel):
    pass

class BaseModelConfig(ModelConfig):
    pass

class BaseModelMLPConfig(BaseModelConfig):
    hidden_sizes: Optional[List[int]]
    model_name: str

class BaseModelTransformerConfig(BaseModelConfig):
    raise NotImplementedError

class SparsifierConfig(PydanticBaseModel):
    type_of_sparsifier: str
    dict_eles_to_input_ratio: float
    k: int

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
    save_every_n_epochs: Optional[int]

class TrainBaseModelConfig(TrainConfig):
    save_dir: Optional[Path]
    base_model: BaseModelConfig
    # Consider adding training run name

class TrainSparsifiedModelConfig(TrainConfig):
    save_dir: Optional[Path]
    feat_sparsity_lambda: float
    weight_sparsity_lambda: float
    sparsifier_inp_out_recon_loss_scale: float
    sparsifiers: SparsifierConfig
    base_model: BaseModelConfig

class TrainTranscodersConfig(TrainConfig):
    save_dir: Optional[Path]
    feat_sparsity_lambda: float
    weight_sparsity_lambda: float
    transcoders: SparsifierConfig
    sparsifiers: SparsifierConfig
    base_model: BaseModelConfig

class TrainMetaSAEsConfig(TrainConfig):
    save_dir: Optional[Path]
    feat_sparsity_lambda: float
    weight_sparsity_lambda: float
    meta_saes: SparsifierConfig
    skeleton: NeuralNetworkSkeletonConfig

class TrainLayerwiseSAEsConfig(TrainConfig):
    save_dir: Optional[Path]
    feat_sparsity_lambda: float
    weight_sparsity_lambda: float
    sparsifiers: SparsifierConfig
    base_model: BaseModelConfig

class WandbConfig(PydanticBaseModel):
    project: str
    entity: str

class Config(PydanticBaseModel):
    seed: int
    model: Optional[BaseModelConfig]
    train: Optional[TrainConfig]
    wandb: Optional[WandbConfig]


def load_config(config_path: Path) -> Config:
    """Load the config from a YAML file into a Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)
    return config
