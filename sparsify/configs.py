"""Config loading and saving.

Takes care of configs for training MLPs on MNIST and then training the 
Modified Models. 

"""
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    hidden_sizes: Optional[List[int]]


class TrainConfig(BaseModel):
    learning_rate: float
    batch_size: int
    epochs: int
    save_dir: Optional[Path]
    model_name: str
    type_of_sparsifier: str
    sparsity_lambda: float
    dict_eles_to_input_ratio: float
    sparsifier_inp_out_recon_loss_scale: float
    k: int
    save_every_n_epochs: Optional[int]


class WandbConfig(BaseModel):
    project: str
    entity: str


class Config(BaseModel):
    seed: int
    model: ModelConfig
    train: TrainConfig
    wandb: Optional[WandbConfig]


def load_config(config_path: Path) -> Config:
    """Load the config from a YAML file into a Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)
    return config