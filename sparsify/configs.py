"""Config loading and saving.

Takes care of configs for training MLPs on MNIST and then training the
Modified Models.

"""

from pathlib import Path
from typing import Any, Literal, Optional

import torch
import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class WandbConfig(BaseModel):
    project: str
    entity: str


StrDtype = Literal["float32", "float64", "bfloat16"]
TORCH_DTYPES: dict[StrDtype, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


class HookedTransformerPreConfig(BaseModel):
    """Pydantic model whose arguments will be passed to a HookedTransformerConfig."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)
    d_model: int
    n_layers: int
    n_ctx: int
    d_head: int
    d_vocab: int
    act_fn: str
    dtype: Optional[torch.dtype]

    @field_validator("dtype", mode="before")
    @classmethod
    def dtype_to_torch_dtype(cls, v: Optional[StrDtype]) -> Optional[torch.dtype]:
        if v is None:
            return None
        return TORCH_DTYPES[v]


class TrainConfig(BaseModel):
    seed: int = 0
    num_epochs: int
    batch_size: int
    effective_batch_size: Optional[int] = None
    lr: float
    scheduler: Optional[str] = None
    warmup_steps: int = 0
    max_grad_norm: Optional[float] = None
    act_sparsity_lambda: Optional[float] = 0.0
    w_sparsity_lambda: Optional[float] = 0.0
    sparsity_p_norm: float = 1.0
    loss_include_sae_inp_orig: bool = True
    loss_include_sae_out_orig: bool = True
    loss_include_sae_inp_sae_out: bool = True
    loss_include_sae_sparsity: bool = True


class SparsifiersConfig(BaseModel):
    type_of_sparsifier: Optional[str] = "sae"
    dict_size_to_input_ratio: float = 1.0
    k: Optional[int] = None
    sae_position_name: str  # TODO will become List[str]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tlens_model_name: Optional[str] = None
    tlens_config: Optional[HookedTransformerPreConfig] = None
    train: TrainConfig
    saes: SparsifiersConfig
    wandb: Optional[WandbConfig] = None

    @model_validator(mode="before")
    @classmethod
    def check_only_one_model_definition(cls, values: dict[str, Any]) -> dict[str, Any]:
        assert (values.get("tlens_model_name") is not None) + (
            values.get("tlens_config") is not None
        ) == 1, "Must specify exactly one of tlens_model_name or tlens_config."
        return values
    
    @model_validator(mode="before")
    @classmethod
    def check_effective_batch_size(cls, values: dict[str, Any]) -> dict[str, Any]:
        assert values["train"]["effective_batch_size"] % values["train"]["batch_size"] == 0, "effective_batch_size must be a multiple of batch_size."
        return values


def load_config(config_path: Path) -> Config:
    """Load the config from a YAML file into a Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)
    return config
