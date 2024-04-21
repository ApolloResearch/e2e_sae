from typing import Any

import yaml

from sparsify.scripts.train_tlens_saes.run_train_tlens_saes import Config
from sparsify.settings import REPO_ROOT
from sparsify.utils import replace_pydantic_model

TINYSTORIES_CONFIG = f"{REPO_ROOT}/sparsify/scripts/train_tlens_saes/tinystories_1M_e2e.yaml"


def get_tinystories_config(*updates: dict[str, Any]) -> Config:
    """Load the tinystories config and update it with the given updates."""
    # Set the wandb_project to null since we never want to log tests to wandb
    updates = updates + ({"wandb_project": None},)
    with open(TINYSTORIES_CONFIG) as f:
        config = Config(**yaml.safe_load(f))
    return replace_pydantic_model(config, *updates)
