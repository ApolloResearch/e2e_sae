from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import BeforeValidator, PlainSerializer

from sparsify.utils import to_root_path

StrDtype = Literal["float32", "float64", "bfloat16"]
TORCH_DTYPES: dict[StrDtype, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}

# This is a type for pydantic configs that will convert all relative paths
# to be relative to the ROOT_DIR of sparsify
RootPath = Annotated[Path, BeforeValidator(to_root_path), PlainSerializer(lambda x: str(x))]
