from typing import Literal

import torch

StrDtype = Literal["float32", "float64", "bfloat16"]
TORCH_DTYPES: dict[StrDtype, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}
