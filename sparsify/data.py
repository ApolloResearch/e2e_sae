from typing import Literal

from datasets import load_dataset
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from sparsify.types import Samples


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_name: str
    is_tokenized: bool = True
    tokenizer_name: str
    streaming: bool = True
    split: Literal["train"]


def create_data_loader(data_config: DataConfig, batch_size: int) -> DataLoader[Samples]:
    """Create a DataLoader for the given dataset."""
    dataset = load_dataset(
        data_config.dataset_name,
        streaming=data_config.streaming,
        split=data_config.split,
    )
    if data_config.is_tokenized:
        torch_dataset: TorchDataset[Samples] = dataset.with_format("torch")  # type: ignore
    else:
        raise NotImplementedError("Tokenization not implemented yet.")

    # Note that a pre-tokenized dataset was shuffled when generated:
    # https://github.com/ai-safety-foundation/sparse_autoencoder/blob/main/sparse_autoencoder/source_data/abstract_dataset.py#L209
    loader = DataLoader[Samples](
        torch_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return loader
