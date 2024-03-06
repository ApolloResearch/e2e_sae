import torch
from datasets import IterableDataset, load_dataset
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformer_lens.utils import tokenize_and_concatenate
from transformers import AutoTokenizer

from sparsify.types import Samples


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_name: str
    is_tokenized: bool = True
    tokenizer_name: str
    streaming: bool = True
    split: str
    n_ctx: int
    column_name: str = "input_ids"
    """The name of the column in the dataset that contains the tokenized samples. Typically
    'input_ids' for datasets stored with sparsify/upload_hf_dataset.py, or "tokens" for datasets
    created in TransformerLens (e.g. NeelNanda/pile-10k)."""


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    train: DatasetConfig
    eval: DatasetConfig | None = None
    seed: int = 0


def create_data_loader(
    dataset_config: DatasetConfig, batch_size: int, seed: int, buffer_size: int = 1000
) -> tuple[DataLoader[Samples], AutoTokenizer]:
    """Create a DataLoader for the given dataset."""
    dataset = load_dataset(
        dataset_config.dataset_name, streaming=dataset_config.streaming, split=dataset_config.split
    )

    if dataset_config.streaming:
        assert isinstance(dataset, IterableDataset)
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    else:
        dataset = dataset.shuffle(seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(dataset_config.tokenizer_name)
    torch_dataset: TorchDataset[Samples]
    if dataset_config.is_tokenized:
        torch_dataset = dataset.with_format("torch")  # type: ignore
        # Get a sample from the dataset and check if it's tokenized and what the n_ctx is
        # Note that the dataset may be streamed, so we can't just index into it
        sample = next(iter(torch_dataset))[dataset_config.column_name]
        assert (
            isinstance(sample, torch.Tensor) and sample.ndim == 1
        ), "Expected the dataset to be tokenized."
        assert len(sample) == dataset_config.n_ctx, "n_ctx does not match the tokenized length."

    else:
        torch_dataset = tokenize_and_concatenate(
            dataset,  # type: ignore
            tokenizer,
            max_length=dataset_config.n_ctx,
            add_bos_token=True,
        )

    # Note that a pre-tokenized dataset was shuffled when generated:
    # https://github.com/ai-safety-foundation/sparse_autoencoder/blob/main/sparse_autoencoder/source_data/abstract_dataset.py#L209
    loader = DataLoader[Samples](
        torch_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return loader, tokenizer
