from typing import Any

import einops
import numpy as np
import torch
from datasets import Dataset, IterableDataset, load_dataset
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from e2e_sae.types import Samples


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_name: str
    is_tokenized: bool = True
    tokenizer_name: str
    streaming: bool = True
    split: str
    n_ctx: int
    seed: int | None = None
    column_name: str = "input_ids"
    """The name of the column in the dataset that contains the data (tokenized or non-tokenized).
    Typically 'input_ids' for datasets stored with e2e_sae/scripts/upload_hf_dataset.py, or "tokens"
    for datasets tokenized in TransformerLens (e.g. NeelNanda/pile-10k)."""


def _keep_single_column(dataset: Dataset, col_name: str) -> Dataset:
    """
    Acts on a HuggingFace dataset to delete all columns apart from a single column name - useful
    when we want to tokenize and mix together different strings.
    """
    for key in dataset.features:  # pyright: ignore[reportAttributeAccessIssue]
        if key != col_name:
            dataset = dataset.remove_columns(key)
    return dataset


def tokenize_and_concatenate(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = False,
    num_proc: int = 10,
) -> Dataset:
    """Helper function to tokenizer and concatenate a dataset of text. This converts the text to
    tokens, concatenates them (separated by EOS tokens) and then reshapes them into a 2D array of
    shape (____, sequence_length), dropping the last batch. Tokenizers are much faster if
    parallelised, so we chop the string into 20, feed it into the tokenizer, in parallel with
    padding, then remove padding at the end.

    NOTE: Adapted from
    https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/utils.py#L267
    to handle IterableDataset.

    TODO: Fix typing of tokenizer

    This tokenization is useful for training language models, as it allows us to efficiently train
    on a large corpus of text of varying lengths (without, eg, a lot of truncation or padding).
    Further, for models with absolute positional encodings, this avoids privileging early tokens
    (eg, news articles often begin with CNN, and models may learn to use early positional
    encodings to predict these)

    Args:
        dataset: The dataset to tokenize, assumed to be a HuggingFace text dataset. Can be a regular
            Dataset or an IterableDataset.
        tokenizer: The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        max_length: The length of the context window of the sequence. Defaults to 1024.
        column_name: The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token: Add BOS token at the beginning of each sequence. Defaults to False as this
            is not done during training.

    Returns:
        Dataset or IterableDataset: Returns the tokenized dataset, as a dataset of tensors, with a
        single column called "input_ids".

    Note: There is a bug when inputting very small datasets (eg, <1 batch per process) where it
    just outputs nothing. I'm not super sure why
    """
    dataset = _keep_single_column(dataset, column_name)
    if tokenizer.pad_token is None:  # pyright: ignore[reportAttributeAccessIssue]
        # We add a padding token, purely to implement the tokenizer. This will be removed before
        # inputting tokens to the model, so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})  # pyright: ignore[reportAttributeAccessIssue]
    # Define the length to chop things up into - leaving space for a bos_token if required
    seq_len = max_length - 1 if add_bos_token else max_length

    def tokenize_function(
        examples: dict[str, list[str]],
    ) -> dict[
        str,
        NDArray[np.signedinteger[Any]],
    ]:
        text = examples[column_name]
        # Concatenate it all into an enormous string, separated by eos_tokens
        full_text = tokenizer.eos_token.join(text)  # pyright: ignore[reportAttributeAccessIssue]
        # Divide into 20 chunks of ~ equal length
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]
        # Tokenize the chunks in parallel. Uses no because HF map doesn't want tensors returned
        tokens = tokenizer(chunks, return_tensors="np", padding=True)["input_ids"].flatten()  # type: ignore
        # Drop padding tokens
        tokens = tokens[tokens != tokenizer.pad_token_id]  # pyright: ignore[reportAttributeAccessIssue]
        num_tokens = len(tokens)
        num_batches = num_tokens // (seq_len)
        # Drop the final tokens if not enough to make a full sequence
        tokens = tokens[: seq_len * num_batches]
        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)  # pyright: ignore[reportAttributeAccessIssue]
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"input_ids": tokens}

    if isinstance(dataset, IterableDataset):
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=[column_name]
        )
    else:
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=[column_name], num_proc=num_proc
        )

    tokenized_dataset = tokenized_dataset.with_format("torch")

    return tokenized_dataset


def create_data_loader(
    dataset_config: DatasetConfig, batch_size: int, buffer_size: int = 1000, global_seed: int = 0
) -> tuple[DataLoader[Samples], AutoTokenizer]:
    """Create a DataLoader for the given dataset.

    Args:
        dataset_config: The configuration for the dataset.
        batch_size: The batch size.
        buffer_size: The buffer size for streaming datasets.
        global_seed: Used for shuffling if dataset_config.seed is None.

    Returns:
        A tuple of the DataLoader and the tokenizer.
    """
    dataset = load_dataset(
        dataset_config.dataset_name, streaming=dataset_config.streaming, split=dataset_config.split
    )
    seed = dataset_config.seed if dataset_config.seed is not None else global_seed
    if dataset_config.streaming:
        assert isinstance(dataset, IterableDataset)
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    else:
        dataset = dataset.shuffle(seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(dataset_config.tokenizer_name)
    torch_dataset: Dataset
    if dataset_config.is_tokenized:
        torch_dataset = dataset.with_format("torch")  # type: ignore
        # Get a sample from the dataset and check if it's tokenized and what the n_ctx is
        # Note that the dataset may be streamed, so we can't just index into it
        sample = next(iter(torch_dataset))[dataset_config.column_name]  # type: ignore
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

    # Note that a pre-tokenized dataset was shuffled when generated
    # see e2e_sae.scripts.upload_hf_dataset.TextDataset.__init__
    loader = DataLoader[Samples](
        torch_dataset,  # type: ignore
        batch_size=batch_size,
        shuffle=False,
    )
    return loader, tokenizer
