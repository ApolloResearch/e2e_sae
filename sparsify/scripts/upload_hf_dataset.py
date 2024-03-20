"""
Taken and adapated from Alan Cooney's
https://github.com/ai-safety-foundation/sparse_autoencoder/tree/main/sparse_autoencoder.
"""

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

from datasets import (
    Dataset,
    DatasetDict,
    VerificationMode,
    load_dataset,
)
from huggingface_hub import HfApi
from jaxtyping import Int
from pydantic import PositiveInt, validate_call
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class GenericTextDataBatch(TypedDict):
    """Generic Text Dataset Batch.

    Assumes the dataset provides a 'text' field with a list of strings.
    """

    text: list[str]
    meta: list[dict[str, dict[str, str]]]  # Optional, depending on the dataset structure.


TokenizedPrompt = list[int]
"""A tokenized prompt."""


class TokenizedPrompts(TypedDict):
    """Tokenized prompts."""

    input_ids: list[TokenizedPrompt]


class TorchTokenizedPrompts(TypedDict):
    """Tokenized prompts prepared for PyTorch."""

    input_ids: Int[Tensor, "batch pos vocab"]


class TextDataset:
    """Generic Text Dataset for any text-based dataset from Hugging Face."""

    tokenizer: PreTrainedTokenizerBase

    def preprocess(
        self,
        source_batch: GenericTextDataBatch,
        *,
        context_size: int,
    ) -> TokenizedPrompts:
        """Preprocess a batch of prompts.

        Tokenizes a batch of text data and packs into context_size samples. An eos token is added
        to the end of each document after tokenization.

        Args:
            source_batch: A batch of source data, including 'text' with a list of strings.
            context_size: Context size for tokenized prompts.

        Returns:
            Tokenized prompts.
        """
        prompts: list[str] = source_batch["text"]

        tokenized_prompts = self.tokenizer(prompts, truncation=False, padding=False)

        all_tokens = []
        for document_tokens in tokenized_prompts[self._dataset_column_name]:  # type: ignore
            all_tokens.extend(document_tokens + [self.tokenizer.eos_token_id])
        # Ignore incomplete chunks
        chunks = [
            all_tokens[i : i + context_size]
            for i in range(0, len(all_tokens), context_size)
            if len(all_tokens[i : i + context_size]) == context_size
        ]

        return {"input_ids": chunks}

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizerBase,
        context_size: PositiveInt = 256,
        load_revision: str = "main",
        dataset_dir: str | None = None,
        dataset_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None,
        dataset_split: str | None = None,
        dataset_column_name: str = "input_ids",
        n_processes_preprocessing: PositiveInt | None = None,
        preprocess_batch_size: PositiveInt = 1000,
    ):
        """Initialize a generic text dataset from Hugging Face.

        Args:
            dataset_path: Path to the dataset on Hugging Face (e.g. `'monology/pile-uncopyright'`).
            tokenizer: Tokenizer to process text data.
            context_size: The context size to use when returning a list of tokenized prompts.
                *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* used
                a context size of 250.
            load_revision: The commit hash or branch name to download from the source dataset.
            dataset_dir: Defining the `data_dir` of the dataset configuration.
            dataset_files: Path(s) to source data file(s).
            dataset_split: Dataset split (e.g., 'train'). If None, process all splits.
            dataset_column_name: The column name for the prompts.
            n_processes_preprocessing: Number of processes to use for preprocessing.
            preprocess_batch_size: Batch size for preprocessing (tokenizing prompts).
        """
        self.tokenizer = tokenizer

        self.context_size = context_size
        self._dataset_column_name = dataset_column_name

        # Load the dataset
        dataset = load_dataset(
            dataset_path,
            revision=load_revision,
            streaming=False,  # We need to pre-download the dataset to upload it to the hub.
            split=dataset_split,
            data_dir=dataset_dir,
            data_files=dataset_files,
            verification_mode=VerificationMode.NO_CHECKS,  # As it fails when data_files is set
        )
        # If split is not None, will return a Dataset instance. Convert to DatasetDict.
        if isinstance(dataset, Dataset):
            assert dataset_split is not None
            dataset = DatasetDict({dataset_split: dataset})
        assert isinstance(dataset, DatasetDict)

        for split in dataset:
            print(f"Processing split: {split}")
            # Setup preprocessing (we remove all columns except for input ids)
            remove_columns: list[str] = list(next(iter(dataset[split])).keys())  # type: ignore
            if "input_ids" in remove_columns:
                remove_columns.remove("input_ids")

            # Tokenize and chunk the prompts
            mapped_dataset = dataset[split].map(
                self.preprocess,
                batched=True,
                batch_size=preprocess_batch_size,
                fn_kwargs={"context_size": context_size},
                remove_columns=remove_columns,
                num_proc=n_processes_preprocessing,
            )
            dataset[split] = mapped_dataset.shuffle()

        self.dataset = dataset

    @validate_call
    def push_to_hugging_face_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload preprocessed dataset using sparse_autoencoder.",
        max_shard_size: str = "500MB",
        revision: str = "main",
        *,
        private: bool = False,
    ) -> None:
        """Share preprocessed dataset to Hugging Face hub.

        Motivation:
            Pre-processing a dataset can be time-consuming, so it is useful to be able to share the
            pre-processed dataset with others. This function allows you to do that by pushing the
            pre-processed dataset to the Hugging Face hub.

        Warning:
            You must be logged into HuggingFace (e.g with `huggingface-cli login` from the terminal)
            to use this.

        Warning:
            This will only work if the dataset is not streamed (i.e. if `pre_download=True` when
            initializing the dataset).

        Args:
            repo_id: Hugging Face repo ID to save the dataset to (e.g. `username/dataset_name`).
            commit_message: Commit message.
            max_shard_size: Maximum shard size (e.g. `'500MB'`).
            revision: Branch to push to.
            private: Whether to save the dataset privately.
        """
        self.dataset.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
            max_shard_size=max_shard_size,
            private=private,
            revision=revision,
        )


@dataclass
class DatasetToPreprocess:
    """Dataset to preprocess info."""

    source_path: str
    """Source path from HF (e.g. `roneneldan/TinyStories`)."""

    tokenizer_name: str
    """HF tokenizer name (e.g. `gpt2`)."""

    load_revision: str = "main"
    """Commit hash or branch name to download from the source dataset."""

    data_dir: str | None = None
    """Data directory to download from the source dataset."""

    data_files: list[str] | None = None
    """Data files to download from the source dataset."""

    hugging_face_username: str = "apollo-research"
    """HF username for the upload."""

    private: bool = False
    """Whether the HF dataset should be private or public."""

    context_size: int = 2048
    """Number of tokens in a single sample. gpt2 uses 1024, pythia uses 2048."""

    split: str | None = None
    """Dataset split to download from the source dataset. If None, process all splits."""

    @property
    def source_alias(self) -> str:
        """Create a source alias for the destination dataset name.

        Returns:
            The modified source path as source alias.
        """
        return self.source_path.replace("/", "-")

    @property
    def tokenizer_alias(self) -> str:
        """Create a tokenizer alias for the destination dataset name.

        Returns:
            The modified tokenizer name as tokenizer alias.
        """
        return self.tokenizer_name.replace("/", "-")

    @property
    def destination_repo_name(self) -> str:
        """Destination repo name.

        Returns:
            The destination repo name.
        """
        split_str = f"{self.split}-" if self.split else ""
        return f"{self.source_alias}-{split_str}tokenizer-{self.tokenizer_alias}"

    @property
    def destination_repo_id(self) -> str:
        """Destination repo ID.

        Returns:
            The destination repo ID.
        """
        return f"{self.hugging_face_username}/{self.destination_repo_name}"


def upload_datasets(datasets_to_preprocess: list[DatasetToPreprocess]) -> None:
    """Upload datasets to HF.

    Warning:
        Assumes you have already created the corresponding repos on HF.

    Args:
        datasets_to_preprocess: List of datasets to preprocess.

    Raises:
        ValueError: If the repo doesn't exist.
    """
    repositories_updating = [dataset.destination_repo_id for dataset in datasets_to_preprocess]
    print("Updating repositories:\n" "\n".join(repositories_updating))

    for dataset in datasets_to_preprocess:
        print("Processing dataset: ", dataset.source_path)

        # Preprocess
        tokenizer = AutoTokenizer.from_pretrained(dataset.tokenizer_name)
        text_dataset = TextDataset(
            dataset_path=dataset.source_path,
            tokenizer=tokenizer,
            dataset_files=dataset.data_files,
            dataset_dir=dataset.data_dir,
            dataset_split=dataset.split,
            context_size=dataset.context_size,
            load_revision=dataset.load_revision,
        )
        # size_in_bytes and info gives info about the whole dataset regardless of the split index,
        # so we just get the first split.
        split = next(iter(text_dataset.dataset))
        print("Dataset info:")
        print(f"Size: {text_dataset.dataset[split].size_in_bytes / 1e9:.2f} GB")  # type: ignore
        print("Info: ", text_dataset.dataset[split].info)

        # Upload
        text_dataset.push_to_hugging_face_hub(
            repo_id=dataset.destination_repo_id, private=dataset.private
        )
        # Also upload the current file to the repo for reproducibility and transparency
        api = HfApi()
        api.upload_file(
            path_or_fileobj=__file__,
            path_in_repo="upload_script.py",
            repo_id=dataset.destination_repo_id,
            repo_type="dataset",
            commit_message="Add upload script",
        )


if __name__ == "__main__":
    # Check that the user is signed in to huggingface-cli
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"], check=True, capture_output=True, text=True
        )
        if "Not logged in" in result.stdout:
            print("Please sign in to huggingface-cli using `huggingface-cli login`.")
            raise Exception("You are not logged in to huggingface-cli.")
    except subprocess.CalledProcessError:
        print("An error occurred while checking the login status.")
        raise

    datasets: list[DatasetToPreprocess] = [
        DatasetToPreprocess(
            source_path="roneneldan/TinyStories",
            # Paper says gpt-neo tokenizer, and e.g. EleutherAI/gpt-neo-125M uses the same tokenizer
            # as gpt2. They also suggest using gpt2 in (https://github.com/EleutherAI/gpt-neo).
            tokenizer_name="gpt2",
            hugging_face_username="apollo-research",
            context_size=512,
        ),
        DatasetToPreprocess(
            source_path="Skylion007/openwebtext",
            tokenizer_name="gpt2",
            hugging_face_username="apollo-research",
            context_size=1024,
        ),
        DatasetToPreprocess(
            source_path="Skylion007/openwebtext",
            tokenizer_name="EleutherAI/gpt-neox-20b",
            hugging_face_username="apollo-research",
            context_size=2048,
        ),
        DatasetToPreprocess(
            source_path="monology/pile-uncopyrighted",
            tokenizer_name="gpt2",
            hugging_face_username="apollo-research",
            context_size=1024,
            # Get the first few (each file is 11GB so this should be enough for a large dataset)
            data_files=[
                "train/00.jsonl.zst",
                "train/01.jsonl.zst",
                "train/02.jsonl.zst",
                "train/03.jsonl.zst",
                "train/04.jsonl.zst",
            ],
        ),
        DatasetToPreprocess(
            source_path="monology/pile-uncopyrighted",
            tokenizer_name="EleutherAI/gpt-neox-20b",
            hugging_face_username="apollo-research",
            private=False,
            context_size=2048,
            data_files=[
                "train/00.jsonl.zst",
                "train/01.jsonl.zst",
                "train/02.jsonl.zst",
                "train/03.jsonl.zst",
                "train/04.jsonl.zst",
            ],
        ),
    ]

    upload_datasets(datasets)
