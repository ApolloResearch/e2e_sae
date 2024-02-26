"""
Taken and adapated from Alan Cooney's
https://github.com/ai-safety-foundation/sparse_autoencoder/tree/main/sparse_autoencoder.
"""

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict, final

from datasets import Dataset, IterableDataset, VerificationMode, load_dataset
from jaxtyping import Int
from pydantic import PositiveInt, validate_call
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
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
        buffer_size: PositiveInt = 1000,
        context_size: PositiveInt = 256,
        load_revision: str = "main",
        dataset_dir: str | None = None,
        dataset_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None,
        dataset_split: str = "train",
        dataset_column_name: str = "input_ids",
        n_processes_preprocessing: PositiveInt | None = None,
        preprocess_batch_size: PositiveInt = 1000,
        *,
        pre_download: bool = False,
    ):
        """Initialize a generic text dataset from Hugging Face.

        Args:
            dataset_path: Path to the dataset on Hugging Face (e.g. `'monology/pile-uncopyright'`).
            tokenizer: Tokenizer to process text data.
            buffer_size: The buffer size to use when shuffling the dataset when streaming. When
                streaming a dataset, this just pre-downloads at least `buffer_size` items and then
                shuffles just that buffer. Note that the generated activations should also be
                shuffled before training the sparse autoencoder, so a large buffer may not be
                strictly necessary here. Note also that this is the number of items in the dataset
                (e.g. number of prompts) and is typically significantly less than the number of
                tokenized prompts once the preprocessing function has been applied.
            context_size: The context size to use when returning a list of tokenized prompts.
                *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* used
                a context size of 250.
            load_revision: The commit hash or branch name to download from the source dataset.
            dataset_dir: Defining the `data_dir` of the dataset configuration.
            dataset_files: Path(s) to source data file(s).
            dataset_split: Dataset split (e.g., 'train').
            dataset_column_name: The column name for the prompts.
            n_processes_preprocessing: Number of processes to use for preprocessing.
            preprocess_batch_size: Batch size for preprocessing (tokenizing prompts).
            pre_download: Whether to pre-download the whole dataset.
        """
        self.tokenizer = tokenizer

        self.context_size = context_size
        self._dataset_column_name = dataset_column_name

        # Load the dataset
        should_stream = not pre_download
        dataset = load_dataset(
            dataset_path,
            revision=load_revision,
            streaming=should_stream,
            split=dataset_split,
            data_dir=dataset_dir,
            data_files=dataset_files,
            verification_mode=VerificationMode.NO_CHECKS,  # As it fails when data_files is set
        )

        # Setup preprocessing (we remove all columns except for input ids)
        remove_columns: list[str] = list(next(iter(dataset)).keys())
        if "input_ids" in remove_columns:
            remove_columns.remove("input_ids")

        if pre_download:
            if not isinstance(dataset, Dataset):
                error_message = (
                    f"Expected Hugging Face dataset to be a Dataset when pre-downloading, but got "
                    f"{type(dataset)}."
                )
                raise TypeError(error_message)

            # Download the whole dataset
            mapped_dataset = dataset.map(
                self.preprocess,
                batched=True,
                batch_size=preprocess_batch_size,
                fn_kwargs={"context_size": context_size},
                remove_columns=remove_columns,
                num_proc=n_processes_preprocessing,
            )
            self.dataset = mapped_dataset.shuffle()
        else:
            # Setup approximate shuffling. As the dataset is streamed, this just pre-downloads at
            # least `buffer_size` items and then shuffles just that buffer.
            # https://huggingface.co/docs/datasets/v2.14.5/stream#shuffle
            if not isinstance(dataset, IterableDataset):
                error_message = (
                    f"Expected Hugging Face dataset to be an IterableDataset when streaming, but "
                    f"got {type(dataset)}."
                )
                raise TypeError(error_message)

            mapped_dataset = dataset.map(
                self.preprocess,
                batched=True,
                batch_size=preprocess_batch_size,
                fn_kwargs={"context_size": context_size},
                remove_columns=remove_columns,
            )
            self.dataset = mapped_dataset.shuffle(buffer_size=buffer_size)  # type: ignore

    @final
    def __iter__(self) -> Any:  # noqa: ANN401
        """Iterate Dunder Method.

        Enables direct access to :attr:`dataset` with e.g. `for` loops.
        """
        return self.dataset.__iter__()

    @final
    def get_dataloader(
        self, batch_size: int, num_workers: int = 0
    ) -> DataLoader[TorchTokenizedPrompts]:
        """Get a PyTorch DataLoader.

        Args:
            batch_size: The batch size to use.
            num_workers: Number of CPU workers.

        Returns:
            PyTorch DataLoader.
        """
        torch_dataset: TorchDataset[TorchTokenizedPrompts] = self.dataset.with_format("torch")  # type: ignore

        return DataLoader[TorchTokenizedPrompts](
            torch_dataset,
            batch_size=batch_size,
            # Shuffle is most efficiently done with the `shuffle` method on the dataset itself, not
            # here.
            shuffle=False,
            num_workers=num_workers,
        )

    @validate_call
    def push_to_hugging_face_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload preprocessed dataset using sparse_autoencoder.",
        max_shard_size: str | None = None,
        n_shards: PositiveInt = 64,
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
            max_shard_size: Maximum shard size (e.g. `'500MB'`). Should not be set if `n_shards`
                is set.
            n_shards: Number of shards to split the dataset into. A high number is recommended
                here to allow for flexible distributed training of SAEs across nodes (where e.g.
                each node fetches its own shard).
            revision: Branch to push to.
            private: Whether to save the dataset privately.

        Raises:
            TypeError: If the dataset is streamed.
        """
        if isinstance(self.dataset, IterableDataset):
            error_message = (
                "Cannot share a streamed dataset to Hugging Face. "
                "Please use `pre_download=True` when initializing the dataset."
            )
            raise TypeError(error_message)

        self.dataset.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
            max_shard_size=max_shard_size,
            num_shards=n_shards,
            private=private,
            revision=revision,
        )


@dataclass
class DatasetToPreprocess:
    """Dataset to preprocess info."""

    source_path: str
    """Source path from HF (e.g. `skeskinen/TinyStories-hf`)."""

    tokenizer_name: str
    """HF tokenizer name (e.g. `gpt2`)."""

    load_revision: str = "main"
    """Commit hash or branch name to download from the source dataset."""

    data_dir: str | None = None
    """Data directory to download from the source dataset."""

    data_files: list[str] | None = None
    """Data files to download from the source dataset."""

    hugging_face_username: str = "alancooney"
    """HF username for the upload."""

    private: bool = False
    """Whether the HF dataset should be private or public."""

    context_size: int = 2048
    """Number of tokens in a single sample. gpt2 uses 1024, pythia uses 2048."""

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
        return f"sae-{self.source_alias}-tokenizer-{self.tokenizer_alias}"

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
            pre_download=True,  # Must be true to upload after pre-processing, to the hub.
            dataset_files=dataset.data_files,
            dataset_dir=dataset.data_dir,
            context_size=dataset.context_size,
            load_revision=dataset.load_revision,
        )
        print("Size: ", text_dataset.dataset.size_in_bytes)
        print("Info: ", text_dataset.dataset.info)

        # Upload
        text_dataset.push_to_hugging_face_hub(
            repo_id=dataset.destination_repo_id, private=dataset.private
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
            # Note that roneneldan/TinyStories has dataset loading issues, so we use skeskinen's
            # which fixes the issue (and explains the issue in the README.md of the repo)
            source_path="skeskinen/TinyStories-hf",
            load_revision="5e877826c63d00ec32d0a93e1110cd764402e9b9",
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
            # Get just the first few (each file is 11GB so this should be enough for a large dataset)
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
