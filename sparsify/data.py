from transformer_lens import evals
from typing import Any
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformer_lens import utils


def get_train_loader(tokenizer, config):
    if config.train.dataset == "pile10k":
        train_loader = evals.make_pile_data_loader(
            tokenizer, batch_size=config.train.batch_size
        )
    elif config.train.dataset == "tinystories":
        tinystories_data = load_dataset("roneneldan/TinyStories", split="train")
        print(len(tinystories_data))
        dataset = utils.tokenize_and_concatenate(tinystories_data, tokenizer)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    else:
        raise ValueError(f"Unknown dataset {config.train.dataset}")
    return train_loader
