from dataclasses import dataclass
import torch
from typing import Optional
import fire
from pathlib import Path
import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from sparsify.configs import load_config, Config
from sparsify.models import HookedTransformer
from transformer_lens import utils, evals

from torchvision import datasets, transforms
from datasets.load import load_dataset


def train(
    model: HookedTransformer,
    config: Config,
    dataloader: DataLoader,
) -> HookedTransformer:
    """
    Trains an HookedTransformer model on an autoregressive language modeling task.
    Args:
        model: The model to train
        config: The training configuration
        dataset: The dataset to train on - this function assumes the dataset is set up for autoregressive language modeling.
    Returns:
        The trained model
    """
    torch.manual_seed(config.seed)
    model.train()
    if config.train.wandb:
        if config.train.wandb_project_name is None:
            config.train.wandb_project_name = "easy-transformer"
        wandb.init(project=config.train.wandb_project_name, config=vars(config))

    if config.train.device is None:
        config.train.device = utils.get_device()

    if config.train.optimizer_name in ["Adam", "AdamW"]:
        # Weight decay in Adam is implemented badly, so use AdamW instead (see PyTorch AdamW docs)
        if config.train.weight_decay is not None:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.train.lr,
                weight_decay=config.train.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.train.lr,
            )
    elif config.train.optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay
            if config.train.weight_decay is not None
            else 0.0,
            momentum=config.train.momentum,
        )
    else:
        raise ValueError(f"Optimizer {config.train.optimizer_name} not supported")

    scheduler = None
    if config.train.warmup_steps > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / config.train.warmup_steps),
        )

    model.to(config.train.device)

    for epoch in tqdm(range(1, config.train.num_epochs + 1)):
        samples = 0
        for step, batch in tqdm(enumerate(dataloader)):
            tokens = batch["tokens"].to(config.train.device)
            loss = model(tokens, return_type="loss")
            loss.backward()
            if config.train.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            if config.train.warmup_steps > 0:
                assert scheduler is not None
                scheduler.step()
            optimizer.zero_grad()

            samples += tokens.shape[0]

            if config.train.wandb:
                wandb.log(
                    {"train_loss": loss.item(), "samples": samples, "epoch": epoch}
                )

            if config.train.print_every is not None and step % config.train.print_every == 0:
                print(f"Epoch {epoch} Samples {samples} Step {step} Loss {loss.item()}")

            if (
                config.train.save_every is not None
                and step % config.train.save_every == 0
                and config.train.save_dir is not None
            ):
                torch.save(model.state_dict(), f"{config.train.save_dir}/model_{step}.pt")

            if config.train.max_steps is not None and step >= config.train.max_steps:
                break

    return model


def get_model(config: Config) -> HookedTransformer:
    """
    Gets a model from a config unless a pretrained model is specified.
    Args:
        config: The config to get the model from
    Returns:
        The model
    """
    if config.pretrained_model_type is not None:
        model = HookedTransformer.from_pretrained(
            config.pretrained_model_type
        )
    else:
        model = HookedTransformer(cfg=config.model)
    
    return model

def get_data(config: Config, tokenizer):
    if config.data.dataset == "pile-10k":
        train_loader = evals.make_pile_data_loader(tokenizer, batch_size=config.train.batch_size)
    else:
        raise ValueError(f"Dataset {config.data.dataset} not supported")
    return train_loader

def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path)
    model = get_model(config)
    dataloader = get_data(config, model.tokenizer)

    train(model, config, dataloader)


if __name__ == "__main__":
    fire.Fire(main)
