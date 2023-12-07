"""Train a model on MNIST.

This script takes ~40 seconds to run for 3 layers and 15 epochs on a CPU.

Usage:
    python scripts/train_mnist.py <path/to/config.yaml>
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import fire
import torch
import wandb
import yaml
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from mypkg.log import logger
from mypkg.models import MLP
from mypkg.utils import save_model


class ModelConfig(BaseModel):
    hidden_sizes: Optional[List[int]]


class TrainConfig(BaseModel):
    learning_rate: float
    batch_size: int
    epochs: int
    save_dir: Optional[Path]
    save_every_n_epochs: Optional[int]


class WandbConfig(BaseModel):
    project: str
    entity: str


class Config(BaseModel):
    seed: int
    model: ModelConfig
    train: TrainConfig
    wandb: Optional[WandbConfig]


def load_config(config_path: Path) -> Config:
    """Load the config from a YAML file into a Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)
    return config


def train(config: Config) -> None:
    """Train the MLP on MNIST.

    If config.wandb is not None, log the results to Weights & Biases.
    """
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    if not config.train.save_dir:
        config.train.save_dir = Path(__file__).parent.parent / ".checkpoints" / "mnist"

    # Load the MNIST dataset
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(
        root=Path(__file__).parent.parent / ".data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=config.train.batch_size, shuffle=True)

    # Initialize the MLP model
    model = MLP(config.model.hidden_sizes, input_size=784, output_size=10)
    model = model.to(device)

    # Define the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

    if config.wandb:
        run_name = f"lr-{config.train.learning_rate}_bs-{config.train.batch_size}"
        wandb.init(
            name=run_name,
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=config.model_dump(),
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = config.train.save_dir / f"{run_name}_{timestamp}"

    samples = 0
    # Training loop
    for epoch in tqdm(range(config.train.epochs), total=config.train.epochs, desc="Epochs"):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            samples += images.shape[0]
            # Flatten the images
            images = images.view(images.shape[0], -1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                logger.info(
                    "Epoch [%d/%d], Step [%d/%d], Loss: %f",
                    epoch + 1,
                    config.train.epochs,
                    i + 1,
                    len(train_loader),
                    loss.item(),
                )

                if config.wandb:
                    wandb.log({"train/loss": loss.item(), "train/samples": samples}, step=samples)

        if config.train.save_every_n_epochs and (epoch + 1) % config.train.save_every_n_epochs == 0:
            save_model(json.loads(config.model_dump_json()), save_dir, model, epoch)

    if not (save_dir / f"model_epoch_{epoch + 1}.pt").exists():
        save_model(json.loads(config.model_dump_json()), save_dir, model, epoch)


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path)
    train(config)


if __name__ == "__main__":
    fire.Fire(main)
