"""Train a model on MNIST.

This script takes ~40 seconds to run for 3 layers and 15 epochs on a CPU.

Usage:
    python run_train_mnist.py <path/to/config.yaml>
"""

import os
from datetime import datetime
from pathlib import Path

import fire
import torch
import wandb
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveFloat, PositiveInt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from e2e_sae.log import logger
from e2e_sae.models.mlp import MLP
from e2e_sae.settings import REPO_ROOT
from e2e_sae.types import RootPath
from e2e_sae.utils import load_config, save_module, set_seed


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hidden_sizes: list[PositiveInt] | None


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    learning_rate: PositiveFloat
    batch_size: PositiveInt
    n_epochs: PositiveInt
    save_dir: RootPath | None = Path(__file__).parent / "out"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: NonNegativeInt
    model: ModelConfig
    train: TrainConfig
    wandb_project: str | None  # If None, don't log to Weights & Biases


def train(config: Config) -> None:
    """Train the MLP on MNIST.

    If config.wandb is not None, log the results to Weights & Biases.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Load the MNIST dataset
    data_path = str(REPO_ROOT / ".data")
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=config.train.batch_size, shuffle=True)
    test_data = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=config.train.batch_size, shuffle=False)
    valid_data = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    valid_loader = DataLoader(valid_data, batch_size=config.train.batch_size, shuffle=False)

    # Initialize the MLP model
    model = MLP(config.model.hidden_sizes, input_size=784, output_size=10)
    model = model.to(device)
    model.train()

    # Define the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

    hidden_repr = (
        "-".join(str(x) for x in config.model.hidden_sizes) if config.model.hidden_sizes else None
    )

    run_name = (
        f"orig-train-lr-{config.train.learning_rate}_bs-{config.train.batch_size}"
        f"_hidden-{hidden_repr}"
    )
    if config.wandb_project:
        load_dotenv()
        wandb.init(
            name=run_name,
            project=config.wandb_project,
            entity=os.getenv("WANDB_ENTITY"),
            config=config.model_dump(mode="json"),
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = config.train.save_dir / f"{run_name}_{timestamp}" if config.train.save_dir else None

    samples = 0
    # Training loop
    for epoch in tqdm(
        range(1, config.train.n_epochs + 1), total=config.train.n_epochs, desc="Epochs"
    ):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            samples += images.shape[0]
            # Flatten the images
            images = images.view(images.shape[0], -1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate accuracy
            _, argmax = torch.max(outputs, 1)
            accuracy = (labels == argmax.squeeze()).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % config.train.n_epochs // 10 == 0:
                logger.info(
                    "Epoch [%d/%d], Step [%d/%d], Loss: %f, Accuracy: %f",
                    epoch,
                    config.train.n_epochs,
                    i + 1,
                    len(train_loader),
                    loss.item(),
                    accuracy.item(),
                )

                if config.wandb_project:
                    wandb.log({"train/loss": loss.item()}, step=samples)

        # Validate the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                images = images.view(images.shape[0], -1)
                outputs = model(images)
                _, argmax = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (labels == argmax.squeeze()).sum().item()

            accuracy = correct / total
            logger.info("Accuracy of the network on the 10000 test images: %f %%", 100 * accuracy)

            if config.wandb_project:
                wandb.log({"valid/accuracy": accuracy}, step=samples)
        model.train()

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.shape[0], -1)
            outputs = model(images)
            _, argmax = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (labels == argmax.squeeze()).sum().item()

        accuracy = correct / total
        logger.info("Accuracy of the network on the 10000 test images: %f %%", 100 * accuracy)

        if config.wandb_project:
            wandb.log({"test/accuracy": accuracy}, step=samples)
    model.train()

    if save_dir:
        save_module(
            config_dict=config.model_dump(mode="json"),
            save_dir=save_dir,
            module=model,
            model_filename=f"epoch_{config.train.n_epochs}.pt",
        )
    if config.wandb_project:
        wandb.finish()


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    set_seed(config.seed)
    train(config)


if __name__ == "__main__":
    fire.Fire(main)
