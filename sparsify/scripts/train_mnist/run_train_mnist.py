"""Train a model on MNIST.

This script takes ~40 seconds to run for 3 layers and 15 epochs on a CPU.

Usage:
    python run_train_mnist.py <path/to/config.yaml>
"""

from datetime import datetime
from pathlib import Path

import fire
import torch
import wandb
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from sparsify.log import logger
from sparsify.models.mlp import MLP
from sparsify.settings import REPO_ROOT
from sparsify.utils import load_config, save_model


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hidden_sizes: list[int] | None


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    learning_rate: float
    batch_size: int
    epochs: int
    save_dir: Path | None
    save_every_n_epochs: int | None


class WandbConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    project: str
    entity: str


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int
    model: ModelConfig
    train: TrainConfig
    wandb: WandbConfig | None


def train(config: Config) -> None:
    """Train the MLP on MNIST.

    If config.wandb is not None, log the results to Weights & Biases.
    """
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    if not config.train.save_dir:
        config.train.save_dir = REPO_ROOT / ".checkpoints" / "mnist"

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

    # Define the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

    run_name = (
        f"orig-train-lr-{config.train.learning_rate}_bs-{config.train.batch_size}"
        f"-{str(config.model.hidden_sizes)}"
    )
    if config.wandb:
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

            # Calculate accuracy
            _, argmax = torch.max(outputs, 1)
            accuracy = (labels == argmax.squeeze()).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % config.train.epochs // 10 == 0:
                logger.info(
                    "Epoch [%d/%d], Step [%d/%d], Loss: %f, Accuracy: %f",
                    epoch + 1,
                    config.train.epochs,
                    i + 1,
                    len(train_loader),
                    loss.item(),
                    accuracy.item(),
                )

                if config.wandb:
                    wandb.log({"train/loss": loss.item(), "train/samples": samples}, step=samples)

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

            if config.wandb:
                wandb.log({"valid/accuracy": accuracy}, step=samples)

        if config.train.save_every_n_epochs and (epoch + 1) % config.train.save_every_n_epochs == 0:
            save_model(
                config_dict=config.model_dump(),
                save_dir=save_dir,
                model=model,
                epoch=epoch,
                sparse=False,
            )

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

        if config.wandb:
            wandb.log({"test/accuracy": accuracy}, step=samples)

    if not (save_dir / f"model_epoch_{config.train.epochs - 1}.pt").exists():
        save_model(
            config_dict=config.model_dump(),
            save_dir=save_dir,
            model=model,
            epoch=config.train.epochs - 1,
            sparse=False,
        )


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    train(config)


if __name__ == "__main__":
    fire.Fire(main)
