"""Collect and analyze max activating dataset examples.

Usage:
    python max_act_data_mnist.py <path/to/config.yaml>
"""

from datetime import datetime
from pathlib import Path

import fire
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from e2e_sae.log import logger
from e2e_sae.models.mlp import MLPMod
from e2e_sae.settings import REPO_ROOT
from e2e_sae.utils import load_config


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hidden_sizes: list[PositiveInt] | None


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    batch_size: PositiveInt
    model_name: str
    save_dir: Path | None


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: NonNegativeInt
    model: ModelConfig
    infer: InferenceConfig


def max_act(config: Config) -> None:
    """Collect and analyze max activating dataset examples."""
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Load the MNIST dataset
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(
        root=str(REPO_ROOT / ".data"), train=True, download=True, transform=transform
    )
    DataLoader(train_data, batch_size=config.infer.batch_size, shuffle=True)
    test_data = datasets.MNIST(
        root=str(REPO_ROOT / ".data"), train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=config.infer.batch_size, shuffle=False)

    # Define model path to load
    model_path = REPO_ROOT / "models" / config.infer.model_name
    model_path = max(model_path.glob("*.pt"), key=lambda x: int(x.stem.split("_")[-1]))
    model_mod = MLPMod(config.model.hidden_sizes, input_size=784, output_size=10)
    model_mod_state_dict = torch.load(model_path)
    model_mod.load_state_dict(model_mod_state_dict)
    model_mod = model_mod.to(device)

    datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Get number of dataset samples in testset
    num_dataset_samples = len(test_loader.dataset)  # type: ignore
    # Get size of each autoencoder dictionary
    sae_sizes = {}
    for key in model_mod.sparsifiers:
        sae_sizes[key] = model_mod.sparsifiers[key].n_dict_components

    # Make a dictionary for each autoencoder to store the cs for each dataset sample
    cs_dict = {}
    for key in model_mod.sparsifiers:
        cs_dict[key] = torch.zeros(num_dataset_samples, sae_sizes[key])

    # Make a dictionary to count the dead cs for each autoencoder
    dead_dict = {}
    for key in model_mod.sparsifiers:
        dead_dict[key] = torch.zeros(sae_sizes[key])

    samples = 0
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        samples += images.shape[0]
        images = images.view(images.shape[0], -1)
        mod_acts, cs, saes_outs = model_mod(images)

        # Identify the dead dictionary elements (the cs with no activations) by summing over the
        # dataset samples and counting the number of zeros for each dictionary element
        for key in cs:
            dead_dict[key] += torch.sum(cs[key], dim=0).cpu()

        # Store the cs for each dataset sample
        for key in cs:
            cs_dict[key][i * config.infer.batch_size : (i + 1) * config.infer.batch_size] = cs[key]

    # Divide the dead dictionary elements by the number of dataset samples to get the percentage of
    # dead dictionary elements
    for key in dead_dict:
        dead_dict[key] = dead_dict[key] / num_dataset_samples

    # Plot a histogram of the dead dictionary elements for each autoencoder
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, len(dead_dict.keys()))
    # for i, key in enumerate(dead_dict.keys()):
    #     axs[i].hist(dead_dict[key].detach().numpy(), bins=20)
    #     axs[i].set_title(key)
    # plt.show()
    # plt.savefig("dead_dict_elements.png")


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)  # TODO make separate config for model_mod
    config = load_config(config_path, config_model=Config)
    max_act(config)


if __name__ == "__main__":
    fire.Fire(main)
