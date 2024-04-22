"""Train SAEs on an MNIST model.

NOTE: To run this, you must first train an MLP model on MNIST. You can do this with the
`e2e_sae/scripts/train_mnist/run_train_mnist.py` script.

Usage:
    python train_mnist_saes.py <path/to/config.yaml>
"""

import os
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import fire
import torch
import wandb
import yaml
from dotenv import load_dotenv
from jaxtyping import Float
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from e2e_sae.log import logger
from e2e_sae.models.mlp import MLP, MLPMod
from e2e_sae.settings import REPO_ROOT
from e2e_sae.types import RootPath
from e2e_sae.utils import load_config, save_module, set_seed


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    learning_rate: float
    batch_size: PositiveInt
    n_epochs: PositiveInt
    save_dir: RootPath | None = Path(__file__).parent / "out"
    type_of_sparsifier: str
    sparsity_lambda: NonNegativeFloat
    dict_eles_to_input_ratio: PositiveFloat
    sparsifier_in_out_recon_loss_scale: NonNegativeFloat
    k: PositiveInt


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: NonNegativeInt
    saved_model_dir: RootPath
    train: TrainConfig
    wandb_project: str | None  # If None, don't log to Weights & Biases


def get_activation(
    name: str, activations: OrderedDict[str, torch.Tensor]
) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
    """function to be called when the forward pass reaches a layer"""

    def hook(model: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations[name] = output.detach()

    return hook


def load_data(config: Config) -> tuple[DataLoader[datasets.MNIST], DataLoader[datasets.MNIST]]:
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(
        root=str(REPO_ROOT / ".data"), train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=config.train.batch_size, shuffle=True)
    test_data = datasets.MNIST(
        root=str(REPO_ROOT / ".data"), train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=config.train.batch_size, shuffle=False)
    return train_loader, test_loader


def get_models(
    config: Config, device: str | torch.device
) -> tuple[MLP, MLPMod, OrderedDict[str, torch.Tensor]]:
    # Load the hidden_sizes form the trained model
    with open(config.saved_model_dir / "final_config.yaml") as f:
        hidden_sizes = yaml.safe_load(f)["model"]["hidden_sizes"]

    latest_model_path = max(
        config.saved_model_dir.glob("*.pt"), key=lambda x: int(x.stem.split("_")[-1])
    )
    # Initialize the MLP model
    model = MLP(hidden_sizes, input_size=784, output_size=10)
    model = model.to(device)
    model_trained_statedict = torch.load(latest_model_path)
    model.load_state_dict(model_trained_statedict)
    model.eval()

    # Add hooks to the model so we can get all intermediate activations
    activations = OrderedDict()
    for name, layer in model.layers.named_children():
        layer.register_forward_hook(get_activation(name, activations))

    # Get the SAEs from the model_mod and put them in the statedict of model_trained
    model_mod = MLPMod(
        hidden_sizes=hidden_sizes,
        input_size=784,
        output_size=10,
        type_of_sparsifier=config.train.type_of_sparsifier,
        k=config.train.k,
        dict_eles_to_input_ratio=config.train.dict_eles_to_input_ratio,
    )
    for k, v in model_mod.state_dict().items():
        if k.startswith("sparsifiers"):
            model_trained_statedict[k] = v

    model_mod.load_state_dict(model_trained_statedict)
    model_mod = model_mod.to(device)
    return model, model_mod, activations


def train(config: Config) -> None:
    """Train the MLP on MNIST.

    If config.wandb is not None, log the results to Weights & Biases.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Load the MNIST dataset
    train_loader, test_loader = load_data(config)

    # Initialize the MLP model and modified model
    model, model_mod, activations = get_models(config, device)

    model_mod.sparsifiers.train()

    for param in model.layers.parameters():
        param.requires_grad = False

    # Define the loss and optimizer
    criterion = nn.MSELoss()
    # Note: only pass the SAE parameters to the optimizer
    optimizer = torch.optim.Adam(model_mod.sparsifiers.parameters(), lr=config.train.learning_rate)

    run_name = (
        f"sae_lambda-{config.train.sparsity_lambda}_lr-{config.train.learning_rate}"
        f"_bs-{config.train.batch_size}"
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

            model(images)  # Consider passing orig outputs as input, so that you can interpolate.
            mod_acts, cs, sparsifiers_outs = model_mod(images)
            # Get final item of dict
            mod_out = mod_acts[list(mod_acts.keys())[-1]]

            # Get loss that compares each item of the hooked activations with the corresponding item
            # of the outputs
            loss: Float[Tensor, ""] = torch.zeros(1, requires_grad=True, device=device)
            assert len(activations) == len(
                mod_acts
            ), "Number of activations and modified activations must be the same"

            # Create dictionaries for the different losses so we can log in wand later
            sp_orig_losses: dict[str, torch.Tensor] = {}
            new_orig_losses: dict[str, torch.Tensor] = {}
            sp_new_losses: dict[str, torch.Tensor] = {}
            sparsity_losses: dict[str, torch.Tensor] = {}
            zeros_counts: dict[str, torch.Tensor] = {}
            zeros_fracs: dict[str, torch.Tensor] = {}

            for layer in range(len(activations)):
                if layer < len(activations) - 1:
                    # sae-orig reconstruction loss
                    sp_orig_losses[str(layer)] = criterion(
                        sparsifiers_outs[str(layer)], activations[str(layer)]
                    )
                    loss = loss + sp_orig_losses[str(layer)]

                # new-orig reconstruction loss
                new_orig_losses[str(layer)] = criterion(
                    mod_acts[str(layer)], activations[str(layer)]
                )
                loss = loss + new_orig_losses[str(layer)]

                # sae-new reconstruction loss
                if layer < len(activations) - 1:
                    if config.train.type_of_sparsifier == "sae":
                        sp_new_losses[str(layer)] = criterion(
                            sparsifiers_outs[str(layer)], mod_acts[str(layer)]
                        )
                    elif config.train.type_of_sparsifier == "codebook":
                        # Auxiliary recon loss described in Tamkin et al. (2023) p. 3
                        sp_new_losses[str(layer)] = criterion(
                            sparsifiers_outs[str(layer)], mod_acts[str(layer)].detach()
                        )
                    loss = loss + (
                        sp_new_losses[str(layer)] * config.train.sparsifier_in_out_recon_loss_scale
                    )

            # Add L_p norm loss
            if config.train.type_of_sparsifier == "sae":
                for layer in range(len(cs)):
                    sparsity_losses[str(layer)] = torch.norm(cs[str(layer)], p=0.6, dim=1).mean()
                    loss = loss + config.train.sparsity_lambda * sparsity_losses[str(layer)]

                # Calculate counts and fractions of zero entries in the saes per batch
                for layer in range(len(cs)):
                    zeros_counts[str(layer)] = torch.sum(cs[str(layer)] == 0)
                    zeros_fracs[str(layer)] = (
                        torch.sum(cs[str(layer)] == 0) / cs[str(layer)].numel()
                    )

            # Calculate accuracy
            _, argmax = torch.max(mod_out, 1)

            accuracy = (labels == argmax.squeeze()).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                logger.info(
                    "Epoch [%d/%d], Step [%d/%d], Loss: %f, Accuracy: %f",
                    epoch,
                    config.train.n_epochs,
                    i + 1,
                    len(train_loader),
                    loss.item(),
                    accuracy,
                )

                if config.wandb_project:
                    wandb.log({"train/loss": loss.item()}, step=samples)
                    wandb.log({"train/accuracy": accuracy}, step=samples)
                    for k, v in sp_orig_losses.items():
                        wandb.log({f"train/loss-sae-orig-{k}": v.item()}, step=samples)
                    for k, v in new_orig_losses.items():
                        wandb.log({f"train/loss-new-orig-{k}": v.item()}, step=samples)
                    for k, v in sp_new_losses.items():
                        wandb.log({f"train/loss-sae-new-{k}": v.item()}, step=samples)
                    for k, v in sparsity_losses.items():
                        wandb.log({f"train/loss-sparsity-loss-{k}": v.item()}, step=samples)
                    for k, v in zeros_counts.items():
                        wandb.log({f"train/zero-counts-{k}": v.item()}, step=samples)
                    for k, v in zeros_fracs.items():
                        wandb.log({f"train/fraction-zeros-{k}": v.item()}, step=samples)

        # Validate the model
        model_mod.sparsifiers.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                images = images.view(images.shape[0], -1)
                mod_acts, cs, sparsifiers_outs = model_mod(images)
                mod_out = mod_acts[list(mod_acts.keys())[-1]]
                _, argmax = torch.max(mod_out, 1)
                total += labels.size(0)
                correct += (labels == argmax.squeeze()).sum().item()

            accuracy = correct / total
            logger.info("Accuracy of the network on the 10000 test images: %f %%", 100 * accuracy)

            if config.wandb_project:
                wandb.log({"valid/accuracy": accuracy}, step=samples)
        model_mod.sparsifiers.train()

    if save_dir:
        save_module(
            config_dict=config.model_dump(mode="json"),
            save_dir=save_dir,
            module=model_mod.sparsifiers,
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
