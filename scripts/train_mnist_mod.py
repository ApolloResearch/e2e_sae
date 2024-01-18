"""Train a model on MNIST.

This script takes ~40 seconds to run for 3 layers and 15 epochs on a CPU.

Usage:
    python scripts/train_mnist.py <path/to/config.yaml>
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, OrderedDict
from collections import OrderedDict

import fire
import torch
import wandb
import yaml
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from sparsify.log import logger
from sparsify.models import MLP
from sparsify.models.mlp import MLPMod
from sparsify.utils import save_model
from sparsify.custom_load import map_state_dict_names_mlp


class ModelConfig(BaseModel):
    hidden_sizes: Optional[List[int]]


class TrainConfig(BaseModel):
    learning_rate: float
    batch_size: int
    epochs: int
    save_dir: Optional[Path]
    model_name: str
    type_of_sparsifier: str
    sparsity_lambda: float
    dict_eles_to_input_ratio: float
    sparsifier_inp_out_recon_loss_scale: float
    k: int
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
        config.train.save_dir = Path(__file__).parent.parent / ".checkpoints" / "mnist_mod"

    # Load the MNIST dataset
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(
        root=Path(__file__).parent.parent / ".data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=config.train.batch_size, shuffle=True)
    test_data = datasets.MNIST(
        root=Path(__file__).parent.parent / ".data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=config.train.batch_size, shuffle=False)
    
    # Define model path to load
    model_path = Path(__file__).parent.parent / "models" / config.train.model_name 
    model_path = max(model_path.glob("*.pt"), key=lambda x: int(x.stem.split("_")[-1]))


    # Initialize the MLP model
    model = MLP(config.model.hidden_sizes, input_size=784, output_size=10)
    model = model.to(device)
    model_trained_statedict = torch.load(model_path)
    model.load_state_dict(model_trained_statedict)
    model.eval()

    # Add hooks to the model so we can get all intermediate activations
    # First, define a function to be called when the forward pass reaches a layer
    def get_activation(name, activations: OrderedDict):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Then, add the hooks to the model
    activations = OrderedDict()
    for name, layer in model.layers.named_children():
        layer.register_forward_hook(get_activation(name, activations))

    # Get the SAEs from the model_mod and put them in the statedict of model_trained
    model_mod = MLPMod(
        config.model.hidden_sizes, 
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

    # Define the loss and optimizer
    criterion = nn.MSELoss()
    # Note: only pass the SAE parameters to the optimizer
    optimizer = torch.optim.Adam(model_mod.sparsifiers.parameters(), lr=config.train.learning_rate)

    if config.wandb:
        run_name = f"sparse-lambda-{config.train.sparsity_lambda}-lr-{config.train.learning_rate}_bs-{config.train.batch_size}-{str(config.model.hidden_sizes)}"
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

            orig_outputs = model(images) # Consider passing orig outputs as input, so that you can interpolate.
            mod_acts, cs, sparsifiers_outs = model_mod(images)
            # Get final item of dict
            mod_out = mod_acts[list(mod_acts.keys())[-1]]

            # Get loss that compares each item of the hooked activations with the corresponding item of the outputs
            loss = 0.0
            assert len(activations) == len(mod_acts), "Number of activations and modified activations must be the same"

            # Create dictionaries for the different losses so we can log in wand later
            sp_orig_losses = {}
            new_orig_losses = {}
            sp_new_losses = {}
            sparsity_losses = {}
            zeros_counts = {}

            for l in range(len(activations)):
                if l < len(activations) - 1:
                    # sae-orig reconstruction loss
                    sp_orig_losses[str(l)] = criterion(sparsifiers_outs[str(l)], activations[str(l)])
                    loss += sp_orig_losses[str(l)]

                # new-orig reconstruction loss
                new_orig_losses[str(l)] = criterion(mod_acts[str(l)], activations[str(l)])
                loss += new_orig_losses[str(l)]

                # sae-new reconstruction loss
                if l < len(activations) - 1:
                    if config.train.type_of_sparsifier == 'sae':
                        sp_new_losses[str(l)] = criterion(sparsifiers_outs[str(l)], mod_acts[str(l)])
                    elif config.train.type_of_sparsifier == 'codebook':
                        # Auxiliary recon loss described in Tamkin et al. (2023) p. 3
                        sp_new_losses[str(l)] = criterion(sparsifiers_outs[str(l)], mod_acts[str(l)].detach())
                    loss += sp_new_losses[str(l)] * config.train.sparsifier_inp_out_recon_loss_scale
            
            # Add sparsity loss
            if config.train.type_of_sparsifier == 'sae':
                for l in range(len(cs)):
                    sparsity_losses[str(l)] = torch.mean(torch.abs(cs[str(l)]))
                    loss += sparsity_losses[str(l)] * config.train.sparsity_lambda

                # Calculate fraction of zero entries in the saes per batch
                for l in range(len(cs)):
                    zeros_counts[str(l)] = torch.sum(cs[str(l)] == 0) / cs[str(l)].numel()

            # Calculate accuracy
            _, argmax = torch.max(mod_out, 1)

            accuracy = (labels == argmax.squeeze()).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i  % 10 == 0:
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
                    wandb.log({"train/accuracy": accuracy.item(), "train/samples": samples}, step=samples)
                    for k, v in sp_orig_losses.items():
                        wandb.log({f"train/loss-sae-orig-{k}": v.item(), "train/samples": samples}, step=samples)
                    for k, v in new_orig_losses.items():
                        wandb.log({f"train/loss-new-orig-{k}": v.item(), "train/samples": samples}, step=samples)
                    for k, v in sp_new_losses.items():
                        wandb.log({f"train/loss-sae-new-{k}": v.item(), "train/samples": samples}, step=samples)
                    for k, v in sparsity_losses.items():
                        wandb.log({f"train/loss-sparsity-loss-{k}": v.item(), "train/samples": samples}, step=samples)
                    for k, v in zeros_counts.items():
                        wandb.log({f"train/fraction-zeros-{k}": v.item(), "train/samples": samples}, step=samples)

        # Validate the model
        model_mod.eval()
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

            if config.wandb:
                wandb.log({"valid/accuracy": accuracy}, step=samples)

    # Comment out because we're not saving mod models right now
    #     if config.train.save_every_n_epochs and (epoch + 1) % config.train.save_every_n_epochs == 0:
    #         save_model(json.loads(config.model_dump_json()), save_dir, model, epoch) # TODO Figure out how to save only saes

    if not (save_dir / f"sparse_model_epoch_{epoch + 1}.pt").exists():
        save_model(json.loads(config.model_dump_json()), save_dir, model_mod, epoch, sparse=True) # TODO Figure out how to save only saes


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str) # TODO make separate config for model_mod
    config = load_config(config_path)
    train(config)


if __name__ == "__main__":
    fire.Fire(main)
