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

from tqdm import tqdm

from sparsify.log import logger
from sparsify.models import MLP
from sparsify.models.models import SparsifiedMLP
from sparsify.data import load_data_to_device, get_data, data_preprocessing, get_num_minibatch_elements
from sparsify.utils import save_trainable_params
from sparsify.model_loading import get_models
from sparsify.configs import load_config, Config



def get_activation(name, activations: OrderedDict):
    """ function to be called when the forward pass reaches a layer"""
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook





def train(config: Config) -> None:
    """Train the MLP on MNIST.

    If config.wandb is not None, log the results to Weights & Biases.
    """
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Load the MNIST dataset
    train_loader, test_loader = get_data(config)

    # Initialize the MLP model and modified model
    base_model, sparsified_model = get_models(config)
    
    # TODO determine which model is in train mode
    trainable_parameters = get_trainable_parameters(config, base_model, sparsified_model)
    inference_model = base_model if sparsified_model is None else sparsified_model # TODO future: make this into a function so it can deal with trancoders and meta_sae

    # TODO here you're going to have to start differentiating between the 
    #  different types of training with regard to loss, optimized parameters, etc.



    run_name = f"sparse-lambda-{config.train.sparsity_lambda}-lr-{config.train.learning_rate}_bs-{config.train.batch_size}-{str(config.model.hidden_sizes)}"
    if config.wandb:
        wandb.init(
            name=run_name,
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=config.model_dump(),
        )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = config.train.save_dir / f"{run_name}_{timestamp}"

    # Define the loss and optimizer # TODO i think in tflens this will be in the model
    criterion = nn.MSELoss()

    if config.optimizer_name in ["Adam", "AdamW"]:
        # Weight decay in Adam is implemented badly, so use AdamW instead (see PyTorch AdamW docs)
        if config.weight_decay is not None:
            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=config.train.lr,
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                trainable_parameters,
                lr=config.lr,
            )
    else:
        raise NotImplementedError(f"Optimizer not implemented: {config.optimizer_name}")

    scheduler = None
    if config.warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / config.warmup_steps),
        )

    samples = 0
    # Training loop
    for epoch in tqdm(range(config.train.num_epochs), total=config.train.num_epochs, desc="Epochs"):
        for step, data in enumerate(train_loader):
            data = load_data_to_device(data, device)
            data = data_preprocessing(data, config)
            samples += get_num_minibatch_elements(data)

            loss = inference_model(data, return_type="loss") # TODO define a new return type? for the different kinds of models with intermediate activations
            loss.backward()
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, config.max_grad_norm)
            optimizer.step()
            if config.warmup_steps > 0:
                assert scheduler is not None
                scheduler.step()
            optimizer.zero_grad()

            if config.wandb:
                wandb.log(
                    {"train_loss": loss.item(), "samples": samples, "epoch": epoch}
                )

            if config.print_every is not None and step % config.print_every == 0:
                print(f"Epoch {epoch} Samples {samples} Step {step} Loss {loss.item()}")

            if (
                config.save_every is not None
                and step % config.save_every == 0
                and config.save_dir is not None
            ):
                # TODO you'll need a more complex saving function so you only save trainable parameters
                #  in a sensible place
                torch.save(inference_model.state_dict(), f"{config.save_dir}/model_{step}.pt")

            if config.max_steps is not None and step >= config.max_steps:
                break

    # Comment out because we're not saving mod models right now
    #     if config.train.save_every_n_epochs and (epoch + 1) % config.train.save_every_n_epochs == 0:
    #         save_model(json.loads(config.model_dump_json()), save_dir, model, epoch) # TODO Figure out how to save only saes

    if not (save_dir / f"sparse_model_epoch_{epoch + 1}.pt").exists():
        save_trainable_params(json.loads(config.model_dump_json()), save_dir, trainable_parameters, epoch, sparse=True) # TODO Figure out how to save only saes


def get_trainable_parameters(config, base_model, sparsified_model):
    """Returns a list of trainable parameters for the optimizer"""
    if sparsified_model is None:
        return base_model.parameters()
    elif sparsified_model is not None:
        return sparsified_model.sparsifiers.parameters()


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str) # TODO make separate config for model_mod
    config = load_config(config_path)
    train(config)


if __name__ == "__main__":
    fire.Fire(main)
