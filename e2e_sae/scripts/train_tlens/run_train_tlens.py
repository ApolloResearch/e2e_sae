"""Train a custom transformerlens model.

Usage:
    python run_train_tlens.py <path/to/config.yaml>
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Self

import fire
import torch
import wandb
from dotenv import load_dotenv
from jaxtyping import Int
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig, evals

from e2e_sae.types import RootPath, TorchDtype
from e2e_sae.utils import load_config, save_module, set_seed


class HookedTransformerPreConfig(BaseModel):
    """Pydantic model whose arguments will be passed to a HookedTransformerConfig."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)
    d_model: PositiveInt
    n_layers: PositiveInt
    n_ctx: PositiveInt
    d_head: PositiveInt
    d_vocab: PositiveInt
    act_fn: str
    dtype: TorchDtype | None
    tokenizer_name: str


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_epochs: PositiveInt
    batch_size: PositiveInt
    effective_batch_size: PositiveInt | None = None
    lr: PositiveFloat
    warmup_samples: NonNegativeInt = 0
    save_dir: RootPath | None = Path(__file__).parent / "out"
    save_every_n_epochs: PositiveInt | None

    @model_validator(mode="after")
    def check_effective_batch_size(self) -> Self:
        if self.effective_batch_size is not None:
            assert (
                self.effective_batch_size % self.batch_size == 0
            ), "effective_batch_size must be a multiple of batch_size."
        return self


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int = 0
    name: str
    tlens_config: HookedTransformerPreConfig
    train: TrainConfig
    wandb_project: str | None  # If None, don't log to Weights & Biases


def train(config: Config, model: HookedTransformer, device: torch.device) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

    effective_batch_size = config.train.effective_batch_size or config.train.batch_size
    n_gradient_accumulation_steps = effective_batch_size // config.train.batch_size

    scheduler = None
    if config.train.warmup_samples > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(
                1.0, (step + 1) / (config.train.warmup_samples // effective_batch_size)
            ),
        )

    train_loader = evals.make_pile_data_loader(model.tokenizer, batch_size=config.train.batch_size)

    # Initialize wandb
    run_name = f"{config.name}_lr-{config.train.lr}_bs-{config.train.batch_size}"
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
    grad_updates = 0
    for epoch in tqdm(
        range(1, config.train.n_epochs + 1), total=config.train.n_epochs, desc="Epochs"
    ):
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Steps"):
            tokens: Int[Tensor, "batch pos"] = batch["tokens"].to(device=device)
            loss = model(tokens, return_type="loss")

            loss = loss / n_gradient_accumulation_steps
            loss.backward()

            if (step + 1) % n_gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                grad_updates += 1

                if config.train.warmup_samples > 0:
                    assert scheduler is not None
                    scheduler.step()

            samples += tokens.shape[0]
            if step == 0 or step % 20 == 0:
                tqdm.write(
                    f"Epoch {epoch} Samples {samples} Step {step} GradUpdates {grad_updates} "
                    f"Loss {loss.item()}"
                )

            if config.wandb_project:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "epoch": epoch,
                        "grad_updates": grad_updates,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=samples,
                )
        if save_dir and (
            (config.train.save_every_n_epochs and epoch % config.train.save_every_n_epochs == 0)
            or epoch == config.train.n_epochs  # Save the last epoch
        ):
            save_module(
                config_dict=config.model_dump(mode="json"),
                save_dir=save_dir,
                module=model,
                model_filename=f"epoch_{epoch}.pt",
            )
            # TODO: Add evaluation loop
    if config.wandb_project:
        wandb.finish()


def main(config_path_str: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path_str, config_model=Config)
    set_seed(config.seed)

    hooked_transformer_config = HookedTransformerConfig(**config.tlens_config.model_dump())
    model = HookedTransformer(hooked_transformer_config)
    model.to(device)

    train(config, model, device=device)


if __name__ == "__main__":
    fire.Fire(main)
