"""Script for training SAEs on top of a transformerlens model.

Usage:
    python run_train_tlens_saes.py <path/to/config.yaml>
"""
import os
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Self, cast

import fire
import torch
import wandb
import yaml
from dotenv import load_dotenv
from jaxtyping import Float, Int
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import lm_accuracy, lm_cross_entropy_loss

from sparsify.data import DataConfig, create_data_loader
from sparsify.losses import LossConfigs, calc_loss
from sparsify.models.sparsifiers import SAE
from sparsify.models.transformers import SAETransformer
from sparsify.scripts.train_tlens.run_train_tlens import HookedTransformerPreConfig
from sparsify.types import RootPath, Samples
from sparsify.utils import load_config, save_model, set_seed


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    save_dir: RootPath | None = Path(__file__).parent / "out"
    save_every_n_samples: PositiveInt | None
    n_samples: PositiveInt | None = None
    batch_size: PositiveInt
    effective_batch_size: PositiveInt | None = None
    lr: PositiveFloat
    scheduler: str | None = None
    warmup_steps: NonNegativeFloat = 0
    max_grad_norm: PositiveFloat | None = None
    loss_configs: LossConfigs

    @model_validator(mode="after")
    def check_effective_batch_size(self) -> Self:
        if self.effective_batch_size is not None:
            assert (
                self.effective_batch_size % self.batch_size == 0
            ), "effective_batch_size must be a multiple of batch_size."
        return self


class SparsifiersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type_of_sparsifier: str | None = "sae"
    dict_size_to_input_ratio: PositiveFloat = 1.0
    k: PositiveInt | None = None  # Only used for codebook sparsifier
    sae_position_name: str  # TODO will become List[str]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: NonNegativeInt = 0
    tlens_model_name: str | None = None
    tlens_model_path: RootPath | None = None
    train: TrainConfig
    data: DataConfig
    saes: SparsifiersConfig
    wandb_project: str | None = None  # If None, don't log to Weights & Biases

    @model_validator(mode="before")
    @classmethod
    def check_only_one_model_definition(cls, values: dict[str, Any]) -> dict[str, Any]:
        assert (values.get("tlens_model_name") is not None) + (
            values.get("tlens_model_path") is not None
        ) == 1, "Must specify exactly one of tlens_model_name or tlens_model_path."
        return values


def sae_hook(
    value: Float[torch.Tensor, "... dim"], hook: HookPoint, sae: SAE, hook_acts: dict[str, Any]
) -> Float[torch.Tensor, "... dim"]:
    """Runs the SAE on the input and stores the output and c in hook_acts."""
    hook_acts["input"] = value
    output, c = sae(value)
    hook_acts["output"] = output
    hook_acts["c"] = c
    return output


@logging_redirect_tqdm()
def train(
    config: Config,
    model: SAETransformer,
    data_loader: DataLoader[Samples],
    device: torch.device,
) -> None:
    model_name = config.tlens_model_name or "custom"

    # TODO make appropriate for transcoders and metaSAEs
    optimizer = torch.optim.Adam(model.saes.parameters(), lr=config.train.lr)

    scheduler = None
    if config.train.warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / config.train.warmup_steps),
        )

    if config.train.n_samples is None:
        # If streaming (i.e. if the dataset is an IterableDataset), we don't know the length
        n_batches = None if isinstance(data_loader.dataset, IterableDataset) else len(data_loader)
    else:
        n_batches = config.train.n_samples // config.train.batch_size

    if config.train.effective_batch_size is not None:
        n_gradient_accumulation_steps = config.train.effective_batch_size // config.train.batch_size
    else:
        n_gradient_accumulation_steps = 1

    def orig_resid_names(name: str) -> bool:
        return model.sae_position_name in name

    # Initialize wandb
    run_name = (
        f"saes_{model_name}_lambda-{config.train.loss_configs.sparsity.coeff}_"
        f"Lp{config.train.loss_configs.sparsity.p_norm}_lr-{config.train.lr}"
    )
    if config.wandb_project:
        load_dotenv(override=True)
        wandb.init(
            name=run_name,
            project=config.wandb_project,
            entity=os.getenv("WANDB_ENTITY"),
            config=config.model_dump(mode="json"),
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = config.train.save_dir / f"{run_name}_{timestamp}" if config.train.save_dir else None

    total_samples = 0
    total_samples_at_last_save = 0
    grad_updates = 0
    grad_norm: float | None = None

    for step, batch in tqdm(enumerate(data_loader), total=n_batches, desc="Steps"):
        tokens: Int[Tensor, "batch pos"] = batch[config.data.column_name].to(device=device)
        # Run model without SAEs
        with torch.inference_mode():
            orig_logits, orig_acts = model.tlens_model.run_with_cache(
                tokens, names_filter=orig_resid_names, return_cache_object=False
            )
        assert isinstance(orig_logits, torch.Tensor)  # Prevent pyright error

        # Run model with SAEs
        sae_acts = {hook_name: {} for hook_name in orig_acts}
        fwd_hooks: list[tuple[str, Callable[..., Float[torch.Tensor, "... d_head"]]]] = [
            (
                hook_name,
                partial(sae_hook, sae=cast(SAE, model.saes[str(i)]), hook_acts=sae_acts[hook_name]),
            )
            for i, hook_name in enumerate(orig_acts)
        ]
        new_logits: Float[Tensor, "batch pos vocab"] = model.tlens_model.run_with_hooks(
            tokens,
            fwd_hooks=fwd_hooks,  # type: ignore
        )
        loss, loss_dict = calc_loss(
            orig_acts=orig_acts,
            sae_acts=sae_acts,
            orig_logits=orig_logits,
            new_logits=new_logits,
            loss_configs=config.train.loss_configs,
        )

        loss = loss / n_gradient_accumulation_steps
        loss.backward()
        if config.train.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.max_grad_norm
            ).item()

        if (step + 1) % n_gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            grad_updates += 1

            if config.train.warmup_steps > 0:
                assert scheduler is not None
                scheduler.step()

        total_samples += tokens.shape[0]
        if step == 0 or step % 20 == 0:
            tqdm.write(
                f"Samples {total_samples} Step {step} GradUpdates {grad_updates} "
                f"Loss {loss.item():.3f}"
            )

        if config.wandb_project:
            log_info = {"grad_updates": grad_updates}
            wandb.log({"train_loss": loss.item(), **log_info})
            for loss_name, loss_value in loss_dict.items():
                wandb.log({loss_name: loss_value.item(), **log_info}, step=total_samples)

            if config.train.max_grad_norm is not None:
                assert grad_norm is not None
                wandb.log({"grad_norm": grad_norm, **log_info}, step=total_samples)
            if step == 0 or step % 5 == 0:
                orig_model_performance_loss = lm_cross_entropy_loss(
                    orig_logits, tokens, per_token=False
                )
                orig_model_performance_acc = lm_accuracy(orig_logits, tokens, per_token=False)
                sae_model_performance_loss = lm_cross_entropy_loss(
                    new_logits, tokens, per_token=False
                )
                sae_model_performance_acc = lm_accuracy(new_logits, tokens, per_token=False)
                # flat_orig_logits = orig_logits.view(-1, orig_logits.shape[-1])
                # flat_new_logits = new_logits.view(-1, new_logits.shape[-1])
                # kl_div = torch.nn.functional.kl_div(
                #     torch.nn.functional.log_softmax(flat_new_logits, dim=-1),
                #     torch.nn.functional.softmax(flat_orig_logits, dim=-1),
                #     reduction="batchmean",
                # ) # Unsure if this is correct. Also it's expensive in terms of memory.

                wandb.log(
                    {
                        "performance/orig_model_performance_loss": orig_model_performance_loss.item(),
                        "performance/orig_model_performance_acc": orig_model_performance_acc.item(),
                        "performance/sae_model_performance_loss": sae_model_performance_loss.item(),
                        "performance/sae_model_performance_acc": sae_model_performance_acc.item(),
                        "performance/difference_loss": (
                            orig_model_performance_loss - sae_model_performance_loss
                        ).item(),
                        "performance/difference_acc": (
                            orig_model_performance_acc - sae_model_performance_acc
                        ).item(),
                        **log_info,
                    },
                    step=total_samples,
                )
        if (
            save_dir
            and config.train.save_every_n_samples
            and total_samples - total_samples_at_last_save >= config.train.save_every_n_samples
        ):
            total_samples_at_last_save = total_samples
            save_model(
                config_dict=config.model_dump(mode="json"),
                save_dir=save_dir,
                model=model,
                model_filename=f"samples_{total_samples}.pt",
            )
        if config.train.n_samples is not None and total_samples >= config.train.n_samples:
            break

    if save_dir:
        save_model(
            config_dict=config.model_dump(mode="json"),
            save_dir=save_dir,
            model=model,
            model_filename=f"samples_{total_samples}.pt",
        )


def load_tlens_model(config: Config) -> HookedTransformer:
    """Load transformerlens model from either HuggingFace or local path."""
    if config.tlens_model_name is not None:
        tlens_model = HookedTransformer.from_pretrained(config.tlens_model_name)
    else:
        assert config.tlens_model_path is not None, "tlens_model_path is None."
        # Load the tlens_config
        with open(config.tlens_model_path / "config.yaml") as f:
            tlens_config = HookedTransformerPreConfig(**yaml.safe_load(f)["tlens_config"])
        hooked_transformer_config = HookedTransformerConfig(**tlens_config.model_dump())

        # Load the model
        tlens_model = HookedTransformer(hooked_transformer_config)
        latest_model_path = max(
            config.tlens_model_path.glob("*.pt"), key=lambda x: int(x.stem.split("_")[-1])
        )
        tlens_model.load_state_dict(torch.load(latest_model_path))

    assert tlens_model.tokenizer is not None
    return tlens_model


def main(config_path_str: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path_str, config_model=Config)
    set_seed(config.seed)

    data_loader, _ = create_data_loader(config.data, batch_size=config.train.batch_size)
    tlens_model = load_tlens_model(config)

    model = SAETransformer(tlens_model, config).to(device=device)
    train(
        config=config,
        model=model,
        data_loader=data_loader,
        device=device,
    )


if __name__ == "__main__":
    fire.Fire(main)
