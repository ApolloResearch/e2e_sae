"""Script for training SAEs on top of a transformerlens model.

Usage:
    python run_train_tlens_saes.py <path/to/config.yaml>
"""
import math
import os
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Self, cast

import fire
import torch
import wandb
from dotenv import load_dotenv
from jaxtyping import Float, Int
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
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
from transformer_lens.hook_points import HookPoint

from sparsify.data import DataConfig, create_data_loader
from sparsify.loader import load_pretrained_saes, load_tlens_model
from sparsify.log import logger
from sparsify.losses import LossConfigs, calc_loss
from sparsify.metrics import DiscreteMetrics, collect_wandb_metrics
from sparsify.models.sparsifiers import SAE
from sparsify.models.transformers import SAETransformer
from sparsify.types import RootPath, Samples
from sparsify.utils import (
    filter_names,
    get_linear_lr_schedule,
    load_config,
    save_module,
    set_seed,
)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    save_dir: RootPath | None = Path(__file__).parent / "out"
    save_every_n_samples: PositiveInt | None
    n_samples: PositiveInt | None = None
    batch_size: PositiveInt
    effective_batch_size: PositiveInt | None = None
    lr: PositiveFloat
    adam_beta1: NonNegativeFloat
    warmup_samples: NonNegativeInt = 0
    cooldown_samples: NonNegativeInt = 0
    max_grad_norm: PositiveFloat | None = None
    log_every_n_grad_steps: PositiveInt = 20
    collect_discrete_metrics_every_n_samples: NonNegativeInt = Field(
        20_000,
        description="Metrics such as activation frequency and alive neurons, are calculated over "
        "discrete periods. This parameter specifies how often to calculate these metrics.",
    )
    discrete_metrics_n_tokens: PositiveInt = Field(
        100_000, description="The number of tokens to caclulate discrete metrics over."
    )
    collect_output_metrics_every_n_samples: NonNegativeInt = Field(
        0,
        description="How many samples between calculating metrics like the cross-entropy loss and "
        "kl divergence between the original and SAE-augmented logits. If training with logits_kl, "
        "these will be calculated every batch regardless of this parameter.",
    )
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
    pretrained_sae_paths: Annotated[
        list[RootPath] | None, BeforeValidator(lambda x: [x] if isinstance(x, str | Path) else x)
    ] = Field(None, description="Path to a pretrained SAE model to load. If None, don't load any.")
    retrain_saes: bool = Field(False, description="Whether to retrain the pretrained SAEs.")
    sae_position_names: Annotated[
        list[str], BeforeValidator(lambda x: [x] if isinstance(x, str) else x)
    ] = Field(
        ...,
        description="The names of the SAE positions to train on. E.g. 'hook_resid_post' or "
        "['hook_resid_post', 'hook_mlp_out'] or ['hook_mlp_out', 'hook_resid_post']",
    )


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: NonNegativeInt = 0
    tlens_model_name: str | None = None
    tlens_model_path: RootPath | None = None
    train: TrainConfig
    data: DataConfig
    saes: SparsifiersConfig
    wandb_project: str | None = None  # If None, don't log to Weights & Biases
    wandb_run_name: str | None = Field(
        None,
        description="If None, a run_name is generated based on (typically) important config "
        "parameters.",
    )
    wandb_run_name_prefix: str = Field("", description="Name that is prepended to the run name")

    @model_validator(mode="before")
    @classmethod
    def check_only_one_model_definition(cls, values: dict[str, Any]) -> dict[str, Any]:
        assert (values.get("tlens_model_name") is not None) + (
            values.get("tlens_model_path") is not None
        ) == 1, "Must specify exactly one of tlens_model_name or tlens_model_path."
        return values


def get_run_name(config: Config) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        coeff_info = f"lpcoeff-{config.train.loss_configs.sparsity.coeff}_"
        if config.train.loss_configs.inp_to_out is not None:
            coeff_info += f"inp-to-out-{config.train.loss_configs.inp_to_out.coeff}_"
        if config.train.loss_configs.logits_kl is not None:
            coeff_info += f"logits-kl-{config.train.loss_configs.logits_kl.coeff}_"

        run_suffix = config.wandb_run_name_prefix + (
            f"{'-'.join(config.saes.sae_position_names)}_"
            f"ratio-{config.saes.dict_size_to_input_ratio}_"
            f"lr-{config.train.lr}_{coeff_info}"
        )
    return config.wandb_run_name_prefix + run_suffix


def sae_hook(
    value: Float[torch.Tensor, "... dim"],
    hook: HookPoint | None,
    sae: SAE | torch.nn.Module,
    hook_acts: dict[str, Any],
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
    trainable_param_names: list[str],
    device: torch.device,
) -> None:
    model.saes.train()

    layerwise = config.train.loss_configs.logits_kl is None

    for name, param in model.named_parameters():
        if name.startswith("saes.") and name.split("saes.")[1] in trainable_param_names:
            param.requires_grad = True
        else:
            param.requires_grad = False
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.train.lr,
        betas=(config.train.adam_beta1, 0.999),
    )

    effective_batch_size = config.train.effective_batch_size or config.train.batch_size
    n_gradient_accumulation_steps = effective_batch_size // config.train.batch_size

    lr_schedule = get_linear_lr_schedule(
        warmup_samples=config.train.warmup_samples,
        cooldown_samples=config.train.cooldown_samples,
        n_samples=config.train.n_samples,
        effective_batch_size=effective_batch_size,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    if config.train.n_samples is None:
        # If streaming (i.e. if the dataset is an IterableDataset), we don't know the length
        n_batches = None if isinstance(data_loader.dataset, IterableDataset) else len(data_loader)
    else:
        n_batches = math.ceil(config.train.n_samples / config.train.batch_size)

    final_layer = None
    if all(name.startswith("blocks.") for name in model.raw_sae_position_names) and layerwise:
        # We don't need to run through the whole model for layerwise runs
        final_layer = max([int(name.split(".")[1]) for name in model.raw_sae_position_names]) + 1

    run_name = get_run_name(config)
    # Initialize wandb
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
    total_tokens = 0
    grad_updates = 0
    grad_norm: float | None = None
    samples_since_discrete_metric_collection: int = 0
    discrete_metrics: DiscreteMetrics | None = None
    samples_since_output_metric_collection: int = 0

    for batch_idx, batch in tqdm(enumerate(data_loader), total=n_batches, desc="Steps"):
        run_entire_model: bool = not layerwise
        is_grad_step: bool = (batch_idx + 1) % n_gradient_accumulation_steps == 0

        if config.train.collect_output_metrics_every_n_samples > 0 and (
            batch_idx == 0
            or (
                samples_since_output_metric_collection
                >= config.train.collect_output_metrics_every_n_samples
            )
        ):
            # Run entire model to collect output metrics
            run_entire_model = True
            samples_since_output_metric_collection = 0

        is_log_step: bool = (
            batch_idx == 0
            or (is_grad_step and (grad_updates + 1) % config.train.log_every_n_grad_steps == 0)
            or (layerwise and run_entire_model)
        )

        tokens: Int[Tensor, "batch pos"] = batch[config.data.column_name].to(device=device)
        # Run model without SAEs
        with torch.inference_mode():
            orig_logits, orig_acts = model.tlens_model.run_with_cache(
                tokens,
                names_filter=model.raw_sae_position_names,
                return_cache_object=False,
                stop_at_layer=None if run_entire_model else final_layer,
            )
            assert isinstance(orig_logits, torch.Tensor)  # Prevent pyright error
        # Get SAE feature activations
        sae_acts = {hook_name: {} for hook_name in orig_acts}
        new_logits: Float[Tensor, "batch pos vocab"] | None = None
        if not run_entire_model:
            # Just run the already-stored activations through the SAEs
            for hook_name in orig_acts:
                sae_hook(
                    value=orig_acts[hook_name].detach().clone(),
                    hook=None,
                    sae=model.saes[hook_name.replace(".", "-")],
                    hook_acts=sae_acts[hook_name],
                )
        else:
            # Run the tokens through the whole SAE-augmented model
            fwd_hooks: list[tuple[str, Callable[..., Float[torch.Tensor, "... d_head"]]]] = [
                (
                    hook_name,
                    partial(
                        sae_hook,
                        sae=cast(SAE, model.saes[hook_name.replace(".", "-")]),
                        hook_acts=sae_acts[hook_name],
                    ),
                )
                for hook_name in orig_acts
            ]
            new_logits = model.tlens_model.run_with_hooks(
                tokens,
                fwd_hooks=fwd_hooks,  # type: ignore
            )
        loss, loss_dict = calc_loss(
            orig_acts=orig_acts,
            sae_acts=sae_acts,
            orig_logits=None if new_logits is None else orig_logits.detach().clone(),
            new_logits=new_logits,
            loss_configs=config.train.loss_configs,
            is_log_step=is_log_step,
        )

        loss = loss / n_gradient_accumulation_steps
        loss.backward()
        if config.train.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.max_grad_norm
            ).item()

        if is_grad_step:
            optimizer.step()
            optimizer.zero_grad()
            grad_updates += 1
            scheduler.step()

        total_samples += tokens.shape[0]
        total_tokens += tokens.shape[0] * tokens.shape[1]
        samples_since_discrete_metric_collection += tokens.shape[0]
        samples_since_output_metric_collection += tokens.shape[0]

        if (
            batch_idx == 0
            or discrete_metrics is None
            and (
                samples_since_discrete_metric_collection
                >= config.train.collect_discrete_metrics_every_n_samples
            )
        ):
            # Start collecting discrete metrics for next config.train.discrete_metrics_n_tokens
            discrete_metrics = DiscreteMetrics(
                dict_sizes={
                    hook_name: sae_acts[hook_name]["c"].shape[-1] for hook_name in sae_acts
                },
                device=device,
            )
            samples_since_discrete_metric_collection = 0

        if discrete_metrics is not None:
            discrete_metrics.update_dict_el_frequencies(
                sae_acts, batch_tokens=tokens.shape[0] * tokens.shape[1]
            )
            if discrete_metrics.tokens_used >= config.train.discrete_metrics_n_tokens:
                # Finished collecting discrete metrics
                metrics = discrete_metrics.collect_for_logging(
                    log_wandb_histogram=config.wandb_project is not None
                )
                metrics["total_tokens"] = total_tokens
                if config.wandb_project:
                    # TODO: Log when not using wandb too
                    wandb.log(metrics, step=total_samples)
                discrete_metrics = None
                samples_since_discrete_metric_collection = 0

        if is_log_step:
            tqdm.write(
                f"Samples {total_samples} Batch_idx {batch_idx} GradUpdates {grad_updates} "
                f"Loss {loss.item():.5f}"
            )

            if config.wandb_project:
                wandb_log_info = collect_wandb_metrics(
                    loss=loss.item(),
                    grad_updates=grad_updates,
                    total_tokens=total_tokens,
                    sae_acts=sae_acts,
                    loss_dict=loss_dict,
                    grad_norm=grad_norm,
                    orig_logits=orig_logits.detach().clone() if new_logits is not None else None,
                    new_logits=new_logits.detach().clone() if new_logits is not None else None,
                    tokens=tokens,
                    lr=optimizer.param_groups[0]["lr"],
                )
                wandb.log(wandb_log_info, step=total_samples)
        if (
            save_dir
            and config.train.save_every_n_samples
            and total_samples - total_samples_at_last_save >= config.train.save_every_n_samples
        ):
            total_samples_at_last_save = total_samples
            save_module(
                config_dict=config.model_dump(mode="json"),
                save_dir=save_dir,
                module=model.saes,
                model_path=save_dir / f"samples_{total_samples}.pt",
            )
            if config.wandb_project:
                wandb.save(save_dir / f"samples_{total_samples}.pt")

        if config.train.n_samples is not None and total_samples >= config.train.n_samples:
            break

    if save_dir:
        save_module(
            config_dict=config.model_dump(mode="json"),
            save_dir=save_dir,
            module=model.saes,
            model_path=save_dir / f"samples_{total_samples}.pt",
        )
        if config.wandb_project:
            wandb.save(save_dir / f"samples_{total_samples}.pt")

    if config.wandb_project:
        wandb.finish()


def main(config_path_or_obj: Path | str | Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path_or_obj, config_model=Config)
    logger.info(config)
    set_seed(config.seed)

    data_loader, _ = create_data_loader(config.data, batch_size=config.train.batch_size)
    tlens_model = load_tlens_model(
        tlens_model_name=config.tlens_model_name, tlens_model_path=config.tlens_model_path
    )

    raw_sae_position_names = filter_names(
        list(tlens_model.hook_dict.keys()), config.saes.sae_position_names
    )

    model = SAETransformer(
        config=config, tlens_model=tlens_model, raw_sae_position_names=raw_sae_position_names
    ).to(device=device)

    all_param_names = [name for name, _ in model.saes.named_parameters()]
    if config.saes.pretrained_sae_paths is not None:
        trainable_param_names = load_pretrained_saes(
            saes=model.saes,
            pretrained_sae_paths=config.saes.pretrained_sae_paths,
            all_param_names=all_param_names,
            retrain_saes=config.saes.retrain_saes,
        )
    else:
        trainable_param_names = all_param_names

    assert len(trainable_param_names) > 0, "No trainable parameters found."
    logger.info(f"Trainable parameters: {trainable_param_names}")
    train(
        config=config,
        model=model,
        data_loader=data_loader,
        trainable_param_names=trainable_param_names,
        device=device,
    )


if __name__ == "__main__":
    fire.Fire(main)
