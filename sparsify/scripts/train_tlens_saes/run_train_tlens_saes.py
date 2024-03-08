"""Script for training SAEs on top of a transformerlens model.

Usage:
    python run_train_tlens_saes.py <path/to/config.yaml>
"""
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Self

import fire
import torch
import wandb
from datasets import IterableDataset
from dotenv import load_dotenv
from jaxtyping import Int
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
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from sparsify.data import DatasetConfig, create_data_loader
from sparsify.hooks import SAEActs
from sparsify.loader import load_pretrained_saes, load_tlens_model
from sparsify.log import logger
from sparsify.losses import LossConfigs, calc_loss
from sparsify.metrics import (
    ActFrequencyMetrics,
    calc_output_metrics,
    calc_sparsity_metrics,
)
from sparsify.models.transformers import SAETransformer
from sparsify.types import RootPath, Samples
from sparsify.utils import (
    filter_names,
    get_linear_lr_schedule,
    load_config,
    save_module,
    set_seed,
)


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
        "['hook_resid_post', 'hook_mlp_out']. Each entry gets matched to all hook positions that "
        "contain the given string.",
    )


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None  # If None, don't log to Weights & Biases
    wandb_run_name: str | None = Field(
        None,
        description="If None, a run_name is generated based on (typically) important config "
        "parameters.",
    )
    wandb_run_name_prefix: str = Field("", description="Name that is prepended to the run name")
    seed: NonNegativeInt = 0
    tlens_model_name: str | None = None
    tlens_model_path: RootPath | None = None
    save_dir: RootPath | None = Path(__file__).parent / "out"
    n_samples: PositiveInt | None = None
    save_every_n_samples: PositiveInt | None
    eval_every_n_samples: PositiveInt | None = Field(
        None, description="If None, don't evaluate. If 0, only evaluate at the end."
    )
    eval_n_samples: PositiveInt | None
    batch_size: PositiveInt
    effective_batch_size: PositiveInt | None = None
    lr: PositiveFloat
    adam_beta1: NonNegativeFloat
    warmup_samples: NonNegativeInt = 0
    cooldown_samples: NonNegativeInt = 0
    max_grad_norm: PositiveFloat | None = None
    log_every_n_grad_steps: PositiveInt = 20
    collect_act_frequency_every_n_samples: NonNegativeInt = Field(
        20_000,
        description="Metrics such as activation frequency and alive neurons are calculated over "
        "fixed number of batches. This parameter specifies how often to calculate these metrics.",
    )
    act_frequency_n_tokens: PositiveInt = Field(
        100_000, description="The number of tokens to caclulate activation frequency metrics over."
    )
    collect_output_metrics_every_n_samples: NonNegativeInt = Field(
        0,
        description="How many samples between calculating metrics like the cross-entropy loss and "
        "kl divergence between the original and SAE-augmented logits. If training with logits_kl, "
        "these will be calculated every batch regardless of this parameter.",
    )
    loss: LossConfigs
    train_data: DatasetConfig
    eval_data: DatasetConfig | None = None
    saes: SparsifiersConfig

    @model_validator(mode="before")
    @classmethod
    def check_only_one_model_definition(cls, values: dict[str, Any]) -> dict[str, Any]:
        assert (values.get("tlens_model_name") is not None) + (
            values.get("tlens_model_path") is not None
        ) == 1, "Must specify exactly one of tlens_model_name or tlens_model_path."
        return values

    @model_validator(mode="after")
    def check_effective_batch_size(self) -> Self:
        if self.effective_batch_size is not None:
            assert (
                self.effective_batch_size % self.batch_size == 0
            ), "effective_batch_size must be a multiple of batch_size."
        return self

    @model_validator(mode="after")
    def verify_valid_eval_settings(self) -> Self:
        """User can't provide eval_every_n_samples without both eval_n_samples and data.eval."""
        if self.eval_every_n_samples is not None:
            assert (
                self.eval_n_samples is not None and self.eval_data is not None
            ), "Must provide eval_n_samples and data.eval when using eval_every_n_samples."
        return self


def get_run_name(config: Config) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        coeff_info = f"lpcoeff-{config.loss.sparsity.coeff}_"
        if config.loss.out_to_in is not None and config.loss.out_to_in.coeff > 0:
            coeff_info += f"inp-to-out-{config.loss.out_to_in.coeff}_"
        if config.loss.logits_kl is not None and config.loss.logits_kl.coeff > 0:
            coeff_info += f"logits-kl-{config.loss.logits_kl.coeff}_"

        run_suffix = config.wandb_run_name_prefix + (
            f"{'-'.join(config.saes.sae_position_names)}_"
            f"ratio-{config.saes.dict_size_to_input_ratio}_lr-{config.lr}_{coeff_info}"
        )
    return config.wandb_run_name_prefix + run_suffix


@torch.inference_mode()
def evaluate(
    config: Config, model: SAETransformer, device: torch.device, cache_positions: list[str] | None
) -> dict[str, float]:
    """Evaluate the model on the eval dataset.

    Accumulates metrics over the entire eval dataset and then divides by the total number of tokens.

    Returns:
        Dictionary of metrics.
    """
    assert config.eval_data is not None, "No eval dataset specified in the config."
    model.saes.eval()
    eval_loader = create_data_loader(config.eval_data, batch_size=config.batch_size)[0]

    if config.eval_n_samples is None:
        # If streaming (i.e. if the dataset is an IterableDataset), we don't know the length
        n_batches = None if isinstance(eval_loader.dataset, IterableDataset) else len(eval_loader)
    else:
        n_batches = math.ceil(config.eval_n_samples / config.batch_size)

    total_tokens = 0
    # Accumulate metrics over the entire eval dataset and later divide by the total number of tokens
    metrics: dict[str, float] = {}

    for batch_idx, batch in tqdm(enumerate(eval_loader), total=n_batches, desc="Eval Steps"):
        if n_batches is not None and batch_idx >= n_batches:
            break

        tokens = batch[config.eval_data.column_name].to(device=device)
        n_tokens = tokens.shape[0] * tokens.shape[1]
        total_tokens += n_tokens

        # Run through the raw transformer without SAEs
        orig_logits, orig_acts = model.forward_raw(
            tokens=tokens, run_entire_model=True, final_layer=None, cache_positions=cache_positions
        )
        # Run through the SAE-augmented model
        new_logits, new_acts = model.forward(
            tokens=tokens,
            sae_positions=model.raw_sae_position_names,
            cache_positions=cache_positions,
        )
        assert new_logits is not None, "new_logits should not be None during evaluation."

        raw_batch_loss_dict = calc_loss(
            orig_acts=orig_acts,
            new_acts=new_acts,
            orig_logits=orig_logits,
            new_logits=new_logits,
            loss_configs=config.loss,
            is_log_step=True,
            train=False,
        )[1]
        batch_loss_dict = {k: v.item() for k, v in raw_batch_loss_dict.items()}
        batch_output_metrics = calc_output_metrics(
            tokens=tokens, orig_logits=orig_logits, new_logits=new_logits, train=False
        )

        sparsity_metrics = calc_sparsity_metrics(new_acts=new_acts, train=False)

        # Update the global metric dictionary
        for k, v in {**batch_loss_dict, **batch_output_metrics, **sparsity_metrics}.items():
            metrics[k] = metrics.get(k, 0.0) + v * n_tokens

    # Get the mean for all metrics
    for key in metrics:
        metrics[key] /= total_tokens

    model.saes.train()
    return metrics


@logging_redirect_tqdm()
def train(
    config: Config,
    model: SAETransformer,
    train_loader: DataLoader[Samples],
    trainable_param_names: list[str],
    device: torch.device,
    cache_positions: list[str] | None = None,
) -> None:
    model.saes.train()

    layerwise = config.loss.logits_kl is None

    for name, param in model.named_parameters():
        if name.startswith("saes.") and name.split("saes.")[1] in trainable_param_names:
            param.requires_grad = True
        else:
            param.requires_grad = False
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        betas=(config.adam_beta1, 0.999),
    )

    effective_batch_size = config.effective_batch_size or config.batch_size
    n_gradient_accumulation_steps = effective_batch_size // config.batch_size

    lr_schedule = get_linear_lr_schedule(
        warmup_samples=config.warmup_samples,
        cooldown_samples=config.cooldown_samples,
        n_samples=config.n_samples,
        effective_batch_size=effective_batch_size,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    if config.n_samples is None:
        # If streaming (i.e. if the dataset is an IterableDataset), we don't know the length
        n_batches = None if isinstance(train_loader.dataset, IterableDataset) else len(train_loader)
    else:
        n_batches = math.ceil(config.n_samples / config.batch_size)

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
    save_dir = config.save_dir / f"{run_name}_{timestamp}" if config.save_dir else None

    total_samples = 0
    total_samples_at_last_save = 0
    total_samples_at_last_eval = 0
    total_tokens = 0
    grad_updates = 0
    grad_norm: float | None = None
    samples_since_act_frequency_collection: int = 0
    act_frequency_metrics: ActFrequencyMetrics | None = None
    samples_since_output_metric_collection: int = 0

    for batch_idx, batch in tqdm(enumerate(train_loader), total=n_batches, desc="Steps"):
        tokens: Int[Tensor, "batch pos"] = batch[config.train_data.column_name].to(device=device)

        total_samples += tokens.shape[0]
        total_tokens += tokens.shape[0] * tokens.shape[1]
        samples_since_act_frequency_collection += tokens.shape[0]
        samples_since_output_metric_collection += tokens.shape[0]

        run_entire_model: bool = not layerwise
        # Note that is_last_batch will always be False for iterable datasets with n_samples=None. In
        # that case, we will never know when the final batch is reached.
        is_last_batch: bool = n_batches is not None and batch_idx == n_batches - 1
        is_grad_step: bool = (batch_idx + 1) % n_gradient_accumulation_steps == 0
        is_eval_step: bool = config.eval_every_n_samples is not None and (
            (batch_idx == 0)
            or total_samples - total_samples_at_last_eval >= config.eval_every_n_samples
            or is_last_batch
        )
        is_collect_act_frequency_step: bool = config.collect_act_frequency_every_n_samples > 0 and (
            batch_idx == 0
            or (
                samples_since_act_frequency_collection
                >= config.collect_act_frequency_every_n_samples
            )
        )
        is_collect_output_metrics_step: bool = (
            config.collect_output_metrics_every_n_samples > 0
            and (
                batch_idx == 0
                or (
                    samples_since_output_metric_collection
                    >= config.collect_output_metrics_every_n_samples
                )
            )
        )
        is_log_step: bool = (
            batch_idx == 0
            or (is_grad_step and (grad_updates + 1) % config.log_every_n_grad_steps == 0)
            or (layerwise and is_collect_output_metrics_step)
            or is_eval_step
            or is_last_batch
        )
        is_save_model_step: bool = save_dir is not None and (
            (
                config.save_every_n_samples
                and total_samples - total_samples_at_last_save >= config.save_every_n_samples
            )
            or is_last_batch
        )

        if is_collect_output_metrics_step:
            # Running the entire model will output non-None new_logits which output metrics are
            # calculated from
            run_entire_model = True
            samples_since_output_metric_collection = 0

        # Run through the raw transformer without SAEs
        orig_logits, orig_acts = model.forward_raw(
            tokens=tokens,
            run_entire_model=run_entire_model,
            final_layer=final_layer,
            cache_positions=cache_positions,
        )
        # Run through the SAE-augmented model
        new_logits, new_acts = model.forward(
            tokens=tokens,
            sae_positions=model.raw_sae_position_names,
            cache_positions=cache_positions,
            orig_acts=None if run_entire_model else orig_acts,
        )

        loss, loss_dict = calc_loss(
            orig_acts=orig_acts,
            new_acts=new_acts,
            orig_logits=None if new_logits is None else orig_logits.detach().clone(),
            new_logits=new_logits,
            loss_configs=config.loss,
            is_log_step=is_log_step,
        )

        loss = loss / n_gradient_accumulation_steps
        loss.backward()
        if config.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            ).item()

        if is_grad_step:
            optimizer.step()
            optimizer.zero_grad()
            grad_updates += 1
            scheduler.step()

        if is_collect_act_frequency_step and act_frequency_metrics is None:
            # Start collecting activation frequency metrics for next config.act_frequency_n_tokens
            act_frequency_metrics = ActFrequencyMetrics(
                dict_sizes={
                    hook_pos: new_act_pos.c.shape[-1]
                    for hook_pos, new_act_pos in new_acts.items()
                    if isinstance(new_act_pos, SAEActs)
                },
                device=device,
            )
            samples_since_act_frequency_collection = 0

        if act_frequency_metrics is not None:
            act_frequency_metrics.update_dict_el_frequencies(
                new_acts, batch_tokens=tokens.shape[0] * tokens.shape[1]
            )
            if act_frequency_metrics.tokens_used >= config.act_frequency_n_tokens:
                # Finished collecting activation frequency metrics
                metrics = act_frequency_metrics.collect_for_logging(
                    log_wandb_histogram=config.wandb_project is not None
                )
                metrics["total_tokens"] = total_tokens
                if config.wandb_project:
                    # TODO: Log when not using wandb too
                    wandb.log(metrics, step=total_samples)
                act_frequency_metrics = None
                samples_since_act_frequency_collection = 0

        if is_log_step:
            tqdm.write(
                f"Samples {total_samples} Batch_idx {batch_idx} GradUpdates {grad_updates} "
                f"Loss {loss.item():.5f}"
            )
            if config.wandb_project:
                log_info = {
                    "loss": loss.item(),
                    "grad_updates": grad_updates,
                    "total_tokens": total_tokens,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                log_info.update({k: v.item() for k, v in loss_dict.items()})
                if grad_norm is not None:
                    log_info["grad_norm"] = grad_norm

                sparsity_metrics = calc_sparsity_metrics(new_acts=new_acts)
                log_info.update(sparsity_metrics)

                if new_logits is not None:
                    train_output_metrics = calc_output_metrics(
                        tokens=tokens,
                        orig_logits=orig_logits.detach().clone(),
                        new_logits=new_logits.detach().clone(),
                    )
                    log_info.update(train_output_metrics)

                if is_eval_step:
                    eval_metrics = evaluate(
                        config=config, model=model, device=device, cache_positions=cache_positions
                    )
                    total_samples_at_last_eval = total_samples
                    log_info.update(eval_metrics)

                wandb.log(log_info, step=total_samples)

        if is_save_model_step:
            assert save_dir is not None
            total_samples_at_last_save = total_samples
            save_module(
                config_dict=config.model_dump(mode="json"),
                save_dir=save_dir,
                module=model.saes,
                model_path=save_dir / f"samples_{total_samples}.pt",
            )
            if config.wandb_project:
                wandb.save(str(save_dir / f"samples_{total_samples}.pt"))

        if is_last_batch:
            break

    # If the model wasn't saved at the last step of training (which may happen if n_samples: null
    # and the dataset is an IterableDataset), save it now.
    if save_dir and not (save_dir / f"samples_{total_samples}.pt").exists():
        save_module(
            config_dict=config.model_dump(mode="json"),
            save_dir=save_dir,
            module=model.saes,
            model_path=save_dir / f"samples_{total_samples}.pt",
        )
        if config.wandb_project:
            wandb.save(str(save_dir / f"samples_{total_samples}.pt"))

    if config.wandb_project:
        wandb.finish()


def main(config_path_or_obj: Path | str | Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path_or_obj, config_model=Config)
    logger.info(config)
    set_seed(config.seed)

    train_loader = create_data_loader(config.train_data, batch_size=config.batch_size)[0]
    tlens_model = load_tlens_model(
        tlens_model_name=config.tlens_model_name, tlens_model_path=config.tlens_model_path
    )

    raw_sae_position_names = filter_names(
        list(tlens_model.hook_dict.keys()), config.saes.sae_position_names
    )
    # TODO: Use consistent naming for sae positions and cache positions (get rid of "names")
    cache_positions: list[str] | None = None
    if config.loss.in_to_orig is not None:
        cache_positions = filter_names(
            list(tlens_model.hook_dict.keys()), config.loss.in_to_orig.hook_positions
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
        train_loader=train_loader,
        trainable_param_names=trainable_param_names,
        device=device,
        cache_positions=cache_positions,
    )


if __name__ == "__main__":
    fire.Fire(main)
