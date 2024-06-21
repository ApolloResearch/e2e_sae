"""Script for training SAEs on top of a transformerlens model.

Usage:
    python run_train_tlens_saes.py <path/to/config.yaml>
"""
import math
import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Literal, Self

import fire
import torch
import wandb
import yaml
from datasets import IterableDataset
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

from e2e_sae.data import DatasetConfig, create_data_loader
from e2e_sae.hooks import SAEActs
from e2e_sae.loader import load_pretrained_saes, load_tlens_model
from e2e_sae.log import logger
from e2e_sae.losses import LossConfigs, calc_loss
from e2e_sae.metrics import (
    ActFrequencyMetrics,
    calc_output_metrics,
    calc_sparsity_metrics,
    collect_act_frequency_metrics,
)
from e2e_sae.models.transformers import SAETransformer
from e2e_sae.types import RootPath, Samples
from e2e_sae.utils import (
    filter_names,
    get_cosine_schedule_with_warmup,
    get_linear_lr_schedule,
    init_wandb,
    load_config,
    replace_pydantic_model,
    save_module,
    set_seed,
)


class SAEsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dict_size_to_input_ratio: PositiveFloat = 1.0
    pretrained_sae_paths: Annotated[
        list[RootPath] | None, BeforeValidator(lambda x: [x] if isinstance(x, str | Path) else x)
    ] = Field(None, description="Path to a pretrained SAEs to load. If None, don't load any.")
    retrain_saes: bool = Field(False, description="Whether to retrain the pretrained SAEs.")
    sae_positions: Annotated[
        list[str], BeforeValidator(lambda x: [x] if isinstance(x, str) else x)
    ] = Field(
        ...,
        description="The names of the hook positions to train SAEs on. E.g. 'hook_resid_post' or "
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
    seed: NonNegativeInt = Field(
        0,
        description="Seed set at start of script. Also used for train_data.seed and eval_data.seed "
        "if they are not set explicitly.",
    )
    tlens_model_name: str | None = None
    tlens_model_path: RootPath | None = Field(
        None,
        description="Path to '.pt' checkpoint. The directory housing this file should also contain "
        "'final_config.yaml' which is output by e2e_sae/scripts/train_tlens/run_train_tlens.py.",
    )
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
    lr_schedule: Literal["linear", "cosine"] = "cosine"
    min_lr_factor: NonNegativeFloat = Field(
        0.1,
        description="The minimum learning rate as a factor of the initial learning rate. Used "
        "in the cooldown phase of a linear or cosine schedule.",
    )
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
    loss: LossConfigs
    train_data: DatasetConfig
    eval_data: DatasetConfig | None = None
    saes: SAEsConfig

    @model_validator(mode="before")
    @classmethod
    def remove_deprecated_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Remove fields that are no longer used."""
        values.pop("collect_output_metrics_every_n_samples", None)
        return values

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

    @model_validator(mode="after")
    def cosine_schedule_requirements(self) -> Self:
        """Cosine schedule must have n_samples set in order to define the cosine curve."""
        if self.lr_schedule == "cosine":
            assert self.n_samples is not None, "Cosine schedule requires n_samples."
            assert self.cooldown_samples == 0, "Cosine schedule must not have cooldown_samples."
        return self


def get_run_name(config: Config) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        coeff_info = f"seed-{config.seed}_lpcoeff-{config.loss.sparsity.coeff}"
        if config.loss.out_to_in is not None and config.loss.out_to_in.coeff > 0:
            coeff_info += f"_in-to-out-{config.loss.out_to_in.coeff}"
        if config.loss.logits_kl is not None and config.loss.logits_kl.coeff > 0:
            coeff_info += f"_logits-kl-{config.loss.logits_kl.coeff}"
        if config.loss.in_to_orig is not None and config.loss.in_to_orig.total_coeff > 0:
            coeff_info += f"_in-to-orig-{config.loss.in_to_orig.total_coeff}"

        run_suffix = (
            f"{coeff_info}_lr-{config.lr}_ratio-{config.saes.dict_size_to_input_ratio}_"
            f"{'-'.join(config.saes.sae_positions)}"
        )
    return config.wandb_run_name_prefix + run_suffix


@torch.inference_mode()
def evaluate(
    config: Config,
    model: SAETransformer,
    device: torch.device,
    cache_positions: list[str] | None,
    log_resid_reconstruction: bool = True,
) -> dict[str, float]:
    """Evaluate the model on the eval dataset.

    Accumulates metrics over the entire eval dataset and then divides by the total number of tokens.

    Args:
        config: The config object.
        model: The SAETransformer model.
        device: The device to run the model on.
        cache_positions: The positions to cache activations at.
        log_resid_reconstruction: Whether to log the reconstruction loss and explained variance
            at hook_resid_post in all layers.
    Returns:
        Dictionary of metrics.
    """
    model.saes.eval()

    eval_config = config
    eval_cache_positions = cache_positions
    eval_loss_config_updates = {}
    if log_resid_reconstruction and config.loss.in_to_orig is None:
        # Update cache_positions with all hook_resid_post positions
        all_resids = [f"blocks.{i}.hook_resid_post" for i in range(model.tlens_model.cfg.n_layers)]
        # Record the reconstruction loss and explained var at hook_resid_post by setting
        # in_to_orig.total_coeff to 0.0
        eval_loss_config_updates.update(
            {"in_to_orig": {"hook_positions": all_resids, "total_coeff": 0.0}}
        )
        eval_cache_positions = list(
            set(all_resids) | (set(cache_positions) if cache_positions else set())
        )
    if config.loss.logits_kl is None:
        # If we're not training with logits_kl ensure that we eval with it
        eval_loss_config_updates.update({"logits_kl": {"coeff": 0.0}})

    # Use a different seed for evaluation than for training if eval seed not explicitly set
    eval_config = replace_pydantic_model(
        config,
        {"loss": eval_loss_config_updates, "seed": config.seed + 42},
    )

    assert eval_config.eval_data is not None, "No eval dataset specified in the config."
    eval_loader = create_data_loader(
        eval_config.eval_data, batch_size=eval_config.batch_size, global_seed=eval_config.seed
    )[0]

    if eval_config.eval_n_samples is None:
        # If streaming (i.e. if the dataset is an IterableDataset), we don't know the length
        n_batches = None if isinstance(eval_loader.dataset, IterableDataset) else len(eval_loader)
    else:
        n_batches = math.ceil(eval_config.eval_n_samples / eval_config.batch_size)

    total_tokens = 0
    # Accumulate metrics over the entire eval dataset and later divide by the total number of tokens
    metrics: dict[str, float] = {}

    for batch_idx, batch in tqdm(enumerate(eval_loader), total=n_batches, desc="Eval Steps"):
        if n_batches is not None and batch_idx >= n_batches:
            break

        tokens = batch[eval_config.eval_data.column_name].to(device=device)
        n_tokens = tokens.shape[0] * tokens.shape[1]
        total_tokens += n_tokens

        # Run through the raw transformer without SAEs
        orig_logits, orig_acts = model.forward_raw(
            tokens=tokens,
            run_entire_model=True,
            final_layer=None,
            cache_positions=eval_cache_positions,
        )
        # Run through the SAE-augmented model
        new_logits, new_acts = model.forward(
            tokens=tokens,
            sae_positions=model.raw_sae_positions,
            cache_positions=eval_cache_positions,
        )
        assert new_logits is not None, "new_logits should not be None during evaluation."

        raw_batch_loss_dict = calc_loss(
            orig_acts=orig_acts,
            new_acts=new_acts,
            orig_logits=orig_logits,
            new_logits=new_logits,
            loss_configs=eval_config.loss,
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

    is_local = config.loss.logits_kl is None and cache_positions is None

    for name, param in model.named_parameters():
        if name.startswith("saes.") and name.split("saes.")[1] in trainable_param_names:
            param.requires_grad = True
        else:
            param.requires_grad = False
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr
    )

    effective_batch_size = config.effective_batch_size or config.batch_size
    n_gradient_accumulation_steps = effective_batch_size // config.batch_size

    if config.lr_schedule == "cosine":
        assert config.n_samples is not None, "Cosine schedule requires n_samples."
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_samples // effective_batch_size,
            num_training_steps=config.n_samples // effective_batch_size,
            min_lr_factor=config.min_lr_factor,
        )
    else:
        assert config.lr_schedule == "linear"
        lr_schedule = get_linear_lr_schedule(
            warmup_samples=config.warmup_samples,
            cooldown_samples=config.cooldown_samples,
            n_samples=config.n_samples,
            effective_batch_size=effective_batch_size,
            min_lr_factor=config.min_lr_factor,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    if config.n_samples is None:
        # If streaming (i.e. if the dataset is an IterableDataset), we don't know the length
        n_batches = None if isinstance(train_loader.dataset, IterableDataset) else len(train_loader)
    else:
        n_batches = math.ceil(config.n_samples / config.batch_size)

    final_layer = None
    if all(name.startswith("blocks.") for name in model.raw_sae_positions) and is_local:
        # We don't need to run through the whole model for local runs
        final_layer = max([int(name.split(".")[1]) for name in model.raw_sae_positions]) + 1

    run_name = get_run_name(config)
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before calling train."
        wandb.run.name = run_name

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

    for batch_idx, batch in tqdm(enumerate(train_loader), total=n_batches, desc="Steps"):
        tokens: Int[Tensor, "batch pos"] = batch[config.train_data.column_name].to(device=device)

        total_samples += tokens.shape[0]
        total_tokens += tokens.shape[0] * tokens.shape[1]
        samples_since_act_frequency_collection += tokens.shape[0]

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
        is_log_step: bool = (
            batch_idx == 0
            or (is_grad_step and (grad_updates + 1) % config.log_every_n_grad_steps == 0)
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

        # Run through the raw transformer without SAEs
        orig_logits, orig_acts = model.forward_raw(
            tokens=tokens,
            run_entire_model=not is_local,
            final_layer=final_layer,
            cache_positions=cache_positions,
        )
        # Run through the SAE-augmented model
        new_logits, new_acts = model.forward(
            tokens=tokens,
            sae_positions=model.raw_sae_positions,
            cache_positions=cache_positions,
            orig_acts=None if not is_local else orig_acts,
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

        if is_grad_step:
            if config.max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.saes.parameters(), config.max_grad_norm
                ).item()
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
                    log_info["grad_norm"] = grad_norm  # Norm of grad before clipping

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
                model_filename=f"samples_{total_samples}.pt",
                config_filename="final_config.yaml",
            )
            if config.wandb_project:
                wandb.save(
                    str(save_dir / f"samples_{total_samples}.pt"), policy="now", base_path=save_dir
                )

        if is_last_batch:
            break

    # If the model wasn't saved at the last step of training (which may happen if n_samples: null
    # and the dataset is an IterableDataset), save it now.
    if save_dir and not (save_dir / f"samples_{total_samples}.pt").exists():
        save_module(
            config_dict=config.model_dump(mode="json"),
            save_dir=save_dir,
            module=model.saes,
            model_filename=f"samples_{total_samples}.pt",
            config_filename="final_config.yaml",
        )
        if config.wandb_project:
            wandb.save(
                str(save_dir / f"samples_{total_samples}.pt"), policy="now", base_path=save_dir
            )

    if config.wandb_project:
        # Collect and log final activation frequency metrics
        metrics = collect_act_frequency_metrics(
            model=model,
            data_config=config.train_data,
            batch_size=config.batch_size // 2,  # Hack to prevent OOM. TODO: Solve this properly
            global_seed=config.seed,
            device=device,
            n_tokens=config.act_frequency_n_tokens,
        )
        wandb.log(metrics)
        wandb.finish()


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)
        # Save the config to wandb
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "final_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config.model_dump(mode="json"), f, indent=2)
            wandb.save(str(config_path), policy="now", base_path=tmp_dir)
            # Unfortunately wandb.save is async, so we need to wait for it to finish before
            # continuing, and wandb python api provides no way to do this.
            # TODO: Find a better way to do this.
            time.sleep(1)

    set_seed(config.seed)
    logger.info(config)

    train_loader = create_data_loader(
        config.train_data, batch_size=config.batch_size, global_seed=config.seed
    )[0]
    tlens_model = load_tlens_model(
        tlens_model_name=config.tlens_model_name, tlens_model_path=config.tlens_model_path
    )

    raw_sae_positions = filter_names(list(tlens_model.hook_dict.keys()), config.saes.sae_positions)
    cache_positions: list[str] | None = None
    if config.loss.in_to_orig is not None:
        assert set(config.loss.in_to_orig.hook_positions).issubset(
            set(tlens_model.hook_dict.keys())
        ), "Some hook_positions in config.loss.in_to_orig.hook_positions are not in the model."
        # Don't add a cache position if there is already an SAE at that position which will cache
        # the inputs anyway
        cache_positions = [
            pos for pos in config.loss.in_to_orig.hook_positions if pos not in raw_sae_positions
        ]

    model = SAETransformer(
        tlens_model=tlens_model,
        raw_sae_positions=raw_sae_positions,
        dict_size_to_input_ratio=config.saes.dict_size_to_input_ratio,
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
