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
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import lm_cross_entropy_loss

from sparsify.data import DataConfig, create_data_loader
from sparsify.log import logger
from sparsify.losses import LossConfigs, calc_loss
from sparsify.models.sparsifiers import SAE
from sparsify.models.transformers import SAETransformer
from sparsify.scripts.train_tlens.run_train_tlens import HookedTransformerPreConfig
from sparsify.types import RootPath, Samples
from sparsify.utils import (
    load_config,
    save_model,
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
    scheduler: str | None = None
    warmup_steps: NonNegativeFloat = 0
    max_grad_norm: PositiveFloat | None = None
    log_every_n_steps: PositiveInt = 20
    collect_discrete_metrics_every_n_samples: PositiveInt = Field(
        20_000,
        description="Metrics such as activation frequency and alive neurons, are calculated over "
        "discrete periods. This parameter specifies how often to calculate these metrics.",
    )
    discrete_metrics_n_tokens: PositiveInt = Field(
        100_000, description="The number of tokens to caclulate discrete metrics over."
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
    model.saes.train()
    # TODO make appropriate for transcoders and metaSAEs
    optimizer = torch.optim.Adam(model.saes.parameters(), lr=config.train.lr)

    scheduler = None
    if config.train.warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / config.train.warmup_steps),
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
        f"{config.saes.sae_position_name}_lr-{config.train.lr}_warm-{config.train.warmup_steps}_"
        f"ratio-{config.saes.dict_size_to_input_ratio}_"
        f"lpcoeff-{config.train.loss_configs.sparsity.coeff}"
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
    samples_since_discrete_metrics_saved: int = 0
    collect_discrete_metrics = False

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
        samples_since_discrete_metrics_saved += tokens.shape[0]

        if not collect_discrete_metrics and (
            samples_since_discrete_metrics_saved
            >= config.train.collect_discrete_metrics_every_n_samples
        ):
            collect_discrete_metrics = True
            samples_since_discrete_metrics_saved = 0
            tokens_in_discrete_metrics = 0
            dict_el_frequencies: dict[str, list[float]] = {
                hook_name: torch.zeros(sae_acts[hook_name]["c"].shape[-1], device=device)
                for hook_name in sae_acts
            }

        if collect_discrete_metrics:
            batch_tokens = tokens.shape[0] * tokens.shape[1]
            tokens_in_discrete_metrics += batch_tokens

            # batch_dict_el_frequencies = calc_batch_dict_el_frequencies(sae_acts)
            # Update dictionary element features with the current batch
            for hook_name in dict_el_frequencies:
                dict_el_frequencies[hook_name] += (sae_acts[hook_name]["c"] != 0).sum(dim=(0, 1))

        if (
            collect_discrete_metrics
            and tokens_in_discrete_metrics >= config.train.discrete_metrics_n_tokens
        ):
            collect_discrete_metrics = False
            samples_since_discrete_metrics_saved = 0

            # Log the discrete metrics
            discrete_metrics: dict[str, float | list[float]] = {}
            for hook_name, sae_act in sae_acts.items():
                # Divide the dict_el_frequencies by the number of tokens since the last save
                dict_el_frequencies[hook_name] /= tokens_in_discrete_metrics

                discrete_metrics["sparsity/alive_dict_elements"] = (
                    (dict_el_frequencies[hook_name] > 0).sum().item()
                )

                if config.wandb_project:
                    data = [[s] for s in dict_el_frequencies[hook_name]]
                    table = wandb.Table(data=data, columns=["dict element activation frequency"])
                    plot = wandb.plot.histogram(
                        table,
                        "dict element activation frequency",
                        title=f"{hook_name} (most_recent_n_tokens={tokens_in_discrete_metrics} "
                        f"dict_size={sae_act['c'].shape[-1]})",
                    )
                    discrete_metrics[f"sparsity/dict_el_frequencies_hist/{hook_name}"] = plot
            if config.wandb_project:
                # TODO: Log when not using wandb too
                wandb.log(discrete_metrics, step=total_samples)
            # update_dict_el_frequencies(
            #     dict_el_frequencies=dict_el_frequencies,
            #     batch_dict_el_frequencies=batch_dict_el_frequencies,
            #     tokens_since_last_freq_save=tokens_since_discrete_metrics_save,
            #     batch_tokens=batch_tokens,
            # )
            # sparsity_metrics, tokens_since_discrete_metrics_save = calc_sparsity_metrics(
            #     sae_acts=sae_acts,
            #     dict_el_frequencies=dict_el_frequencies,
            #     tokens_since_discrete_metrics_save=tokens_since_discrete_metrics_save,
            #     discrete_metrics_n_tokens=config.train.discrete_metrics_n_tokens,
            #     batch_tokens=batch_tokens,
            #     create_wandb_hist=True,
            # )

        if step == 0 or step % config.train.log_every_n_steps == 0:
            tqdm.write(
                f"Samples {total_samples} Step {step} GradUpdates {grad_updates} "
                f"Loss {loss.item():.5f}"
            )

            if config.wandb_project:
                # sparsity_metrics, tokens_since_discrete_metrics_save = calc_sparsity_metrics(
                #     sae_acts=sae_acts,
                #     dict_el_frequencies=dict_el_frequencies,
                #     tokens_since_discrete_metrics_save=tokens_since_discrete_metrics_save,
                #     discrete_metrics_n_tokens=config.train.discrete_metrics_n_tokens,
                #     batch_tokens=tokens.shape[0] * tokens.shape[1],
                #     create_wandb_hist=True,
                # )
                sparsity_metrics: dict[str, float | list[float]] = {}
                for name, sae_act in sae_acts.items():
                    # Record L_0 norm of the cs
                    l_0_norm = torch.norm(sae_act["c"], p=0, dim=-1).mean()
                    sparsity_metrics[f"sparsity/L_0/{name}"] = l_0_norm

                    # Record fraction of zeros in the cs
                    frac_zeros = (sae_act["c"] == 0).sum() / sae_act["c"].numel()
                    sparsity_metrics[f"sparsity/frac_zeros/{name}"] = frac_zeros

                wandb_log_info = {
                    "loss": loss.item(),
                    "grad_updates": grad_updates,
                    **sparsity_metrics,
                }

                for loss_name, loss_value in loss_dict.items():
                    wandb_log_info[loss_name] = loss_value.item()

                if config.train.max_grad_norm is not None:
                    assert grad_norm is not None
                    wandb_log_info["grad_norm"] = grad_norm
                if step == 0 or step % 5 == 0:
                    orig_logits_logging = orig_logits.detach().clone()
                    new_logits_logging = new_logits.detach().clone()
                    orig_model_performance_loss = lm_cross_entropy_loss(
                        orig_logits_logging, tokens, per_token=False
                    )
                    sae_model_performance_loss = lm_cross_entropy_loss(
                        new_logits_logging, tokens, per_token=False
                    )

                    wandb_log_info.update(
                        {
                            "performance/orig_model_ce_loss": orig_model_performance_loss.item(),
                            "performance/sae_model_ce_loss": sae_model_performance_loss.item(),
                            "performance/difference_loss": (
                                orig_model_performance_loss - sae_model_performance_loss
                            ).item(),
                        },
                    )
                wandb.log(wandb_log_info, step=total_samples)
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
    if config.wandb_project:
        wandb.finish()


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


def main(config_path_or_obj: Path | str | Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path_or_obj, config_model=Config)
    logger.info(config)
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
