from functools import partial
from typing import Any

import fire
import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import lm_cross_entropy_loss, lm_accuracy

from sparsify.configs import Config
from sparsify.losses import calc_loss
from sparsify.models.sparsifiers import SAE
from sparsify.models.transformers import SAETransformer
from sparsify.data import get_train_loader
from sparsify.utils import load_config
import wandb
from tqdm.auto import tqdm


def sae_hook(
    value: Float[torch.Tensor, "... d_head"],
    hook: HookPoint,
    sae: SAE,
    hook_acts: dict[str, Any],
) -> Float[torch.Tensor, "... d_head"]:
    """Runs the SAE on the input and stores the output and c in hook_acts."""
    hook_acts["input"] = value
    output, c = sae(value)
    hook_acts["output"] = output
    hook_acts["c"] = c
    return output


def train(config: Config, model: SAETransformer, device: torch.device) -> None:
    torch.manual_seed(config.train.seed)

    if config.wandb:
        timestamp = wandb.util.generate_id()
        run_name = f"{config.tlens_model_name}_lambda{config.train.act_sparsity_lambda}_Lp{config.train.sparsity_p_norm}_lr{config.train.lr}_{timestamp}"
        wandb.init(project=config.wandb.project, config=vars(config), name=run_name)

    optimizer = torch.optim.Adam(
        model.saes.parameters(), lr=config.train.lr
    )  # TODO make appropriate for transcoders and metaSAEs

    scheduler = None
    if config.train.warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / config.train.warmup_steps),
        )

    tokenizer = model.tlens_model.tokenizer
    assert tokenizer is not None, "Tokenizer must be defined for training."

    train_loader = get_train_loader(tokenizer, config)

    sae_acts = {layer: {} for layer in model.saes.keys()}

    samples = 0
    grad_updates = 0
    n_gradient_accumulation_steps = (
        config.train.effective_batch_size // config.train.batch_size
    )
    orig_resid_names = lambda name: model.sae_position_name in name
    for epoch in tqdm(range(1, config.train.num_epochs + 1)):
        # Now do gradient accumulation

        for step, batch in tqdm(enumerate(train_loader)):
            update_step_bool = (step + 1) % n_gradient_accumulation_steps == 0
            tokens = batch["tokens"].to(device=device)  # (B, T)

            # Run model without SAEs
            with torch.inference_mode():
                orig_logits, orig_acts = model.tlens_model.run_with_cache(
                    tokens, names_filter=orig_resid_names
                )  # (B, T, d_model), dict[str, Tensor]

            # Run model with SAEs
            sae_acts = {hook_name: {} for hook_name in orig_acts}
            fwd_hooks = [
                (
                    hook_name,
                    partial(
                        sae_hook, sae=model.saes[str(i)], hook_acts=sae_acts[hook_name]
                    ),
                )
                for i, hook_name in enumerate(orig_acts)
            ]
            new_logits = model.tlens_model.run_with_hooks(
                tokens, fwd_hooks=fwd_hooks
            )  # (B, T, d_model)

            loss, loss_dict = calc_loss(  # Directly optimized
                config=config,
                orig_acts=orig_acts,
                sae_acts=sae_acts,
                orig_logits=orig_logits,
                new_logits=new_logits,
            )

            loss = loss / n_gradient_accumulation_steps

            loss.backward()
            if config.train.max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.train.max_grad_norm
                )

            if update_step_bool:
                optimizer.step()
                optimizer.zero_grad()
                grad_updates += 1

                if config.train.warmup_steps > 0:
                    assert scheduler is not None
                    scheduler.step()

            samples += tokens.shape[0]

            if step == 0 or update_step_bool:
                print(
                    f"Epoch {epoch} Samples {samples} Step {step} GradUpdates {grad_updates} Loss {loss.item()}"
                )

            if config.wandb and (step == 0 or update_step_bool):
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "samples": samples,
                        "epoch": epoch,
                        "grad_updates": grad_updates,
                    }
                )
                for loss_name, loss_value in loss_dict.items():
                    wandb.log(
                        {
                            loss_name: loss_value.item(),
                            "samples": samples,
                            "epoch": epoch,
                            "grad_updates": grad_updates,
                        }
                    )

                if config.train.max_grad_norm is not None:
                    wandb.log(
                        {
                            "grad_norm": grad_norm.item(),
                            "samples": samples,
                            "epoch": epoch,
                            "grad_updates": grad_updates,
                        }
                    )

                orig_model_performance_loss = lm_cross_entropy_loss(
                    orig_logits, tokens, per_token=False
                )
                orig_model_performance_acc = lm_accuracy(
                    orig_logits, tokens, per_token=False
                )
                sae_model_performance_loss = lm_cross_entropy_loss(
                    new_logits, tokens, per_token=False
                )
                sae_model_performance_acc = lm_accuracy(
                    new_logits, tokens, per_token=False
                )
                # flat_orig_logits = orig_logits.view(-1, orig_logits.shape[-1])
                # flat_new_logits = new_logits.view(-1, new_logits.shape[-1])
                # kl_div = torch.nn.functional.kl_div(
                #     torch.nn.functional.log_softmax(flat_new_logits, dim=-1),
                #     torch.nn.functional.softmax(flat_orig_logits, dim=-1),
                #     reduction="batchmean",
                # ) # Unsure if this is correct. Also it's expensive in terms of memory.

                # Log performance metrics
                time_marker_dict = {
                    "samples": samples,
                    "epoch": epoch,
                    "grad_updates": grad_updates,
                }
                performance_dict = {
                    "orig_model_performance_loss": orig_model_performance_loss.item(),
                    "orig_model_performance_acc": orig_model_performance_acc.item(),
                    "sae_model_performance_loss": sae_model_performance_loss.item(),
                    "sae_model_performance_acc": sae_model_performance_acc.item(),
                    "difference_loss": (
                        orig_model_performance_loss - sae_model_performance_loss
                    ).item(),
                    "difference_acc": (
                        orig_model_performance_acc - sae_model_performance_acc
                    ).item(),
                    # "kl_div": kl_div.item(),
                }
                for metric_name, metric_value in performance_dict.items():
                    metric_dict = {
                        f"performance/{metric_name}": metric_value,
                    }
                    metric_dict.update(time_marker_dict)
                    wandb.log(metric_dict)

            # TODO sae saving


def main(config_path_str: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path_str, config_model=Config)

    seed = config.train.seed
    torch.manual_seed(seed)
    if config.tlens_model_name is not None:
        tlens_model = HookedTransformer.from_pretrained(config.tlens_model_name)
    else:
        hooked_transformer_config = HookedTransformerConfig(
            **config.tlens_config.model_dump()
        )
        tlens_model = HookedTransformer(hooked_transformer_config)

    model = SAETransformer(tlens_model, config).to(device=device)
    train(config, model, device=device)


if __name__ == "__main__":
    fire.Fire(main)
