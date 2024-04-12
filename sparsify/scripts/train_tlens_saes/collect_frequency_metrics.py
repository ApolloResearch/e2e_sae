"""Script for calculating frequency metrics for a trained SAETransformer model.

This script will resume a wandb run, store the metrics, and then finish the run again.

Usage:
    python collect_frequency_metrics.py <wandb_project> <wandb_run_id> \
        [--batch_size=<batch_size>] [--n_tokens=<n_tokens>]
"""


import fire
import torch
import wandb
import yaml
from wandb.apis.public import Run

from sparsify.loader import load_tlens_model
from sparsify.log import logger
from sparsify.metrics import collect_act_frequency_metrics
from sparsify.models.transformers import SAETransformer
from sparsify.scripts.train_tlens_saes.run_train_tlens_saes import Config as TrainConfig
from sparsify.utils import filter_names, set_seed


@torch.inference_mode()
def main(
    wandb_project: str,
    wandb_run_id: str,
    batch_size: int | None = None,
    n_tokens: int | None = None,
):
    """Load a trained SAETransformer model from wandb and calculate frequency metrics.

    Frequency matrics are uploaded to the same wandb run.

    Args:
        wandb_project: The wandb project name.
        wandb_run_id: The wandb run ID.
        batch_size: The batch size to use for calculating the frequency metrics. If not provided,
            the batch size from the training run (batch_size) is used.
        n_tokens: The number of tokens to use for calculating the frequency metrics. If not
            provided, the number of tokens from the training run (act_frequency_n_tokens) is used.
    """
    wandb_run_name = f"{wandb_project}/{wandb_run_id}"

    logger.info(f"Collecting frequency metrics for {wandb_run_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    api = wandb.Api()
    run: Run = api.run(wandb_run_name)

    train_config_files = [file for file in run.files() if file.name.endswith("final_config.yaml")]

    assert len(train_config_files) == 1
    train_config = TrainConfig(
        **yaml.safe_load(
            train_config_files[0].download(replace=False, exist_ok=True, root=f"/tmp/{run.id}/")
        )
    )

    tlens_model = load_tlens_model(
        tlens_model_name=train_config.tlens_model_name,
        tlens_model_path=train_config.tlens_model_path,
    )

    raw_sae_positions = filter_names(
        list(tlens_model.hook_dict.keys()), train_config.saes.sae_positions
    )
    model = SAETransformer(
        tlens_model=tlens_model,
        raw_sae_positions=raw_sae_positions,
        dict_size_to_input_ratio=train_config.saes.dict_size_to_input_ratio,
    ).to(device=device)

    # Weights file should be the largest .pt file. All have format (samples_*.pt)
    weight_files = [file for file in run.files() if file.name.endswith(".pt")]
    # Latest checkpoint
    weight_file = sorted(weight_files, key=lambda x: int(x.name.split(".pt")[0].split("_")[-1]))[-1]
    latest_checkpoint = wandb.restore(
        weight_file.name, run_path=wandb_run_name, root=f"/tmp/{wandb_run_id}", replace=False
    )
    assert latest_checkpoint is not None
    model.saes.load_state_dict(torch.load(latest_checkpoint.name, map_location=device))
    model.saes.eval()

    n_tokens = n_tokens or train_config.act_frequency_n_tokens
    batch_size = batch_size or train_config.batch_size
    set_seed(train_config.seed)

    metrics = collect_act_frequency_metrics(
        model=model,
        data_config=train_config.train_data,
        batch_size=batch_size,
        global_seed=train_config.seed,
        device=device,
        n_tokens=n_tokens,
    )
    # Resume wandb run and log metrics
    wandb.init(
        project=wandb_project,
        id=wandb_run_id,
        resume=True,
    )
    wandb.log(metrics)
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
