"""
This is a script to download the SAE files from the wandb runs.

It's not fully reliable:
* Doesn't currently handle recon_ runs with both e2e and reconstruction loss
* Doesn't properly download config for all runs, as the wandb configs aren't reliable
"""

# %%
from pathlib import Path
from typing import Literal

import wandb
import yaml
from wandb.apis.public import Run


def filter_runs(
    runs: list[Run],
    type: Literal["e2e", "layerwise"] | None,
    block: Literal[2, 6, 10] | None,
    seed: Literal[0, 1] | None = 0,
    lp_coeff: float | None = None,
) -> list[Run]:
    filtered = []
    for run in runs:
        # don't handle for now, can incorporate later
        if run.name.startswith("recon_"):
            pass
        correct_seed = (seed is None) or (run.config["seed"] == seed)
        correct_block = (block is None) or (
            run.config["saes"]["sae_positions"] == f"blocks.{block}.hook_resid_pre"
        )
        correct_type = (type is None) or (("logits-kl" in run.name) == (type == "e2e"))
        correct_lp_coeff = (lp_coeff is None) or (
            run.config["loss"]["sparsity"]["coeff"] == lp_coeff
        )
        if correct_seed and correct_block and correct_type and correct_lp_coeff:
            filtered.append(run)
    return filtered


def download_saes_for_run(run: Run, out_dir: Path):
    sae_file = run.file(f"samples_{run.config['n_samples']}.pt")
    sae_file.download(out_dir / run.name, replace=True)  # type: ignore
    with open(out_dir / run.name / "config.yaml", "w") as f:
        yaml.dump(run.config, f)
    print("Downloaded into ", out_dir / run.name)


def main(
    type: Literal["e2e", "layerwise"],
    block: Literal[2, 6, 10],
    seed: Literal[0, 1] = 0,
    lp_coeff: float | None = None,
    out_dir: Path | None = None,
):
    out_dir = Path(__file__).parent.parent / "train_tlens_saes/out" if out_dir is None else out_dir
    api = wandb.Api()
    runs = api.runs("sparsify/gpt2")
    filtered_runs = filter_runs(runs, type, block, seed, lp_coeff)
    print(len(filtered_runs), filtered_runs)
    print(out_dir)
    for run in filtered_runs:
        print(run.name)
    # download_saes_for_run(run, out_dir)


main("e2e", block=6, seed=0)


# %%
# %%
