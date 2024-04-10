import pandas as pd
from wandb.apis.public.runs import Runs


def _get_run_type(kl_coeff: float | None, in_to_orig_coeff: float | None) -> str:
    if (
        kl_coeff is not None
        and in_to_orig_coeff is not None
        and kl_coeff > 0
        and in_to_orig_coeff > 0
    ):
        return "e2e-recon"
    if kl_coeff is not None and kl_coeff > 0:
        return "e2e"
    return "local"


def create_run_df(runs: Runs) -> pd.DataFrame:
    run_info = []
    for run in runs:
        if run.state != "finished":
            print(f"Run {run.name} is not finished, skipping")
            continue
        sae_pos = run.config["saes"]["sae_positions"]
        if isinstance(sae_pos, list):
            if len(sae_pos) > 1:
                raise ValueError("More than one SAE position found")
            sae_pos = sae_pos[0]
        sae_layer = int(sae_pos.split(".")[1])

        kl_coeff = None
        in_to_orig_coeff = None
        if "logits_kl" in run.config["loss"] and run.config["loss"]["logits_kl"] is not None:
            kl_coeff = run.config["loss"]["logits_kl"]["coeff"]
        if "in_to_orig" in run.config["loss"] and run.config["loss"]["in_to_orig"] is not None:
            in_to_orig_coeff = run.config["loss"]["in_to_orig"]["total_coeff"]

        run_type = _get_run_type(kl_coeff, in_to_orig_coeff)

        # The explained variance at each layer
        explained_var_layers = {
            f"explained_var_layer-{key.split('blocks.')[1].split('.')[0]}": value
            for key, value in run.summary_metrics.items()
            if key.startswith("loss/eval/in_to_orig/explained_variance/blocks")
        }

        # Explained variance ln at each layer
        explained_var_ln_layers = {
            f"explained_var_ln_layer-{key.split('blocks.')[1].split('.')[0]}": value
            for key, value in run.summary_metrics.items()
            if key.startswith("loss/eval/in_to_orig/explained_variance_ln/blocks")
        }

        # Reconstruction loss at each layer
        recon_loss_layers = {
            f"recon_loss_layer-{key.split('blocks.')[1].split('.')[0]}": value
            for key, value in run.summary_metrics.items()
            if key.startswith("loss/eval/in_to_orig/blocks")
        }

        if "dict_size_to_input_ratio" in run.config["saes"]:
            ratio = float(run.config["saes"]["dict_size_to_input_ratio"])
        else:
            # local runs didn't store the ratio in the config for these runs
            ratio = float(run.name.split("ratio-")[1].split("_")[0])

        out_to_in = None
        explained_var = None
        explained_var_ln = None
        if f"loss/eval/out_to_in/{sae_pos}" in run.summary_metrics:
            out_to_in = run.summary_metrics[f"loss/eval/out_to_in/{sae_pos}"]
            explained_var = run.summary_metrics[f"loss/eval/out_to_in/explained_variance/{sae_pos}"]
            explained_var_ln = run.summary_metrics[
                f"loss/eval/out_to_in/explained_variance_ln/{sae_pos}"
            ]

        run_info.append(
            {
                "name": run.name,
                "id": run.id,
                "run_type": run_type,
                "layer": sae_layer,
                "seed": run.config["seed"],
                "n_samples": run.config["n_samples"],
                "lr": run.config["lr"],
                "ratio": ratio,
                "sparsity_coeff": run.config["loss"]["sparsity"]["coeff"],
                "in_to_orig_coeff": in_to_orig_coeff,
                "kl_coeff": kl_coeff,
                "out_to_in": out_to_in,
                "L0": run.summary_metrics[f"sparsity/eval/L_0/{sae_pos}"],
                "explained_var": explained_var,
                "explained_var_ln": explained_var_ln,
                "CE_diff": run.summary_metrics["performance/eval/difference_ce_loss"],
                "alive_dict_elements": run.summary_metrics[
                    f"sparsity/alive_dict_elements/{sae_pos}"
                ],
                **explained_var_layers,
                **explained_var_ln_layers,
                **recon_loss_layers,
                "sum_recon_loss": sum(recon_loss_layers.values()),
                "kl": run.summary_metrics["loss/eval/logits_kl"],
            }
        )
    df = pd.DataFrame(run_info)
    return df
