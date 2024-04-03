from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
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
    return "layerwise"


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
            # layerwise runs didn't store the ratio in the config for these runs
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


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_file: Path,
    z: str | None = None,
    xlim: tuple[float | None, float | None] = (None, None),
    ylim: tuple[float | None, float | None] = (None, None),
):
    """Plot a scatter plot with the specified x and y variables, colored by run type or z.

    Args:
        df: DataFrame containing the data.
        x: The variable to plot on the x-axis.
        y: The variable to plot on the y-axis.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        out_file: The filename which the plot will be saved as.
        z: The variable to use for coloring the points. If not provided, points will be
            colored by run type.
        xlim: The x-axis limits.
        ylim: The y-axis limits.
    """
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#f5f6fc"})
    fig, ax = plt.subplots(figsize=(8, 6))
    run_type_map = {
        "e2e": ("End-to-end", "o"),
        "e2e-recon": ("End-to-end-recon", "s"),
        "layerwise": ("Layerwise", "^"),
    }

    cmap = "plasma_r"
    norm, vmin, vmax = None, None, None
    if z is not None:
        vmin = int(10 ** np.floor(np.log10(df[z].min())))
        vmax = int(10 ** np.ceil(np.log10(df[z].max())))
        norm = LogNorm(vmin=vmin, vmax=vmax)

    for run_type, (label, marker) in run_type_map.items():
        data = df[df["run_type"] == run_type]
        assert isinstance(data, pd.DataFrame)
        if not data.empty:
            if z is None:
                sns.scatterplot(
                    data=data,
                    x=x,
                    y=y,
                    label=label,
                    marker=marker,
                    s=95,
                    linewidth=1.1,
                    ax=ax,
                )
            else:
                sns.scatterplot(
                    data=data,
                    x=x,
                    y=y,
                    c=data[z],
                    marker=marker,
                    s=95,
                    linewidth=1.1,
                    cmap=cmap,
                    norm=norm,
                    ax=ax,
                )

    for _, row in df.iterrows():
        ax.text(
            row[x] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01,
            float(row[y]),
            f"{row['sparsity_coeff']}",
            fontsize=8,
            ha="left",
            va="center",
            color="black",
            alpha=0.8,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if z is None:
        ax.legend(title="Run Type", loc="best")
    else:
        mappable = ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, label=z)
        assert vmin is not None
        assert vmax is not None
        num_ticklabels = int(np.log10(vmax) - np.log10(vmin)) + 1
        cbar.set_ticks(np.logspace(np.log10(vmin), np.log10(vmax), num=num_ticklabels))
        cbar.set_ticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in cbar.get_ticks()])

        # Create legend handles manually so we can color the markers black
        legend_handles = []
        for run_type, (label, marker) in run_type_map.items():
            if df[df["run_type"] == run_type].empty:
                continue
            legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    marker=marker,
                    color="black",
                    linestyle="None",
                    markersize=8,
                    label=label,
                )
            )
        ax.legend(handles=legend_handles, title="Run Type", loc="best")

    plt.tight_layout()
    plt.savefig(out_file)


if __name__ == "__main__":
    # Plot gpt2 performance
    api = wandb.Api()
    project = "sparsify/gpt2"
    runs = api.runs(project)

    df = create_run_df(runs)

    # These runs were done with the following config:
    for col in ["lr", "n_samples", "ratio"]:
        assert df[col].nunique() == 1
    assert df["lr"].unique()[0] == 5e-4
    assert df["n_samples"].unique()[0] == 400_000
    assert df["ratio"].unique()[0] == 60

    # Only use seed=0
    df = df[df["seed"] == 0]
    print(df)
    assert isinstance(df, pd.DataFrame)

    unique_layers = list(df["layer"].unique())

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    for layer in unique_layers:
        layer_df = df[df["layer"] == layer]
        assert isinstance(layer_df, pd.DataFrame)
        plot_scatter(
            layer_df,
            x="L0",
            y="CE_diff",
            title=f"Layer {layer}: L0 vs CE Loss Difference (label: sparsity coeff)",
            xlabel="L0",
            ylabel="CE loss difference\n(original model - model with sae)",
            out_file=out_dir / f"l0_vs_ce_loss_layer_{layer}.png",
        )
        plot_scatter(
            layer_df,
            x="alive_dict_elements",
            y="CE_diff",
            title=f"Layer {layer}: Alive Dictionary Elements vs CE Loss Difference (label: sparsity coeff)",
            xlabel="Alive Dictionary Elements",
            ylabel="CE loss difference\n(original model - model with sae)",
            out_file=out_dir / f"alive_elements_vs_ce_loss_layer_{layer}.png",
        )
        plot_scatter(
            layer_df,
            x="out_to_in",
            y="CE_diff",
            title=f"Layer {layer}: Out-to-In Loss vs CE Loss Difference (label: sparsity coeff)",
            xlabel="Out-to-In Loss",
            ylabel="CE loss difference\n(original model - model with sae)",
            out_file=out_dir / f"out_to_in_vs_ce_loss_layer_{layer}.png",
        )
        plot_scatter(
            layer_df,
            x="sum_recon_loss",
            y="CE_diff",
            title=f"Layer {layer}: Future Reconstruction Loss vs CE Loss Difference (label: sparsity coeff)",
            xlabel="Future Reconstruction Loss",
            ylabel="CE loss difference\n(original model - model with sae)",
            out_file=out_dir / f"future_recon_vs_ce_loss_layer_{layer}.png",
        )
        plot_scatter(
            layer_df,
            x="explained_var_ln",
            y="CE_diff",
            z="L0",
            title=f"Layer {layer}: Explained Variance LN vs CE Loss Difference (label: sparsity coeff)",
            xlabel="Explained Variance LN",
            ylabel="CE loss difference\n(original model - model with sae)",
            out_file=out_dir / f"explained_var_ln_vs_ce_loss_layer_{layer}.png",
        )
