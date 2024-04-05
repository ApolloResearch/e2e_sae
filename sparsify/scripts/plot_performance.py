from collections.abc import Sequence
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

RUN_TYPE_MAP = {
    "e2e": ("End-to-end", "o"),
    "e2e-recon": ("End-to-end-recon", "s"),
    "layerwise": ("Layerwise", "^"),
}


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
    out_file: str | Path | None = None,
    z: str | None = None,
    xlim: tuple[float | None, float | None] = (None, None),
    ylim: tuple[float | None, float | None] = (None, None),
    run_types: tuple[str, ...] = ("e2e", "layerwise"),
    sparsity_label: bool = False,
) -> None:
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
        run_types: The run types to include in the plot.
        sparsity_label: Whether to label the points with the sparsity coefficient.
    """
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#f5f6fc"})
    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = "plasma_r"
    norm, vmin, vmax = None, None, None
    if z is not None:
        vmin = int(10 ** np.floor(np.log10(df[z].min())))
        vmax = int(10 ** np.ceil(np.log10(df[z].max())))
        norm = LogNorm(vmin=vmin, vmax=vmax)

    for run_type in run_types:
        if run_type not in RUN_TYPE_MAP:
            raise ValueError(f"Invalid run type: {run_type}")
        label, marker = RUN_TYPE_MAP[run_type]
        data = df.loc[df["run_type"] == run_type]
        if not data.empty:
            scatter_kwargs = {
                "data": data,
                "x": x,
                "y": y,
                "marker": marker,
                "s": 95,
                "linewidth": 1.1,
                "ax": ax,
            }
            if z is None:
                scatter_kwargs["label"] = label
            else:
                scatter_kwargs = {**scatter_kwargs, "c": data[z], "cmap": cmap, "norm": norm}
            sns.scatterplot(**scatter_kwargs)  # type: ignore

    if sparsity_label:
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
        for run_type in run_types:
            if df.loc[df["run_type"] == run_type].empty:
                continue
            legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    marker=RUN_TYPE_MAP[run_type][1],
                    color="black",
                    linestyle="None",
                    markersize=8,
                    label=RUN_TYPE_MAP[run_type][0],
                )
            )
        ax.legend(handles=legend_handles, title="Run Type", loc="best")

    plt.tight_layout()
    if out_file is not None:
        plt.savefig(out_file)
    plt.close(fig)


def plot_per_layer_metric(
    df: pd.DataFrame,
    sae_layer: int,
    metric: str,
    n_layers: int = 8,
    out_file: str | Path | None = None,
    ylim: tuple[float | None, float | None] = (None, None),
    legend_label_cols: list[str] | None = None,
    run_types: Sequence[str] = ("e2e", "layerwise", "e2e-recon"),
) -> None:
    """
    Plot the per-layer metric (explained variance or reconstruction loss) for different run types.

    Args:
        df: DataFrame containing the filtered data for the specific layer.
        sae_layer: The layer where the SAE is applied.
        metric: The metric to plot ('explained_var' or 'recon_loss').
        n_layers: The number of layers in the transformer model.
        out_file: The filename which the plot will be saved as.
        ylim: The y-axis limits.
        legend_label_cols: Columns in df that should be used for the legend. Added in addition to
            the run type.
        run_types: The run types to include in the plot.
    """
    metric_names = {
        "explained_var": "Explained Variance",
        "explained_var_ln": "Layernormed Explained Variance",
        "recon_loss": "Reconstruction MSE",
    }
    metric_name = metric_names[metric] if metric in metric_names else metric

    color_e2e, color_lws, color_e2e_recon = sns.color_palette()[:3]
    color_map = {"e2e": color_e2e, "layerwise": color_lws, "e2e-recon": color_e2e_recon}

    plt.figure(figsize=(10, 6))
    xs = np.arange(sae_layer, n_layers)

    def plot_metric(runs: pd.DataFrame, marker: str):
        for _, row in runs.iterrows():
            run_type = row["run_type"]
            assert isinstance(run_type, str)
            legend_label = run_type
            if legend_label_cols is not None:
                assert all(
                    col in row for col in legend_label_cols
                ), f"Legend label cols not found in row: {row}"
                metric_strings = [f"{col}={format(row[col], '.3f')}" for col in legend_label_cols]
                legend_label += f" ({', '.join(metric_strings)})"
            ys = [row[f"{metric}_layer-{i}"] for i in range(sae_layer, n_layers)]
            plt.plot(
                xs,
                ys,
                label=legend_label,
                color=color_map[run_type],
                alpha=0.7,
                marker=marker,
            )

    for run_type in run_types:
        plot_metric(df.loc[df["run_type"] == run_type], RUN_TYPE_MAP[run_type][1])

    # Ensure that the x-axis are only whole numbers
    plt.xticks(xs, [str(x) for x in xs])
    plt.ylim(ylim)
    plt.legend(title="Run Type")
    plt.title(f"{metric_name} per Layer (SAE layer={sae_layer})")
    plt.xlabel("Layer")
    plt.ylabel(metric_name)
    plt.tight_layout()
    if out_file is not None:
        plt.savefig(out_file)
    plt.close()


if __name__ == "__main__":
    # Plot gpt2 performance
    api = wandb.Api()
    project = "sparsify/gpt2"
    runs = api.runs(project)

    n_layers = 12
    # These runs (keyed by layer number, value is the sparsity coeff), have similar CE loss diff
    constant_ce_runs = {
        2: {"e2e": 0.5, "layerwise": 0.8},
        6: {"e2e": 3, "layerwise": 4},
        10: {"e2e": 0.5, "layerwise": 4},
    }
    # These have similar L0
    constant_l0_runs = {
        2: {"e2e": 1.5, "layerwise": 4},
        6: {"e2e": 1.5, "layerwise": 6},
        10: {"e2e": 1.5, "layerwise": 10},
    }

    df = create_run_df(runs)

    # These runs were done with the following config:
    for col in ["lr", "n_samples", "ratio"]:
        assert df[col].nunique() == 1
    assert df["lr"].unique()[0] == 5e-4
    assert df["n_samples"].unique()[0] == 400_000
    assert df["ratio"].unique()[0] == 60

    # Only use seed=0
    df = df.loc[df["seed"] == 0]
    # print(df)

    unique_layers = list(df["layer"].unique())

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    for layer in unique_layers:
        layer_df = df.loc[df["layer"] == layer]
        plot_scatter(
            layer_df,
            x="L0",
            y="CE_diff",
            title=f"Layer {layer}: L0 vs CE Loss Difference (label: sparsity coeff)",
            xlabel="L0",
            ylabel="CE loss difference\n(original model - model with sae)",
            out_file=out_dir / f"l0_vs_ce_loss_layer_{layer}.png",
            sparsity_label=True,
        )
        plot_scatter(
            layer_df,
            x="alive_dict_elements",
            y="CE_diff",
            title=f"Layer {layer}: Alive Dictionary Elements vs CE Loss Difference (label: sparsity coeff)",
            xlabel="Alive Dictionary Elements",
            ylabel="CE loss difference\n(original model - model with sae)",
            out_file=out_dir / f"alive_elements_vs_ce_loss_layer_{layer}.png",
            sparsity_label=True,
        )
        plot_scatter(
            layer_df,
            x="out_to_in",
            y="CE_diff",
            title=f"Layer {layer}: Out-to-In Loss vs CE Loss Difference (label: sparsity coeff)",
            xlabel="Out-to-In Loss",
            ylabel="CE loss difference\n(original model - model with sae)",
            out_file=out_dir / f"out_to_in_vs_ce_loss_layer_{layer}.png",
            sparsity_label=True,
        )
        plot_scatter(
            layer_df,
            x="sum_recon_loss",
            y="CE_diff",
            title=f"Layer {layer}: Future Reconstruction Loss vs CE Loss Difference (label: sparsity coeff)",
            xlabel="Summed Future Reconstruction Loss",
            ylabel="CE loss difference\n(original model - model with sae)",
            out_file=out_dir / f"future_recon_vs_ce_loss_layer_{layer}.png",
            sparsity_label=True,
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
            sparsity_label=True,
        )

        # Per layer plots. Note that per-layer metrics are all taken at hook_resid_post
        layer_constant_ce_df = layer_df[
            (
                (layer_df["sparsity_coeff"] == constant_ce_runs[layer]["e2e"])
                & (layer_df["run_type"] == "e2e")
            )
            | (
                (layer_df["sparsity_coeff"] == constant_ce_runs[layer]["layerwise"])
                & (layer_df["run_type"] == "layerwise")
            )
        ]
        plot_per_layer_metric(
            layer_constant_ce_df,
            sae_layer=layer,
            metric="explained_var_ln",
            n_layers=n_layers,
            out_file=out_dir / f"explained_var_ln_per_layer_sae_layer_{layer}.png",
            ylim=(None, 1),
            legend_label_cols=["sparsity_coeff", "CE_diff"],
        )
        plot_per_layer_metric(
            layer_constant_ce_df,
            sae_layer=layer,
            metric="recon_loss",
            n_layers=n_layers,
            out_file=out_dir / f"recon_loss_per_layer_sae_layer_{layer}.png",
            legend_label_cols=["sparsity_coeff", "CE_diff"],
        )
