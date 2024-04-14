from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable

from sparsify.analysis import create_run_df

RUN_TYPE_MAP = {
    "e2e": ("End-to-end", "o"),
    "e2e-recon": ("End-to-end-recon", "X"),
    "local": ("Local", "^"),
}

# These runs (keyed by layer number, value is the sparsity coeff), have similar CE loss diff
CONSTANT_CE_RUNS = {
    2: {"e2e": 0.5, "local": 0.8, "e2e-recon": 10},
    6: {"e2e": 3, "local": 4, "e2e-recon": 50},
    10: {"e2e": 1.5, "local": 6, "e2e-recon": 25},
}
# These have similar L0
CONSTANT_L0_RUNS = {
    2: {"e2e": 1.5, "local": 4},
    6: {"e2e": 1.5, "local": 6},
    10: {"e2e": 1.5, "local": 10},
}


def plot_scatter_or_line(
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
    run_types: tuple[str, ...] = ("e2e", "local", "e2e-recon"),
    sparsity_label: bool = False,
    plot_type: Literal["scatter", "line"] | None = None,
) -> None:
    """Plot a scatter or line plot with the specified x and y variables, colored by run type or z.

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
        plot_type: The type of plot to create. Either 'scatter' or 'line'.
    """
    plot_type = plot_type if plot_type is not None else ("line" if z is None else "scatter")
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#f5f6fc"})
    fig, ax = plt.subplots(figsize=(8, 6))

    marker_size = 95 if len(run_types) < 3 else 60

    cmap = "plasma_r"
    norm, vmin, vmax = None, None, None
    if z is not None:
        vmin = int(10 ** np.floor(np.log10(df[z].min())))
        vmax = int(10 ** np.ceil(np.log10(df[z].max())))
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    for run_type in run_types:
        if run_type not in RUN_TYPE_MAP:
            raise ValueError(f"Invalid run type: {run_type}")
        label, marker = RUN_TYPE_MAP[run_type]
        data = df.loc[df["run_type"] == run_type]
        if not data.empty:
            plot_kwargs = {
                "data": data,
                "x": x,
                "y": y,
                "marker": marker,
                "linewidth": 1.1,
                "ax": ax,
                "alpha": 0.8,
            }
            if plot_type == "scatter":
                plot_kwargs["s"] = marker_size
            elif plot_type == "line":
                plot_kwargs["orient"] = "y"
            if z is None:
                plot_kwargs["label"] = label
            else:
                plot_kwargs = {**plot_kwargs, "c": data[z], "cmap": cmap, "norm": norm}
            if plot_type == "scatter":
                sns.scatterplot(**plot_kwargs)  # type: ignore
            elif plot_type == "line":
                sns.lineplot(**plot_kwargs)  # type: ignore
            else:
                raise ValueError(f"Invalid plot type: {plot_type}")

    if sparsity_label:
        for _, row in df.iterrows():
            if row["run_type"] in run_types:
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

    # plt.tight_layout()
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
    legend_label_cols_and_precision: list[tuple[str, int]] | None = None,
    run_types: Sequence[str] = ("e2e", "local", "e2e-recon"),
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
        legend_label_cols_and_precision: Columns in df that should be used for the legend, along
            with their precision. Added in addition to the run type.
        run_types: The run types to include in the plot.
    """
    metric_names = {
        "explained_var": "Explained Variance",
        "explained_var_ln": "Layernormed Explained Variance",
        "recon_loss": "Reconstruction MSE",
    }
    metric_name = metric_names[metric] if metric in metric_names else metric

    color_e2e, color_lws, color_e2e_recon = sns.color_palette()[:3]
    color_map = {"e2e": color_e2e, "local": color_lws, "e2e-recon": color_e2e_recon}

    plt.figure(figsize=(10, 6))
    xs = np.arange(sae_layer, n_layers)

    def plot_metric(runs: pd.DataFrame, marker: str):
        for _, row in runs.iterrows():
            run_type = row["run_type"]
            assert isinstance(run_type, str)
            legend_label = run_type
            if legend_label_cols_and_precision is not None:
                assert all(
                    col in row for col, _ in legend_label_cols_and_precision
                ), f"Legend label cols not found in row: {row}"
                metric_strings = [
                    f"{col}={format(row[col], f'.{prec}f')}"
                    for col, prec in legend_label_cols_and_precision
                ]
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


def plot_seed_comparison(df: pd.DataFrame, out_dir: Path, run_types: Sequence[str]) -> None:
    """Plot the CE loss difference vs L0 for each layer, comparing different seeds.

    Args:
        df: DataFrame containing the data.
        out_dir: The directory to save the plots to.
        run_types: The run types to include in the plot.
    """
    model_name = df["model_name"].unique()[0]
    seed_dir = out_dir / f"seed_comparison_{model_name}"
    seed_dir.mkdir(exist_ok=True, parents=True)

    seeds = df["seed"].unique()
    colors = sns.color_palette("tab10", n_colors=len(seeds))

    for run_type in run_types:
        layer_df = df.loc[(df["run_type"] == run_type)]

        # Get only the layer-sparsity pairs that have more than one seed
        layer_df = layer_df.groupby(["layer", "sparsity_coeff"]).filter(
            lambda x: x["seed"].nunique() > 1
        )
        if layer_df.empty:
            continue
        plt.figure(figsize=(8, 6))
        for i, seed in enumerate(seeds):
            seed_df = layer_df.loc[layer_df["seed"] == seed]
            plt.scatter(
                seed_df["L0"],
                seed_df["CE_diff"],
                label=f"Seed {seed}",
                color=colors[i],
                alpha=0.8,
            )
        layers = ",".join([str(l) for l in layer_df["layer"].unique()])
        plt.title(f"Layers {layers}: L0 vs CE Loss Difference (run_type={run_type})")
        plt.xlabel("L0")
        plt.ylabel("CE loss difference\n(original model - model with sae)")
        plt.legend(title="Seed", loc="best")
        plt.grid()
        plt.tight_layout()
        plt.savefig(seed_dir / f"l0_vs_ce_loss_layers_{layers}_{run_type}.png")
        plt.close()


if __name__ == "__main__":
    # Plot gpt2 performance
    # run_types = ("e2e", "local")
    run_types = ("e2e", "local", "e2e-recon")
    # out_dir = Path(__file__).resolve().parent / "out" / "e2e_local"
    out_dir = Path(__file__).resolve().parent / "out" / "e2e_local_e2e-recon"
    out_dir.mkdir(exist_ok=True)

    api = wandb.Api()
    project = "sparsify/gpt2"
    runs = api.runs(project)

    n_layers = 12
    d_resid = 768

    df = create_run_df(runs)

    # These runs were done with the following config:
    for col in ["lr", "n_samples", "ratio"]:
        assert df[col].nunique() == 1
    assert df["lr"].unique()[0] == 5e-4
    assert df["n_samples"].unique()[0] == 400_000
    assert df["ratio"].unique()[0] == 60
    assert df["model_name"].nunique() == 1

    # Ignore runs that have an L0 bigger than d_resid
    df = df.loc[df["L0"] <= d_resid]
    # Only use the e2e+recon run in layer 10 that has kl_coeff=0.75
    df = df.loc[~((df["layer"] == 10) & (df["run_type"] == "e2e-recon") & (df["kl_coeff"] != 0.75))]

    plot_seed_comparison(df, out_dir=Path(__file__).resolve().parent / "out", run_types=run_types)

    # Only use seed=0 for remaining plots
    df = df.loc[df["seed"] == 0]
    # print(df)

    # ylims for plots with ce_diff on the y axis
    ce_diff_ylims = {
        2: (-0.2, 0),
        6: (-0.4, 0),
        10: (-0.4, 0),
    }
    l0_diff_xlims = {
        2: (0, 200),
        6: (0, 600),
        10: (0, 600),
    }
    unique_layers = list(df["layer"].unique())
    for layer in unique_layers:
        layer_df = df.loc[df["layer"] == layer]

        plot_scatter_or_line(
            layer_df,
            x="L0",
            y="CE_diff",
            xlim=l0_diff_xlims[layer],
            ylim=ce_diff_ylims[layer],
            title=f"Layer {layer}: L0 vs CE Loss Difference",
            xlabel="L0",
            ylabel="CE loss difference\n(original model - model with sae)",
            out_file=out_dir / f"l0_vs_ce_loss_layer_{layer}.png",
            sparsity_label=True,
            run_types=run_types,
        )
        plot_scatter_or_line(
            layer_df,
            x="alive_dict_elements",
            y="CE_diff",
            z="L0",
            ylim=ce_diff_ylims[layer],
            title=f"Layer {layer}: Alive Dictionary Elements vs CE Loss Difference",
            xlabel="Alive Dictionary Elements",
            ylabel="CE loss difference\n(original model - model with sae)",
            out_file=out_dir / f"alive_elements_vs_ce_loss_layer_{layer}.png",
            sparsity_label=False,
            run_types=run_types,
        )
        plot_scatter_or_line(
            layer_df,
            x="L0",
            y="alive_dict_elements",
            title=f"Layer {layer}: L0 vs Alive Dictionary Elements",
            xlabel="L0",
            ylabel="Alive Dictionary Elements",
            out_file=out_dir / f"l0_vs_alive_dict_elements_layer_{layer}.png",
            xlim=l0_diff_xlims[layer],
            sparsity_label=False,
            run_types=run_types,
            plot_type="scatter",
        )
        plot_scatter_or_line(
            layer_df,
            x="out_to_in",
            y="CE_diff",
            ylim=ce_diff_ylims[layer],
            title=f"Layer {layer}: Out-to-In Loss vs CE Loss Difference",
            xlabel="Out-to-In Loss",
            ylabel="CE loss difference\n(original model - model with sae)",
            out_file=out_dir / f"out_to_in_vs_ce_loss_layer_{layer}.png",
            sparsity_label=False,
            run_types=run_types,
            plot_type="scatter",
        )
        plot_scatter_or_line(
            layer_df,
            x="sum_recon_loss",
            y="CE_diff",
            ylim=ce_diff_ylims[layer],
            title=f"Layer {layer}: Future Reconstruction Loss vs CE Loss Difference",
            xlabel="Summed Future Reconstruction Loss",
            ylabel="CE loss difference\n(original model - model with sae)",
            out_file=out_dir / f"future_recon_vs_ce_loss_layer_{layer}.png",
            sparsity_label=False,
            run_types=run_types,
            plot_type="scatter",
        )
        plot_scatter_or_line(
            layer_df,
            x="explained_var_ln",
            y="CE_diff",
            z="L0",
            ylim=ce_diff_ylims[layer],
            title=f"Layer {layer}: Explained Variance LN vs CE Loss Difference",
            xlabel="Explained Variance LN",
            ylabel="CE loss difference\n(original model - model with sae)",
            out_file=out_dir / f"explained_var_ln_vs_ce_loss_layer_{layer}.png",
            sparsity_label=False,
            run_types=run_types,
        )

        # Per layer plots. Note that per-layer metrics are all taken at hook_resid_post
        layer_constant_ce_df = layer_df[
            (
                (layer_df["sparsity_coeff"] == CONSTANT_CE_RUNS[layer]["e2e"])
                & (layer_df["run_type"] == "e2e")
            )
            | (
                (layer_df["sparsity_coeff"] == CONSTANT_CE_RUNS[layer]["local"])
                & (layer_df["run_type"] == "local")
            )
            | (
                (layer_df["sparsity_coeff"] == CONSTANT_CE_RUNS[layer]["e2e-recon"])
                & (layer_df["run_type"] == "e2e-recon")
            )
        ]
        plot_per_layer_metric(
            layer_constant_ce_df,
            sae_layer=layer,
            metric="explained_var_ln",
            n_layers=n_layers,
            out_file=out_dir / f"explained_var_ln_per_layer_sae_layer_{layer}.png",
            ylim=(None, 1),
            legend_label_cols_and_precision=[("L0", 0), ("CE_diff", 3)],
            run_types=run_types,
        )
        plot_per_layer_metric(
            layer_constant_ce_df,
            sae_layer=layer,
            metric="recon_loss",
            n_layers=n_layers,
            out_file=out_dir / f"recon_loss_per_layer_sae_layer_{layer}.png",
            legend_label_cols_and_precision=[("L0", 0), ("CE_diff", 3)],
            run_types=run_types,
        )
