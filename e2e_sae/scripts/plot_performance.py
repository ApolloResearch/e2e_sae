from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from numpy.typing import NDArray

from e2e_sae.analysis import create_run_df
from e2e_sae.log import logger

RUN_TYPE_MAP = {
    "e2e": ("End-to-end", "o"),
    "e2e-recon": ("End-to-end-recon", "X"),
    "local": ("Local", "^"),
}

# Runs with constant CE loss increase for each layer. Values represent wandb run IDs.
CONSTANT_CE_RUNS = {
    2: {"e2e": "ovhfts9n", "local": "ue3lz0n7", "e2e-recon": "visi12en"},
    6: {"e2e": "zgdpkafo", "local": "1jy3m5j0", "e2e-recon": "2lzle2f0"},
    10: {"e2e": "8crnit9h", "local": "m2hntlav", "e2e-recon": "cvj5um2h"},
}


def plot_scatter_or_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    out_file: str | Path | None = None,
    z: str | None = None,
    xlim: Mapping[int, tuple[float | None, float | None]] | None = None,
    ylim: Mapping[int, tuple[float | None, float | None]] | None = None,
    run_types: tuple[str, ...] = ("e2e", "local", "e2e-recon"),
    sparsity_label: bool = False,
    plot_type: Literal["scatter", "line"] | None = None,
    layers: Sequence[int] | None = None,
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
        xlim: The x-axis limits for each layer.
        ylim: The y-axis limits for each layer.
        run_types: The run types to include in the plot.
        sparsity_label: Whether to label the points with the sparsity coefficient.
        plot_type: The type of plot to create. Either 'scatter' or 'line'.
        layers: The layers to include in the plot. If None, all layers in the df will be included.
    """

    if layers is None:
        layers = sorted(df["layer"].unique())
    n_layers = len(layers)

    if xlim is None:
        xlim = {layer: (None, None) for layer in layers}
    if ylim is None:
        ylim = {layer: (None, None) for layer in layers}

    plot_type = plot_type if plot_type is not None else ("line" if z is None else "scatter")
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#f5f6fc"})
    cmap = "plasma_r"

    fig, axs = plt.subplots(n_layers, 1, figsize=(8, 4 * n_layers))
    axs = np.atleast_1d(axs)

    marker_size = 95 if len(run_types) < 3 else 60

    for i, layer in enumerate(layers):
        layer_df = df.loc[df["layer"] == layer]
        ax = axs[i]
        norm, vmin, vmax = None, None, None
        if z is not None:
            vmin = int(10 ** np.floor(np.log10(layer_df[z].min())))
            vmax = int(10 ** np.ceil(np.log10(layer_df[z].max())))
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

        for run_type in run_types:
            if run_type not in RUN_TYPE_MAP:
                raise ValueError(f"Invalid run type: {run_type}")
            label, marker = RUN_TYPE_MAP[run_type]
            data = layer_df.loc[layer_df["run_type"] == run_type]
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
            for _, row in layer_df.iterrows():
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
        ax.set_xlim(xmin=xlim[layer][0], xmax=xlim[layer][1])
        ax.set_ylim(ymin=ylim[layer][0], ymax=ylim[layer][1])
        ax.set_title(f"SAE Layer {layer}", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

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
                if layer_df.loc[layer_df["run_type"] == run_type].empty:
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

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    if out_file is not None:
        plt.savefig(out_file)
        plt.savefig(Path(out_file).with_suffix(".svg"))
    plt.close(fig)


def plot_per_layer_metric(
    df: pd.DataFrame,
    run_ids: Mapping[int, Mapping[str, str]],
    metric: str,
    final_layer: int = 8,
    out_file: str | Path | None = None,
    ylim: tuple[float | None, float | None] = (None, None),
    legend_label_cols_and_precision: list[tuple[str, int]] | None = None,
    run_types: Sequence[str] = ("e2e", "local", "e2e-recon"),
) -> None:
    """
    Plot the per-layer metric (explained variance or reconstruction loss) for different run types.

    Args:
        df: DataFrame containing the filtered data for the specific layer.
        run_ids: The run IDs to use. Format: {layer: {run_type: run_id}}.
        sae_layer: The layer where the SAE is applied.
        metric: The metric to plot ('explained_var' or 'recon_loss').
        n_model_layers: The number of layers in the transformer model.
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
    metric_name = metric_names.get(metric, metric)

    sae_layers = list(CONSTANT_CE_RUNS.keys())
    n_sae_layers = len(sae_layers)

    color_e2e, color_lws, color_e2e_recon = sns.color_palette()[:3]
    color_map = {"e2e": color_e2e, "local": color_lws, "e2e-recon": color_e2e_recon}

    fig, axs = plt.subplots(n_sae_layers, 1, figsize=(8, 4 * n_sae_layers))
    axs = np.atleast_1d(axs)

    def plot_metric(
        ax: plt.Axes,
        plot_df: pd.DataFrame,
        marker: str,
        sae_layer: int,
        xs: NDArray[np.signedinteger[Any]],
    ) -> None:
        for _, row in plot_df.iterrows():
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
            ys = [row[f"{metric}_layer-{i}"] for i in range(sae_layer, final_layer + 1)]
            ax.plot(
                xs,
                ys,
                label=legend_label,
                color=color_map[run_type],
                alpha=0.7,
                marker=marker,
            )

    for i, sae_layer in enumerate(sae_layers):
        valid_ids = [int(v) for v in run_ids[sae_layer].values()]
        layer_df = df.loc[df["id"].isin(valid_ids)]

        ax = axs[i]

        xs = np.arange(sae_layer, final_layer + 1)
        for run_type in run_types:
            if run_type not in RUN_TYPE_MAP:
                raise ValueError(f"Invalid run type: {run_type}")
            label, marker = RUN_TYPE_MAP[run_type]
            plot_metric(ax, layer_df.loc[layer_df["run_type"] == run_type], marker, sae_layer, xs)

            ax.set_title(f"SAE Layer {sae_layer}", fontweight="bold")
            ax.set_xlabel("Model Layer")
            ax.set_ylabel(metric_name)
            ax.legend(title="Run Type", loc="best")
            ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in xs])
            ax.set_ylim(ylim)

    plt.tight_layout()
    if out_file is not None:
        plt.savefig(out_file)
        # Save as svg also
        plt.savefig(Path(out_file).with_suffix(".svg"))
    plt.close()


def plot_two_axes_line_single_run_type(
    df: pd.DataFrame,
    run_type: str,
    out_dir: Path,
    title: str,
    iter_var: str,
    iter_vals: list[float],
) -> None:
    """Plot the CE loss difference vs L0 and alive_dict_elements for different values of iter_var.

    Note that these plots are only for a single run type.

    Args:
        df: DataFrame containing the data.
        run_type: The run type to include in the plot.
        out_dir: The directory to save the plots to.
        title: The title of the plot.
        iter_var: The variable in the DataFrame to iterate over.
        iter_vals: The values to iterate over.
    """
    colors = sns.color_palette("tab10", n_colors=len(iter_vals))
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#f5f6fc"})
    fig, axs = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={"wspace": 0.15, "top": 0.84})
    for i, vals in enumerate(iter_vals):
        vals_df = df.loc[df[iter_var] == vals]
        for j, (ax, x) in enumerate(zip(axs, ["L0", "alive_dict_elements"], strict=True)):
            # Sort the data by CE_diff to ensure consistent line plotting
            iter_var_df_sorted = vals_df.sort_values("CELossIncrease")
            ax.scatter(
                iter_var_df_sorted[x],
                iter_var_df_sorted["CELossIncrease"],
                label=f"{vals}",
                color=colors[i],
                alpha=0.8,
            )
            # Plot a line between the points
            ax.plot(
                iter_var_df_sorted[x],
                iter_var_df_sorted["CELossIncrease"],
                color=colors[i],
                alpha=0.5,
                linewidth=1,
            )
    axs[0].legend(loc="best", title=iter_var)

    # the L0 axis (left) should always be decreasing, reverse it if it is not
    if axs[0].get_xlim()[0] < axs[0].get_xlim()[1]:
        axs[0].invert_xaxis()
    # The alive_dict_elements axis (right) should always be increasing, reverse it if it is not
    if axs[1].get_xlim()[0] > axs[1].get_xlim()[1]:
        axs[1].invert_xaxis()

    # CE loss increase should always be decreasing, reverse it if it is not
    if axs[0].get_ylim()[0] < axs[0].get_ylim()[1]:
        axs[0].invert_yaxis()
        axs[1].invert_yaxis()

    axs[0].set_xlabel("L0")
    axs[1].set_xlabel("Alive Dictionary Elements")
    axs[0].set_ylabel("CE Loss Increase")
    axs[1].set_ylabel("CE Loss Increase")
    # move y-axis to right of second subplot
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.set_ticks_position("right")
    axs[1].yaxis.set_tick_params(color="white")
    # add "Better" text annotations
    axs[0].text(
        s="Better →", x=1, y=1.02, ha="right", va="bottom", fontsize=10, transform=axs[0].transAxes
    )
    axs[1].text(
        s="← Better", x=0, y=1.02, ha="left", va="bottom", fontsize=10, transform=axs[1].transAxes
    )
    axs[0].text(
        s="Better →",
        x=1.075,
        y=1,
        ha="center",
        va="top",
        fontsize=10,
        transform=axs[0].transAxes,
        rotation=90,
    )
    fig.suptitle(title)
    plt.savefig(out_dir / f"l0_alive_elements_vs_ce_loss_{run_type}.png")
    plt.close(fig)


def plot_two_axes_line(
    df: pd.DataFrame,
    x1: str,
    x2: str,
    y: str,
    xlabel1: str,
    xlabel2: str,
    ylabel: str,
    title: str | None = None,
    out_file: str | Path | None = None,
    run_types: Sequence[str] = ("e2e", "local", "e2e-recon"),
    xlim1: Mapping[int, tuple[float | None, float | None]] | None = None,
    xlim2: Mapping[int, tuple[float | None, float | None]] | None = None,
    xticks1: tuple[list[float], list[str]] | None = None,
    xticks2: tuple[list[float], list[str]] | None = None,
    ylim: Mapping[int, tuple[float | None, float | None]] | None = None,
    layers: Sequence[int] | None = None,
) -> None:
    """Line plot with two x-axes and one y-axis between them. One line for each run type.

    Args:
        df: DataFrame containing the data.
        x1: The variable to plot on the first x-axis.
        x2: The variable to plot on the second x-axis.
        y: The variable to plot on the y-axis.
        title: The title of the plot.
        xlabel1: The label for the first x-axis.
        xlabel2: The label for the second x-axis.
        ylabel: The label for the y-axis.
        out_file: The filename which the plot will be saved as.
        run_types: The run types to include in the plot.
        xlim1: The x-axis limits for the first x-axis for each layer.
        xlim2: The x-axis limits for the second x-axis for each layer.
        xticks1: The x-ticks for the first x-axis.
        xticks2: The x-ticks for the second x-axis.
        ylim: The y-axis limits for each layer.
        layers: The layers to include in the plot. If None, all layers in the df will be included.
    """

    if layers is None:
        layers = sorted(df["layer"].unique())
    n_layers = len(layers)

    if xlim1 is None:
        xlim1 = {layer: (None, None) for layer in layers}
    if xlim2 is None:
        xlim2 = {layer: (None, None) for layer in layers}
    if ylim is None:
        ylim = {layer: (None, None) for layer in layers}

    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#f5f6fc"})
    fig = plt.figure(figsize=(8, 4 * n_layers), constrained_layout=True)
    subfigs = fig.subfigures(n_layers)
    subfigs = np.atleast_1d(subfigs)
    for subfig, layer in zip(subfigs, layers, strict=False):
        layer_df = df.loc[df["layer"] == layer]
        axs = subfig.subplots(1, 2)
        for run_type in run_types:
            if run_type not in RUN_TYPE_MAP:
                raise ValueError(f"Invalid run type: {run_type}")
            label, marker = RUN_TYPE_MAP[run_type]
            data = layer_df.loc[layer_df["run_type"] == run_type]
            if not data.empty:
                # draw the lines between points based on the y value
                data = data.sort_values(y)
                for i, (ax, x) in enumerate(zip(axs, [x1, x2], strict=False)):
                    ax.plot(
                        data[x],
                        data[y],
                        marker=marker,
                        linewidth=1.1,
                        alpha=0.8,
                        label=label if i == 0 else None,
                    )

        axs[0].legend(loc="best")

        axs[0].set_xlim(xmin=xlim1[layer][0], xmax=xlim1[layer][1])
        axs[1].set_xlim(xmin=xlim2[layer][0], xmax=xlim2[layer][1])
        axs[0].set_ylim(ymin=ylim[layer][0], ymax=ylim[layer][1])
        axs[1].set_ylim(ymin=ylim[layer][0], ymax=ylim[layer][1])

        # Set a title above axs[0] and axs[1] to show the layer number
        subfig.suptitle(f"SAE Layer {layer}", fontweight="bold")
        axs[0].set_xlabel(xlabel1)
        axs[1].set_xlabel(xlabel2)
        axs[0].set_ylabel(ylabel)
        axs[1].set_ylabel(ylabel)

        # move y-axis to right of second subplot
        axs[1].yaxis.set_label_position("right")
        axs[1].yaxis.set_ticks_position("right")
        axs[1].yaxis.set_tick_params(color="white")

        if xticks1 is not None:
            axs[0].set_xticks(xticks1[0], xticks1[1])
        if xticks2 is not None:
            axs[1].set_xticks(xticks2[0], xticks2[1])

        # add "Better" text annotations
        axs[0].text(
            s="Better →",
            x=1,
            y=1.02,
            ha="right",
            va="bottom",
            fontsize=10,
            transform=axs[0].transAxes,
        )
        axs[1].text(
            s="← Better",
            x=0,
            y=1.02,
            ha="left",
            va="bottom",
            fontsize=10,
            transform=axs[1].transAxes,
        )
        axs[0].text(
            s="Better →",
            x=1.075,
            y=1,
            ha="center",
            va="top",
            fontsize=10,
            transform=axs[0].transAxes,
            rotation=90,
        )

    if title is not None:
        fig.suptitle(title)

    if out_file is not None:
        plt.savefig(out_file)
        # Save as svg also
        plt.savefig(Path(out_file).with_suffix(".svg"))
    plt.close(fig)


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
        plt.tight_layout()
        plt.savefig(seed_dir / f"l0_vs_ce_loss_layers_{layers}_{run_type}.png")
        plt.close()

        # Also write all the "id"s to file
        ids = layer_df["id"].unique().tolist()
        with open(seed_dir / f"ids_layers_{layers}_{run_type}.txt", "w") as f:
            f.write(",".join(ids))


def plot_ratio_comparison(df: pd.DataFrame, out_dir: Path, run_types: Sequence[str]) -> None:
    """Plots alive_dict_elements and L0 vs CE loss increase for different dictionary ratios.

    Args:
        df: DataFrame containing the data.
        out_dir: The directory to save the plots to.
        run_types: The run types to include in the plot.
    """
    ratios_dir = out_dir / "ratio_comparison"
    ratios_dir.mkdir(exist_ok=True, parents=True)
    ratios_runs = df.loc[df["ratio"] != 60]
    for run_type in run_types:
        run_type_df = ratios_runs.loc[ratios_runs["run_type"] == run_type]
        if run_type_df.empty:
            continue
        # Should have the same kl_coeff, n_samples, sae_pos, and layer
        for col in ["kl_coeff", "n_samples", "sae_pos", "layer"]:
            assert run_type_df[col].nunique() <= 1, f"Multiple {col} values found"
        sparsity_coeffs = run_type_df["sparsity_coeff"].unique()
        kl_coeff = run_type_df["kl_coeff"].unique()[0]
        n_samples = run_type_df["n_samples"].unique()[0]
        sae_pos = run_type_df["sae_pos"].unique()[0]
        layer = run_type_df["layer"].unique()[0]
        # Also get the run with ratio=60 that has the same sparsity_coeff, out_to_in, kl_coeff,
        # and n_samples
        kl_mask = (
            (df["kl_coeff"] == kl_coeff) if not np.isnan(kl_coeff) else df["kl_coeff"].isnull()
        )
        runs_ratio_60 = df.loc[
            (df["ratio"] == 60)
            & kl_mask
            & (df["n_samples"] == n_samples)
            & (df["sae_pos"] == sae_pos)
            & (df["layer"] == layer)
            & (df["sparsity_coeff"].isin(sparsity_coeffs))
        ]
        combined_df = pd.concat([run_type_df, runs_ratio_60])
        ratios = sorted(combined_df["ratio"].unique())

        plot_two_axes_line_single_run_type(
            df=combined_df,
            run_type=run_type,
            out_dir=ratios_dir,
            title=f"run_type={run_type}: L0 and Alive Dict Elements vs CE Loss Diff in {sae_pos}",
            iter_var="ratio",
            iter_vals=ratios,
        )

        # Also write all the "id"s to file
        ids = combined_df["id"].unique().tolist()
        with open(ratios_dir / f"ids_ratio_comparison_{run_type}.txt", "w") as f:
            f.write(",".join(ids))


def plot_n_samples_comparison(df: pd.DataFrame, out_dir: Path, run_types: Sequence[str]) -> None:
    """Plots alive_dict_elements and L0 vs CE loss increase for different n_samples.

    Args:
        df: DataFrame containing the data.
        out_dir: The directory to save the plots to.
        run_types: The run types to include in the plot.
    """

    n_samples_dir = out_dir / "n_samples_comparison"
    n_samples_dir.mkdir(exist_ok=True, parents=True)

    n_samples_runs = df.loc[df["n_samples"] != 400_000]
    for run_type in run_types:
        run_type_df = n_samples_runs.loc[n_samples_runs["run_type"] == run_type]
        if run_type_df.empty:
            continue

        # Should have the same kl_coeff, ratio, sae_pos, and layer
        for col in ["kl_coeff", "ratio", "sae_pos", "layer"]:
            # Note that there may be a NaN value in the kl_coeff column
            assert run_type_df[col].nunique() <= 1, f"Multiple {col} values found"
        sparsity_coeffs = run_type_df["sparsity_coeff"].unique()
        kl_coeff = run_type_df["kl_coeff"].unique()[0]
        ratio = run_type_df["ratio"].unique()[0]
        sae_pos = run_type_df["sae_pos"].unique()[0]
        layer = run_type_df["layer"].unique()[0]
        # Also get the run with 400k samples that has the same sparsity_coeff, out_to_in, kl_coeff,
        # and ratio
        kl_mask = (
            (df["kl_coeff"] == kl_coeff) if not np.isnan(kl_coeff) else df["kl_coeff"].isnull()
        )
        runs_samples_400k = df.loc[
            (df["n_samples"] == 400_000)
            & kl_mask
            & (df["ratio"] == ratio)
            & (df["sae_pos"] == sae_pos)
            & (df["layer"] == layer)
            & (df["sparsity_coeff"].isin(sparsity_coeffs))
        ]
        combined_df = pd.concat([run_type_df, runs_samples_400k])

        samples = sorted(combined_df["n_samples"].unique())

        title = f"run_type={run_type}: L0 and Alive Dict Elements vs CE Loss Diff in {sae_pos}"
        plot_two_axes_line_single_run_type(
            df=combined_df,
            run_type=run_type,
            out_dir=n_samples_dir,
            title=title,
            iter_var="n_samples",
            iter_vals=samples,
        )

        # Also write all the "id"s to file
        ids = combined_df["id"].unique().tolist()
        with open(n_samples_dir / f"ids_n_samples_comparison_{run_type}.txt", "w") as f:
            f.write(",".join(ids))


def get_df_gpt2() -> pd.DataFrame:
    api = wandb.Api()
    project = "sparsify/gpt2"
    runs = api.runs(project)

    d_resid = 768

    df = create_run_df(runs)

    assert df["lr"].nunique() == 1 and df["lr"].unique()[0] == 5e-4
    assert df["model_name"].nunique() == 1

    # Ignore runs that have an L0 bigger than d_resid
    df = df.loc[df["L0"] <= d_resid]
    # Only use the e2e+recon run in layer 10 that has kl_coeff=0.75
    # df = df.loc[~((df["layer"] == 10) & (df["run_type"] == "e2e-recon") & (df["kl_coeff"] != 0.75))]
    df = df.loc[
        ~(
            (df["layer"] == 10)
            & (df["run_type"] == "e2e-recon")
            & ((df["kl_coeff"] != 0.5) | (df["in_to_orig_coeff"] != 0.05))
        )
    ]
    return df


def gpt2_plots():
    run_types = ("e2e", "local", "e2e-recon")
    n_layers = 12

    out_dir = Path(__file__).resolve().parent / "out" / "_".join(run_types)
    out_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving plots to {out_dir}")

    df = get_df_gpt2()

    plot_n_samples_comparison(
        df=df.loc[(df["ratio"] == 60) & (df["seed"] == 0)],
        out_dir=Path(__file__).resolve().parent / "out",
        run_types=run_types,
    )

    plot_seed_comparison(
        df=df.loc[df["n_samples"] == 400_000],
        out_dir=Path(__file__).resolve().parent / "out",
        run_types=run_types,
    )

    plot_ratio_comparison(
        df=df.loc[(df["seed"] == 0) & (df["n_samples"] == 400_000)],
        out_dir=Path(__file__).resolve().parent / "out",
        run_types=run_types,
    )

    performance_df = df.loc[(df["ratio"] == 60) & (df["seed"] == 0) & (df["n_samples"] == 400_000)]
    # For layer 2 we filter out the runs with L0 > 200. Otherwise we end up with point in one
    # subplot but not the other
    performance_df = performance_df.loc[
        ~((performance_df["L0"] > 200) & (performance_df["layer"] == 2))
    ]

    # ylims for plots with ce_diff on the y axis
    loss_increase_lims = {2: (0.2, 0.0), 6: (0.4, 0.0), 10: (0.4, 0.0)}
    # xlims for plots with L0 on the x axis
    l0_diff_xlims = {2: (200.0, 0.0), 6: (600.0, 0.0), 10: (600.0, 0.0)}

    # Plot two axes line plots with L0 and alive_dict_elements on the x-axis
    plot_two_axes_line(
        performance_df,
        x1="L0",
        x2="alive_dict_elements",
        y="CELossIncrease",
        xlabel1="L0",
        xlabel2="Alive Dictionary Elements",
        ylabel="CE Loss Increase",
        out_file=out_dir / "l0_alive_dict_elements_vs_ce_loss.png",
        run_types=run_types,
        xlim1=l0_diff_xlims,
        xticks2=([0, 10_000, 20_000, 30_000, 40_000], ["0", "10k", "20k", "30k", "40k"]),
        ylim=loss_increase_lims,
    )
    plot_scatter_or_line(
        performance_df,
        x="out_to_in",
        y="CELossIncrease",
        ylim=loss_increase_lims,
        title="Out-to-In Loss vs CE Loss Increase",
        xlabel="Out-to-In Loss",
        ylabel="CE loss increase\n(original model - model with sae)",
        out_file=out_dir / "out_to_in_vs_ce_loss.png",
        sparsity_label=False,
        run_types=run_types,
        plot_type="scatter",
    )
    plot_scatter_or_line(
        performance_df,
        x="sum_recon_loss",
        y="CELossIncrease",
        ylim=loss_increase_lims,
        title="Future Reconstruction Loss vs CE Loss Increase",
        xlabel="Summed Future Reconstruction Loss",
        ylabel="CE loss increase\n(original model - model with sae)",
        out_file=out_dir / "future_recon_vs_ce_loss.png",
        sparsity_label=False,
        run_types=run_types,
        plot_type="scatter",
    )
    plot_scatter_or_line(
        performance_df,
        x="explained_var_ln",
        y="CELossIncrease",
        z="L0",
        ylim=loss_increase_lims,
        title="Explained Variance LN vs CE Loss Increase",
        xlabel="Explained Variance LN",
        ylabel="CE loss increase\n(original model - model with sae)",
        out_file=out_dir / "explained_var_ln_vs_ce_loss.png",
        sparsity_label=False,
        run_types=run_types,
    )

    # We didn't track metrics for hook_resid_post in the final model layer in e2e+recon, though
    # perhaps we should have (including using the final layer's hook_resid_post in the loss)
    final_layer = n_layers - 1
    plot_per_layer_metric(
        performance_df,
        run_ids=CONSTANT_CE_RUNS,
        metric="explained_var_ln",
        final_layer=final_layer,
        out_file=out_dir / "explained_var_ln_per_layer.png",
        ylim=(None, 1),
        legend_label_cols_and_precision=[("L0", 0), ("CE_diff", 3)],
        run_types=run_types,
    )
    plot_per_layer_metric(
        performance_df,
        run_ids=CONSTANT_CE_RUNS,
        metric="recon_loss",
        final_layer=final_layer,
        out_file=out_dir / "recon_loss_per_layer.png",
        ylim=(0, None),
        legend_label_cols_and_precision=[("L0", 0), ("CE_diff", 3)],
        run_types=run_types,
    )


def get_tinystories_1m_df() -> pd.DataFrame:
    api = wandb.Api()
    project = "sparsify/tinystories-1m-2"
    runs = api.runs(project)

    d_resid = 64

    df = create_run_df(runs, per_layer_metrics=False, use_run_name=True)

    assert df["model_name"].nunique() == 1

    # Our "final" runs have the following properties:
    e2e_properties = (
        (df["lr"] == 0.001)
        & (df["n_samples"] == 450_000)
        & (~df["name"].str.contains("nogradnorm"))
        & (df["run_type"] == "e2e")
    )
    local_properties = (
        (df["lr"] == 0.01) & (df["n_samples"] == 400_000) & (df["run_type"] == "local")
    )
    # Note how we only have runs with 250k samples for e2e-recon
    e2e_recon_properties = (
        (df["lr"] == 0.001)
        & (df["n_samples"] == 250_000)
        & (df["run_type"] == "e2e-recon")
        & (df["in_to_orig_coeff"] == 1000)  # This is seemed the best for kl_coeff=0.5
    )
    df = df.loc[
        (e2e_properties | local_properties | e2e_recon_properties)
        & (df["ratio"] == 50)
        & (df["seed"] == 0)
    ]

    # Ignore runs that have an L0 bigger than d_resid
    df = df.loc[df["L0"] <= d_resid]

    return df


def tinystories_1m_plots():
    # Plot tinystories_1m performance
    run_types = ("e2e", "local", "e2e-recon")
    out_dir = Path(__file__).resolve().parent / "out" / "tinystories-1m" / "_".join(run_types)
    out_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving plots to {out_dir}")

    df = get_tinystories_1m_df()

    # ylims for plots with ce_diff on the y axis
    loss_increase_lims = {0: (0.4, 0), 3: (0.6, 0), 6: (0.6, 0)}
    # xlims for plots with L0 on the x axis
    l0_diff_xlims = {0: (40, 0), 3: (64, 0), 6: (64, 0)}
    # xlims for plots with alive_dict_elements on the x axis
    alive_dict_elements_xlims = {0: (None, None), 3: (1250, None), 6: (1000, None)}

    plot_two_axes_line(
        df,
        x1="L0",
        x2="alive_dict_elements",
        y="CELossIncrease",
        xlabel1="L0",
        xlabel2="Alive Dictionary Elements",
        ylabel="CE Loss Increase",
        out_file=out_dir / "l0_alive_dict_elements_vs_ce_loss.png",
        run_types=run_types,
        xlim1=l0_diff_xlims,
        xlim2=alive_dict_elements_xlims,
        ylim=loss_increase_lims,
    )


if __name__ == "__main__":
    gpt2_plots()
    tinystories_1m_plots()
