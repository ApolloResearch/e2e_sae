import json
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
from e2e_sae.scripts.geometric_analysis import COLOR_MAP

RUN_TYPE_MAP = {
    "e2e": ("End-to-end", "o"),
    "downstream": ("Downstream", "X"),
    "local": ("Local", "^"),
}

# Runs with constant CE loss increase for each layer. Values represent wandb run IDs.
CONSTANT_CE_RUNS = {
    2: {"e2e": "ovhfts9n", "local": "ue3lz0n7", "downstream": "visi12en"},
    6: {"e2e": "zgdpkafo", "local": "1jy3m5j0", "downstream": "2lzle2f0"},
    10: {"e2e": "8crnit9h", "local": "m2hntlav", "downstream": "cvj5um2h"},
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
    run_types: tuple[str, ...] = ("local", "e2e", "downstream"),
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
    run_types: Sequence[str] = ("local", "e2e", "downstream"),
    horz_layout: bool = False,
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
        horz_layout: Whether to use a horizontal layout for the subplots. Requires sae_layers to be
            exactly [2, 6, 10]. Ignores legend_label_cols_and_precision if True.
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
    color_map = {"e2e": color_e2e, "local": color_lws, "downstream": color_e2e_recon}

    if horz_layout:
        assert sae_layers == [2, 6, 10]
        fig, axs = plt.subplots(
            1, n_sae_layers, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 2, 1.2]}
        )
        legend_label_cols_and_precision = None
    else:
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
        layer_df = df.loc[df["id"].isin(list(run_ids[sae_layer].values()))]

        ax = axs[i]

        xs = np.arange(sae_layer, final_layer + 1)
        for run_type in run_types:
            if run_type not in RUN_TYPE_MAP:
                raise ValueError(f"Invalid run type: {run_type}")
            label, marker = RUN_TYPE_MAP[run_type]
            plot_metric(ax, layer_df.loc[layer_df["run_type"] == run_type], marker, sae_layer, xs)

        ax.set_title(f"SAE Layer {sae_layer}", fontweight="bold")
        ax.set_xlabel("Model Layer")
        if (not horz_layout) or i == 0:
            ax.legend(title="Run Type", loc="best")
            ax.set_ylabel(metric_name)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_ylim(ylim)

    plt.tight_layout()
    if out_file is not None:
        plt.savefig(out_file)
        # Save as svg also
        plt.savefig(Path(out_file).with_suffix(".svg"))
    plt.close()


def _format_two_axes(axs: Sequence[plt.Axes]) -> None:
    """Adds better arrows, and moves the y-axis to the right of the second subplot."""
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


def plot_two_axes_line_single_run_type(
    df: pd.DataFrame,
    run_type: str,
    out_dir: Path,
    title: str,
    iter_var: str,
    iter_vals: list[float],
    filename_prefix: str = "",
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
        filename_prefix: The prefix to add to the filename.
    """
    colors = sns.color_palette("tab10", n_colors=len(iter_vals))
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

    # add better labels and move right axis
    _format_two_axes((axs[0], axs[1]))

    fig.suptitle(title)
    filename = f"{filename_prefix}_l0_alive_elements_vs_ce_loss_{run_type}.png"
    plt.savefig(out_dir / filename)
    plt.savefig(out_dir / filename.replace(".png", ".svg"))
    plt.close(fig)
    logger.info(f"Saved to {out_dir / filename}")


# TODO: replace calls with plot_two_axes_line_facet (which is a bit more general)
def plot_two_axes_line_facet(
    df: pd.DataFrame,
    x1: str,
    x2: str,
    y: str,
    facet_by: str,
    line_by: str,
    xlabel1: str | None = None,
    xlabel2: str | None = None,
    ylabel: str | None = None,
    suptitle: str | None = None,
    facet_vals: Sequence[Any] | None = None,
    xlim1: Mapping[Any, tuple[float | None, float | None]] | None = None,
    xlim2: Mapping[Any, tuple[float | None, float | None]] | None = None,
    xticks1: tuple[list[float], list[str]] | None = None,
    xticks2: tuple[list[float], list[str]] | None = None,
    ylim: Mapping[Any, tuple[float | None, float | None]] | None = None,
    styles: Mapping[Any, Mapping[str, Any]] | None = None,
    title: Mapping[Any, str] | None = None,
    out_file: str | Path | None = None,
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
    if facet_vals is None:
        facet_vals = sorted(df[facet_by].unique())

    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#f5f6fc"})
    fig = plt.figure(figsize=(8, 4 * len(facet_vals)), constrained_layout=True)
    subfigs = fig.subfigures(len(facet_vals))
    subfigs = np.atleast_1d(subfigs)

    for subfig, facet_val in zip(subfigs, facet_vals, strict=False):
        axs = subfig.subplots(1, 2)
        facet_df = df.loc[df[facet_by] == facet_val]
        for line_val, data in facet_df.groupby(line_by):
            line_style = {"label": line_val, "marker": "o", "linewidth": 1.1}  # default
            line_style.update({} if styles is None else styles[line_val])  # specific overrides
            if not data.empty:
                # draw the lines between points based on the y value
                data = data.sort_values(y)
                axs[0].plot(data[x1], data[y], **line_style)
                axs[1].plot(data[x2], data[y], **line_style)

        if facet_val == facet_vals[-1]:
            axs[0].legend(title=line_by, loc="lower left")

        if xlim1 is not None:
            axs[0].set_xlim(xmin=xlim1[facet_val][0], xmax=xlim1[facet_val][1])
        if xlim2 is not None:
            axs[1].set_xlim(xmin=xlim2[facet_val][0], xmax=xlim2[facet_val][1])
        if ylim is not None:
            axs[0].set_ylim(ymin=ylim[facet_val][0], ymax=ylim[facet_val][1])
            axs[1].set_ylim(ymin=ylim[facet_val][0], ymax=ylim[facet_val][1])

        # Set a title above axs[0] and axs[1] to show the layer number
        row_title = title[facet_val] if title is not None else None
        subfig.suptitle(row_title, fontweight="bold")
        axs[0].set_xlabel(xlabel1 or x1)
        axs[1].set_xlabel(xlabel2 or x2)
        axs[0].set_ylabel(ylabel or y)
        axs[1].set_ylabel(ylabel or y)

        if xticks1 is not None:
            axs[0].set_xticks(xticks1[0], xticks1[1])
        if xticks2 is not None:
            axs[1].set_xticks(xticks2[0], xticks2[1])

        # add better labels and move right axis
        _format_two_axes(axs)

    if suptitle is not None:
        fig.suptitle(suptitle)

    if out_file is not None:
        plt.savefig(out_file)
        plt.savefig(Path(out_file).with_suffix(".svg"))
        logger.info(f"Saved to {out_file}")

    plt.close(fig)


def plot_two_axes_line(
    df: pd.DataFrame,
    x1: str,
    x2: str,
    y: str,
    xlabel1: str,
    xlabel2: str,
    ylabel: str,
    out_file: str | Path,
    title: str | None = None,
    run_types: Sequence[str] = ("local", "e2e", "downstream"),
    xlim1: Mapping[int, tuple[float | None, float | None]] | None = None,
    xlim2: Mapping[int, tuple[float | None, float | None]] | None = None,
    xticks1: tuple[list[float], list[str]] | None = None,
    xticks2: tuple[list[float], list[str]] | None = None,
    ylim: Mapping[int, tuple[float | None, float | None]] | None = None,
    layers: Sequence[int] | None = None,
    color_map: Mapping[str, Any] | None = None,
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
            color = COLOR_MAP[run_type] if color_map is None else color_map[run_type]
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
                        alpha=1,
                        label=label if i == 0 else None,
                        color=color,
                        markersize=5,
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

        if xticks1 is not None:
            axs[0].set_xticks(xticks1[0], xticks1[1])
        if xticks2 is not None:
            axs[1].set_xticks(xticks2[0], xticks2[1])

        # add better labels and move right axis
        _format_two_axes(axs)

    if title is not None:
        fig.suptitle(title)

    plt.savefig(out_file)
    plt.savefig(Path(out_file).with_suffix(".svg"))
    plt.close(fig)
    logger.info(f"Saved to {out_file}")


def calc_summary_metric(
    df: pd.DataFrame,
    x1: str,
    x2: str,
    out_file: str | Path,
    interpolate_layer: int,
    x1_interpolation_range: tuple[float, float],
    x2_interpolation_range: tuple[float, float],
    y: str = "CELossIncrease",
    run_types: Sequence[str] = ("local", "e2e", "downstream"),
) -> None:
    """Calculate and save the summary metric for the ratio difference in the y-axis.

    Args:
        df: DataFrame containing the data.
        x1: The variable to plot on the first x-axis.
        x2: The variable to plot on the second x-axis.
        out_file: The filename which the summary metric will be saved as.
        interpolate_layer: The layer to interpolate x values for.
        x1_interpolation_range: The range of x1 values to interpolate over.
        x2_interpolation_range: The range of x2 values to interpolate over.
        y: The variable to plot on the y-axis.
        run_types: The run types to include in the calculation.
    """
    # Take 20 points between the interpolation range
    x1_interpolation_lines = np.linspace(
        x1_interpolation_range[0], x1_interpolation_range[1], num=20
    )
    x2_interpolation_lines = np.linspace(
        x2_interpolation_range[0], x2_interpolation_range[1], num=20
    )

    layer_df = df.loc[df["layer"] == interpolate_layer]
    intersections = {x1: {}, x2: {}}
    for run_type in run_types:
        if run_type not in RUN_TYPE_MAP:
            raise ValueError(f"Invalid run type: {run_type}")
        data = layer_df.loc[layer_df["run_type"] == run_type]
        if not data.empty:
            data = data.sort_values(y)
            for metric, vertical_lines in zip(
                [x1, x2], [x1_interpolation_lines, x2_interpolation_lines], strict=True
            ):
                # np.interp requires monotically increasing data.
                if np.all(np.diff(data[metric]) > 0):
                    interp_x = data[metric].copy()
                    interp_y = data[y].copy()
                elif np.all(np.diff(data[metric]) < 0):
                    # Reverse the order of xs and ys
                    interp_x = data[metric].copy()[::-1]
                    interp_y = data[y].copy()[::-1]
                else:
                    raise ValueError(
                        f"Data in column {metric} is not monotonic, cannot interpolate"
                    )
                intersections[metric][run_type] = np.interp(vertical_lines, interp_x, interp_y)

    # Calculate the summary statistic for the ratio difference in the y-axis
    # Our summary metrics are always compared to the "local" run type
    comparison_cols = [run_type for run_type in run_types if run_type != "local"]
    ce_ratios = {}
    for col in comparison_cols:
        ce_ratios[col] = {}
        for metric in [x1, x2]:
            ce_ratios[col][metric] = (
                (intersections[metric][col] / intersections[metric]["local"]).mean().item()
            )
    with open(out_file, "w") as f:
        json.dump(ce_ratios, f)
    logger.info(f"Summary saved to {out_file}")


def plot_seed_comparison(
    df: pd.DataFrame,
    run_ids: Sequence[tuple[str, str]],
    out_dir: Path,
) -> None:
    """Plot the CE loss difference vs L0 for all runs, comparing seeds.

    Args:
        df: DataFrame containing the data.
        run_ids: List of run id tuples indicating the same runs with different seeds.
        out_dir: The directory to save the plots to.
    """
    out_dir.mkdir(exist_ok=True, parents=True)

    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#f5f6fc"})
    fig, ax = plt.subplots(figsize=(8, 6))
    assert isinstance(ax, plt.Axes)

    for run_id1, run_id2 in run_ids:
        run1 = df.loc[df["id"] == run_id1]
        run2 = df.loc[df["id"] == run_id2]
        assert not run1.empty and not run2.empty, f"Run ID not found: {run_id1} or {run_id2}"
        assert run1["run_type"].nunique() == 1 and run2["run_type"].nunique() == 1
        run_type = run1["run_type"].iloc[0]
        assert run_type == run2["run_type"].iloc[0]
        color = sns.color_palette()[list(RUN_TYPE_MAP.keys()).index(run_type)]

        ax.scatter(
            [run1["L0"].iloc[0], run2["L0"].iloc[0]],
            [run1["CELossIncrease"].iloc[0], run2["CELossIncrease"].iloc[0]],
            color=color,
            label=run_type,
        )

        ax.plot(
            [run1["L0"].iloc[0], run2["L0"].iloc[0]],
            [run1["CELossIncrease"].iloc[0], run2["CELossIncrease"].iloc[0]],
            color=color,
        )

    ax.set_xlabel("L0")
    ax.set_ylabel("CE Loss Increase")
    # Ensure that there is only one legend entry for each run type, but ensure the colours are correct
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for label, handle in zip(labels, handles, strict=False):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    ax.legend(unique_handles, unique_labels, title="Run Type", loc="lower right")

    ax.set_title("L0 vs CE Loss Increase (Seed Comparison)")

    plt.tight_layout()
    out_file = out_dir / "seed_comparison.png"
    plt.savefig(out_file)
    plt.savefig(out_file.with_suffix(".svg"))
    plt.close()
    logger.info(f"Saved to {out_file}")


def plot_ratio_comparison(df: pd.DataFrame, out_dir: Path, run_types: Sequence[str]) -> None:
    """Plots alive_dict_elements and L0 vs CE loss increase for different dictionary ratios.

    Args:
        df: DataFrame containing the data.
        out_dir: The directory to save the plots to.
        run_types: The run types to include in the plot.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    ratios_runs = df.loc[df["ratio"] != 60]

    dfs_to_plot = []
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
        dfs_to_plot.append(pd.concat([run_type_df, runs_ratio_60]))

    combined_df = pd.concat(dfs_to_plot)
    plot_two_axes_line_facet(
        df=combined_df,
        x1="L0",
        x2="alive_dict_elements",
        y="CELossIncrease",
        facet_by="run_type",
        facet_vals=run_types,
        line_by="ratio",
        xlim1={run_type: (300, 0) for run_type in run_types},
        xlim2={run_type: (0, 45_000) for run_type in run_types},
        xticks2=([0, 10_000, 20_000, 30_000, 40_000], ["0", "10k", "20k", "30k", "40k"]),
        ylim={run_type: (0.75, 0) for run_type in run_types},
        title={run_type: run_type for run_type in run_types},
        out_file=out_dir / "ratio_comparison.png",
        xlabel2="Alive Dictionary Elements",
        ylabel="CE Loss Increase",
    )


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

    dfs_to_plot = []

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
        dfs_to_plot.append(pd.concat([run_type_df, runs_samples_400k]))

    combined_df = pd.concat(dfs_to_plot)
    plot_two_axes_line_facet(
        df=combined_df,
        x1="L0",
        x2="alive_dict_elements",
        y="CELossIncrease",
        facet_by="run_type",
        facet_vals=run_types,
        line_by="n_samples",
        xlim1={run_type: (400, 0) for run_type in run_types},
        xlim2={run_type: (0, 48_000) for run_type in run_types},
        xticks2=([0, 10_000, 20_000, 30_000, 40_000], ["0", "10k", "20k", "30k", "40k"]),
        ylim={run_type: (0.75, 0) for run_type in run_types},
        title={run_type: run_type for run_type in run_types},
        out_file=out_dir / "n_samples_comparison.png",
        xlabel2="Alive Dictionary Elements",
        ylabel="CE Loss Increase",
    )


def get_df_gpt2() -> pd.DataFrame:
    api = wandb.Api()
    project = "sparsify/gpt2"
    runs = api.runs(project)

    d_resid = 768

    df = create_run_df(runs)

    # df for all run_types except for local should only have one unique lr (5e-4)
    assert (
        df.loc[df["run_type"] != "local"]["lr"].nunique() == 1
        and df.loc[df["run_type"] != "local"]["lr"].unique()[0] == 5e-4
    )

    # assert df["lr"].nunique() == 1 and df["lr"].unique()[0] == 5e-4
    assert df["model_name"].nunique() == 1

    # Ignore runs that have an L0 bigger than d_resid
    df = df.loc[df["L0"] <= d_resid]
    # Only use the e2e+recon run in layer 10 that has kl_coeff=0.75
    # df = df.loc[~((df["layer"] == 10) & (df["run_type"] == "downstream") & (df["kl_coeff"] != 0.75))]
    df = df.loc[
        ~(
            (df["layer"] == 10)
            & (df["run_type"] == "downstream")
            & ((df["kl_coeff"] != 0.5) | (df["in_to_orig_coeff"] != 0.05))
        )
    ]
    return df


def plot_local_lr_comparison(df: pd.DataFrame, out_dir: Path, run_types: Sequence[str]) -> None:
    """Plot a two axes line plot with L0 and alive_dict_elements on the x-axis and CE loss increase
    on the y-axis. Colored by learning rate.
    """
    out_dir.mkdir(exist_ok=True, parents=True)

    lrs = df["lr"].unique()
    for layer in df["layer"].unique():
        layer_df = df.loc[df["layer"] == layer]
        if layer_df.empty:
            continue
        # Get all the local runs. These should be the runs that have kl_coeff of 0.0 or nan
        local_df = layer_df.loc[(layer_df["kl_coeff"] == 0.0) | layer_df["kl_coeff"].isnull()]
        if local_df.empty:
            continue

        sae_pos = local_df["sae_pos"].unique()[0]
        lrs = sorted(local_df["lr"].unique())
        # Plot a two axes line plot with L0 and alive_dict_elements on the x-axis and CE loss
        # increase on the y-axis. Colored by learning rate

        plot_two_axes_line_single_run_type(
            df=local_df,
            run_type="local",
            out_dir=out_dir,
            title=f"Local: L0 + Alive Elements vs CE Loss Increase {sae_pos}",
            iter_var="lr",
            iter_vals=lrs,
            filename_prefix="lr",
        )


def gpt2_plots():
    run_types = ("local", "e2e", "downstream")
    n_layers = 12

    df = get_df_gpt2()

    plot_n_samples_comparison(
        df=df.loc[(df["ratio"] == 60) & (df["seed"] == 0)],
        out_dir=Path(__file__).resolve().parent / "out",
        run_types=run_types,
    )

    # 1 seed in each layer
    local_seed_ids = [("ue3lz0n7", "d8vgjnyc"), ("1jy3m5j0", "uqfp43ti"), ("m2hntlav", "77bp68uk")]
    e2e_seed_ids = [("ovhfts9n", "slxwr007"), ("tvj2owza", "atfccmo3"), ("jnjpmyqk", "ac9i1g6v")]
    e2e_recon_ids = [("y8sca507", "hqo5azo2")]
    seed_ids = local_seed_ids + e2e_seed_ids + e2e_recon_ids
    plot_seed_comparison(
        df=df,
        run_ids=seed_ids,
        out_dir=Path(__file__).resolve().parent / "out" / "seed_comparison",
    )

    plot_ratio_comparison(
        df=df.loc[(df["seed"] == 0) & (df["n_samples"] == 400_000)],
        out_dir=Path(__file__).resolve().parent / "out" / "ratio_comparison",
        run_types=run_types,
    )

    # Layer 6 local learning rate comparison
    plot_local_lr_comparison(
        df=df.loc[
            (df["seed"] == 0)
            & (df["n_samples"] == 400_000)
            & (df["ratio"] == 60)
            & (df["run_type"] == "local")
            & (df["layer"] == 6)
        ],
        out_dir=Path(__file__).resolve().parent / "out" / "lr_comparison",
        run_types="local",
    )

    out_dir = Path(__file__).resolve().parent / "out" / "_".join(run_types)
    out_dir.mkdir(exist_ok=True, parents=True)

    performance_df = df.loc[(df["ratio"] == 60) & (df["seed"] == 0) & (df["n_samples"] == 400_000)]
    # For layer 2 we filter out the runs with L0 > 200. Otherwise we end up with point in one
    # subplot but not the other
    performance_df = performance_df.loc[
        ~((performance_df["L0"] > 200) & (performance_df["layer"] == 2))
    ]
    # Some runs with seed-comparison in the name are duplicates, ignore those
    performance_df = performance_df.loc[~performance_df["name"].str.contains("seed-comparison")]

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
    # Calculate the summary metric for the CE ratio difference
    calc_summary_metric(
        df=performance_df,
        x1="L0",
        x2="alive_dict_elements",
        out_file=out_dir / "l0_alive_dict_elements_vs_ce_loss_summary.json",
        interpolate_layer=6,
        x1_interpolation_range=(500, 50),
        x2_interpolation_range=(23000, 35000),
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
        horz_layout=True,
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
        horz_layout=True,
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
    # Note how we only have runs with 250k samples for downstream
    e2e_recon_properties = (
        (df["lr"] == 0.001)
        & (df["n_samples"] == 250_000)
        & (df["run_type"] == "downstream")
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
    run_types = ("local", "e2e", "downstream")
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
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#f5f6fc"})
    gpt2_plots()
    tinystories_1m_plots()
