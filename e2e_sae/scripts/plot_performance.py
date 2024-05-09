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
from e2e_sae.scripts.plot_settings import (
    SIMILAR_CE_RUNS,
    SIMILAR_RUN_INFO,
    STYLE_MAP,
)


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
            if run_type not in STYLE_MAP:
                raise ValueError(f"Invalid run type: {run_type}")
            label, marker = STYLE_MAP[run_type]["label"], STYLE_MAP[run_type]["marker"]
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
                        marker=STYLE_MAP[run_type]["marker"],
                        color="black",
                        linestyle="None",
                        markersize=8,
                        label=STYLE_MAP[run_type]["label"],
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
    logger.info(f"Saved to {out_file}")


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

    sae_layers = list(run_ids.keys())
    n_sae_layers = len(sae_layers)

    if horz_layout:
        assert sae_layers == [2, 6, 10]
        fig, axs = plt.subplots(
            1, n_sae_layers, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 2, 1.2]}
        )
        legend_label_cols_and_precision = None
    else:
        fig, axs = plt.subplots(n_sae_layers, 1, figsize=(6, 4 * n_sae_layers))
        axs = np.atleast_1d(axs)

    def plot_metric(
        ax: plt.Axes,
        plot_df: pd.DataFrame,
        sae_layer: int,
        xs: NDArray[np.signedinteger[Any]],
    ) -> None:
        for _, row in plot_df.iterrows():
            run_type = row["run_type"]
            assert isinstance(run_type, str)
            legend_label = STYLE_MAP[run_type]["label"]
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
                **STYLE_MAP[run_type],  # type: ignore[reportArgumentType]
            )

    for i, sae_layer in enumerate(sae_layers):
        layer_df = df.loc[df["id"].isin(list(run_ids[sae_layer].values()))]

        ax = axs[i]

        xs = np.arange(sae_layer, final_layer + 1)
        for run_type in run_types:
            plot_metric(ax, layer_df.loc[layer_df["run_type"] == run_type], sae_layer, xs)

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
        logger.info(f"Saved to {out_file}")
    plt.close()


def _format_two_axes(axs: Sequence[plt.Axes], better_labels: bool) -> None:
    """Adds better arrows, and moves the y-axis to the right of the second subplot."""
    # move y-axis to right of second subplot
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.set_ticks_position("right")
    axs[1].yaxis.set_tick_params(color="white")

    if better_labels:
        axs[0].text(
            s="← Better",
            x=0.5,
            y=1.02,
            ha="center",
            va="bottom",
            fontsize=10,
            transform=axs[0].transAxes,
        )
        axs[1].text(
            s="← Better",
            x=0.5,
            y=1.02,
            ha="center",
            va="bottom",
            fontsize=10,
            transform=axs[1].transAxes,
        )
        axs[0].text(
            s="Better →",
            x=1.075,
            y=0.625,
            ha="center",
            va="top",
            fontsize=10,
            transform=axs[0].transAxes,
            rotation=90,
        )


def plot_two_axes_line_facet(
    df: pd.DataFrame,
    x1: str,
    x2: str,
    y: str,
    facet_by: str,
    line_by: str,
    sort_by: str | None = None,
    xlabel1: str | None = None,
    xlabel2: str | None = None,
    ylabel: str | None = None,
    suptitle: str | None = None,
    facet_vals: Sequence[Any] | None = None,
    xlim1: Mapping[Any, tuple[float | None, float | None]] | None = None,
    xlim2: Mapping[Any, tuple[float | None, float | None]] | None = None,
    xticks1: tuple[list[float], list[str]] | None = None,
    xticks2: tuple[list[float], list[str]] | None = None,
    yticks: tuple[list[float], list[str]] | None = None,
    ylim: Mapping[Any, tuple[float | None, float | None]] | None = None,
    styles: Mapping[Any, Mapping[str, Any]] | None = None,
    title: Mapping[Any, str] | None = None,
    legend_title: str | None = None,
    better_labels: bool = True,
    out_file: str | Path | None = None,
) -> None:
    """Line plot with two x-axes and one y-axis between them. One line for each run type.

    Args:
        df: DataFrame containing the data.
        x1: The variable to plot on the first x-axis.
        x2: The variable to plot on the second x-axis.
        y: The variable to plot on the y-axis.
        facet_by: The variable to facet the plot by.
        line_by: The variable to draw lines for.
        sort_by: The variable governing how lines are drawn between points. If None, lines will be
            drawn based on the y value.
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
        yticks: The y-ticks for the y-axis.
        ylim: The y-axis limits for each layer.
        styles: The styles to use for each line. If None, default styles will be used.
        title: The title for each row of the plot.
        legend_title: The title for the legend.
        better_labels: Whether to add "Better" text annotations.
        out_file: The filename which the plot will be saved as.
    """
    if facet_vals is None:
        facet_vals = sorted(df[facet_by].unique())
    if sort_by is None:
        sort_by = y

    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#f5f6fc"})
    fig = plt.figure(figsize=(8, 4 * len(facet_vals)), constrained_layout=True)
    subfigs = fig.subfigures(len(facet_vals))
    subfigs = np.atleast_1d(subfigs)

    # Get all unique line values from the entire DataFrame
    all_line_vals = df[line_by].unique()
    # Sort the line values based on the type of the line_by column
    if line_by == "run_type":
        sorted_line_vals = ["local", "e2e", "downstream"]
    else:
        sorted_line_vals = sorted(all_line_vals, key=str if df[line_by].dtype == object else float)
    # Get all unique line values from the entire DataFrame
    all_line_vals = df[line_by].unique()
    for subfig, facet_val in zip(subfigs, facet_vals, strict=False):
        axs = subfig.subplots(1, 2)
        facet_df = df.loc[df[facet_by] == facet_val]
        for line_val in sorted_line_vals:
            data = facet_df.loc[facet_df[line_by] == line_val]
            # for line_val, data in facet_df.groupby(line_by):
            line_style = {"label": line_val, "marker": "o", "linewidth": 1.1}  # default
            line_style.update(
                {} if styles is None else styles.get(line_val, {})
            )  # specific overrides
            if not data.empty:
                # draw the lines between points based on the y value
                data = data.sort_values(sort_by)
                axs[0].plot(data[x1], data[y], **line_style)
                axs[1].plot(data[x2], data[y], **line_style)
            else:
                # Add empty plots for missing line values to ensure they appear in the legend
                axs[0].plot([], [], **line_style)
                axs[1].plot([], [], **line_style)

        if facet_val == facet_vals[-1]:
            axs[0].legend(title=legend_title or line_by, loc="lower right")

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
        if yticks is not None:
            axs[0].set_yticks(yticks[0], yticks[1])
            axs[1].set_yticks(yticks[0], yticks[1])

        # add better labels and move right axis
        _format_two_axes(axs, better_labels)

    if suptitle is not None:
        fig.suptitle(suptitle)

    if out_file is not None:
        plt.savefig(out_file)
        plt.savefig(Path(out_file).with_suffix(".svg"))
        logger.info(f"Saved to {out_file}")

    plt.close(fig)


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
        if run_type not in STYLE_MAP:
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

    fig, ax = plt.subplots(figsize=(8, 6))
    assert isinstance(ax, plt.Axes)

    for run_id1, run_id2 in run_ids:
        run1 = df.loc[df["id"] == run_id1]
        run2 = df.loc[df["id"] == run_id2]
        assert not run1.empty and not run2.empty, f"Run ID not found: {run_id1} or {run_id2}"
        assert run1["run_type"].nunique() == 1 and run2["run_type"].nunique() == 1
        run_type = run1["run_type"].iloc[0]
        assert run_type == run2["run_type"].iloc[0]

        ax.scatter(
            [run1["L0"].iloc[0], run2["L0"].iloc[0]],
            [run1["CELossIncrease"].iloc[0], run2["CELossIncrease"].iloc[0]],
            **STYLE_MAP[run_type],  # type: ignore[reportArgumentType]
        )

        ax.plot(
            [run1["L0"].iloc[0], run2["L0"].iloc[0]],
            [run1["CELossIncrease"].iloc[0], run2["CELossIncrease"].iloc[0]],
            **STYLE_MAP[run_type],  # type: ignore[reportArgumentType]
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

    layers = list(sorted(df["layer"].unique()))
    layers_str = "-".join(map(str, layers))
    ax.set_title(f"Seed Comparison layers {layers_str}: L0 vs CE Loss Increase")

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
        xlim1={run_type: (0, 200) for run_type in run_types},
        xlim2={run_type: (0, 45_000) for run_type in run_types},
        xticks2=([0, 10_000, 20_000, 30_000, 40_000], ["0", "10k", "20k", "30k", "40k"]),
        ylim={run_type: (0.5, 0) for run_type in run_types},
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
        xlim1={run_type: (0, 200) for run_type in run_types},
        xlim2={run_type: (10_000, 48_000) for run_type in run_types},
        xticks2=([10_000, 20_000, 30_000, 40_000], ["10k", "20k", "30k", "40k"]),
        ylim={run_type: (0.75, 0) for run_type in run_types},
        title={run_type: run_type for run_type in run_types},
        out_file=n_samples_dir / "n_samples_comparison.png",
        xlabel2="Alive Dictionary Elements",
        ylabel="CE Loss Increase",
    )


def create_summary_latex_tables(df: pd.DataFrame, out_dir: Path) -> None:
    """Create summary for similar CE and similar L0 runs and print as latex tables.

    Args:
        df: DataFrame containing the data.
        out_dir: The directory to save the tables to.
    """
    col_map = {
        "layer": "Layer",
        "run_type": "RunType",
        "sparsity_coeff": "$\\lambda$",
        "L0": "$L_0$",
        "alive_dict_elements": "AliveElements",
        "mean_grad_norm": "GradNorm",
        "CELossIncrease": "CELossIncrease",
    }
    for similar_run_var, run_group in SIMILAR_RUN_INFO.items():
        run_group_name = f"constant_{similar_run_var}"
        layer_dfs = {}
        run_types: list[str] | None = None
        for layer, run_info in run_group.items():
            if run_types is None:
                run_types = list(run_info.keys())
            layer_df = df.loc[df["layer"] == layer]
            layer_df = layer_df.loc[layer_df["id"].isin(run_info.values()), col_map.keys()]
            layer_dfs[layer] = layer_df
        run_group_df = pd.concat(layer_dfs.values())

        # Sort by layer and run_type. Layer should be in numerical order.
        # Run type should be in the order "local", "e2e", "downstream"
        layer_order = {layer: i for i, layer in enumerate(run_group.keys())}
        assert run_types is not None
        run_type_order = {run_type: len(layer_order) + i for i, run_type in enumerate(run_types)}
        order = {**layer_order, **run_type_order}
        run_group_df = run_group_df.sort_values(["layer", "run_type"], key=lambda x: x.map(order))

        # Format the values in the DataFrame
        run_group_df["sparsity_coeff"] = run_group_df["sparsity_coeff"].apply(lambda x: f"{x:.2f}")
        run_group_df["L0"] = run_group_df["L0"].apply(lambda x: f"{x:.1f}")
        run_group_df["alive_dict_elements"] = run_group_df["alive_dict_elements"].apply(
            lambda x: f"{round(x, -3) // 1000}k"
        )
        run_group_df["mean_grad_norm"] = run_group_df["mean_grad_norm"].apply(lambda x: f"{x:.2f}")
        run_group_df["CELossIncrease"] = run_group_df["CELossIncrease"].apply(lambda x: f"{x:.3f}")

        run_group_df = run_group_df.rename(columns=col_map)
        # Create the LaTeX table
        latex_str = "\\begin{table}[h]\n\\centering\n"
        # The grey column will be L0 if run_group_name is constant_l0, and CE Loss otherwise
        if run_group_name == "constant_l0":
            latex_str += "\\begin{tabular}{|c|c|c|>{\\columncolor[gray]{0.9}}c|c|c|c|}\n\\hline\n"
        else:
            latex_str += "\\begin{tabular}{|c|c|c|c|c|c|>{\\columncolor[gray]{0.9}}c|}\n\\hline\n"
        # Same as above but have the c's depend on the number of columns
        latex_str += " & ".join(run_group_df.columns) + " \\\\\n\\hline\n"

        for _, layer_df in run_group_df.groupby("Layer"):
            for _, row in layer_df.iterrows():
                latex_str += (
                    " & ".join(["" if pd.isna(val) else str(val) for val in row.values]) + " \\\\\n"
                )
            latex_str += "\\hline\n"

        latex_str += "\\end{tabular}\n"
        latex_str += f"\\caption{{Comparison of runs with similar {'$L_0$' if run_group_name == 'constant_l0' else 'CE Loss'} for each block}}\n"
        latex_str += f"\\label{{tab:{run_group_name}}}\n"
        latex_str += "\\end{table}\n"

        out_file = out_dir / run_group_name / f"{run_group_name}_summary.tex"
        out_file.parent.mkdir(exist_ok=True, parents=True)
        with open(out_file, "w") as f:
            f.write(latex_str)
        logger.info(f"Saved to {out_file}")


def get_df_gpt2() -> pd.DataFrame:
    api = wandb.Api()
    project = "sparsify/gpt2"
    runs = api.runs(project)

    d_resid = 768

    df = create_run_df(runs)

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


def gpt2_plots():
    run_types = ("local", "e2e", "downstream")
    n_layers = 12

    df = get_df_gpt2()

    layers = list(sorted(df["layer"].unique()))

    plot_n_samples_comparison(
        df=df.loc[(df["ratio"] == 60) & (df["seed"] == 0) & (df["lr"] == 5e-4)],
        out_dir=Path(__file__).resolve().parent / "out",
        run_types=run_types,
    )
    # # 1 seed in each layer
    local_seed_ids = [("ue3lz0n7", "d8vgjnyc"), ("1jy3m5j0", "uqfp43ti"), ("m2hntlav", "77bp68uk")]
    e2e_seed_ids = [("ovhfts9n", "slxwr007"), ("tvj2owza", "atfccmo3"), ("jnjpmyqk", "ac9i1g6v")]
    e2e_recon_ids = [("y8sca507", "hqo5azo2")]
    seed_ids = local_seed_ids + e2e_seed_ids + e2e_recon_ids
    plot_seed_comparison(
        df=df.loc[df["lr"] == 5e-4],
        run_ids=seed_ids,
        out_dir=Path(__file__).resolve().parent / "out" / "seed_comparison",
    )
    plot_ratio_comparison(
        df=df.loc[(df["seed"] == 0) & (df["n_samples"] == 400_000) & (df["lr"] == 5e-4)],
        out_dir=Path(__file__).resolve().parent / "out" / "ratio_comparison",
        run_types=run_types,
    )

    # Local lr comparison
    local_lr_df = df.loc[
        (df["seed"] == 0)
        & (df["n_samples"] == 400_000)
        & (df["ratio"] == 60)
        & (df["run_type"] == "local")
        & ~((df["L0"] > 300) & (df["layer"] == 2))  # Avoid points in L0 but not alive_dict subplot
    ]
    plot_two_axes_line_facet(
        df=local_lr_df,
        x1="L0",
        x2="alive_dict_elements",
        y="CELossIncrease",
        facet_by="layer",
        facet_vals=layers,
        line_by="lr",
        xlabel1="L0",
        xlabel2="Alive Dictionary Elements",
        ylabel="CE Loss Increase",
        xlim1={2: (0.0, 200.0), 6: (0.0, 200.0), 10: (0.0, 300.0)},
        xlim2={layer: (0, 45_000) for layer in layers},
        xticks2=([0, 10_000, 20_000, 30_000, 40_000], ["0", "10k", "20k", "30k", "40k"]),
        ylim={layer: (0.4, 0) for layer in layers},
        title={layer: f"Layer {layer}" for layer in layers},
        out_file=Path(__file__).resolve().parent
        / "out"
        / "lr_comparison"
        / "local_lr_comparison.png",
    )
    # e2e lr comparison
    e2e_lr_df = df.loc[
        (df["seed"] == 0)
        & (df["n_samples"] == 400_000)
        & (df["ratio"] == 60)
        & (df["run_type"] == "e2e")
        & ~((df["L0"] > 300) & (df["layer"] == 2))  # Avoid points in L0 but not alive_dict subplot
        & (~df["name"].str.contains("seed-comparison"))
    ]
    plot_two_axes_line_facet(
        df=e2e_lr_df,
        x1="L0",
        x2="alive_dict_elements",
        y="CELossIncrease",
        facet_by="layer",
        facet_vals=layers,
        line_by="lr",
        xlabel1="L0",
        xlabel2="Alive Dictionary Elements",
        ylabel="CE Loss Increase",
        xlim1={2: (0.0, 200.0), 6: (0.0, 200.0), 10: (0.0, 300.0)},
        xlim2={layer: (0, 45_000) for layer in layers},
        xticks2=([0, 10_000, 20_000, 30_000, 40_000], ["0", "10k", "20k", "30k", "40k"]),
        ylim={layer: (0.4, 0) for layer in layers},
        title={layer: f"Layer {layer}" for layer in layers},
        out_file=Path(__file__).resolve().parent
        / "out"
        / "lr_comparison"
        / "e2e_lr_comparison.png",
    )
    out_dir = Path(__file__).resolve().parent / "out" / "_".join(run_types)
    out_dir.mkdir(exist_ok=True, parents=True)

    performance_df = df.loc[(df["ratio"] == 60) & (df["seed"] == 0) & (df["n_samples"] == 400_000)]
    # For layer 2 we filter out the runs with L0 > 200. Otherwise we end up with point in one
    # subplot but not the other
    performance_df = performance_df.loc[
        ~((performance_df["L0"] > 200) & (performance_df["layer"] == 2))
    ]
    # Ignore specialised runs
    performance_df = performance_df.loc[
        ~performance_df["name"].str.contains("seed-comparison")
        & ~performance_df["name"].str.contains("lr-comparison")
        & ~performance_df["name"].str.contains("lower-downstream")
        & -performance_df["name"].str.contains("e2e-local")
        & ~performance_df["name"].str.contains("recon-all")
        & ~performance_df["name"].str.contains("misc_")
    ]

    create_summary_latex_tables(df=performance_df, out_dir=Path(__file__).resolve().parent / "out")

    # ylims for plots with ce_diff on the y axis
    loss_increase_lims = {2: (0.2, 0.0), 6: (0.4, 0.0), 10: (0.4, 0.0)}
    # xlims for plots with L0 on the x axis
    l0_diff_xlims = {2: (0.0, 200.0), 6: (0.0, 600.0), 10: (0.0, 600.0)}

    plot_two_axes_line_facet(
        performance_df,
        x1="mean_grad_norm",
        x2="CELossIncrease",
        y="alive_dict_elements",
        facet_by="layer",
        facet_vals=layers,
        line_by="run_type",
        sort_by="CELossIncrease",
        xlabel1="Mean Grad Norm",
        xlabel2="CE Loss Increase",
        ylabel="Alive Dictionary Elements",
        xlim2={2: (0.0, 0.2), 6: (0.0, 0.4), 10: (0.0, 0.4)},
        yticks=([0, 10_000, 20_000, 30_000, 40_000], ["0", "10k", "20k", "30k", "40k"]),
        title={layer: f"Layer {layer}" for layer in layers},
        out_file=out_dir / "grad_norm_vs_ce_loss_vs_alive_dict_elements.png",
        styles=STYLE_MAP,
        better_labels=False,
        legend_title="Run Type",
    )
    # Pareto curve plots (two axes line plots with L0 and alive_dict_elements on the x-axis)
    # all layers
    plot_two_axes_line_facet(
        performance_df,
        x1="L0",
        x2="alive_dict_elements",
        y="CELossIncrease",
        facet_by="layer",
        facet_vals=layers,
        line_by="run_type",
        xlabel1="L0",
        xlabel2="Alive Dictionary Elements",
        ylabel="CE Loss Increase",
        xlim1=l0_diff_xlims,
        xlim2={layer: (0, 45_000) for layer in layers},
        xticks2=([0, 10_000, 20_000, 30_000, 40_000], ["0", "10k", "20k", "30k", "40k"]),
        ylim=loss_increase_lims,
        title={layer: f"Layer {layer}" for layer in layers},
        out_file=out_dir / "l0_alive_dict_elements_vs_ce_loss.png",
        styles=STYLE_MAP,
        legend_title="Run Type",
    )
    # just layer 6
    plot_two_axes_line_facet(
        performance_df,
        x1="L0",
        x2="alive_dict_elements",
        y="CELossIncrease",
        facet_by="layer",
        facet_vals=[6],
        line_by="run_type",
        xlabel1="L0",
        xlabel2="Alive Dictionary Elements",
        ylabel="CE Loss Increase",
        out_file=out_dir / "l0_alive_dict_elements_vs_ce_loss_layer_6.png",
        xlim1=l0_diff_xlims,
        xticks2=([0, 10_000, 20_000, 30_000, 40_000], ["0", "10k", "20k", "30k", "40k"]),
        ylim=loss_increase_lims,
        styles=STYLE_MAP,
        legend_title="Run Type",
    )
    # Calculate the summary metric for the CE ratio difference
    calc_summary_metric(
        df=performance_df,
        x1="L0",
        x2="alive_dict_elements",
        out_file=out_dir / "l0_alive_dict_elements_vs_ce_loss_summary.json",
        interpolate_layer=6,
        x1_interpolation_range=(300, 50),
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
        run_ids=SIMILAR_CE_RUNS,
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
        run_ids=SIMILAR_CE_RUNS,
        metric="recon_loss",
        final_layer=final_layer,
        out_file=out_dir / "recon_loss_per_layer.png",
        ylim=(0, None),
        run_types=run_types,
        horz_layout=True,
    )
    plot_per_layer_metric(
        performance_df,
        run_ids={6: SIMILAR_CE_RUNS[6]},
        metric="recon_loss",
        final_layer=final_layer,
        out_file=out_dir / "recon_loss_per_layer_layer_6.png",
        ylim=(0, 25),
        run_types=run_types,
        horz_layout=False,
    )


def get_tinystories_1m_df() -> pd.DataFrame:
    api = wandb.Api()
    project = "sparsify/tinystories-1m-2"
    runs = api.runs(project)

    d_resid = 64

    df = create_run_df(runs, per_layer_metrics=False, use_run_name=True, grad_norm=False)

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
    l0_diff_xlims = {0: (0, 40), 3: (0, 64), 6: (0, 64)}
    # xlims for plots with alive_dict_elements on the x axis

    plot_two_axes_line_facet(
        df,
        x1="L0",
        x2="alive_dict_elements",
        y="CELossIncrease",
        facet_by="layer",
        facet_vals=[3],
        line_by="run_type",
        xlabel1="L0",
        xlabel2="Alive Dictionary Elements",
        ylabel="CE Loss Increase",
        out_file=out_dir / "l0_alive_dict_elements_vs_ce_loss_layer_3.png",
        xlim1=l0_diff_xlims,
        xlim2={3: (0, None)},
        ylim={3: (0.6, 0)},
        styles=STYLE_MAP,
        legend_title="Run Type",
    )


if __name__ == "__main__":
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#f5f6fc"})
    gpt2_plots()
    tinystories_1m_plots()
