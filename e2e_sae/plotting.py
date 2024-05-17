from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

from e2e_sae.log import logger


def plot_per_layer_metric(
    df: pd.DataFrame,
    run_ids: Mapping[int, Mapping[str, str]],
    metric: str,
    final_layer: int,
    run_types: Sequence[str],
    out_file: str | Path | None = None,
    ylim: tuple[float | None, float | None] = (None, None),
    legend_label_cols_and_precision: list[tuple[str, int]] | None = None,
    legend_title: str | None = None,
    styles: Mapping[str, Mapping[str, Any]] | None = None,
    horz_layout: bool = False,
    show_ax_titles: bool = True,
    save_svg: bool = True,
) -> None:
    """
    Plot the per-layer metric (explained variance or reconstruction loss) for different run types.

    Args:
        df: DataFrame containing the filtered data for the specific layer.
        run_ids: The run IDs to use. Format: {layer: {run_type: run_id}}.
        metric: The metric to plot ('explained_var' or 'recon_loss').
        final_layer: The final layer to plot up to.
        run_types: The run types to include in the plot.
        out_file: The filename which the plot will be saved as.
        ylim: The y-axis limits.
        legend_label_cols_and_precision: Columns in df that should be used for the legend, along
            with their precision. Added in addition to the run type.
        legend_title: The title of the legend.
        styles: The styles to use.
        horz_layout: Whether to use a horizontal layout for the subplots. Requires sae_layers to be
            exactly [2, 6, 10]. Ignores legend_label_cols_and_precision if True.
        show_ax_titles: Whether to show titles for each subplot.
        save_svg: Whether to save the plot as an SVG file in addition to PNG. Default is True.
    """
    metric_names = {
        "explained_var": "Explained Variance",
        "explained_var_ln": "Explained Variance\nof Normalized Activations",
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
        fig, axs = plt.subplots(n_sae_layers, 1, figsize=(5, 3.5 * n_sae_layers))
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
            legend_label = styles[run_type]["label"] if styles is not None else run_type
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
            kwargs = styles[run_type] if styles is not None else {}
            ax.plot(xs, ys, **kwargs)

    for i, sae_layer in enumerate(sae_layers):
        layer_df = df.loc[df["id"].isin(list(run_ids[sae_layer].values()))]

        ax = axs[i]

        xs = np.arange(sae_layer, final_layer + 1)
        for run_type in run_types:
            plot_metric(ax, layer_df.loc[layer_df["run_type"] == run_type], sae_layer, xs)

        if show_ax_titles:
            ax.set_title(f"SAE Layer {sae_layer}", fontweight="bold")
        ax.set_xlabel("Model Layer")
        if (not horz_layout) or i == 0:
            ax.legend(title=legend_title, loc="best")
            ax.set_ylabel(metric_name)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_ylim(ylim)

    plt.tight_layout()
    if out_file is not None:
        plt.savefig(out_file, dpi=400)
        logger.info(f"Saved to {out_file}")
        if save_svg:
            plt.savefig(Path(out_file).with_suffix(".svg"))
    plt.close()


def plot_facet(
    df: pd.DataFrame,
    xs: Sequence[str],
    y: str,
    facet_by: str,
    line_by: str,
    line_by_vals: Sequence[str] | None = None,
    sort_by: str | None = None,
    xlabels: Sequence[str | None] | None = None,
    ylabel: str | None = None,
    suptitle: str | None = None,
    facet_vals: Sequence[Any] | None = None,
    xlims: Sequence[Mapping[Any, tuple[float | None, float | None]] | None] | None = None,
    xticks: Sequence[tuple[list[float], list[str]] | None] | None = None,
    yticks: tuple[list[float], list[str]] | None = None,
    ylim: Mapping[Any, tuple[float | None, float | None]] | None = None,
    styles: Mapping[Any, Mapping[str, Any]] | None = None,
    title: Mapping[Any, str] | None = None,
    legend_title: str | None = None,
    legend_pos: str = "lower right",
    axis_formatter: Callable[[Sequence[plt.Axes]], None] | None = None,
    out_file: str | Path | None = None,
    plot_type: Literal["line", "scatter"] = "line",
    save_svg: bool = True,
) -> None:
    """Line plot with multiple x-axes and one y-axis between them. One line for each run type.

    Args:
        df: DataFrame containing the data.
        xs: The variables to plot on the x-axes.
        y: The variable to plot on the y-axis.
        facet_by: The variable to facet the plot by.
        line_by: The variable to draw lines for.
        line_by_vals: The values to draw lines for. If None, all unique values will be used.
        sort_by: The variable governing how lines are drawn between points. If None, lines will be
            drawn based on the y value.
        title: The title of the plot.
        xlabel: The labels for the x-axes.
        ylabel: The label for the y-axis.
        out_file: The filename which the plot will be saved as.
        run_types: The run types to include in the plot.
        xlims: The x-axis limits for each x-axis for each layer.
        xticks: The x-ticks for each x-axis.
        yticks: The y-ticks for the y-axis.
        ylim: The y-axis limits for each layer.
        styles: The styles to use for each line. If None, default styles will be used.
        title: The title for each row of the plot.
        legend_title: The title for the legend.
        axis_formatter: A function to format the axes, e.g. to add "better" labels.
        out_file: The filename which the plot will be saved as.
        plot_type: The type of plot to create. Either "line" or "scatter".
        save_svg: Whether to save the plot as an SVG file in addition to png. Default is True.
    """

    num_axes = len(xs)
    if facet_vals is None:
        facet_vals = sorted(df[facet_by].unique())
    if sort_by is None:
        sort_by = y

    # TODO: For some reason the title is not centered at x=0.5. Fix
    xtitle_pos = 0.513

    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#f5f6fc"})
    fig_width = 4 * num_axes
    fig = plt.figure(figsize=(fig_width, 4 * len(facet_vals)), constrained_layout=True)
    subfigs = fig.subfigures(len(facet_vals))
    subfigs = np.atleast_1d(subfigs)

    # Get all unique line values from the entire DataFrame
    all_line_vals = df[line_by].unique()
    if line_by_vals is not None:
        assert all(
            val in all_line_vals for val in line_by_vals
        ), f"Invalid line values: {line_by_vals}"
        sorted_line_vals = line_by_vals
    else:
        sorted_line_vals = sorted(all_line_vals, key=str if df[line_by].dtype == object else float)

    colors = sns.color_palette("tab10", n_colors=len(sorted_line_vals))
    for subfig, facet_val in zip(subfigs, facet_vals, strict=False):
        axs = subfig.subplots(1, num_axes)
        facet_df = df.loc[df[facet_by] == facet_val]
        for line_val, color in zip(sorted_line_vals, colors, strict=True):
            data = facet_df.loc[facet_df[line_by] == line_val]
            line_style = {
                "label": line_val,
                "marker": "o",
                "linewidth": 1.1,
                "color": color,
                "linestyle": "-" if plot_type == "line" else "None",
            }  # default
            line_style.update(
                {} if styles is None else styles.get(line_val, {})
            )  # specific overrides
            if not data.empty:
                # draw the lines between points based on the y value
                data = data.sort_values(sort_by)
                for i in range(num_axes):
                    if plot_type == "scatter":
                        axs[i].scatter(data[xs[i]], data[y], **line_style)
                    elif plot_type == "line":
                        axs[i].plot(data[xs[i]], data[y], **line_style)
                    else:
                        raise ValueError(f"Unknown plot type: {plot_type}")
            else:
                # Add empty plots for missing line values to ensure they appear in the legend
                for i in range(num_axes):
                    axs[i].plot([], [], **line_style)

        if facet_val == facet_vals[-1]:
            axs[0].legend(title=legend_title or line_by, loc=legend_pos)

        for i in range(num_axes):
            if xlims is not None and xlims[i] is not None:
                xmin, xmax = xlims[i][facet_val]  # type: ignore
                axs[i].set_xlim(xmin=xmin, xmax=xmax)
            if ylim is not None:
                ymin, ymax = ylim[facet_val]
                axs[i].set_ylim(ymin=ymin, ymax=ymax)

        # Set a title above the subplots to show the layer number
        row_title = title[facet_val] if title is not None else None
        subfig.suptitle(row_title, fontweight="bold", x=xtitle_pos)
        for i in range(num_axes):
            axs[i].set_xlabel(xlabels[i] if xlabels is not None else xs[i])
            if i == 0:
                axs[i].set_ylabel(ylabel or y)
            if xticks is not None and xticks[i] is not None:
                ticks, labels = xticks[i]  # type: ignore
                axs[i].set_xticks(ticks, labels=labels)
            if yticks is not None:
                axs[i].set_yticks(yticks[0], yticks[1])

        if axis_formatter is not None:
            axis_formatter(axs)

    if suptitle is not None:
        fig.suptitle(suptitle, fontweight="bold", x=xtitle_pos)

    if out_file is not None:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, dpi=400)
        logger.info(f"Saved to {out_file}")
        if save_svg:
            plt.savefig(Path(out_file).with_suffix(".svg"))

    plt.close(fig)
