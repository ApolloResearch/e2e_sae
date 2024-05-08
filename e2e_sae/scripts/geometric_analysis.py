import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import umap
import wandb
from jaxtyping import Float
from matplotlib import colors as mcolors
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict
from scipy import stats
from torch import Tensor
from wandb.apis.public import Run

from e2e_sae.log import logger
from e2e_sae.settings import REPO_ROOT

CONSTANT_CE_RUNS = {
    2: {"local": "ue3lz0n7", "e2e": "ovhfts9n", "downstream": "visi12en"},
    6: {"local": "1jy3m5j0", "e2e": "zgdpkafo", "downstream": "2lzle2f0"},
    10: {"local": "m2hntlav", "e2e": "8crnit9h", "downstream": "cvj5um2h"},
}
CONSTANT_L0_RUNS = {
    2: {"local": "6vtk4k51", "e2e": "bst0prdd", "downstream": "e26jflpq"},
    6: {"local": "jup3glm9", "e2e": "tvj2owza", "downstream": "2lzle2f0"},
    10: {"local": "5vmpdgaz", "e2e": "8crnit9h", "downstream": "cvj5um2h"},
}


COLOR_MAP = {
    "local": "#f0a70a",
    "e2e": "#518c31",
    "downstream": plt.get_cmap("tab20b").colors[2],  # type: ignore[reportAttributeAccessIssue]
}


class RegionCoords(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    xmin: float
    xmax: float
    ymin: float
    ymax: float


class Region(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    coords: RegionCoords
    description: str
    letter: str | None = None


class LayerRegions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    filename: str
    regions: list[Region]


# Regions of interest keyed by run types and layer number
REGIONS: dict[str, dict[int, LayerRegions]] = {
    "e2e_local": {
        2: LayerRegions(
            filename="e2e_local_umap_blocks.2.hook_resid_pre.pt",
            regions=[
                Region(
                    coords=RegionCoords(xmin=-2.7, xmax=-2.3, ymin=1.3, ymax=1.6),
                    description="Local outlier region on left",
                ),
                Region(
                    coords=RegionCoords(xmin=9.0, xmax=9.5, ymin=5.5, ymax=6.7),
                    description="Mostly-local line at top",
                ),
                Region(
                    coords=RegionCoords(xmin=10.5, xmax=11, ymin=3.6, ymax=3.9),
                    description="E2e cluster near top",
                ),
                Region(
                    coords=RegionCoords(xmin=7.5, xmax=8, ymin=0.3, ymax=0.6),
                    description="Local cluster on middle left",
                ),
                Region(
                    coords=RegionCoords(xmin=11, xmax=11.2, ymin=-1, ymax=-0.8),
                    description="Random cluster",
                ),
            ],
        ),
        6: LayerRegions(
            filename="e2e_local_umap_blocks.6.hook_resid_pre.pt",
            regions=[
                # Region(coords=RegionCoords(xmax=3), description="E2e outlier area in bottom left"),
                Region(
                    coords=RegionCoords(xmin=1.7, xmax=2.2, ymin=5, ymax=5.3),
                    description="E2e outlier area in bottom left",
                ),
                Region(
                    coords=RegionCoords(xmin=3.5, xmax=3.8, ymin=9, ymax=9.4),
                    description="Mostly-local outlier cluster on left",
                ),
                Region(
                    coords=RegionCoords(xmin=9.5, xmax=11, ymin=2.5, ymax=4),
                    description="Local line structure at bottom",
                ),
                Region(
                    coords=RegionCoords(xmin=10.7, xmax=11, ymin=5.5, ymax=6),
                    description="E2e region above Local line",
                ),
                Region(
                    coords=RegionCoords(xmin=10.5, xmax=11, ymin=8.5, ymax=9),
                    description="Random cluster in middle",
                ),
            ],
        ),
        10: LayerRegions(
            filename="e2e_local_umap_blocks.10.hook_resid_pre.pt",
            regions=[
                Region(
                    coords=RegionCoords(xmin=2, xmax=2.2, ymin=2.3, ymax=2.5),
                    description="E2e cluster in middle",
                ),
                Region(
                    coords=RegionCoords(xmin=6.5, xmax=6.7, ymin=3.5, ymax=3.7),
                    description="Local cluster on right",
                ),
                Region(
                    coords=RegionCoords(xmin=0, xmax=0.5, ymin=8, ymax=8.3),
                    description="Mixed cluster in top left",
                ),
                Region(
                    coords=RegionCoords(xmin=1, xmax=1.2, ymin=5, ymax=5.2),
                    description="Random cluster in middle",
                ),
            ],
        ),
    },
    "downstream_local": {
        2: LayerRegions(
            filename="downstream_local_umap_blocks.2.hook_resid_pre.pt",
            regions=[
                Region(
                    coords=RegionCoords(xmin=-2.8, xmax=-2.4, ymin=-0.4, ymax=-0.1),
                    description="Local outlier region on left",
                ),
                Region(
                    coords=RegionCoords(xmin=9.6, xmax=10, ymin=6, ymax=6.5),
                    description="Mixed line at top",
                ),
                Region(
                    coords=RegionCoords(xmin=10.2, xmax=10.4, ymin=1, ymax=1.1),
                    description="Local cluster in middle",
                ),
                Region(
                    coords=RegionCoords(xmin=14.2, xmax=14.4, ymin=0, ymax=0.4),
                    description="Mostly downstream cluster on right",
                ),
                Region(
                    coords=RegionCoords(xmin=9, xmax=9.5, ymin=-0.5, ymax=0.0),
                    description="Mixed in middle",
                ),
            ],
        ),
        6: LayerRegions(
            filename="downstream_local_umap_blocks.6.hook_resid_pre.pt",
            regions=[
                Region(
                    coords=RegionCoords(xmin=-0.5, xmax=0, ymin=-0.5, ymax=-0.1),
                    description="Mostly-local line at bottom",
                    letter="A",
                ),
                Region(
                    coords=RegionCoords(xmin=7.8, xmax=8.5, ymin=1, ymax=1.5),
                    description="downstream outlier bottom right",
                    letter="B",
                ),
                Region(
                    coords=RegionCoords(xmin=-5.5, xmax=-5, ymin=6.2, ymax=6.5),
                    description="Mostly-local cluster on left",
                    letter="C",
                ),
                Region(
                    coords=RegionCoords(xmin=5.5, xmax=6, ymin=8.5, ymax=8.8),
                    description="Mixed cluster on right",
                    letter="D",
                ),
                Region(
                    coords=RegionCoords(xmin=0.0, xmax=0.5, ymin=2.45, ymax=2.7),
                    description="Local cluster above bottom line touching downstream cluster",
                    letter="E",
                ),
                Region(
                    coords=RegionCoords(xmin=0.0, xmax=0.5, ymin=2.2, ymax=2.45),
                    description="downstream cluster above bottome line touching local cluster",
                    letter="F",
                ),
                Region(
                    coords=RegionCoords(xmin=-1, xmax=-0.5, ymin=8.5, ymax=8.7),
                    description="Mixed in middle",
                    letter="G",
                ),
            ],
        ),
        10: LayerRegions(
            filename="downstream_local_umap_blocks.10.hook_resid_pre.pt",
            regions=[
                Region(
                    coords=RegionCoords(xmin=5.5, xmax=6, ymin=-0.1, ymax=0.2),
                    description="Local dense cluster on right",
                ),
                Region(
                    coords=RegionCoords(xmin=6, xmax=6.5, ymin=1, ymax=1.5),
                    description="downstream outlier right",
                ),
                Region(
                    coords=RegionCoords(xmin=3.7, xmax=4, ymin=0.3, ymax=0.5),
                    description="Mostly downstream in right/middle",
                ),
                Region(
                    coords=RegionCoords(xmin=-2.5, xmax=-2.3, ymin=-2.3, ymax=-2),
                    description="Mixed hanging off bottom left",
                ),
                Region(
                    coords=RegionCoords(xmin=-0.5, xmax=-0.2, ymin=2, ymax=2.3),
                    description="Mixed in middle",
                ),
            ],
        ),
    },
}


class EmbedInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    alive_elements_per_dict: tuple[int, int]
    embedding: Float[Tensor, "n_both_dicts 2"]
    sae_pos: str
    raw_dict_sizes: tuple[int, int]
    all_alive_indices: tuple[list[int], list[int]]


class AliveElements(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    alive_dict_elements: Float[Tensor, "n_dense n_dict_elements"]
    sae_pos: str
    raw_dict_size: int
    alive_indices: list[int]


def get_dict_indices_for_embedding_range(
    embed_info: EmbedInfo,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
) -> tuple[list[int], list[int]]:
    """Get the indices of the embeddings that fall within the specified region.

    Args:
        embed_info: Named tuple containing the embedding and other information.
        xmin: The minimum x value to filter by.
        xmax: The maximum x value to filter by.
        ymin: The minimum y value to filter by.
        ymax: The maximum y value to filter by.

    Returns:
        - Indices of the first dict whose values lie in the specified region.
        - Indices of the second dict whose values lie in the specified region.
    """

    x = embed_info.embedding[:, 0]
    y = embed_info.embedding[:, 1]

    mask = torch.ones_like(x, dtype=torch.bool)
    if xmin is not None:
        mask &= x >= xmin
    if xmax is not None:
        mask &= x <= xmax
    if ymin is not None:
        mask &= y >= ymin
    if ymax is not None:
        mask &= y <= ymax

    output = mask.nonzero().squeeze()

    # Now get the indices of the first and second dict
    first_alive_dict_indices = output[output < embed_info.alive_elements_per_dict[0]].tolist()
    raw_second_alive_dict_indices = output[output >= embed_info.alive_elements_per_dict[0]].tolist()
    # We have to subtract the number of elements in the first dict to get the indices in the second
    second_alive_dict_indices = [
        i - embed_info.alive_elements_per_dict[0] for i in raw_second_alive_dict_indices
    ]

    # Convert the alive dict indices to the original dict indices using all_alive_indices
    first_dict_indices = [embed_info.all_alive_indices[0][i] for i in first_alive_dict_indices]
    second_dict_indices = [embed_info.all_alive_indices[1][i] for i in second_alive_dict_indices]

    return first_dict_indices, second_dict_indices


def plot_max_cosine_similarity(
    cosine_sim: Float[Tensor, "e2e_n_dict local_n_dict"], title: str, outfile: Path
):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    # Get the maximum cosine similarity for each e2e feature
    max_cosine_sim, _ = torch.max(cosine_sim, dim=1)

    # Plot a histogram of the max cosine similarities
    plt.hist(max_cosine_sim.detach().numpy(), bins=50)
    plt.xlabel("Max Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.savefig(outfile)
    plt.close()


def analyze_cosine_similarity(
    cosine_sim: Float[Tensor, "e2e_n_dict local_n_dict"],
    threshold_high: float,
    threshold_low: float,
):
    # Get the maximum cosine similarity for each e2e feature
    max_cosine_sim, _ = torch.max(cosine_sim, dim=1)

    # Plot a histogram of the max cosine similarities
    # Make the x-axis range from 0 to 1
    plt.hist(max_cosine_sim.detach().numpy(), bins=50)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Histogram of Max Cosine Similarities")
    plt.show()

    # Count the number of e2e features with cosine similarity > threshold_high
    num_high_sim = torch.sum(max_cosine_sim > threshold_high).item()
    total_e2e_features = cosine_sim.shape[0]
    percentage_high_sim = num_high_sim / total_e2e_features * 100

    # Find the indices of e2e features with cosine similarity < threshold_low
    low_sim_indices = torch.where(torch.all(cosine_sim < threshold_low, dim=1))[0].tolist()
    num_low_sim = len(low_sim_indices)
    percentage_low_sim = num_low_sim / total_e2e_features * 100

    print(
        f"{percentage_high_sim:.2f}% of the e2e features have a cosine similarity"
        f"> {threshold_high} with at least 1 local feature."
    )
    print(
        f"There are {num_low_sim} ({percentage_low_sim:.2f}%) e2e features that have a"
        f"cosine similarity < {threshold_low} from any other local feature."
    )
    print("Indices of low similarity e2e features:", low_sim_indices)


def get_cosine_similarity(
    dict_elements_1: Float[Tensor, "n_dense n_dict_1"],
    dict_elements_2: Float[Tensor, "n_dense n_dict_2"],
) -> Float[Tensor, "n_dict_1 n_dict_2"]:
    """Get the cosine similarity between the alive dictionary elements of two runs.

    Args:
        dict_elements_1: The alive dictionary elements of the first run.
        dict_elements_2: The alive dictionary elements of the second run.

    Returns:
        The cosine similarity between the alive dictionary elements of the two runs.
    """
    # Compute cosine similarity in pytorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dict_elements_1 = dict_elements_1.to(device)
    dict_elements_2 = dict_elements_2.to(device)

    # Normalize the tensors
    dict_elements_1 = F.normalize(dict_elements_1, p=2, dim=0)
    dict_elements_2 = F.normalize(dict_elements_2, p=2, dim=0)

    # Compute cosine similarity using matrix multiplication
    cosine_sim: Float[Tensor, "n_dict_1 n_dict_2"] = torch.mm(dict_elements_1.T, dict_elements_2)

    return cosine_sim.cpu()


def plot_cosine_similarity_heatmap(
    cosine_sim: Float[Tensor, "e2e_n_dict local_n_dict"],
    labels: list[str],
    sae_pos: str,
):
    """Plot a cosine similarity heatmap between the alive dictionary elements of e2e and local runs.

    Args:
        cosine_sim: The cosine similarity between the alive dictionary elements of the e2e and local runs.
        labels: The labels for each set of alive dictionary elements.
        sae_pos: The SAE position.
    """

    # TODO: Think about how we want to visualize/analyse these cosine sims

    cosine_sim = cosine_sim.detach().cpu().numpy()

    # Make all values below 0.4 white to make it easier to see the differences
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap",
        [(0, "white"), (0.4, "white"), (1, "red")],
        N=256,
    )

    plt.figure(figsize=(10, 10))
    sns.heatmap(cosine_sim, cmap=cmap, square=True, cbar_kws={"shrink": 0.5}, vmin=0, vmax=1)
    plt.xlabel(labels[1])
    plt.ylabel(labels[0])
    plt.title(f"Cosine Similarity Heatmap of Alive Dictionary Elements in {sae_pos}")
    plt.tight_layout()
    plt.savefig(f"cosine_sim_heatmap_{sae_pos}_{labels[0]}_{labels[1]}.png")


def compute_umap_embedding(
    alive_elements_pair: tuple[AliveElements, AliveElements], seed: int = 1
) -> EmbedInfo:
    """Compute the UMAP embedding of the alive dictionary elements and save it to file.

    Args:
        alive_elements: The alive dictionary elements object for the two runs.
        seed: The random seed to use for UMAP.
    """
    alive_elements_per_dict: tuple[int, int] = (
        alive_elements_pair[0].alive_dict_elements.shape[1],
        alive_elements_pair[1].alive_dict_elements.shape[1],
    )
    alive_elements_combined = (
        torch.cat([ae.alive_dict_elements for ae in alive_elements_pair], dim=1).T.detach().numpy()
    )
    assert alive_elements_pair[0].sae_pos == alive_elements_pair[1].sae_pos, (
        f"The SAE positions are different: "
        f"{alive_elements_pair[0].sae_pos} and {alive_elements_pair[1].sae_pos}"
    )

    reducer = umap.UMAP(random_state=seed)
    embedding = reducer.fit_transform(alive_elements_combined)
    embed_info = EmbedInfo(
        alive_elements_per_dict=alive_elements_per_dict,
        embedding=torch.tensor(embedding),
        sae_pos=alive_elements_pair[0].sae_pos,
        raw_dict_sizes=(alive_elements_pair[0].raw_dict_size, alive_elements_pair[1].raw_dict_size),
        all_alive_indices=(
            alive_elements_pair[0].alive_indices,
            alive_elements_pair[1].alive_indices,
        ),
    )
    return embed_info


def plot_umap(
    embed_info: EmbedInfo,
    labels: Sequence[str],
    run_types: Sequence[str],
    out_file: Path,
    lims: dict[str, tuple[float | None, float | None]] | None = None,
    grid: bool = False,
    regions: list[Region] | None = None,
    legend_title: str | None = None,
):
    """Plot the UMAP embedding of the alive dictionary elements, colored by the labels.

    Saves the plot to the output directory.

    Args:
        embedding_file: The file containing the UMAP embedding.
        labels: The labels for each set of alive dictionary elements.
        run_types: The types of runs being compared.
        out_file: The output file to save the plot to.
        lims: The x and y limits for the plot.
        grid: Whether to plot a grid in the background.
        regions: The regions to draw boxes around.
    """
    embedding = embed_info.embedding.detach().clone()
    alive_elements_per_dict = embed_info.alive_elements_per_dict

    # Ignore all embeddings that are outside the limits
    if lims is not None:
        mask = torch.ones(embedding.shape[0], dtype=torch.bool)
        if lims["x"][0] is not None:
            mask &= embedding[:, 0] >= lims["x"][0]
        if lims["x"][1] is not None:
            mask &= embedding[:, 0] <= lims["x"][1]
        if lims["y"][0] is not None:
            mask &= embedding[:, 1] >= lims["y"][0]
        if lims["y"][1] is not None:
            mask &= embedding[:, 1] <= lims["y"][1]
        embedding = embedding[mask]

    plt.figure(figsize=(8, 8), dpi=600)
    for i, label in enumerate(labels):
        plt.scatter(
            embedding[i * alive_elements_per_dict[i] : (i + 1) * alive_elements_per_dict[i], 0],  # type: ignore
            embedding[i * alive_elements_per_dict[i] : (i + 1) * alive_elements_per_dict[i], 1],  # type: ignore
            label=label,
            s=1,
            color=COLOR_MAP[run_types[i]],
            alpha=0.3,
        )
    # Create legend elements with larger point size
    colors = [COLOR_MAP[run_type] for run_type in run_types]
    legend_elements = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=label, markerfacecolor=color, markersize=10
        )
        for label, color in zip(labels, colors, strict=False)
    ]
    plt.legend(handles=legend_elements, loc="lower right", title=legend_title)
    # run_type_str = " and ".join(run_types)
    # plt.title(f"UMAP of alive dictionary elements in {sae_pos}: {run_type_str}")

    if grid:
        # Put ticks every 0.5 points
        plt.xticks(np.arange(int(plt.xlim()[0]), int(plt.xlim()[1]), 0.5))
        plt.yticks(np.arange(int(plt.ylim()[0]), int(plt.ylim()[1]), 0.5))
        # Make the tick text size smaller
        plt.tick_params(axis="both", which="major", labelsize=8)
        plt.grid()
    else:
        plt.xticks([])
        plt.yticks([])

    # Draw square boxes around the identified regions
    if regions is not None:
        for region in regions:
            xmin, xmax = region.coords.xmin, region.coords.xmax
            ymin, ymax = region.coords.ymin, region.coords.ymax
            # Don't plot the region if it's outside the limits
            xlim_min, xlim_max = plt.xlim()
            ylim_min, ylim_max = plt.ylim()
            if xmin < xlim_min or xmax > xlim_max or ymin < ylim_min or ymax > ylim_max:
                continue
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle(
                (xmin, ymin), width, height, linewidth=1, edgecolor="r", facecolor="none"
            )
            plt.gca().add_patch(rect)  # type: ignore

            # Add the letter to the left of the box
            if region.letter is not None:
                plt.text(
                    xmin - 0.3,
                    (ymin + ymax) / 2,
                    region.letter,
                    color="r",
                    fontsize=12,
                    fontweight="bold",
                    va="center",
                )
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.savefig(out_file.with_suffix(".svg"), bbox_inches="tight")
    logger.info(f"Saved UMAP plot to {out_file}")


def get_alive_dict_elements(
    api: wandb.Api,
    project_name: str,
    run_id: str,
) -> AliveElements:
    """Get the alive dictionary elements for a run.

    Args:
        api: The wandb API.
        project_name: The name of the wandb project.
        run_id: The ID of the run.

    Returns:
        AliveElements object containing containing the alive dictionary elements and related info
        for the run
    """

    run: Run = api.run(f"{project_name}/{run_id}")

    # Weights file should be the largest .pt file. All have format (samples_*.pt)
    weight_files = [file for file in run.files() if file.name.endswith(".pt")]
    # Latest checkpoint
    weight_file = sorted(weight_files, key=lambda x: int(x.name.split(".pt")[0].split("_")[-1]))[-1]

    cache_dir = Path(os.environ.get("SAE_CACHE_DIR", "/tmp/"))
    model_cache_dir = cache_dir / project_name / run.id
    latest_checkpoint = wandb.restore(
        weight_file.name,
        run_path=f"{project_name}/{run.id}",
        root=model_cache_dir,
        replace=False,
    )
    assert latest_checkpoint is not None
    weights = torch.load(latest_checkpoint.name, map_location="cpu")
    decoder_keys = [k for k in weights if k.endswith("decoder.weight")]
    assert len(decoder_keys) == 1, "Only one decoder should be present in the SAE"

    # decoder_keys[0] is of the form "blocks-2-hook_resid_pre.decoder.weight"
    decoder = weights[decoder_keys[0]]
    # We were forced to use hyphens for the sae_pos in our moduledict, convert it back
    sae_pos = decoder_keys[0].split(".")[0].replace("-", ".")

    dict_elements = F.normalize(decoder, dim=0)

    alive_indices = run.summary_metrics[f"sparsity/alive_dict_elements_indices_final/{sae_pos}"]

    return AliveElements(
        alive_dict_elements=dict_elements[:, alive_indices],
        sae_pos=sae_pos,
        raw_dict_size=dict_elements.shape[1],
        alive_indices=alive_indices,
    )


def create_umap_plots(
    api: wandb.Api,
    project: str,
    run_types: tuple[str, str],
    compute_umaps: bool = True,
    constant_val: Literal["CE", "l0"] = "CE",
    lims: dict[int, dict[str, tuple[float | None, float | None]]] | None = None,
    grid: bool = False,
):
    run_types_str = "_".join(run_types)
    run_dict = CONSTANT_L0_RUNS if constant_val == "l0" else CONSTANT_CE_RUNS  # type: ignore
    out_dir = Path(__file__).parent / "out" / "umap" / f"constant_{constant_val}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if lims is None:
        lims = {
            2: {"x": (None, None), "y": (None, None)},
            6: {"x": (None, None), "y": (None, None)},
            10: {"x": (None, None), "y": (None, None)},
        }
    for layer_num in [2, 6, 10]:
        run_ids = run_dict[layer_num]

        embed_file = out_dir / REGIONS[run_types_str][layer_num].filename

        if compute_umaps:
            all_alive_elements = []
            for run_type in run_types:
                alive_elements = get_alive_dict_elements(
                    api=api, project_name=project, run_id=run_ids[run_type]
                )
                all_alive_elements.append(alive_elements)
            assert len(all_alive_elements) == 2, "Only two runs can be compared at a time"

            embed_info = compute_umap_embedding(alive_elements_pair=tuple(all_alive_elements))
            torch.save(embed_info.model_dump(), embed_file)

        embed_info = EmbedInfo(**torch.load(embed_file))
        umap_file = embed_file.with_suffix(".png")

        # Get indices for all embeddings
        dict_1_indices, dict_2_indices = get_dict_indices_for_embedding_range(embed_info)
        assert len(dict_1_indices) + len(dict_2_indices) == embed_info.embedding.shape[0]
        dict_1_embeds = embed_info.embedding[: len(dict_1_indices)]
        dict_2_embeds = embed_info.embedding[len(dict_1_indices) :]

        # Create a csv file with columns: RunType, EmbeddingIndex, X, Y.
        # Useful for neuronpedia to upload to their website
        with open(embed_file.with_suffix(".csv"), "w") as f:
            f.write("RunType,EmbeddingIndex,X,Y\n")
            for i in range(len(embed_info.embedding)):
                if i < len(dict_1_indices):
                    f.write(
                        f"{run_types[0]},{dict_1_indices[i]},{dict_1_embeds[i][0]},{dict_1_embeds[i][1]}\n"
                    )
                else:
                    idx = i - len(dict_1_indices)
                    f.write(
                        f"{run_types[1]},{dict_2_indices[idx]},{dict_2_embeds[idx][0]},{dict_2_embeds[idx][1]}\n"
                    )

        # The above but handle arbitrary number of run types
        labels = [f"{run_type}-{run_ids[run_type]}" for run_type in run_types]
        plot_umap(
            embed_info,
            labels=run_types,
            run_types=run_types,
            out_file=umap_file,
            lims=lims[layer_num],
            grid=grid,
            regions=REGIONS[run_types_str][layer_num].regions,
            legend_title=f"Layer {layer_num}",
        )
        for i, region in enumerate(REGIONS[run_types_str][layer_num].regions):
            region_dict_1_indices, region_dict_2_indices = get_dict_indices_for_embedding_range(
                embed_info, **region.coords.model_dump()
            )

            region_filename = REGIONS[run_types_str][layer_num].filename
            path_from_repo_root = (out_dir / region_filename).relative_to(REPO_ROOT)
            with open(f"{embed_file}_region_{i}.json", "w") as f:
                json.dump(
                    {
                        "embedding_file": str(path_from_repo_root),
                        "run_labels": labels,
                        "description": region.description,
                        "coords": region.coords.model_dump(),
                        run_types[0]: region_dict_1_indices,
                        run_types[1]: region_dict_2_indices,
                    },
                    f,
                    indent=2,
                )


# First-level keys are layer numbers, second-level keys are run types
MaxPairwiseSimilarities = dict[int, dict[str, Float[Tensor, "n_dict"]]]  # noqa: F821


def get_max_pairwise_similarities(
    api: wandb.Api,
    project: str,
    run_dict: Mapping[int, Mapping[str, str]],
    run_types: Sequence[str],
    out_file: Path,
    from_file: bool = False,
) -> MaxPairwiseSimilarities:
    """For each run, calculate the max pairwise cosine similarity for each dictionary element.

    The dictionaries are given by the run_types and run_dict arguments.

    Args:
        api: The wandb API.
        project: The name of the wandb project.
        run_dict: A dictionary mapping layer numbers to dictionaries of run IDs.
        out_file: The file to save or load the max pairwise similarities to.
        from_file: Whether to load the max pairwise similarities from file.

    Returns:
        A dictionary mapping layer numbers to dictionaries of max pairwise cosine similarities,
        where the inner dictionary is keyed by run type.
    """
    if from_file:
        return torch.load(out_file, map_location="cpu")

    max_pairwise_similarities: MaxPairwiseSimilarities = {}
    for layer_num in run_dict:
        max_pairwise_similarities[layer_num] = {}
        for run_type in run_types:
            run_id = run_dict[layer_num][run_type]
            alive_elements = get_alive_dict_elements(api=api, project_name=project, run_id=run_id)
            pairwise_similarities = get_cosine_similarity(
                alive_elements.alive_dict_elements, alive_elements.alive_dict_elements
            )
            # Ignore the cosine similarity of the same vector with itself
            pairwise_similarities.fill_diagonal_(float("-inf"))
            # Get the max pairwise cosine similarity for each dictionary element
            max_cosine_sim, _ = torch.max(pairwise_similarities, dim=1)
            max_pairwise_similarities[layer_num][run_type] = max_cosine_sim

    torch.save(max_pairwise_similarities, out_file)
    return max_pairwise_similarities


def create_subplot_hists(
    sim_list: Sequence[Float[Tensor, "n_dict"]],  # noqa: F821
    titles: Sequence[str | None],
    colors: Sequence[Any] | None = None,
    bins: int = 50,
    fig: Figure | None = None,
    figsize: tuple[float, float] = (5, 4),
    xlim: tuple[float, float] = (0, 1),
    xlabel: str = "Cosine Similarity",
    suptitle: str | None = None,
    out_file: Path | None = None,
    alpha=0.8,
):
    fig = fig or plt.figure(figsize=figsize, layout="constrained")
    axs = fig.subplots(len(sim_list), 1, sharex=True, gridspec_kw={"hspace": 0.1})
    axs = np.atleast_1d(axs)
    colors = colors or [None for _ in sim_list]
    for ax, sims, title, color in zip(axs, sim_list, titles, colors, strict=True):
        ax.hist(sims.flatten().detach().numpy(), range=xlim, bins=bins, color=color, alpha=alpha)
        ax.set_title(title, pad=2)
        ax.set_yticks([])

    axs[-1].set_xlim(xlim)
    axs[-1].set_xlabel(xlabel)
    # plt.tight_layout()
    if suptitle is not None:
        fig.suptitle(suptitle, fontweight="bold")
    if out_file:
        plt.savefig(out_file, dpi=500)
        plt.savefig(out_file.with_suffix(".svg"))
        logger.info(f"Saved plot to {out_file}")


def create_subplot_hists_short(
    sim_list: Sequence[Float[Tensor, "n_dict"]],  # noqa: F821
    titles: Sequence[str | None],
    colors: Sequence[Any] | None = None,
    bins: int = 50,
    fig: Figure | None = None,
    figsize: tuple[float, float] = (3, 2),
    xlim: tuple[float, float] = (0, 1),
    xlabel: str = "Cosine Similarity",
    suptitle: str | None = None,
    out_file: Path | None = None,
    alpha=0.8,
    left_pad=0.35,
    text_offset_pts=35,
):
    fig = fig or plt.figure(figsize=figsize)
    axs = fig.subplots(len(sim_list), 1, sharex=True)
    plt.subplots_adjust(left=left_pad, top=0.95, bottom=0.25, hspace=0.2, right=0.95)
    axs = np.atleast_1d(axs)
    colors = colors or [None for _ in sim_list]
    for ax, sims, title, color in zip(axs, sim_list, titles, colors, strict=True):
        ax.hist(sims.flatten().detach().numpy(), bins=bins, color=color, alpha=alpha, density=True)
        ax.axvline(sims.mean().item(), color="k", linestyle="solid", alpha=0.8, lw=1)
        # ax.text(-0.28, 0.5, title, va="center", ha="center", transform=ax.transAxes, fontsize=10)
        ax.annotate(
            title,
            (0, 0.5),
            (-text_offset_pts, 0),
            xycoords="axes fraction",
            textcoords="offset points",
            va="center",
            ha="center",
        )
        ax.set_yticks([])

    axs[-1].set_xlim(xlim)
    axs[-1].set_xlabel(xlabel)
    axs[-1].set_xticks([0, 0.5, 1])
    # plt.tight_layout()
    if suptitle is not None:
        fig.suptitle(suptitle, fontweight="bold")
    if out_file:
        plt.savefig(out_file)
        plt.savefig(out_file.with_suffix(".svg"))
        logger.info(f"Saved plot to {out_file}")


def permutation_mean_diff(sample_a: ArrayLike, sample_b: ArrayLike, permutations: int):
    """Compute the mean difference between two samples and its confidence interval.

    Args:
        sample_a: The first sample.
        sample_b: The second sample.
        permutations: The number of permutations to perform.

    Returns:
        A scipy BootstrapResults object
    """

    def mean_diff(resamp_a: ArrayLike, resamp_b: ArrayLike, axis=-1) -> float:
        return np.mean(resamp_a, axis=axis) - np.mean(resamp_b, axis=axis)  # type: ignore[reportCallIssue]

    return stats.bootstrap((sample_a, sample_b), mean_diff, n_resamples=permutations, batch=1000)


def create_within_sae_similarity_plots(api: wandb.Api, project: str):
    for constant_val in ["CE", "l0"]:
        run_dict = CONSTANT_L0_RUNS if constant_val == "l0" else CONSTANT_CE_RUNS
        out_dir = Path(__file__).parent / "out" / f"constant_{constant_val}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"max_pairwise_similarities_constant_{constant_val}.pt"
        pairwise_similarities = get_max_pairwise_similarities(
            api=api,
            project=project,
            run_dict=run_dict,
            run_types=("local", "e2e", "downstream"),
            out_file=out_file,
            from_file=True,
        )

        # plot all layers
        fig = plt.figure(figsize=(8, 4), layout="constrained")
        subfigs = fig.subfigures(1, len(pairwise_similarities), wspace=0.05)
        for i, (layer_num, layer_similarities) in enumerate(pairwise_similarities.items()):
            create_subplot_hists(
                sim_list=list(layer_similarities.values()),
                titles=["local", "end-to-end", "e2e + downstream"],
                colors=[COLOR_MAP[run_type] for run_type in layer_similarities],
                fig=subfigs[i],
                suptitle=f"Layer {layer_num}",
            )
            # subfigs[i].suptitle(f"Layer {layer_num}")
        fig.suptitle(
            f"Within SAE Similarities (Constant {constant_val.upper()})", fontweight="bold"
        )
        out_file = out_dir / f"within_sae_similarities_{constant_val}_all_layers.png"
        plt.savefig(out_file)
        plt.savefig(out_file.with_suffix(".svg"))
        logger.info(f"Saved plot to {out_file}")

        if constant_val == "CE":
            # Just layer 6
            fig = create_subplot_hists_short(
                sim_list=list(pairwise_similarities[6].values()),
                titles=["local", "end-to-end", "e2e + \ndownstream"],
                colors=[COLOR_MAP[run_type] for run_type in pairwise_similarities[6]],
                out_file=out_dir / "within_sae_similarities_CE_layer_6.png",
            )

            logger.info(
                "Mean self-similarities: "
                + ", ".join(
                    f"{run_type}: {sims.mean().item():.2f}"
                    for run_type, sims in pairwise_similarities[6].items()
                )
            )

            for run_type in ["e2e", "downstream"]:
                # Permutation test for mean difference
                bootstrap_result = permutation_mean_diff(
                    pairwise_similarities[6]["local"],
                    pairwise_similarities[6][run_type],
                    permutations=1000,
                )
                diff = (
                    pairwise_similarities[6]["local"].mean()
                    - pairwise_similarities[6][run_type].mean()
                )
                low, high = bootstrap_result.confidence_interval
                logger.info(
                    f"Permutation test for mean difference between local and {run_type}:"
                    f"\n\t{diff:.3f} [95\\%CI: {low:.3f}-{high:.3f}]"
                )


def get_cross_max_similarities(
    api: wandb.Api, project_name: str, run_ids: tuple[str, str]
) -> tuple[Float[Tensor, "n_dict_1"], list[list[int]]]:  # noqa: F821
    """Get the max pairwise cosine similarity between the alive dictionary elements of two runs.

    Args:
        api: The wandb API.
        project_name: The name of the wandb project.
        run_ids: The IDs of the two runs to compare.

    Returns:
        - The max pairwise cosine similarity between the alive dictionary elements of the two runs.
        - The indices of the alive dictionary elements for each run.
    """
    alive_elements: list[AliveElements] = []
    for run_id in run_ids:
        alive_elements.append(
            get_alive_dict_elements(api=api, project_name=project_name, run_id=run_id)
        )
    assert len(alive_elements) == 2, "Only two runs can be compared at a time"

    cross_similarities = get_cosine_similarity(
        alive_elements[0].alive_dict_elements, alive_elements[1].alive_dict_elements
    )
    # Get the max pairwise cosine similarity for each dictionary element
    max_cosine_sim, _ = torch.max(cross_similarities, dim=1)

    alive_indices = [ae.alive_indices for ae in alive_elements]

    return max_cosine_sim, alive_indices


CrossMaxSimilarity = dict[int, dict[str, Float[Tensor, "n_dict_1"]]]  # noqa: F821


def create_cross_type_similarity_plots(
    api: wandb.Api,
    project: str,
    constant_val: str = "CE",
    from_file: bool = False,
):
    """Create plots comparing the max similarities between different run types."""
    run_dict = CONSTANT_L0_RUNS if constant_val == "l0" else CONSTANT_CE_RUNS
    out_dir = Path(__file__).parent / "out" / f"constant_{constant_val}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get pairwise similarities between the two runtypes for each layer
    # dict_1_alive_indices: dict[int, list[int]] = {}
    if from_file and (out_dir / "similarities.pt").exists():
        cross_max_similarity = torch.load(out_dir / "similarities.pt")
    else:
        cross_max_similarity: CrossMaxSimilarity = {}
        for layer_num in run_dict:
            cross_max_similarity[layer_num] = {}
            for comparison_type in ["e2e", "downstream"]:
                cross_max_similarity[layer_num][comparison_type] = get_cross_max_similarities(
                    api=api,
                    project_name=project,
                    run_ids=(run_dict[layer_num][comparison_type], run_dict[layer_num]["local"]),
                )[0]
        torch.save(cross_max_similarity, out_dir / "similarities.pt")

    # All layers
    fig = plt.figure(figsize=(8, 3), layout="constrained")
    subfigs = fig.subfigures(1, len(cross_max_similarity), wspace=0.05)
    for i, (layer_num, max_sim) in enumerate(cross_max_similarity.items()):
        create_subplot_hists(
            sim_list=list(max_sim.values()),
            fig=subfigs[i],
            suptitle=f"Layer {layer_num}",
            colors=[COLOR_MAP["e2e"], COLOR_MAP["downstream"]],
            titles=["end-to-end → local", "e2e + downstream → local"],
        )
    fig.suptitle(f"Cross Type Similarities (Constant {constant_val.upper()})", fontweight="bold")
    out_file = out_dir / "cross_type_similarities_all_layers.png"
    plt.savefig(out_file)
    plt.savefig(out_file.with_suffix(".svg"))
    logger.info(f"Saved plot to {out_file}")

    # Just layer 6
    fig = create_subplot_hists_short(
        sim_list=list(cross_max_similarity[6].values()),
        figsize=(3.5, 1.8),
        colors=[COLOR_MAP["e2e"], COLOR_MAP["downstream"]],
        titles=["end-to-end\n→ local", "e2e + downstream\n → local"],
        out_file=out_dir / "cross_type_similarities_layer_6.png",
        left_pad=0.43,
        text_offset_pts=50,
    )


def create_cross_seed_similarity_plot(
    api: wandb.Api,
    project: str,
    run_ids: dict[str, tuple[str, str]],
):
    sim_dict = {
        run_type: get_cross_max_similarities(
            api=api,
            project_name=project,
            run_ids=run_ids[run_type],
        )[0]
        for run_type in run_ids
    }

    out_dir = Path(__file__).parent / "out" / "seed_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    create_subplot_hists_short(
        sim_list=list(sim_dict.values()),
        titles=["local", "end-to-end", "e2e + \ndownstream"],
        colors=[COLOR_MAP[run_type] for run_type in sim_dict],
        out_file=out_dir / "cross_seed_all_types.png",
    )


if __name__ == "__main__":
    project = "gpt2"
    api = wandb.Api()

    # create_within_sae_similarity_plots(api, project)

    # create_cross_type_similarity_plots(api, project, constant_val="CE")

    # # These three are also relevent but not in the paper
    # # Sparsity coeff 1.5
    # create_seed_max_similarity_comparison_plots(
    #     api, project, run_ids=("bok0t1sw", "tuzvyysg"), layer=6, run_type="e2e"
    # )
    # create_seed_max_similarity_comparison_plots(
    #     api, project, run_ids=("atfccmo3", "tvj2owza"), layer=6, run_type="e2e"
    # )
    # # This is far less similar. Has lower sparsity too (0.2)
    # create_seed_max_similarity_comparison_plots(
    #     api, project, run_ids=("hbjl3zwy", "wzzcimkj"), layer=6, run_type="e2e"
    # )

    # run_ids: dict[str, tuple[str, str]] = {
    #     "local": ("1jy3m5j0", "uqfp43ti"),
    #     "e2e": ("pzelh1s8", "ir00gg9g"),
    #     "downstream": ("y8sca507", "hqo5azo2"),
    # }
    # create_cross_seed_similarity_plot(api, project, run_ids)

    # Post-hoc ignore the identified outliers in e2e-local umap
    e2e_local_ce_lims: dict[int, dict[str, tuple[float | None, float | None]]] = {
        2: {"x": (-2.0, None), "y": (None, None)},
        6: {"x": (4.0, None), "y": (None, None)},
        10: {"x": (None, None), "y": (None, None)},
    }
    try:
        create_umap_plots(
            api,
            project,
            run_types=("e2e", "local"),
            compute_umaps=False,
            constant_val="CE",
            lims=e2e_local_ce_lims,
        )
    except FileNotFoundError:
        logger.warning("Could not create e2e-local UMAP plot")

    downstream_local_ce_lims: dict[int, dict[str, tuple[float | None, float | None]]] = {
        2: {"x": (4, None), "y": (None, None)},
        6: {"x": (None, None), "y": (None, None)},
        10: {"x": (None, None), "y": (None, None)},
    }
    create_umap_plots(
        api,
        project,
        run_types=("downstream", "local"),
        compute_umaps=False,
        constant_val="CE",
        lims=downstream_local_ce_lims,
        grid=False,
    )

# %%
