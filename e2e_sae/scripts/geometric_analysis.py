import json
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
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from wandb.apis.public import Run

from e2e_sae.log import logger
from e2e_sae.settings import REPO_ROOT

CONSTANT_CE_RUNS = {
    2: {"e2e": "ovhfts9n", "local": "ue3lz0n7", "e2e-recon": "visi12en"},
    6: {"e2e": "zgdpkafo", "local": "1jy3m5j0", "e2e-recon": "2lzle2f0"},
    10: {"e2e": "8crnit9h", "local": "m2hntlav", "e2e-recon": "cvj5um2h"},
}
CONSTANT_L0_RUNS = {
    2: {"e2e": "bst0prdd", "local": "6vtk4k51", "e2e-recon": "e26jflpq"},
    6: {"e2e": "tvj2owza", "local": "jup3glm9", "e2e-recon": "2lzle2f0"},
    10: {"e2e": "8crnit9h", "local": "5vmpdgaz", "e2e-recon": "cvj5um2h"},
}


COLOR_E2E, COLOR_LWS, COLOR_E2E_RECON = sns.color_palette()[:3]
COLOR_MAP = {"e2e": COLOR_E2E, "local": COLOR_LWS, "e2e-recon": COLOR_E2E_RECON}


class RegionCoords(BaseModel):
    xmin: float
    xmax: float
    ymin: float
    ymax: float


class Region(BaseModel):
    coords: RegionCoords
    description: str


class LayerRegions(BaseModel):
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
    "e2e-recon_local": {
        2: LayerRegions(
            filename="e2e-recon_local_umap_blocks.2.hook_resid_pre.pt",
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
                    description="Mostly e2e-recon cluster on right",
                ),
                Region(
                    coords=RegionCoords(xmin=9, xmax=9.5, ymin=-0.5, ymax=0.0),
                    description="Mixed in middle",
                ),
            ],
        ),
        6: LayerRegions(
            filename="e2e-recon_local_umap_blocks.6.hook_resid_pre.pt",
            regions=[
                Region(
                    coords=RegionCoords(xmin=-0.5, xmax=0, ymin=-0.5, ymax=-0.1),
                    description="Mostly-local line at bottom",
                ),
                Region(
                    coords=RegionCoords(xmin=7.8, xmax=8.5, ymin=1, ymax=1.5),
                    description="e2e-recon outlier bottom right",
                ),
                Region(
                    coords=RegionCoords(xmin=-5.5, xmax=-5, ymin=6.2, ymax=6.5),
                    description="Mostly-local cluster on left",
                ),
                Region(
                    coords=RegionCoords(xmin=5.5, xmax=6, ymin=8.5, ymax=8.8),
                    description="Mixed cluster on right",
                ),
                Region(
                    coords=RegionCoords(xmin=0.0, xmax=0.5, ymin=2.45, ymax=2.7),
                    description="Local cluster above bottom line touching e2e-recon cluster",
                ),
                Region(
                    coords=RegionCoords(xmin=0.0, xmax=0.5, ymin=2.2, ymax=2.45),
                    description="e2e-recon cluster above bottome line touching local cluster",
                ),
                Region(
                    coords=RegionCoords(xmin=-1, xmax=-0.5, ymin=8.5, ymax=8.7),
                    description="Mixed in middle",
                ),
            ],
        ),
        10: LayerRegions(
            filename="e2e-recon_local_umap_blocks.10.hook_resid_pre.pt",
            regions=[
                Region(
                    coords=RegionCoords(xmin=5.5, xmax=6, ymin=-0.1, ymax=0.2),
                    description="Local dense cluster on right",
                ),
                Region(
                    coords=RegionCoords(xmin=6, xmax=6.5, ymin=1, ymax=1.5),
                    description="e2e-recon outlier right",
                ),
                Region(
                    coords=RegionCoords(xmin=3.7, xmax=4, ymin=0.3, ymax=0.5),
                    description="Mostly e2e-recon in right/middle",
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
        f"{percentage_high_sim:.2f}% of the e2e features have a cosine similarity > {threshold_high} with at least 1 local feature."
    )
    print(
        f"There are {num_low_sim} ({percentage_low_sim:.2f}%) e2e features that have a cosine similarity < {threshold_low} from any other local feature."
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
    sae_pos = embed_info.sae_pos

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

    colors = sns.color_palette()[: len(labels)]
    plt.figure(figsize=(10, 10), dpi=600)
    for i, label in enumerate(labels):
        plt.scatter(
            embedding[i * alive_elements_per_dict[i] : (i + 1) * alive_elements_per_dict[i], 0],  # type: ignore
            embedding[i * alive_elements_per_dict[i] : (i + 1) * alive_elements_per_dict[i], 1],  # type: ignore
            label=label,
            s=1,
            color=colors[i],
            alpha=0.3,
        )
    # Create legend elements with larger point size
    legend_elements = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=label, markerfacecolor=color, markersize=10
        )
        for label, color in zip(labels, colors, strict=False)
    ]
    plt.legend(handles=legend_elements)
    run_type_str = " and ".join(run_types)
    plt.title(f"UMAP of alive dictionary elements in {sae_pos}: {run_type_str}")

    if grid:
        # Put ticks every 0.5 points
        plt.xticks(np.arange(int(plt.xlim()[0]), int(plt.xlim()[1]), 0.5))
        plt.yticks(np.arange(int(plt.ylim()[0]), int(plt.ylim()[1]), 0.5))
        # Make the tick text size smaller
        plt.tick_params(axis="both", which="major", labelsize=8)
        plt.grid()

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

    plt.savefig(out_file, dpi=300)
    plt.savefig(out_file.with_suffix(".svg"))
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
    latest_checkpoint = wandb.restore(
        weight_file.name,
        run_path=f"{project_name}/{run.id}",
        root=f"/tmp/{run.id}/",
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
    out_dir = Path(__file__).parent / "out" / f"constant_{constant_val}"
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
            labels=labels,
            run_types=run_types,
            out_file=umap_file,
            lims=lims[layer_num],
            grid=grid,
            regions=REGIONS[run_types_str][layer_num].regions,
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
    sim_list: Sequence[Float[Tensor, "n_dict"]],
    titles: Sequence[str | None],
    colors: Sequence[Any] | None = None,
    bins: int = 100,
    fig: Figure | None = None,
    figsize: tuple[float, float] = (5, 4),
    xlim: tuple[float, float] = (0, 1),
    xlabel: str = "Cosine Similarity",
    suptitle: str | None = None,
    out_file: Path | None = None,
):
    fig = fig or plt.figure(figsize=figsize, layout="constrained")
    axs = fig.subplots(len(sim_list), 1, sharex=True)
    axs = np.atleast_1d(axs)
    colors = colors or [None for _ in sim_list]
    for ax, sims, title, color in zip(axs, sim_list, titles, colors, strict=True):
        ax.hist(sims.flatten().detach().numpy(), bins=bins, color=color, alpha=0.5)
        ax.set_title(title)
        ax.set_yticks([])

    axs[-1].set_xlim(xlim)
    axs[-1].set_xlabel(xlabel)
    # plt.tight_layout()
    if suptitle is not None:
        fig.suptitle(suptitle, fontweight="bold")
    if out_file:
        plt.savefig(out_file)
        logger.info(f"Saved subplot histograms to {out_file}")


def plot_max_pairwise_similarities(
    max_pairwise_similarities: MaxPairwiseSimilarities,
    constant_val: str,
    out_file: Path,
):
    """Make a KDE for each layer showing max pairwise cosine similarities for each run type."""

    # Make a single plot for all layers using plt.subplots
    fig, axs = plt.subplots(1, len(max_pairwise_similarities), figsize=(15, 5))
    for i, (layer_num, layer_similarities) in enumerate(max_pairwise_similarities.items()):
        ax = axs[i]
        for run_type, max_cosine_sim in layer_similarities.items():
            label = f"{run_type} Constant {constant_val}"
            sns.kdeplot(max_cosine_sim.flatten().detach().numpy(), ax=ax, label=label)
        ax.set_title(f"Layer {layer_num}", fontweight="bold")
        ax.set_xlabel("Max Cosine Similarity")
        ax.set_ylabel("Density")
        ax.legend()
    plt.suptitle(out_file.stem, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.savefig(out_file.with_suffix(".svg"))
    logger.info(f"Saved max pairwise similarity to {out_file}")


def create_max_pairwise_similarity_plots(api: wandb.Api, project: str):
    for constant_val in ["CE", "l0"]:
        run_dict = CONSTANT_L0_RUNS if constant_val == "l0" else CONSTANT_CE_RUNS
        out_dir = Path(__file__).parent / "out" / f"constant_{constant_val}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"max_pairwise_similarities_constant_{constant_val}.pt"
        pairwise_similarities = get_max_pairwise_similarities(
            api=api,
            project=project,
            run_dict=run_dict,
            run_types=("e2e", "local", "e2e-recon"),
            out_file=out_file,
            from_file=False,
        )

        # plot all layers
        fig = plt.figure(figsize=(10, 4), layout="constrained")
        subfigs = fig.subfigures(1, len(pairwise_similarities), wspace=0.05)
        for i, (layer_num, layer_similarities) in enumerate(pairwise_similarities.items()):
            create_subplot_hists(
                sim_list=list(layer_similarities.values()),
                titles=list(layer_similarities.keys()),
                colors=[COLOR_MAP[run_type] for run_type in layer_similarities],
                fig=subfigs[i],
                suptitle=f"Layer {layer_num}",
            )
            # subfigs[i].suptitle(f"Layer {layer_num}")
        fig.suptitle(
            f"Within SAE Similarities (Constant {constant_val.upper()})", fontweight="bold"
        )
        out_file = out_dir / f"within_sae_similarities_{constant_val}_all_layers.png"

        if constant_val == "CE":
            # Just layer 6
            fig = create_subplot_hists(
                sim_list=list(pairwise_similarities[6].values()),
                titles=list(pairwise_similarities[6].keys()),
                colors=[COLOR_MAP[run_type] for run_type in pairwise_similarities[6]],
                figsize=(4, 4),
                out_file=out_dir / "within_sae_similarities_CE_layer_6.png",
                suptitle="Within SAE Similarities",
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


CrossMaxSimilarity = dict[int, Float[Tensor, "n_dict_1"]]  # noqa: F821


def plot_cross_max_similarity(
    cross_max_similarity: CrossMaxSimilarity,
    out_file: Path,
    plot_type: Literal["kde", "hist"] = "kde",
    cut_kde: bool = False,
):
    """Make a single plot with the layers side by side showing max cross similarities."""
    fig, axs = plt.subplots(1, len(cross_max_similarity), figsize=(15, 5))
    axs = np.atleast_1d(axs)
    for i, (layer_num, max_cosine_sim) in enumerate(cross_max_similarity.items()):
        ax = axs[i]
        if plot_type == "hist":
            ax.hist(max_cosine_sim.flatten().detach().numpy(), bins=50)
        elif plot_type == "kde":
            if cut_kde:
                sns.kdeplot(max_cosine_sim.flatten().detach().numpy(), ax=ax, cut=0)
            else:
                sns.kdeplot(max_cosine_sim.flatten().detach().numpy(), ax=ax)
        ax.set_title(f"Layer {layer_num}", fontweight="bold")
        ax.set_xlabel("Max Cosine Similarity")
        ax.set_ylabel("Density")
    plt.suptitle(out_file.stem, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.savefig(out_file.with_suffix(".svg"))
    logger.info(f"Saved cross max similarity to {out_file}")


def create_cross_max_similarity_plots(
    api: wandb.Api,
    project: str,
    run_types: tuple[str, str],
    constant_val: str = "CE",
    write_csv: bool = False,
):
    """Create plots comparing the max similarities between different run types."""
    run_dict = CONSTANT_L0_RUNS if constant_val == "l0" else CONSTANT_CE_RUNS
    out_dir = Path(__file__).parent / "out" / f"constant_{constant_val}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get pairwise similarities between the two runtypes for each layer
    cross_max_similarity: CrossMaxSimilarity = {}
    dict_1_alive_indices: dict[int, list[int]] = {}
    for layer_num in run_dict:
        max_cosine_sim, alive_indices = get_cross_max_similarities(
            api=api,
            project_name=project,
            run_ids=(run_dict[layer_num][run_types[0]], run_dict[layer_num][run_types[1]]),
        )
        cross_max_similarity[layer_num] = max_cosine_sim
        dict_1_alive_indices[layer_num] = alive_indices[0]

    plot_cross_max_similarity(
        cross_max_similarity,
        out_file=out_dir / f"cross_max_similarities_{run_types[0]}_{run_types[1]}.png",
    )

    if write_csv:
        # Create a csv with columns Layer, Dict1Index, MaxCosineSim
        with open(out_dir / f"cross_max_similarities_{run_types[0]}_{run_types[1]}.csv", "w") as f:
            f.write("Layer,Dict1Index,MaxCosineSim\n")
            for layer_num, max_cosine_sim in cross_max_similarity.items():
                for i, max_sim in enumerate(max_cosine_sim.flatten().detach().numpy()):
                    f.write(f"{layer_num},{dict_1_alive_indices[layer_num][i]},{max_sim}\n")


def cross_seed_similarity_all_types(
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
    fig, axs = plt.subplots(3, 1, figsize=(5, 4), sharex=True)

    for i, (run_type, max_cosine_sim) in enumerate(sim_dict.items()):
        sims = max_cosine_sim.flatten().detach().numpy()
        axs[i].hist(
            sims,
            range=(0, 1),
            bins=100,
            color=COLOR_MAP[run_type],
            density=True,
            alpha=0.7,
        )
        axs[i].set_yticks([])
        axs[i].set_title(run_type)

    plt.xlabel("Max Cosine Similarity")
    plt.xlim(0, 1)
    plt.tight_layout(h_pad=0.3)

    out_dir = Path(__file__).parent / "out" / "seed_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "cross_seed_all_types.png")
    plt.savefig(out_dir / "cross_seed_all_types.svg")


def create_seed_max_similarity_comparison_plots(
    api: wandb.Api, project: str, run_ids: tuple[str, str], layer: int, run_type: str
):
    """Create plots comparing the max similarities between different seeds."""
    out_dir = Path(__file__).parent / "out" / "seed_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get pairwise similarities between the two seeds for each layer
    layer_cross_max_similarity, _ = get_cross_max_similarities(
        api=api,
        project_name=project,
        run_ids=run_ids,
    )

    plot_cross_max_similarity(
        {layer: layer_cross_max_similarity},
        out_file=out_dir / f"cross_max_similarities_{run_type}_{run_ids[0]}_{run_ids[1]}.png",
        plot_type="kde",
        cut_kde=True,
    )


def create_random_cross_max_similarity_plots(
    api: wandb.Api, project: str, run_id: str, layer: int, run_type: str
):
    """Compare the max similarities between a run and random vectors."""
    out_dir = Path(__file__).parent / "out" / "random_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get the alive dictionary elements for the run
    alive_elements = get_alive_dict_elements(api=api, project_name=project, run_id=run_id)

    # Create a random set of the same size as the alive dictionary elements
    rand_dict = torch.randn_like(alive_elements.alive_dict_elements)
    rand_dict = F.normalize(rand_dict, dim=0)

    # Get the max pairwise cosine similarity between the alive dictionary elements and the random
    # set

    cross_similarities = get_cosine_similarity(alive_elements.alive_dict_elements, rand_dict)

    # Get the max pairwise cosine similarity for each dictionary element
    max_cosine_sim, _ = torch.max(cross_similarities, dim=1)

    plot_cross_max_similarity(
        {layer: max_cosine_sim},
        out_file=out_dir / f"cross_max_similarities_{run_type}_{run_id}_random.png",
        plot_type="kde",
        cut_kde=True,
    )


if __name__ == "__main__":
    project = "gpt2"
    api = wandb.Api()

    create_random_cross_max_similarity_plots(
        api, project, run_id="zgdpkafo", layer=6, run_type="e2e-recon"
    )

    create_max_pairwise_similarity_plots(api, project)

    for run_type in [
        ("e2e", "local"),
        ("e2e", "e2e-recon"),
        ("local", "e2e-recon"),
        ("local", "e2e"),
        ("e2e-recon", "e2e"),
        ("e2e-recon", "local"),
    ]:
        # Store csv of indices and max similarities for each layer for e2e-recon vs local
        write_csv = run_type in [("e2e-recon", "local"), ("local", "e2e-recon")]
        create_cross_max_similarity_plots(
            api, project, run_types=run_type, constant_val="CE", write_csv=write_csv
        )

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

    run_ids: dict[str, tuple[str, str]] = {
        "e2e": ("atfccmo3", "tvj2owza"),
        "local": ("pzelh1s8", "ir00gg9g"),
        "e2e-recon": ("y8sca507", "hqo5azo2"),
    }
    cross_seed_similarity_all_types(api, project, run_ids)

    # Post-hoc ignore the identified outliers in e2e-local umap
    e2e_local_ce_lims: dict[int, dict[str, tuple[float | None, float | None]]] = {
        2: {"x": (-2.0, None), "y": (None, None)},
        6: {"x": (4.0, None), "y": (None, None)},
        10: {"x": (None, None), "y": (None, None)},
    }
    create_umap_plots(
        api,
        project,
        run_types=("e2e", "local"),
        compute_umaps=False,
        constant_val="CE",
        lims=e2e_local_ce_lims,
    )

    e2e_recon_local_ce_lims: dict[int, dict[str, tuple[float | None, float | None]]] = {
        2: {"x": (4, None), "y": (None, None)},
        6: {"x": (None, None), "y": (None, None)},
        10: {"x": (None, None), "y": (None, None)},
    }
    create_umap_plots(
        api,
        project,
        run_types=("e2e-recon", "local"),
        compute_umaps=False,
        constant_val="CE",
        lims=e2e_recon_local_ce_lims,
        grid=False,
    )
