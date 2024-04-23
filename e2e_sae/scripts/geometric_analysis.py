import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

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


# Regions of interest keyed by layer number
REGIONS: dict[int, LayerRegions] = {
    2: LayerRegions(
        filename="umap_embeds_blocks.2.hook_resid_pre.pt",
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
        filename="umap_embeds_blocks.6.hook_resid_pre.pt",
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
        filename="umap_embeds_blocks.10.hook_resid_pre.pt",
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
}


class EmbedInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    alive_elements_per_dict: tuple[int, int]
    embedding: Float[Tensor, "n_both_dicts 2"]
    sae_pos: str
    raw_dict_sizes: tuple[int, int]
    all_alive_indices: tuple[list[int], list[int]]


# class AliveElements(BaseModel):
#     model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
#     dict_1: Float[Tensor, "n_dense n_dict_elements1"]
#     dict_2: Float[Tensor, "n_dense n_dict_elements2"]
#     sae_pos: str
#     raw_dict_sizes: tuple[int, int]
#     all_alive_indices: tuple[list[int], list[int]]
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

    return cosine_sim


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
    labels: tuple[str, str],
    run_types: tuple[str, str],
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
    plt.figure(figsize=(10, 10))
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

    plt.savefig(out_file)
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


def create_umap_plots(api: wandb.Api, project: str, compute_umaps: bool = True):
    constant_val: Literal["CE", "l0"] = "CE"
    # Must chose two run types from ("e2e", "local", "e2e-recon") to compare
    run_types = ("e2e", "local")

    run_dict = CONSTANT_L0_RUNS if constant_val == "l0" else CONSTANT_CE_RUNS  # type: ignore
    out_dir = Path(__file__).parent / "out" / f"constant_{constant_val}_{'_'.join(run_types)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Post-hoc ignore the identified outliers
    lims: dict[int, dict[str, tuple[float | None, float | None]]] = {
        2: {"x": (-2.0, None), "y": (None, None)},
        6: {"x": (4.0, None), "y": (None, None)},
        10: {"x": (None, None), "y": (None, None)},
    }
    for layer_num in [2, 6, 10]:
        run_ids = run_dict[layer_num]

        embed_file = out_dir / REGIONS[layer_num].filename

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
        all_e2e_indices, all_local_indices = get_dict_indices_for_embedding_range(embed_info)
        assert len(all_e2e_indices) + len(all_local_indices) == embed_info.embedding.shape[0]
        e2e_embeds = embed_info.embedding[: len(all_e2e_indices)]
        local_embeds = embed_info.embedding[len(all_e2e_indices) :]

        # Create a csv file with columns: RunType, EmbeddingIndex, X, Y.
        # Useful for neuronpedia to upload to their website
        run_types_str = "-".join(run_types)
        with open(
            out_dir / f"layer-{layer_num}_constant_{constant_val}_{run_types_str}_embeds.csv", "w"
        ) as f:
            f.write("RunType,EmbeddingIndex,X,Y\n")
            for i in range(len(embed_info.embedding)):
                if i < len(all_e2e_indices):
                    f.write(
                        f"{run_types[0]},{all_e2e_indices[i]},{e2e_embeds[i][0]},{e2e_embeds[i][1]}\n"
                    )
                else:
                    idx = i - len(all_e2e_indices)
                    f.write(
                        f"{run_types[1]},{all_local_indices[idx]},{local_embeds[idx][0]},{local_embeds[idx][1]}\n"
                    )

        labels = (
            f"{run_types[0]}-{run_ids[run_types[0]]}",
            f"{run_types[1]}-{run_ids[run_types[1]]}",
        )
        plot_umap(
            embed_info,
            labels=labels,
            run_types=run_types,
            out_file=umap_file,
            lims=lims[layer_num],
            grid=False,
            regions=REGIONS[layer_num].regions,
        )
        for i, region in enumerate(REGIONS[layer_num].regions):
            e2e_indices, local_indices = get_dict_indices_for_embedding_range(
                embed_info, **region.coords.model_dump()
            )

            region_filename = REGIONS[layer_num].filename
            path_from_repo_root = (out_dir / region_filename).relative_to(REPO_ROOT)
            with open(out_dir / f"layer-{layer_num}_region-{i}.json", "w") as f:
                json.dump(
                    {
                        "embedding_file": str(path_from_repo_root),
                        "run_labels": labels,
                        "description": region.description,
                        "coords": region.coords.model_dump(),
                        "e2e": e2e_indices,
                        "local": local_indices,
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
    run_types: tuple[str, str],
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
            run_types=("e2e", "local"),
            out_file=out_file,
            from_file=False,
        )
        plot_max_pairwise_similarities(
            pairwise_similarities,
            constant_val=constant_val,
            out_file=out_file.with_suffix(".png"),
        )
        logger.info(f"Saved max pairwise similarity to {out_file.with_suffix('.png')}")


def get_cross_max_similarities(api: wandb.Api, project_name: str, run_ids: tuple[str, str]):
    """Get the max pairwise cosine similarity between the alive dictionary elements of two runs.

    Args:
        api: The wandb API.
        project_name: The name of the wandb project.
        run_ids: The IDs of the two runs to compare.

    Returns:
        The max pairwise cosine similarity between the alive dictionary elements of the two runs.
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

    return max_cosine_sim


CrossMaxSimilarity = dict[int, Float[Tensor, "n_dict_1"]]  # noqa: F821


def plot_cross_max_similarity(
    cross_max_similarity: CrossMaxSimilarity,
    out_file: Path,
):
    """Make a single plot with the layers side by side showing max cross similarities."""
    fig, axs = plt.subplots(1, len(cross_max_similarity), figsize=(15, 5))
    for i, (layer_num, max_cosine_sim) in enumerate(cross_max_similarity.items()):
        ax = axs[i]
        sns.kdeplot(max_cosine_sim.flatten().detach().numpy(), ax=ax)
        ax.set_title(f"Layer {layer_num}", fontweight="bold")
        ax.set_xlabel("Max Cosine Similarity")
        ax.set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.savefig(out_file.with_suffix(".svg"))
    logger.info(f"Saved cross max similarity to {out_file}")


def create_cross_max_similarity_plots(
    api: wandb.Api, project: str, run_types: tuple[str, str], constant_val: str = "CE"
):
    """Create plots comparing the max similarities between different run types."""
    run_dict = CONSTANT_L0_RUNS if constant_val == "l0" else CONSTANT_CE_RUNS
    out_dir = Path(__file__).parent / "out" / f"constant_{constant_val}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get pairwise similarities between the two runtypes for each layer
    cross_max_similarity: CrossMaxSimilarity = {}
    for layer_num in run_dict:
        cross_max_similarity[layer_num] = get_cross_max_similarities(
            api=api,
            project_name=project,
            run_ids=(run_dict[layer_num][run_types[0]], run_dict[layer_num][run_types[1]]),
        )

    plot_cross_max_similarity(
        cross_max_similarity,
        out_file=out_dir / f"cross_max_similarities_{run_types[0]}_{run_types[1]}.png",
    )


if __name__ == "__main__":
    project = "gpt2"
    api = wandb.Api()

    create_max_pairwise_similarity_plots(api, project)

    create_cross_max_similarity_plots(api, project, run_types=("e2e", "local"), constant_val="CE")

    create_umap_plots(api, project, compute_umaps=False)

    constant_val: Literal["CE", "l0"] = "l0"
    # Must chose two run types from ("e2e", "local", "e2e-recon") to compare
    run_types = ("e2e", "local")

    api = wandb.Api()
    run_dict = CONSTANT_L0_RUNS if constant_val == "l0" else CONSTANT_CE_RUNS  # type: ignore
    out_dir = Path(__file__).parent / "out" / f"constant_{constant_val}_{'_'.join(run_types)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer_num in [2, 6, 10]:
        run_ids = run_dict[layer_num]
        run_info = (
            f"({run_types[0]}-{run_ids[run_types[0]]}, {run_types[1]}-{run_ids[run_types[1]]})"
        )
        ### Compare seed 0 with seed 1 for e2e run on block 6. Note that we can't do
        ### this for e2e easily because we don't have final_config.yaml for those.
        # seed0_block6_local_id = "1jy3m5j0"
        # seed1_block6_local_id = "uqfp43ti"
        # if layer_num != 6:
        #     continue
        # # run_types = ("e2e-seed0", "e2e-seed1")
        # # run_ids = {"e2e-seed0": "atfccmo3", "e2e-seed1": "tvj2owza"}
        # # run_info = f"(seed0-{seed0_block6_e2e_id}, seed1-{seed1_block6_e2e_id})"
        # run_types = ("local-seed0", "local-seed1")
        # run_ids = {"local-seed0": seed0_block6_local_id, "local-seed1": seed1_block6_local_id}
        # run_info = f"(seed0-{seed0_block6_local_id}, seed1-{seed1_block6_local_id})"
        # out_dir = Path(__file__).parent / "out" / "local_seed0_seed1"
        # ###

        # alive_elements = get_alive_dict_elements(
        #     api=api,
        #     project_name=project,
        #     runs=[(run_type, run_ids[run_type]) for run_type in run_types],
        # )

        ###  Make dict_elements2 a random set of the same size as alive_elements.dict_1
        # run_info = f"({run_types[0]}-{run_ids[run_types[0]]}, (random vectors))"
        # rand_dict = torch.randn_like(alive_elements.dict_1)
        # alive_elements._replace(dict_2=F.normalize(rand_dict, dim=0))

        # out_dir = Path(__file__).parent / "out" / "e2e_random"
        ###

        # cosine_sim = get_cosine_similarity(dict_elements_1, dict_elements_2)
        # ###
        # plot_max_cosine_similarity(
        #     cosine_sim,
        #     title=f"{sae_pos} constant {constant_val} {run_info}",
        #     outfile=out_dir / f"max_cosine_sim_{sae_pos}.png",
        # )

        # analyze_cosine_similarity(cosine_sim, threshold_high=0.8, threshold_low=0.2)
        # plot_cosine_similarity_heatmap(
        #     cosine_sim,
        #     labels=[f"e2e-{e2e_id}", f"local-{local_id}"],
        #     sae_pos=sae_pos,
        # )
