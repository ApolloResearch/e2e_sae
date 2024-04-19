import json
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
import yaml
from jaxtyping import Float
from matplotlib import colors as mcolors
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from wandb.apis.public import Run

from sparsify.loader import load_tlens_model
from sparsify.models.transformers import SAE, SAETransformer
from sparsify.scripts.train_tlens_saes.run_train_tlens_saes import Config
from sparsify.settings import REPO_ROOT
from sparsify.utils import filter_names

CONSTANT_CE_RUNS = {
    2: {"e2e": "ovhfts9n", "local": "ue3lz0n7", "e2e-recon": "visi12en"},
    6: {"e2e": "zgdpkafo", "local": "1jy3m5j0", "e2e-recon": "2lzle2f0"},
    10: {"e2e": "8crnit9h", "local": "m2hntlav", "e2e-recon": "1ei8azro"},
}
CONSTANT_L0_RUNS = {
    2: {"e2e": "bst0prdd", "local": "6vtk4k51", "e2e-recon": "e26jflpq"},
    6: {"e2e": "tvj2owza", "local": "jup3glm9", "e2e-recon": "2lzle2f0"},
    10: {"e2e": "8crnit9h", "local": "5vmpdgaz", "e2e-recon": "c3nr4wce"},
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
    raw_sae_pos: str
    raw_dict_sizes: tuple[int, int]
    all_alive_indices: tuple[list[int], list[int]]


class AliveElements(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    dict_1: Float[Tensor, "n_dense n_dict_elements1"]
    dict_2: Float[Tensor, "n_dense n_dict_elements2"]
    raw_sae_pos: str
    raw_dict_sizes: tuple[int, int]
    all_alive_indices: tuple[list[int], list[int]]


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


def compute_umap_embedding(alive_elements: AliveElements, seed: int = 1) -> EmbedInfo:
    """Compute the UMAP embedding of the alive dictionary elements and save it to file.

    Args:
        alive_elements: The alive dictionary elements object for the two runs.
        seed: The random seed to use for UMAP.
    """
    alive_elements_per_dict: tuple[int, int] = (
        alive_elements.dict_1.shape[1],
        alive_elements.dict_2.shape[1],
    )
    all_alive_elements = (
        torch.cat([alive_elements.dict_1, alive_elements.dict_2], dim=1).T.detach().numpy()
    )
    reducer = umap.UMAP(random_state=seed)
    embedding = reducer.fit_transform(all_alive_elements)
    embed_info = EmbedInfo(
        alive_elements_per_dict=alive_elements_per_dict,
        embedding=torch.tensor(embedding),
        raw_sae_pos=alive_elements.raw_sae_pos,
        raw_dict_sizes=alive_elements.raw_dict_sizes,
        all_alive_indices=alive_elements.all_alive_indices,
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
    raw_sae_pos = embed_info.raw_sae_pos

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
    plt.title(f"UMAP of alive dictionary elements in {raw_sae_pos}: {run_type_str}")

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
            plt.gca().add_patch(rect)

    plt.savefig(out_file)


def get_alive_dict_elements(
    api: wandb.Api,
    project_name: str,
    runs: list[tuple[str, str]],
) -> AliveElements:
    """Get the alive dictionary elements and sae_pos for the e2e and local runs.

    Args:
        api: The wandb API.
        project_name: The name of the wandb project.
        runs: A list of (run_type, run_id) tuples.

    Returns:
        AliveElements object containing containing the alive dictionary elements and related info
        for the two runs.
    """

    assert len(runs) == 2, "Only two runs can be compared at a time"

    raw_sae_pos = None
    all_alive_elements = []
    dict_sizes = []
    all_alive_indices = []
    for _, run_id in runs:
        run: Run = api.run(f"{project_name}/{run_id}")
        config_files = [file for file in run.files() if "final_config.yaml" in file.name]

        assert len(config_files) == 1
        config = Config(
            **yaml.safe_load(
                config_files[0].download(replace=False, exist_ok=True, root=f"/tmp/{run.id}/")
            )
        )
        tlens_model = load_tlens_model(
            tlens_model_name=config.tlens_model_name, tlens_model_path=config.tlens_model_path
        )

        raw_sae_positions = filter_names(
            list(tlens_model.hook_dict.keys()), config.saes.sae_positions
        )
        assert len(raw_sae_positions) == 1
        if raw_sae_pos is None:
            raw_sae_pos = raw_sae_positions[0]
        else:
            assert raw_sae_pos == raw_sae_positions[0], "SAE positions are not the same"

        model = SAETransformer(
            tlens_model=tlens_model,
            raw_sae_positions=raw_sae_positions,
            dict_size_to_input_ratio=config.saes.dict_size_to_input_ratio,
        )
        # Weights file should be the largest .pt file. All have format (samples_*.pt)
        weight_files = [file for file in run.files() if file.name.endswith(".pt")]
        # Latest checkpoint
        weight_file = sorted(
            weight_files, key=lambda x: int(x.name.split(".pt")[0].split("_")[-1])
        )[-1]
        latest_checkpoint = wandb.restore(
            weight_file.name,
            run_path=f"{project_name}/{run.id}",
            root=f"/tmp/{run.id}/",
            replace=False,
        )
        assert latest_checkpoint is not None
        model.saes.load_state_dict(torch.load(latest_checkpoint.name, map_location="cpu"))
        model.saes.eval()

        # Get the alive dictionary element indices
        alive_indices = run.summary_metrics[
            f"sparsity/alive_dict_elements_indices_final/{raw_sae_pos}"
        ]

        sae: SAE = model.saes[model.all_sae_positions[0]]  # type: ignore
        alive_elements = sae.dict_elements[:, alive_indices]

        all_alive_elements.append(alive_elements)
        dict_sizes.append(sae.dict_elements.shape[1])
        all_alive_indices.append(alive_indices)

    assert raw_sae_pos is not None
    return AliveElements(
        dict_1=all_alive_elements[0],
        dict_2=all_alive_elements[1],
        raw_sae_pos=raw_sae_pos,
        raw_dict_sizes=tuple(dict_sizes),
        all_alive_indices=tuple(all_alive_indices),
    )


if __name__ == "__main__":
    project = "gpt2"
    constant_val: Literal["CE", "l0"] = "CE"
    # Must chose two run types from ("e2e", "local", "e2e-recon") to compare
    run_types = ("e2e", "local")
    # run_types = ("local", "e2e")

    # Post-hoc ignore the outliers for gpt2 plots
    lims: dict[int, dict[str, tuple[float | None, float | None]]] = {
        2: {"x": (-2.0, None), "y": (None, None)},
        6: {"x": (4.0, None), "y": (None, None)},
        10: {"x": (None, None), "y": (None, None)},
    }

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

        alive_elements = get_alive_dict_elements(
            api=api,
            project_name=project,
            runs=[(run_type, run_ids[run_type]) for run_type in run_types],
        )

        ### Get pairwise cosine similarity for dict1. Note that we have make the diagonals
        ### very negative because we don't want to compare the same vector with itself.
        # alive_elements._replace(dict_2=alive_elements.dict_1)
        # run_info = f"({run_types[0]}-{run_ids[run_types[0]]}) alive={alive_elements.dict_1.shape[1]}"
        # out_dir = Path(__file__).parent / "out" / f"constant_{constant_val}_{run_types[0]}-pairwise"
        ###

        ###  Make dict_elements2 a random set of the same size as alive_elements.dict_1
        # run_info = f"({run_types[0]}-{run_ids[run_types[0]]}, (random vectors))"
        # rand_dict = torch.randn_like(alive_elements.dict_1)
        # alive_elements._replace(dict_2=F.normalize(rand_dict, dim=0))

        # out_dir = Path(__file__).parent / "out" / "e2e_random"
        ###

        # embed_file = out_dir / f"umap_embeds_{alive_elements.raw_sae_pos}.pt"
        embed_file = out_dir / REGIONS[layer_num].filename
        # Can comment this slow part out if just iterating on plotting
        # embed_info = compute_umap_embedding(alive_elements=alive_elements)
        # torch.save(embed_info.model_dump(), embed_file)

        embed_info = EmbedInfo(**torch.load(embed_file))
        umap_file = embed_file.with_suffix(".png")
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
            print(f"Region: {region.description} ({region.coords})")
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

        # cosine_sim = get_cosine_similarity(dict_elements_1, dict_elements_2)
        # ### make the diagonal very negative so that it doesn't show up when maxing for getting
        # ### Pairwise cosine similarity
        # # cosine_sim.fill_diagonal_(float("-inf"))
        # ###
        # plot_max_cosine_similarity(
        #     cosine_sim,
        #     title=f"{raw_sae_pos} constant {constant_val} {run_info}",
        #     outfile=out_dir / f"max_cosine_sim_{raw_sae_pos}.png",
        # )

        # analyze_cosine_similarity(cosine_sim, threshold_high=0.8, threshold_low=0.2)
        # plot_cosine_similarity_heatmap(
        #     cosine_sim,
        #     labels=[f"e2e-{e2e_id}", f"local-{local_id}"],
        #     sae_pos=raw_sae_pos,
        # )
