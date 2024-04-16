"""
Script to select regions of embeddings from a UMAP and get the feature indices in them.
"""

import json
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from jaxtyping import Float
from pydantic import BaseModel
from torch import Tensor

from sparsify.settings import REPO_ROOT


class RegionCoords(BaseModel):
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None


class Region(BaseModel):
    coords: RegionCoords
    description: str


class LayerRegions(BaseModel):
    filename: str
    regions: list[Region]


# Regions keyed by layer number
REGIONS: dict[int, LayerRegions] = {
    2: LayerRegions(
        filename="umap_embeds_blocks.2.hook_resid_pre.pt",
        regions=[
            Region(coords=RegionCoords(ymin=5.5), description="Mostly-orange line at top"),
            Region(
                coords=RegionCoords(xmin=10.5, xmax=11, ymin=3.6, ymax=3.9),
                description="Blue cluster near top",
            ),
            Region(
                coords=RegionCoords(xmin=7.5, xmax=8, ymin=0.3, ymax=0.6),
                description="Orange cluster on middle left",
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
            Region(coords=RegionCoords(xmax=3), description="Blue area in bottom left"),
            Region(
                coords=RegionCoords(xmin=9, xmax=11.2, ymax=4),
                description="Orange line structure at bottom",
            ),
            Region(
                coords=RegionCoords(xmin=10.7, xmax=11, ymin=5.5, ymax=6),
                description="Blue region above orange line",
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
                description="Blue cluster in middle",
            ),
            Region(
                coords=RegionCoords(xmin=6.5, ymin=3.5, ymax=3.7),
                description="Orange cluster on right",
            ),
            Region(
                coords=RegionCoords(xmin=0, xmax=0.5, ymin=8, ymax=8.5),
                description="Mixed cluster in top left",
            ),
            Region(
                coords=RegionCoords(xmin=1, xmax=1.2, ymin=5, ymax=5.2),
                description="Random cluster in middle",
            ),
        ],
    ),
}


class UmapEmbedInfo(NamedTuple):
    alive_elements_per_dict: tuple[int, int]
    embedding: Float[Tensor, "n_both_dicts 2"]
    labels: list[str]
    raw_sae_pos: str
    raw_dict_sizes: tuple[int, int]
    all_alive_indices: tuple[list[int], list[int]]


def get_dict_indices_for_embedding_range(
    embed_info: UmapEmbedInfo,
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


def plot_with_grid(embed_info: UmapEmbedInfo) -> None:
    """Plot the embeddings with a grid in the background.

    Args:
        embed_info: Named tuple containing the embedding and other information.
    """
    alive_elements = embed_info.alive_elements_per_dict

    plt.figure(figsize=(10, 10))
    colors = sns.color_palette()[:2]
    for i in range(2):
        plt.scatter(
            embed_info.embedding[i * alive_elements[i] : (i + 1) * alive_elements[i], 0],
            embed_info.embedding[i * alive_elements[i] : (i + 1) * alive_elements[i], 1],
            s=1,
            color=colors[i],
            alpha=0.6,
        )
    # Put ticks every 0.5 points
    plt.xticks(np.arange(int(plt.xlim()[0]), int(plt.xlim()[1]), 0.5))
    plt.yticks(np.arange(int(plt.ylim()[0]), int(plt.ylim()[1]), 0.5))
    # Make the tick text size smaller
    plt.tick_params(axis="both", which="major", labelsize=8)

    plt.grid()
    plt.show()


if __name__ == "__main__":
    for layer in [2, 6, 10]:
        filename = REGIONS[layer].filename

        embed_dir = Path(__file__).parent / "out" / "constant_CE_e2e_local"
        embeds = torch.load(embed_dir / filename)
        embeds["embedding"] = torch.tensor(embeds["embedding"])

        embed_info = UmapEmbedInfo(**embeds)

        # Uncomment to plot the embeddings with a grid. This was used to select the regions.
        # plot_with_grid(embed_info)

        for i, region in enumerate(REGIONS[layer].regions):
            print(f"Region: {region.description} ({region.coords})")
            e2e_indices, local_indices = get_dict_indices_for_embedding_range(
                embed_info, **region.coords.model_dump()
            )

            path_from_repo_root = (embed_dir / filename).relative_to(REPO_ROOT)
            with open(embed_dir / f"layer-{layer}_region-{i}.json", "w") as f:
                json.dump(
                    {
                        "embedding_file": str(path_from_repo_root),
                        "run_labels": embed_info.labels,
                        "description": region.description,
                        "coords": region.coords.model_dump(),
                        "e2e": e2e_indices,
                        "local": local_indices,
                    },
                    f,
                    indent=2,
                )
