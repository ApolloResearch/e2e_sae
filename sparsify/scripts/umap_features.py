"""
Script to select regions of embeddings from a UMAP and get the feature indices in them.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from jaxtyping import Float
from torch import Tensor

# Regions keyed by layer number
REGIONS = {
    2: {
        "file": "umap_embeds_blocks.2.hook_resid_pre.pt",
        "regions": [
            {"coords": {"ymin": 5.5}, "description": "Mostly-orange line at top"},
            {
                "coords": {"xmin": 10.5, "xmax": 11, "ymin": 3.6, "ymax": 3.9},
                "description": "Blue cluster near top",
            },
            {
                "coords": {"xmin": 7.5, "xmax": 8, "ymin": 0.3, "ymax": 0.6},
                "description": "Orange cluster on middle left",
            },
            {
                "coords": {"xmin": 11, "xmax": 11.2, "ymin": -1, "ymax": -0.8},
                "description": "Random cluster",
            },
        ],
    },
    6: {
        "file": "umap_embeds_blocks.6.hook_resid_pre.pt",
        "regions": [
            {"coords": {"xmax": 3}, "description": "Blue area in bottom left"},
            {
                "coords": {"xmin": 9, "xmax": 11.2, "ymax": 4},
                "description": "Orange line structure at bottom",
            },
            {
                "coords": {"xmin": 10.7, "xmax": 11, "ymin": 5.5, "ymax": 6},
                "description": "Blue region above orange line",
            },
            {
                "coords": {"xmin": 10.5, "xmax": 11, "ymin": 8.5, "ymax": 9},
                "description": "Random cluster in middle",
            },
        ],
    },
    10: {
        "file": "umap_embeds_blocks.10.hook_resid_pre.pt",
        "regions": [
            {
                "coords": {"xmin": 2, "xmax": 2.2, "ymin": 2.3, "ymax": 2.5},
                "description": "Blue cluster in middle",
            },
            {
                "coords": {"xmin": 6.5, "ymin": 3.5, "ymax": 3.7},
                "description": "Orange cluster on right",
            },
            {
                "coords": {"xmin": 0, "xmax": 0.5, "ymin": 8, "ymax": 8.5},
                "description": "Mixed cluster in top left",
            },
            {
                "coords": {"xmin": 1, "xmax": 1.2, "ymin": 5, "ymax": 5.2},
                "description": "Random cluster in middle",
            },
        ],
    },
}


def get_dict_indices_for_embedding_range(
    embeds: dict[str, tuple[int, int] | Float[Tensor, "n_both_dicts 2"]],
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
) -> tuple[list[int], list[int]]:
    """Get the indices of the embeddings that fall within the specified region.

    Args:
        embeds: Dictionary of n_elements_per_dict and embedding.
        xmin: The minimum x value to filter by.
        xmax: The maximum x value to filter by.
        ymin: The minimum y value to filter by.
        ymax: The maximum y value to filter by.

    Returns:
        - Indices of the first dict whose values lie in the specified region.
        - Indices of the second dict whose values lie in the specified region.
    """
    embedding = embeds["embedding"]
    assert isinstance(embedding, torch.Tensor)
    x = embedding[:, 0]
    y = embedding[:, 1]

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
    first_dict_indices = output[output < embeds["n_elements_per_dict"][0]].tolist()
    second_dict_indices = output[output >= embeds["n_elements_per_dict"][0]].tolist()

    return first_dict_indices, second_dict_indices


def plot_with_grid(embeds: dict[str, tuple[int, int] | Float[Tensor, "n_both_dicts 2"]]):
    """Plot the embeddings with a grid in the background.

    Args:
        embeds: Dictionary of n_elements_per_dict and embedding.
    """
    embedding = embeds["embedding"]
    assert isinstance(embedding, torch.Tensor)

    plt.figure(figsize=(10, 10))
    colors = sns.color_palette()[:2]
    for i in range(2):
        plt.scatter(
            embedding[
                i * embeds["n_elements_per_dict"][i] : (i + 1) * embeds["n_elements_per_dict"][i], 0
            ],
            embedding[
                i * embeds["n_elements_per_dict"][i] : (i + 1) * embeds["n_elements_per_dict"][i], 1
            ],
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
    # Select your layer here
    layer = 6

    # TODO: Use a dataclass/pydantic base model for the regions to avoid type errors
    filename = REGIONS[layer]["file"]
    assert isinstance(filename, str)

    embed_dir = Path(__file__).parent / "out" / "constant_CE_e2e_local"
    embeds = torch.load(embed_dir / filename)
    embeds["embedding"] = torch.tensor(embeds["embedding"])

    # Uncomment to plot the embeddings with a grid. This was used to select the regions.
    # plot_with_grid(embeds)
    regions = REGIONS[layer]["regions"]
    assert isinstance(regions, list)
    for i, region in enumerate(regions):
        description = region["description"]
        assert isinstance(description, str)
        coords = region["coords"]
        assert isinstance(coords, dict)
        print(f"Region: {description} ({coords})")
        e2e_indices, local_indices = get_dict_indices_for_embedding_range(embeds, **coords)
        with open(embed_dir / f"layer-{layer}_region-{i}.json", "w") as f:
            json.dump(
                {
                    "embedding_file": REGIONS[layer]["file"],
                    "description": description,
                    "coords": coords,
                    "e2e": e2e_indices,
                    "local": local_indices,
                },
                f,
                indent=2,
            )
