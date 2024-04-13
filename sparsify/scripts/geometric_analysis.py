from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import umap
import wandb
import yaml
from jaxtyping import Float
from matplotlib import colors as mcolors
from torch import Tensor
from wandb.apis.public import Run

from sparsify.analysis import create_run_df
from sparsify.loader import load_tlens_model
from sparsify.models.transformers import SAE, SAETransformer
from sparsify.scripts.train_tlens_saes.run_train_tlens_saes import Config
from sparsify.utils import filter_names

# These runs (keyed by layer number, value is the sparsity coeff), have similar CE loss diff
# CONSTANT_CE_RUNS = {
#     2: {"e2e": 0.5, "local": 0.8, "e2e-recon": 10},
#     6: {"e2e": 3, "local": 4, "e2e-recon": 50},
#     10: {"e2e": 1.5, "local": 6, "e2e-recon": 25},
# }
# # These have similar L0
# CONSTANT_L0_RUNS = {
#     2: {"e2e": 1.5, "local": 4, "e2e-recon": 35},
#     6: {"e2e": 1.5, "local": 6, "e2e-recon": 35},
#     10: {"e2e": 1.5, "local": 10, "e2e-recon": 25},
# }

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


def plot_umap(
    all_alive_elements: tuple[
        Float[Tensor, "n_dense n_dict_elements1"], Float[Tensor, "n_dense n_dict_elements2"]
    ],
    labels: tuple[str, str],
    sae_pos: str,
    outdir: Path,
    seed: int = 1,
):
    """Plot the UMAP of the alive dictionary elements, colored by the labels.

    Saves the plot and the embedding to the output directory.

    Args:
        all_alive_elements: A tuple of alive dictionary elements for each label.
        labels: The labels for each set of alive dictionary elements.
        sae_pos: The SAE position.
        outdir: The output directory to save the plot and embedding.
        seed: The random seed to use for UMAP.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    n_elements_per_dict: tuple[int, int] = (
        all_alive_elements[0].shape[1],
        all_alive_elements[1].shape[1],
    )
    all_alive_elements = torch.cat(all_alive_elements, dim=1).T.detach().numpy()
    reducer = umap.UMAP(random_state=seed)
    embedding = reducer.fit_transform(all_alive_elements)

    colors = sns.color_palette()[: len(labels)]
    plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels):
        plt.scatter(
            embedding[i * n_elements_per_dict[i] : (i + 1) * n_elements_per_dict[i], 0],  # type: ignore
            embedding[i * n_elements_per_dict[i] : (i + 1) * n_elements_per_dict[i], 1],  # type: ignore
            label=label,
            s=1,
            color=colors[i],
            alpha=0.3,
        )
    plt.legend()
    label_elements = "_".join(
        [f"{label}-{n}" for label, n in zip(labels, n_elements_per_dict, strict=False)]
    )
    plt.title(f"UMAP of alive dictionary elements in {sae_pos}: {label_elements}")
    plt.savefig(outdir / f"umap_{sae_pos}.png")
    embeds = {"n_elements_per_dict": n_elements_per_dict, "embedding": embedding, "labels": labels}
    torch.save(embeds, outdir / f"umap_embeds_{sae_pos}.pt")


def get_run_ids(
    api: wandb.Api,
    project_name: str,
    layer_num: int,
    constant_val: Literal["CE", "l0"],
    run_types: tuple[str, ...],
    seed: int = 0,
) -> dict[str, str]:
    """Get the run IDs for the e2e, local and e2e+recon runs for a given layer number.

    Args:
        api: The wandb API.
        project_name: The name of the wandb project.
        layer_num: The layer number.
        constant_val: The constant value to use for the CE loss or L0.
        run_types: The run types to get the IDs for.
        seed: The seed to use for the runs.

    Returns:
        The run IDs for the e2e, local and e2e+recon runs.
    """
    runs = api.runs(project_name)

    df = create_run_df(runs)

    # Only use seed=0
    df = df.loc[df["seed"] == seed]

    constant_dict = CONSTANT_CE_RUNS if constant_val == "CE" else CONSTANT_L0_RUNS

    run_info = {}
    for run_type in run_types:
        assert (
            run_type in constant_dict[layer_num]
        ), f"Run type {run_type} not found in constant dict"

        series = df.loc[
            (df["run_type"] == run_type)
            & (df["sparsity_coeff"] == constant_dict[layer_num][run_type])
            & (df["layer"] == layer_num)
        ]["id"].values
        assert len(series) == 1
        run_info[run_type] = series[0]

    return run_info


def get_alive_dict_elements(
    api: wandb.Api,
    project_name: str,
    runs: list[tuple[str, str]],
) -> tuple[dict[str, Float[Tensor, "n_dense n_dict_elements"]], str]:
    """Get the alive dictionary elements and sae_pos for the e2e and local runs.

    Args:
        api: The wandb API.
        project_name: The name of the wandb project.
        runs: A list of (run_type, run_id) tuples.

    Returns:
        - dictionary of alive dictionary elements for each run, keyed by run type
        - The raw SAE position common to both runs.
    """

    raw_sae_pos: str | None = None
    all_alive_elements = {}
    for run_type, run_id in runs:
        run: Run = api.run(f"{project_name}/{run_id}")
        config_files = [file for file in run.files() if "final_config.yaml" in file.name]
        # if len(config_files) == 0:
        #     # Some seed=1 runs don't have a final_config.yaml file. Get the required vals from the
        #     # partial config saved in run.config
        #     # Use a named tuple to mimic the Config class with keys "tlens_model_name",
        #     # "tlens_model_path", "saes"
        #     assert run.config["seed"] == 1
        #     saes = namedtuple("saes", ["sae_positions", "dict_size_to_input_ratio"])(
        #         sae_positions=run.config["saes"]["sae_positions"],
        #         dict_size_to_input_ratio=run.config["saes"]["dict_size_to_input_ratio"],
        #     )
        #     config = namedtuple("Config", ["tlens_model_name", "tlens_model_path", "saes"])(
        #         tlens_model_name=run.config["tlens_model_name"],
        #         tlens_model_path=run.config["tlens_model_path"],
        #         saes=saes,
        #     )
        # else:
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
            weight_file.name, run_path=f"{project}/{run.id}", root=f"/tmp/{run.id}/", replace=False
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
        all_alive_elements[run_type] = alive_elements

    assert raw_sae_pos is not None
    return all_alive_elements, raw_sae_pos


if __name__ == "__main__":
    project = "gpt2"
    constant_val: Literal["CE", "l0"] = "CE"
    # Must chose two run types from ("e2e", "local", "e2e-recon") to compare
    run_types = ("e2e", "local")
    # run_types = ("local", "e2e")

    api = wandb.Api()
    run_dict = CONSTANT_L0_RUNS if constant_val == "l0" else CONSTANT_CE_RUNS  # type: ignore
    out_dir = Path(__file__).parent / "out" / f"constant_{constant_val}_{'_'.join(run_types)}"

    for layer_num in [2, 6, 10]:
        # run_ids = get_run_ids(
        #     api=api,
        #     project_name="gpt2",
        #     layer_num=layer_num,
        #     constant_val=constant_val,
        #     run_types=run_types,
        # )
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

        alive_dict_elements, raw_sae_pos = get_alive_dict_elements(
            api=api,
            project_name=project,
            runs=[(run_type, run_ids[run_type]) for run_type in run_types],
        )
        dict_elements_1 = alive_dict_elements[run_types[0]]
        dict_elements_2 = alive_dict_elements[run_types[1]]

        ### Get pairwise cosine similarity for dict1. Note that we have make the diagonals
        ### very negative because we don't want to compare the same vector with itself.
        # dict_elements_2 = dict_elements_1
        # run_info = f"({run_types[0]}-{run_ids[run_types[0]]}) alive={dict_elements_1.shape[1]}"
        # out_dir = Path(__file__).parent / "out" / f"constant_{constant_val}_{run_types[0]}-pairwise"
        ###

        ###  Make dict_elements2 a random set of the same size as dict_elements_1
        # run_info = f"({run_types[0]}-{run_ids[run_types[0]]}, (random vectors))"
        # dict_elements_2 = torch.randn_like(dict_elements_1)
        # dict_elements_2 = F.normalize(dict_elements_2, dim=0)

        # out_dir = Path(__file__).parent / "out" / "e2e_random"
        ###
        plot_umap(
            (dict_elements_1, dict_elements_2),
            labels=(
                f"{run_types[0]}-{run_ids[run_types[0]]}",
                f"{run_types[1]}-{run_ids[run_types[1]]}",
            ),
            sae_pos=raw_sae_pos,
            outdir=out_dir,
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
