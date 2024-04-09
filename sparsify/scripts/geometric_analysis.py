import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import umap
import wandb
import yaml
from jaxtyping import Float
from torch import Tensor
from wandb.apis.public import Run

from sparsify.analysis import create_run_df
from sparsify.loader import load_tlens_model
from sparsify.models.transformers import SAE, SAETransformer
from sparsify.scripts.train_tlens_saes.run_train_tlens_saes import Config
from sparsify.utils import filter_names

# These runs (keyed by layer number, value is the sparsity coeff), have similar CE loss diff
CONSTANT_CE_RUNS = {
    2: {"e2e": 0.5, "local": 0.8},
    6: {"e2e": 3, "local": 4},
    10: {"e2e": 0.5, "local": 4},
}
# These have similar L0
CONSTANT_L0_RUNS = {
    2: {"e2e": 1.5, "local": 4},
    6: {"e2e": 1.5, "local": 6},
    10: {"e2e": 1.5, "local": 10},
}


def plot_cosine_similarity_heatmap(
    e2e_dict_elements: Float[Tensor, "n_dense n_dict_elements"],
    local_dict_elements: Float[Tensor, "n_dense n_dict_elements"],
    labels: list[str],
    sae_pos: str,
):
    """Plot a cosine similarity heatmap between the alive dictionary elements of e2e and local runs.

    Args:
        all_alive_elements: A list of alive dictionary elements for each label.
        labels: The labels for each set of alive dictionary elements.
        sae_pos: The SAE position.
    """
    # Compute cosine similarity in pytorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e2e_dict_elements = e2e_dict_elements.to(device)
    local_dict_elements = local_dict_elements.to(device)

    # Normalize the tensors
    e2e_dict_elements = F.normalize(e2e_dict_elements, p=2, dim=0)
    local_dict_elements = F.normalize(local_dict_elements, p=2, dim=0)

    # Compute cosine similarity using matrix multiplication
    cosine_sim = torch.mm(e2e_dict_elements.T, local_dict_elements)

    # TODO: Think about how we want to visualize/analyse these cosine sims

    # cosine_sim = cosine_sim.detach().cpu().numpy()

    # # Make all values below 0.4 white to make it easier to see the differences
    # cmap = mcolors.LinearSegmentedColormap.from_list(
    #     "custom_cmap",
    #     [(0, "white"), (0.4, "white"), (1, "red")],
    #     N=256,
    # )

    # plt.figure(figsize=(10, 10))
    # sns.heatmap(cosine_sim, cmap=cmap, square=True, cbar_kws={"shrink": 0.5}, vmin=0, vmax=1)
    # plt.xlabel(labels[1])
    # plt.ylabel(labels[0])
    # plt.title(f"Cosine Similarity Heatmap of Alive Dictionary Elements in {sae_pos}")
    # plt.tight_layout()
    # plt.savefig(f"cosine_sim_heatmap_{sae_pos}_{labels[0]}_{labels[1]}.png")


def plot_umap(
    all_alive_elements: list[torch.Tensor], labels: list[str], sae_pos: str, seed: int = 1
):
    """Plot the UMAP of the alive dictionary elements, colored by the labels.

    Args:
        all_alive_elements: A list of alive dictionary elements for each label.
        labels: The labels for each set of alive dictionary elements.
        sae_pos: The SAE position.
        seed: The random seed to use for UMAP.
    """
    n_elements_per_label: list[int] = [elements.shape[1] for elements in all_alive_elements]
    all_alive_elements = torch.cat(all_alive_elements, dim=1).T.detach().numpy()
    reducer = umap.UMAP(random_state=seed)
    embedding = reducer.fit_transform(all_alive_elements)

    colors = sns.color_palette()[: len(labels)]
    plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels):
        embed = embedding[i * n_elements_per_label[i] : (i + 1) * n_elements_per_label[i]]  # type: ignore
        plt.scatter(
            embed,
            embed,
            label=label,
            s=1,
            color=colors[i],
            alpha=0.6,
        )
    plt.legend()
    label_elements = "_".join(
        [f"{label}-{n}" for label, n in zip(labels, n_elements_per_label, strict=False)]
    )
    plt.title(f"UMAP of alive dictionary elements in {sae_pos}: {label_elements}")
    plt.savefig(f"umap_{sae_pos}_{label_elements}.png")


def get_run_ids(api: wandb.Api, project_name: str, layer_num: int) -> tuple[str, str]:
    """Get the run IDs for the e2e and local runs for a given layer number.

    Args:
        api: The wandb API.
        project_name: The name of the wandb project.
        layer_num: The layer number.

    Returns:
        The run IDs for the e2e and local runs.
    """
    runs = api.runs(project_name)

    df = create_run_df(runs)

    # Only use seed=0
    df = df.loc[df["seed"] == 0]

    # Get the block 6 id runs with the constant L0
    e2e_series = df.loc[
        (df["run_type"] == "e2e")
        & (df["sparsity_coeff"] == CONSTANT_CE_RUNS[layer_num]["e2e"])
        & (df["layer"] == layer_num)
    ]["id"].values
    local_series = df.loc[
        (df["run_type"] == "layerwise")
        & (df["sparsity_coeff"] == CONSTANT_CE_RUNS[layer_num]["local"])
        & (df["layer"] == layer_num)
    ]["id"].values
    assert len(e2e_series) == 1
    assert len(local_series) == 1
    e2e_id = e2e_series[0]
    local_id = local_series[0]

    return e2e_id, local_id


def get_alive_dict_elements(
    api: wandb.Api, project_name: str, e2e_id: str, local_id: str
) -> tuple[Float[Tensor, "n_dense n_dict_elements"], Float[Tensor, "n_dense n_dict_elements"], str]:
    """Get the alive dictionary elements and sae_pos for the e2e and local runs.

    Args:
        api: The wandb API.
        project_name: The name of the wandb project.
        e2e_id: The run ID for the e2e run.
        local_id: The run ID for the local run.

    Returns:
        - Alive dictionary elements for the e2e run
        - Alive dictionary elements for the local run.
        - The raw SAE position common to both runs.
    """

    raw_sae_pos: str | None = None
    all_alive_elements = {}
    for run_id in [e2e_id, local_id]:
        run_type = "e2e" if run_id == e2e_id else "local"
        run: Run = api.run(f"{project_name}/{run_id}")
        config_file = [file for file in run.files() if "final_config.yaml" in file.name][0]
        config = Config(
            **yaml.safe_load(
                config_file.download(replace=False, exist_ok=True, root=f"/tmp/{run.id}/")
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
    return all_alive_elements["e2e"], all_alive_elements["local"], raw_sae_pos


if __name__ == "__main__":
    project = "gpt2"
    api = wandb.Api()
    for layer_num in [2, 6, 10]:
        e2e_id, local_id = get_run_ids(api=api, project_name="gpt2", layer_num=layer_num)

        e2e_dict_elements, local_dict_elements, raw_sae_pos = get_alive_dict_elements(
            api=api,
            project_name=project,
            e2e_id=e2e_id,
            local_id=local_id,
        )
        plot_umap(
            [e2e_dict_elements, local_dict_elements],
            labels=[f"e2e-{e2e_id}", f"local-{local_id}"],
            sae_pos=raw_sae_pos,
        )
        plot_cosine_similarity_heatmap(
            e2e_dict_elements,
            local_dict_elements,
            labels=[f"e2e-{e2e_id}", f"local-{local_id}"],
            sae_pos=raw_sae_pos,
        )
