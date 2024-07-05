import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
from jaxtyping import Float, Int
from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict
from wandb.apis.public import Run

from e2e_sae.data import DatasetConfig, create_data_loader
from e2e_sae.hooks import SAEActs
from e2e_sae.log import logger
from e2e_sae.models.transformers import SAETransformer
from e2e_sae.scripts.analysis.geometric_analysis import create_subplot_hists
from e2e_sae.scripts.analysis.plot_settings import SIMILAR_CE_RUNS, STYLE_MAP

ActTensor = Float[torch.Tensor, "batch seq hidden"]
LogitTensor = Float[torch.Tensor, "batch seq vocab"]

OUT_DIR = Path(__file__).parent / "out" / "activation_analysis"


class Acts(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokens: Int[torch.Tensor, "batch seq"] = torch.empty(0, 1024, dtype=torch.long)
    orig: ActTensor = torch.empty(0, 1024, 768)
    recon: ActTensor = torch.empty(0, 1024, 768)
    kl: Float[torch.Tensor, "batch seq"] = torch.empty(0, 1024)
    c_idxs: Int[torch.Tensor, "idxs"] | None = None  # noqa: F821
    c: Float[torch.Tensor, "batch seq c"] | None = None
    orig_pred_tok_ids: Int[torch.Tensor, "batch seq k"] | None = None
    orig_pred_log_probs: Float[torch.Tensor, "batch seq k"] | None = None
    recon_pred_tok_ids: Int[torch.Tensor, "batch seq k"] | None = None
    recon_pred_log_probs: Float[torch.Tensor, "batch seq k"] | None = None
    k: int = 5

    def add(
        self,
        tokens: torch.Tensor,
        acts: SAEActs,
        kl: Float[torch.Tensor, "batch seq"],
        c_idxs: Int[torch.Tensor, "idxs"] | None = None,  # noqa: F821
        orig_logits: LogitTensor | None = None,
        sae_logits: LogitTensor | None = None,
    ):
        self.tokens = torch.cat([self.tokens, tokens.cpu()])
        self.orig = torch.cat([self.orig, acts.input.cpu()])
        self.recon = torch.cat([self.recon, acts.output.cpu()])
        self.kl = torch.cat([self.kl, kl.cpu()])
        if c_idxs is not None:
            if self.c is None:
                self.c_idxs = c_idxs
                self.c = torch.empty(0, 1024, len(c_idxs), dtype=acts.c.dtype)
            self.c = torch.cat([self.c, acts.c[:, :, c_idxs].cpu()])
        if orig_logits is not None:
            if self.orig_pred_tok_ids is None:
                self.orig_pred_tok_ids = torch.empty(0, 1024, self.k, dtype=torch.long)
            if self.orig_pred_log_probs is None:
                self.orig_pred_log_probs = torch.empty(0, 1024, self.k)
            orig_logprobs = F.log_softmax(orig_logits, dim=-1)
            orig_pred_log_probs, orig_pred_tok_ids = orig_logprobs.topk(self.k, dim=-1)
            self.orig_pred_tok_ids = torch.cat([self.orig_pred_tok_ids, orig_pred_tok_ids.cpu()])
            self.orig_pred_log_probs = torch.cat(
                [self.orig_pred_log_probs, orig_pred_log_probs.cpu()]
            )
        if sae_logits is not None:
            if self.recon_pred_tok_ids is None:
                self.recon_pred_tok_ids = torch.empty(0, 1024, self.k, dtype=torch.long)
            if self.recon_pred_log_probs is None:
                self.recon_pred_log_probs = torch.empty(0, 1024, self.k)
            recon_logprobs = F.log_softmax(sae_logits, dim=-1)
            recon_pred_log_probs, recon_pred_tok_ids = recon_logprobs.topk(self.k, dim=-1)
            self.recon_pred_tok_ids = torch.cat([self.recon_pred_tok_ids, recon_pred_tok_ids.cpu()])
            self.recon_pred_log_probs = torch.cat(
                [self.recon_pred_log_probs, recon_pred_log_probs.cpu()]
            )

    def __len__(self):
        return len(self.tokens)

    def __str__(self) -> str:
        return f"Acts(len={len(self)})"


def kl_div(new_logits: LogitTensor, old_logits: LogitTensor) -> Float[torch.Tensor, "batch seq"]:
    return F.kl_div(
        F.log_softmax(new_logits, dim=-1), F.softmax(old_logits, dim=-1), reduction="none"
    ).sum(-1)


@torch.no_grad()
def get_acts(
    run: Run,
    batch_size=5,
    batches=1,
    device: str = "cuda",
    c_idxs: Int[torch.Tensor, "idxs"] | None = None,  # noqa: F821
    load_cache: bool = True,
    save_cache: bool = True,
) -> Acts:
    cache_file = OUT_DIR / "cache" / f"{run.id}.pt"
    if load_cache and cache_file.exists():
        cached_acts = Acts(**torch.load(cache_file))
        if len(cached_acts) >= batches * batch_size:
            logger.info(f"Loaded cached acts from {cache_file}")
            return cached_acts
        logger.info(f"Cache file {cache_file} is incomplete, generating new acts...")

    model = SAETransformer.from_wandb(f"{run.project}/{run.id}")
    model.to(device)
    data_config = DatasetConfig(**run.config["eval_data"])
    loader, _ = create_data_loader(data_config, batch_size=batch_size, global_seed=22)
    acts = Acts()
    assert len(model.raw_sae_positions) == 1
    sae_pos = model.raw_sae_positions[0]

    loader_iter = iter(loader)
    for _ in tqdm.trange(batches, disable=(batches == 1)):
        tokens = next(loader_iter)["input_ids"].to(device)
        orig_logits, _ = model.forward_raw(tokens, run_entire_model=True, final_layer=None)
        sae_logits, sae_cache = model.forward(tokens, [sae_pos])
        sae_acts = sae_cache[sae_pos]
        assert isinstance(sae_acts, SAEActs)
        assert sae_logits is not None
        acts.add(
            tokens,
            sae_acts,
            kl=kl_div(sae_logits, orig_logits),
            c_idxs=c_idxs,
            orig_logits=orig_logits,
            sae_logits=sae_logits,
        )

    if save_cache:
        torch.save(acts.model_dump(), cache_file)

    return acts


def norm_scatterplot(
    acts: Acts,
    xlim: tuple[float | None, float | None] = (0, None),
    ylim: tuple[float | None, float | None] = (0, None),
    inset_extent: int = 150,
    out_file: Path | None = None,
    inset_pos: tuple[float, float, float, float] = (0.2, 0.2, 0.6, 0.7),
    figsize: tuple[float, float] = (6, 5),
    main_plot_diag_line: bool = False,
    scatter_alphas: tuple[float, float] = (0.3, 0.1),
):
    orig_norm = torch.norm(acts.orig.flatten(0, 1), dim=-1)
    recon_norms = torch.norm(acts.recon.flatten(0, 1), dim=-1)

    plt.subplots(figsize=figsize)
    ax = plt.gca()
    ax.scatter(orig_norm, recon_norms, alpha=scatter_alphas[0], s=3, c="k")
    ax.set_xlim(xlim)  # type: ignore
    ax.set_ylim(ylim)  # type: ignore
    ax.set_aspect("equal")
    if xlim[1] is not None:
        ax.set_xticks(range(0, xlim[1] + 1, 1000))  # type: ignore[reportCallIssue]
    if ylim[1] is not None:
        ax.set_yticks(range(0, ylim[1] + 1, 1000))  # type: ignore[reportCallIssue]

    axins = plt.gca().inset_axes(  # type: ignore
        inset_pos,
        xlim=(0, inset_extent),
        ylim=(0, inset_extent),
        xticks=[0, inset_extent],
        yticks=[0, inset_extent],
    )
    axins.scatter(orig_norm, recon_norms, alpha=scatter_alphas[1], s=1, c="k")
    axins.plot([0, inset_extent], [0, inset_extent], "k--", alpha=1, lw=0.8)
    axins.set_aspect("equal")
    if main_plot_diag_line:
        assert xlim[1] is not None
        ax.plot([0, xlim[1]], [0, xlim[1]], "k--", alpha=0.5, lw=0.8)

    ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.xlabel("Norm of Original Acts, $||a(x)||_2$")
    plt.ylabel("Norm of Reconstructed Acts, $||\\hat{a}(x)||_2$")

    if out_file is not None:
        plt.savefig(out_file, bbox_inches="tight")
        plt.savefig(Path(out_file).with_suffix(".svg"))
        logger.info(f"Saved plot to {out_file}")


def get_norm_ratios(acts: Acts) -> dict[str, float]:
    norm_ratios = torch.norm(acts.recon, dim=-1) / torch.norm(acts.orig, dim=-1)
    return {"pos0": norm_ratios[:, 0].mean().item(), "pos_gt_0": norm_ratios[:, 1:].mean().item()}


ActsDict = dict[tuple[int, str], Acts]


def get_acts_from_layer_type(layer: int, run_type: str, n_batches: int = 1):
    run_id = SIMILAR_CE_RUNS[layer][run_type]
    run = wandb.Api().run(f"sparsify/gpt2/{run_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return get_acts(run, batch_size=5, batches=n_batches, device=device)


def cosine_sim_plot(acts_dict: ActsDict, run_types: list[str], out_file: Path | None = None):
    colors = [STYLE_MAP[run_type]["color"] for run_type in run_types]

    def get_sims(acts: Acts):
        orig = acts.orig.flatten(0, 1)
        recon = acts.recon.flatten(0, 1)
        return F.cosine_similarity(orig, recon, dim=-1)

    fig = plt.figure(figsize=(8, 4), layout="constrained")
    subfigs = fig.subfigures(1, 3, wspace=0.05)

    for subfig, layer in zip(subfigs, [2, 6, 10], strict=True):
        sims = [get_sims(acts_dict[layer, run_type]) for run_type in run_types]
        create_subplot_hists(
            sim_list=sims,
            titles=[STYLE_MAP[run_type]["label"] for run_type in run_types],
            colors=colors,
            fig=subfig,
            suptitle=f"Layer {layer}",
        )
        # subfigs[i].suptitle(f"Layer {layer_num}")
    fig.suptitle("Input-Output Similarities", fontweight="bold")
    if out_file is not None:
        plt.savefig(out_file)
        plt.savefig(out_file.with_suffix(".svg"))
        logger.info(f"Saved plot to {out_file}")


def pca(x: Float[torch.Tensor, "n emb"], n_dims: int | None) -> Float[torch.Tensor, "emb emb"]:
    x = x - x.mean(0)
    cov_matrix = torch.cov(x.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues, eigenvectors = eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]

    explained_var = eigenvalues[:n_dims].sum() / eigenvalues.sum()
    print(f"Explaining {explained_var.item():.2%} of variance")

    return eigenvectors[:, :n_dims]


def pos_dir_plot(acts: Acts, out_file: Path | None = None):
    pca_dirs = pca(acts.orig.flatten(0, 1), n_dims=None).T

    seqpos_arr = torch.arange(acts.orig.shape[1]).expand((len(acts), -1))

    fig, axs = plt.subplots(3, 1, figsize=(6, 4), sharex=True)
    axs = np.atleast_1d(axs)  # type: ignore
    for dir_idx, ax in zip(range(1, 4), axs, strict=True):
        ax.plot(
            seqpos_arr.flatten(),
            acts.orig.flatten(0, 1) @ pca_dirs[dir_idx],
            ".k",
            ms=1,
            alpha=0.02,
            label="orig",
        )
        ax.plot(
            seqpos_arr.flatten(),
            acts.recon.flatten(0, 1) @ pca_dirs[dir_idx],
            ".r",
            ms=1,
            alpha=0.02,
            label="recon",
        )
        ax.set_ylabel(f"PCA dir {dir_idx}")
        ax.set_yticks([])

    plt.xlabel("Seqence Position")

    leg = axs[0].legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)
        lh.set_markersize(5)

    plt.tight_layout()
    if out_file is not None:
        plt.savefig(out_file)
        plt.savefig(out_file.with_suffix(".svg"))
        logger.info(f"Saved plot to {out_file}")


def create_latex_table(data: dict[int, dict[str, dict[str, float]]]):
    """Formats norms into the appropriate latex table body"""
    body = ""
    for sae_type in ["local", "e2e", "downstream"]:
        row = [sae_type]
        for pos in ["pos0", "pos_gt_0"]:
            row.extend([f"{data[layer][sae_type][pos]:.2f}" for layer in [2, 6, 10]])
        body += " & ".join(row) + " \\\\\n"
    return body


if __name__ == "__main__":
    run_types = list(SIMILAR_CE_RUNS[6].keys())
    acts_dict: ActsDict = {
        (layer, run_type): get_acts_from_layer_type(layer, run_type, n_batches=20)
        for layer in SIMILAR_CE_RUNS
        for run_type in run_types
    }

    acts_6_e2e = acts_dict[6, "e2e"]
    assert acts_6_e2e is not None
    # norm_scatterplot(
    #     acts_6_e2e, xlim=(0, 3200), ylim=(0, 2000), out_file=OUT_DIR / "norm_scatter.png"
    # )

    norm_scatterplot(
        acts_dict[6, "downstream"],
        xlim=(0, 3200),
        ylim=(0, 3200),
        figsize=(5, 5),
        main_plot_diag_line=True,
        inset_pos=(0.2, 0.34, 0.6, 0.6),
        inset_extent=150,
        scatter_alphas=(0.1, 0.05),
        out_file=OUT_DIR / "norm_scatter_downstream.png",
    )

    norms = {
        layer: {run_type: get_norm_ratios(acts_dict[(layer, run_type)]) for run_type in run_types}
        for layer in [2, 6, 10]
    }

    # Generate LaTeX table
    print(create_latex_table(norms))

    with open(OUT_DIR / "norm_ratios.json", "w") as f:
        json.dump(norms, f)

    cosine_sim_plot(acts_dict, run_types, out_file=OUT_DIR / "cosine_similarity.png")

    pos_dir_plot(acts_dict[6, "e2e"], out_file=OUT_DIR / "pos_dir_e2e_l6.png")
    pos_dir_plot(acts_dict[6, "downstream"], out_file=OUT_DIR / "pos_dir_downstream_l6.png")
    pos_dir_plot(acts_dict[10, "e2e"], out_file=OUT_DIR / "pos_dir_e2e_l10.png")
    pos_dir_plot(acts_dict[10, "downstream"], out_file=OUT_DIR / "pos_dir_downstream_l10.png")
