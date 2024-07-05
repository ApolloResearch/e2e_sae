"""Analysis of PCA dir 0 in layer 10"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from einops import repeat
from jaxtyping import Float
from scipy.stats import pearsonr, spearmanr
from torch.nn.functional import cosine_similarity, normalize

from e2e_sae.log import logger
from e2e_sae.scripts.analysis.activation_analysis import Acts, get_acts, pca
from e2e_sae.scripts.analysis.geometric_analysis import (
    EmbedInfo,
    get_alive_dict_elements,
)
from e2e_sae.scripts.analysis.plot_settings import SIMILAR_CE_RUNS, STYLE_MAP

# %%
local_run_id = SIMILAR_CE_RUNS[10]["local"]
downstream_run_id = SIMILAR_CE_RUNS[10]["downstream"]

analysis_dir = Path(__file__).parent
umap_data_dir = Path(analysis_dir / "out/umap")
out_dir = analysis_dir / "out/pca_dir_0"

umap_file = umap_data_dir / "constant_CE/downstream_local_umap_blocks.10.hook_resid_pre.pt"
umap_info = EmbedInfo(**torch.load(umap_file))

api = wandb.Api()

local_acts = get_acts(api.run(f"sparsify/gpt2/{local_run_id}"))
local_dictionary = get_alive_dict_elements(api, "gpt2", local_run_id)
local_embeds = umap_info.embedding[umap_info.alive_elements_per_dict[0] :, :]

downstream_acts = get_acts(api.run(f"sparsify/gpt2/{downstream_run_id}"))
downstream_dictionary = get_alive_dict_elements(api, "gpt2", downstream_run_id)
downstream_embeds = umap_info.embedding[: umap_info.alive_elements_per_dict[0]]

pca_dirs = pca(local_acts.orig.flatten(0, 1), n_dims=None).T

outlier_pos_0_dir = normalize(local_acts.orig[:, 0, :].mean(0), p=2, dim=0)
print(cosine_similarity(pca_dirs[0], outlier_pos_0_dir).item())

# %%
######## UMAP IN DIR PLOT ########


def umaps_in_dir(
    dir: Float[torch.Tensor, "emb"],  # noqa: F821
    vabs: float | None = None,
    outfile: Path | None = None,
):
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5), layout="constrained")
    axs = np.atleast_1d(axs)  # type: ignore
    local_sims = cosine_similarity(local_dictionary.alive_dict_elements.T, dir, dim=-1)
    downstream_sims = cosine_similarity(downstream_dictionary.alive_dict_elements.T, dir, dim=-1)

    vabs = vabs or max(local_sims.abs().max().item(), downstream_sims.abs().max().item())

    dir = pca_dirs[1]
    kwargs = {"s": 1, "vmin": -vabs, "vmax": vabs, "alpha": 0.3, "cmap": "coolwarm_r"}
    axs[0].scatter(local_embeds[:, 0], local_embeds[:, 1], c=local_sims, **kwargs)
    mappable = axs[1].scatter(
        downstream_embeds[:, 0], downstream_embeds[:, 1], c=downstream_sims, **kwargs
    )
    cbar = plt.colorbar(mappable=mappable, label="Similarity to 0th PCA direction", shrink=0.8)
    cbar.solids.set(alpha=1)  # type: ignore[reportOptionalMemberAccess]

    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    axs[0].set_title("Local SAE in layer 10")
    axs[1].set_title("Downstream SAE in layer 10")

    axs[0].set_box_aspect(1)
    axs[1].set_box_aspect(1)

    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.savefig(outfile.with_suffix(".svg"), bbox_inches="tight")
        logger.info(f"Saved UMAP plot to {outfile}")


umaps_in_dir(pca_dirs[0], outfile=out_dir / "umap_pos0_dir.png", vabs=0.5)

# %%
######## ACTIVATION HISTOGRAM ########

fig, axs = plt.subplots(1, 2, figsize=(7, 2), sharey=True)
axs = np.atleast_1d(axs)  # type: ignore
fig.subplots_adjust(wspace=0.05)  # adjust space between axes

acts_in_dir = local_acts.orig @ pca_dirs[0]
eot_token = local_acts.tokens.max()
eot_mask = local_acts.tokens == eot_token
n_batch, n_seq = local_acts.tokens.shape
pos0_mask = repeat(torch.arange(n_seq), "seq -> batch seq", batch=n_batch) == 0

ax0_xlim = (-20, 220)
ax1_xlim = (2980, 3220)

colors = plt.get_cmap("tab10").colors  # type: ignore[reportAttributeAccessIssue]

axs[1].hist(
    acts_in_dir[pos0_mask],
    label="position 0",
    density=True,
    bins=40,
    range=ax1_xlim,
    color=colors[4],
)
axs[0].hist(
    acts_in_dir[eot_mask & ~pos0_mask],
    label="end-of-text",
    density=True,
    bins=40,
    range=ax0_xlim,
    color=colors[6],
)
axs[0].hist(
    acts_in_dir[~eot_mask & ~pos0_mask],
    label="other",
    density=True,
    bins=40,
    range=ax0_xlim,
    color=colors[5],
)

# hide the spines between ax and ax2
axs[0].spines.right.set_visible(False)
axs[1].spines.left.set_visible(False)
axs[1].set_yticks([])

axs[0].set_xlim(*ax0_xlim)
axs[1].set_xlim(*ax1_xlim)

# axis break symbols
d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(
    marker=[(-d, -1), (d, 1)],
    markersize=12,
    linestyle="none",
    color="k",
    mec="k",
    mew=1,
    clip_on=False,
)
axs[0].plot([1, 1], [0, 1], transform=axs[0].transAxes, **kwargs)
axs[1].plot([0, 0], [0, 1], transform=axs[1].transAxes, **kwargs)

# legend
lines = axs[1].get_legend_handles_labels()[0] + axs[0].get_legend_handles_labels()[0]
labels = axs[1].get_legend_handles_labels()[1] + axs[0].get_legend_handles_labels()[1]
axs[1].legend(lines, labels)

axs[1].set_xlabel("Activation in PCA direction 0", x=-0.025, horizontalalignment="center")

plt.savefig(out_dir / "activation_hist.png", dpi=300, bbox_inches="tight")
plt.savefig(out_dir / "activation_hist.svg", bbox_inches="tight")

# %%
######## INPUT-OUTPUT CORRELATION IN PCA DIR 0 ########


def corr_in_dir(
    acts: Acts, direction: torch.Tensor, spearman: bool = False, include_pos0: bool = True
) -> float:
    pos_idx = 0 if include_pos0 else 1
    orig_in_d = (acts.orig[:, pos_idx:] @ direction).flatten()
    recon_in_d = (acts.recon[:, pos_idx:] @ direction).flatten()
    if spearman:
        return spearmanr(orig_in_d, recon_in_d).statistic  # type: ignore[reportAttributeAccessIssue]
    else:
        return pearsonr(orig_in_d, recon_in_d).statistic  # type: ignore[reportAttributeAccessIssue]


print(
    "Local SAE, corr in 0th pca:",
    f"{corr_in_dir(local_acts, pca_dirs[0], include_pos0=False):.3f}",
)
print(
    "Downstream SAE, corr in 0th pca:",
    f"{corr_in_dir(downstream_acts, pca_dirs[0], include_pos0=False):.3f}",
)

print(
    "Downstream SAE, corr in 0th pca at position 0",
    pearsonr(
        (downstream_acts.orig[:, 0] @ pca_dirs[0]).flatten(),
        (downstream_acts.recon[:, 0] @ pca_dirs[0]).flatten(),
    ).statistic,  # type: ignore[reportAttributeAccessIssue]
)

# %%
######## INPUT-OUTPUT CORRELATION IN PCA DIRS PLOT ########

xs = range(25)
corrs = {
    "local": [corr_in_dir(local_acts, pca_dirs[i], include_pos0=False) for i in range(50)],
    "downstream": [
        corr_in_dir(downstream_acts, pca_dirs[i], include_pos0=False) for i in range(50)
    ],
}

plt.plot(xs, corrs["local"], **STYLE_MAP["local"])  # type: ignore[reportArgumentType]
plt.plot(xs, corrs["downstream"], **STYLE_MAP["downstream"])  # type: ignore[reportArgumentType]
plt.ylabel("input-output correlation")
plt.xlabel("PCA direction")
plt.legend(loc="lower right", title="SAE type")
plt.ylim(0, 1)
plt.gcf().set_size_inches(4, 3)
plt.xlim(-1, None)
plt.savefig(out_dir / "input_output_corr.png", dpi=300, bbox_inches="tight")
plt.savefig(out_dir / "input_output_corr.svg", bbox_inches="tight")
