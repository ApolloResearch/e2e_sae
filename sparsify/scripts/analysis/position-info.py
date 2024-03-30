# %%
import matplotlib.pyplot as plt
import torch
from jaxtyping import Float

from sparsify.scripts.analysis.get_acts import get_acts

acts = get_acts(batches=15)
# %%


def pca(x: Float[torch.Tensor, "n emb"], n_dims: int) -> Float[torch.Tensor, "emb emb"]:
    x = x - x.mean(0)
    cov_matrix = torch.cov(x.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues, eigenvectors = eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]

    explained_var = eigenvalues[:n_dims].sum() / eigenvalues.sum()
    print(f"Explaining {explained_var.item():.2%} of variance")

    return eigenvectors[:, :n_dims]


mean_by_pos = acts.orig.mean(0)
pos_dirs = pca(mean_by_pos, n_dims=5).T

# mean_by_pos_pca = mean_by_pos @ pca_dirs

# %%
seqpos_arr = torch.arange(acts.orig.shape[1]).expand((len(acts), -1))

fig, axs = plt.subplots(5, 1, figsize=(6, 6), sharex=True)
for i, ax in enumerate(axs):
    ax.plot(
        seqpos_arr.flatten(),
        (acts.orig @ pos_dirs[i]).flatten(),
        ".k",
        ms=1,
        alpha=0.1,
        label="orig",
    )
    ax.plot(
        seqpos_arr.flatten(),
        (acts.recon @ pos_dirs[i]).flatten(),
        ".r",
        ms=1,
        alpha=0.1,
        label="recon",
    )
    ax.set_ylabel(f"Pos dir {i}")
    ax.set_yticks([])

plt.xlabel("Seqence Position")

leg = axs[0].legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)
    lh.set_markersize(5)

plt.suptitle("Recon vs Original activations in GPT2 position directions")
plt.tight_layout()

# %%
i = 2
plt.scatter(acts.orig @ pos_dirs[i], acts.recon @ pos_dirs[i], s=2, c=seqpos_arr, alpha=1)
plt.colorbar(label="Sequence Position")


def add_xy_line(ax: Optional[plt.Axes] = None, style="k-"):
    ax = ax or plt.gca()
    xlims, ylims = ax.get_xlim(), ax.get_ylim()
    min_ext = min(xlims[0], ylims[0])
    max_ext = max(xlims[1], ylims[1])
    ax.plot([min_ext, max_ext], [min_ext, max_ext], style)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)


add_xy_line()
plt.xlabel(f"Original activation in PCA direction {i}")
plt.ylabel(f"Reconstructed activation in PCA direction {i}")
