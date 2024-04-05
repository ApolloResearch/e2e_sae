# %%
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from jaxtyping import Float

from sparsify.scripts.analysis.get_acts import ActTensor, get_acts, load_sae_from_path

# %%
plts_out_dir = Path("/mnt/ssd-interp/nix/sparsify/sparsify/scripts/analysis/out")
sae_out_dir = Path("/mnt/ssd-interp/nix/sparsify/sparsify/scripts/train_tlens_saes/out")

# e2e SAE at layer 6
sae_name = "seed-0_lpcoeff-1.5_logits-kl-1.0_lr-0.0005_ratio-60.0_blocks.6.hook_resid_pre"

# layerwise SAE at layer 6
# sae_name = "seed-0_lpcoeff-6.0_in-to-out-1.0_lr-0.0005_ratio-60.0_blocks.6.hook_resid_pre"

model, config = load_sae_from_path(sae_out_dir / sae_name / "samples_400000.pt")
model.to("cuda")
acts = get_acts(model, config, batch_size=5, batches=10)
# %%


def ln(x: ActTensor):
    return torch.nn.functional.layer_norm(x, [768])


def sq_err(orig: ActTensor, recon: ActTensor):
    return (orig - recon).pow(2).sum(dim=-1)


def exp_var(orig: ActTensor, recon: ActTensor):
    total_variance = (orig - orig.mean(0)).pow(2).sum(-1)
    return (1 - sq_err(orig, recon) / total_variance).mean()


# %%
##### FIGURE 1: Norm of activations

orig_norm = torch.norm(acts.orig.flatten(0, 1), dim=-1)
recon_norms = torch.norm(acts.recon.flatten(0, 1), dim=-1)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(orig_norm, recon_norms, alpha=0.3, s=3, c="k")
ax.set_ylim(0, 2000)
ax.set_xlim(0, None)
ax.set_aspect("equal")
ax.set_xticks([0, 1000, 2000, 3000])
ax.set_yticks([0, 1000, 2000])

inset_extent = 150
axins = plt.gca().inset_axes(
    (0.2, 0.2, 0.6, 0.7),
    xlim=(0, inset_extent),
    ylim=(0, inset_extent),
    xticks=[0, inset_extent],
    yticks=[0, inset_extent],
)
axins.scatter(orig_norm, recon_norms, alpha=0.1, s=1, c="k")
axins.plot([0, inset_extent], [0, inset_extent], "k--", alpha=1, lw=0.8)
axins.set_aspect("equal")

ax.indicate_inset_zoom(axins, edgecolor="black")

plt.xlabel("Norm of Original Acts")
plt.ylabel("Norm of Reconstructed Acts")

plt.savefig(plts_out_dir / "orig-vs-recon-norms.png", dpi=300, bbox_inches="tight")

# %%
##### FIGURE 2: Cos Sim of activations

sims = torch.nn.functional.cosine_similarity(ln(acts.orig), ln(acts.recon), dim=-1)
plt.hist(sims.cpu().flatten(), bins=200, density=True)
plt.title("Cosine Similarity of Original and Reconstructed Acts")
plt.xlim(0, 1)
plt.yticks([])
plt.gcf().set_size_inches(6, 2.5)


# %%
##### FIGURE 3: Reconstruction by PCA direction


def pca(x: Float[torch.Tensor, "n emb"], n_dims: int | None) -> Float[torch.Tensor, "emb emb"]:
    x = x - x.mean(0)
    cov_matrix = torch.cov(x.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues, eigenvectors = eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]

    explained_var = eigenvalues[:n_dims].sum() / eigenvalues.sum()
    print(f"Explaining {explained_var.item():.2%} of variance")

    return eigenvectors[:, :n_dims]


def flat(x: ActTensor) -> Float[torch.Tensor, "n emb"]:
    return x.flatten(0, 1)


pca_dirs = pca(flat(acts.orig), n_dims=None).T

corrs = []
for i in range(768):
    o_acts_in_dir = flat(acts.orig) @ pca_dirs[i]
    r_acts_in_dir = flat(acts.recon) @ pca_dirs[i]
    corrs.append(torch.corrcoef(torch.stack([o_acts_in_dir, r_acts_in_dir]))[0, 1])


# %%
plt.plot(corrs, ".k", ms=2)
plt.xlabel("PCA Direction")
# plt.ylabel("r")
plt.gcf().set_size_inches(6, 4)
plt.ylim(None, 1)
plt.title("Correlation of original and reconstructed activations")
plt.tight_layout()

# %%
##### FIGURE 4: Position directions

seqpos_arr = torch.arange(acts.orig.shape[1]).expand((len(acts), -1))

fig, axs = plt.subplots(3, 1, figsize=(6, 4), sharex=True)
for dir_idx, ax in zip(range(1, 4), axs, strict=True):
    ax.plot(
        seqpos_arr.flatten(),
        flat(acts.orig) @ pca_dirs[dir_idx],
        ".k",
        ms=1,
        alpha=0.02,
        label="orig",
    )
    ax.plot(
        seqpos_arr.flatten(),
        flat(acts.recon) @ pca_dirs[dir_idx],
        ".r",
        ms=1,
        alpha=0.02,
        label="recon",
    )
    ax.set_ylabel(f"PCA dir {dir_idx}")
    ax.set_yticks([])

plt.xlabel("Seqence Position")

leg = axs[0].legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)
    lh.set_markersize(5)

plt.tight_layout()


# %%
####### MISC EXPERIMENTS, e.g. with projecting out certain residual stream directions ######


def summary(orig: ActTensor, recon: ActTensor):
    print("Mean norm:")
    print(f"  Orig: {torch.norm(orig.flatten(0, 1), dim=-1).mean().item():.2f}")
    print(f"  Recon: {torch.norm(recon.flatten(0, 1), dim=-1).mean().item():.2f}")
    print(f"Explained variance: {exp_var(orig, recon).item():.2f}")
    print(f"Sq err: {sq_err(orig, recon).mean().item():.2f}")


print("Unaltered")
summary(acts.orig, acts.recon)

print("\nLayerNormed")
summary(ln(acts.orig), ln(acts.recon))


# %%

ln_pca_dirs = pca(ln(acts.orig).flatten(0, 1), n_dims=999).T

P = (
    ln_pca_dirs.T
    @ torch.where(
        ~torch.isin(torch.arange(768), torch.tensor([1, 2])) & (torch.arange(768) < 50), 1.0, 0.0
    ).diag()
    # @ torch.where(~torch.isin(torch.arange(768), torch.tensor([1, 2])), 1.0, 0.0).diag()
    @ ln_pca_dirs
)

print("Projected LayerNormed")
summary(ln(acts.orig) @ P, ln(acts.recon) @ P)


# %%


sims = torch.nn.functional.cosine_similarity(ln(acts.orig) @ P, ln(acts.recon) @ P, dim=-1)
plt.hist(sims.cpu().flatten(), bins=200, density=True)
plt.title("Cosine Similarity of Original and Reconstructed Acts")
plt.xlim(0, 1)
plt.yticks([])
plt.gcf().set_size_inches(6, 2.5)

# %%
x = ln(acts.orig) @ P
(x - x.mean(0)).pow(2).sum(-1)


# %%
def var(x: ActTensor):
    return (x - x.mean(0)).pow(2).sum(-1)


# var(acts.orig).mean(), var(ln(acts.orig)).mean(), var(ln(acts.orig) @ P).mean()

# %%
corrs = []
for i in range(768):
    o_acts_in_dir = ln(acts.orig) @ ln_pca_dirs[i]
    r_acts_in_dir = ln(acts.recon) @ ln_pca_dirs[i]
    corrs.append(
        torch.corrcoef(torch.stack([o_acts_in_dir.flatten(), r_acts_in_dir.flatten()]))[0, 1]
    )
# %%
plt.plot(corrs, ".")
plt.xlim(0, 15)
# %%

dir_i = 12
plt.scatter(
    ln(acts.orig) @ ln_pca_dirs[dir_i],
    ln(acts.recon) @ ln_pca_dirs[dir_i],
    s=2,
    c="k",
    alpha=0.2,
)

o_acts_in_dir = ln(acts.orig) @ ln_pca_dirs[dir_i]
r_acts_in_dir = ln(acts.recon) @ ln_pca_dirs[dir_i]
torch.corrcoef(torch.stack([o_acts_in_dir.flatten(), r_acts_in_dir.flatten()]))

# %%

pca_dirs = pca(acts.orig.flatten(0, 1), n_dims=999).T

# %%
dir_i = -3
plt.scatter(
    ln(acts.orig) @ pca_dirs[dir_i],
    ln(acts.recon) @ pca_dirs[dir_i],
    s=2,
    c="k",
    alpha=0.2,
)

o_acts_in_dir = ln(acts.orig) @ pca_dirs[dir_i]
r_acts_in_dir = ln(acts.recon) @ pca_dirs[dir_i]
torch.corrcoef(torch.stack([o_acts_in_dir.flatten(), r_acts_in_dir.flatten()]))
# %%
corrs = []
for i in range(768):
    o_acts_in_dir = ln(acts.orig) @ ln_pca_dirs[i]
    r_acts_in_dir = ln(acts.recon) @ ln_pca_dirs[i]
    corrs.append(
        torch.corrcoef(torch.stack([o_acts_in_dir.flatten(), r_acts_in_dir.flatten()]))[0, 1]
    )
# %%
plt.plot(corrs, ".")
# plt.xlim(0, 15)
