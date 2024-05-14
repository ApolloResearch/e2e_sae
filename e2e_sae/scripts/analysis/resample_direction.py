from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import tqdm
import transformer_lens as tl
import wandb
from jaxtyping import Float
from transformer_lens import HookedTransformer

from e2e_sae.data import DatasetConfig, create_data_loader
from e2e_sae.scripts.analysis.activation_analysis import get_acts, kl_div, pca
from e2e_sae.scripts.analysis.plot_settings import SIMILAR_CE_RUNS

ActTensor = Float[torch.Tensor, "batch seq hidden"]
LogitTensor = Float[torch.Tensor, "batch seq vocab"]
DirTensor = Float[torch.Tensor, "hidden"]

# %%

api = wandb.Api()
local_run_id = SIMILAR_CE_RUNS[10]["local"]
run = api.run(f"sparsify/gpt2/{local_run_id}")
acts = get_acts(run)
pca_dirs = pca(acts.orig.flatten(0, 1), n_dims=None).T


# %%

dataset_config = DatasetConfig(
    dataset_name="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
    is_tokenized=True,
    tokenizer_name="gpt2",
    streaming=True,
    split="train",
    n_ctx=1024,
    seed=100,
    column_name="input_ids",
)

# %%
gpt2 = HookedTransformer.from_pretrained("gpt2")
hook_point_name = "blocks.10.hook_resid_pre"

# %%

eot_id = 50256


data_loader, _ = create_data_loader(dataset_config, batch_size=30)
device = "cuda" if torch.cuda.is_available() else "cpu"
batch = next(iter(data_loader))["input_ids"].to(device)


def shuffle_tensor(x: torch.Tensor) -> torch.Tensor:
    return x[torch.randperm(x.shape[0])]


def apply_fn_to_mask(
    x: torch.Tensor, mask: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    x_out = x[:]
    x_out[mask] = fn(x[mask])
    return x_out


def resample_direction_hook(
    x: ActTensor, hook: tl.hook_points.HookPoint, dir: DirTensor
) -> ActTensor:
    seqpos_arr = torch.arange(x.shape[1]).repeat([len(x), 1]).to(device)
    mask = seqpos_arr > 0
    # mask = (seqpos_arr > 0) & (batch != eot_id)

    x_within_dir = (x @ dir).unsqueeze(-1) * dir
    resid = x - x_within_dir
    shuffled_x_within_dir = apply_fn_to_mask(x_within_dir, mask, shuffle_tensor)
    return resid + shuffled_x_within_dir


@torch.no_grad()
def get_kl_diff_permuting_dir(dir: DirTensor):
    orig_logits = gpt2(batch)
    partial_hook = partial(resample_direction_hook, dir=dir.to(device))
    hooked_logits = gpt2.run_with_hooks(batch, fwd_hooks=[(hook_point_name, partial_hook)])

    return kl_div(orig_logits, hooked_logits).mean().item()


xs = range(50)
kls = [get_kl_diff_permuting_dir(dir=pca_dirs[x]) for x in tqdm.tqdm(xs)]

# %%
xlim = 25
plt.plot(xs[:xlim], kls[:xlim], "o-k")
plt.ylim(0, None)
plt.xlabel("PCA direction")
plt.ylabel("KL divergence")
# plt.title("How much does permuting activations mess up the model?")
plt.gcf().set_size_inches(4, 3)

scripts_dir = Path(__file__).parent
out_dir = scripts_dir / "out/pca_dir_0"

plt.savefig(out_dir / "resample_sensitivity.png", dpi=300, bbox_inches="tight")
plt.savefig(out_dir / "resample_sensitivity.svg", bbox_inches="tight")

# %%
