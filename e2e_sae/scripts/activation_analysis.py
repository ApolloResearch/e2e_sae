import json
from pathlib import Path
from typing import cast

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
from e2e_sae.scripts.geometric_analysis import COLOR_MAP, CONSTANT_CE_RUNS, create_subplot_hists

ActTensor = Float[torch.Tensor, "batch seq hidden"]
LogitTensor = Float[torch.Tensor, "batch seq vocab"]


def kl_div(new_logits: LogitTensor, old_logits: LogitTensor) -> Float[torch.Tensor, "batch seq"]:
    return F.kl_div(
        F.log_softmax(new_logits, dim=-1), F.softmax(old_logits, dim=-1), reduction="none"
    ).sum(-1)


class Acts(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokens: Int[torch.Tensor, "batch seq"] = torch.empty(0, 1024, dtype=torch.long)
    orig: ActTensor = torch.empty(0, 1024, 768)
    recon: ActTensor = torch.empty(0, 1024, 768)
    orig_logits: LogitTensor = torch.empty(0, 1024, 50257)
    new_logits: LogitTensor = torch.empty(0, 1024, 50257)
    kl: Float[torch.Tensor, "batch seq"] = torch.empty(0, 1024)

    def add(
        self, tokens: torch.Tensor, acts: SAEActs, orig_logits: LogitTensor, new_logits: LogitTensor
    ):
        self.tokens = torch.cat([self.tokens, tokens.cpu()])
        self.orig = torch.cat([self.orig, acts.input.cpu()])
        self.recon = torch.cat([self.recon, acts.output.cpu()])
        self.orig_logits = torch.cat([self.orig_logits, orig_logits.cpu()])
        self.new_logits = torch.cat([self.new_logits, new_logits.cpu()])
        self.kl = torch.cat([self.kl, kl_div(new_logits, orig_logits).cpu()])

    def __len__(self):
        return len(self.tokens)

    def __str__(self) -> str:
        return f"Acts(len={len(self)})"


@torch.no_grad()
def get_acts(run: Run, batch_size=5, batches=1, device: str = "cuda") -> Acts:
    model = SAETransformer.from_wandb(f"{run.project}/{run.id}")
    model.to(device)
    data_config = DatasetConfig(**run.config["eval_data"])
    loader, _ = create_data_loader(data_config, batch_size=batch_size)
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
        acts.add(tokens, sae_acts, orig_logits=orig_logits, new_logits=sae_logits)

    return acts


def norm_scatterplot(
    acts: Acts,
    xlim: tuple[int | None, int | None] = (0, None),
    ylim: tuple[int | None, int | None] = (0, None),
    inset_extent: int = 150,
    out_file: Path | None = None,
    inset_pos: tuple[float, float, float, float] = (0.2, 0.2, 0.6, 0.7),
    figsize: tuple[float, float] = (6, 5),
):
    orig_norm = torch.norm(acts.orig.flatten(0, 1), dim=-1)
    recon_norms = torch.norm(acts.recon.flatten(0, 1), dim=-1)

    plt.subplots(figsize=figsize)
    ax = cast(plt.Axes, plt.gca())
    ax.scatter(orig_norm, recon_norms, alpha=0.3, s=3, c="k")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    if xlim[1] is not None:
        ax.set_xticks(range(0, xlim[1] + 1, 1000))  # type: ignore[reportCallIssue]
    if ylim[1] is not None:
        ax.set_yticks(range(0, ylim[1] + 1, 1000))  # type: ignore[reportCallIssue]

    inset_extent = 150
    axins = plt.gca().inset_axes(  # type: ignore
        inset_pos,
        xlim=(0, inset_extent),
        ylim=(0, inset_extent),
        xticks=[0, inset_extent],
        yticks=[0, inset_extent],
    )
    axins.scatter(orig_norm, recon_norms, alpha=0.1, s=1, c="k")
    axins.plot([0, inset_extent], [0, inset_extent], "k--", alpha=1, lw=0.8)
    axins.set_aspect("equal")

    ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.xlabel("Norm of Original Acts, $||a(x)||_2$")
    plt.ylabel("Norm of Reconstructed Acts, $||\\hat{a}(x)||_2$")

    if out_file is not None:
        plt.savefig(out_file, bbox_inches="tight")
        plt.savefig(Path(out_file).with_suffix(".svg"))


def get_norm_ratios(acts: Acts) -> tuple[float, float]:
    norm_ratios = torch.norm(acts.recon, dim=-1) / torch.norm(acts.orig, dim=-1)
    return norm_ratios[:, 1:].mean().item(), norm_ratios[:, 0].mean().item()


ActsDict = dict[tuple[int, str], Acts]


def get_acts_from_layer_type(layer: int, run_type: str, n_batches: int = 1):
    run_id = CONSTANT_CE_RUNS[layer][run_type]
    run = wandb.Api().run(f"sparsify/gpt2/{run_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return get_acts(run, batch_size=5, batches=n_batches, device=device)


def cosine_sim_plot(acts_dict: ActsDict, out_file: Path | None = None):
    def get_sims(acts: Acts):
        orig = acts.orig.flatten(0, 1)
        recon = acts.recon.flatten(0, 1)
        return F.cosine_similarity(orig, recon, dim=-1)

    fig = plt.figure(figsize=(8, 4), layout="constrained")
    subfigs = fig.subfigures(1, 3, wspace=0.05)

    for subfig, layer in zip(subfigs, [2, 6, 10], strict=True):
        sims = [get_sims(acts_dict[layer, run_type]) for run_type in ["local", "e2e", "downstream"]]
        create_subplot_hists(
            sim_list=sims,
            titles=["Local", "End-to-end", "E2e + downstream"],
            colors=list(COLOR_MAP.values()),
            fig=subfig,
            suptitle=f"Layer {layer}",
        )
        # subfigs[i].suptitle(f"Layer {layer_num}")
    fig.suptitle("Input-Output Similarities", fontweight="bold")
    if out_file is not None:
        plt.savefig(out_file)
        plt.savefig(out_file.with_suffix(".svg"))
        logger.info(f"Saved plot to {out_file}")


def create_latex_table(data: dict[int, dict[str, tuple[float, float]]]):
    """Formats norms into the appropriate latex table body"""
    layers = ["local", "e2e", "downstream"]
    body = ""
    for layer in layers:
        row = [layer]
        for pos in range(2):
            for key in sorted(data.keys(), key=int):
                row.append(f"{data[key][layer][pos]:.3f}")
        body += " & ".join(row) + " \\\\\n"
    return body


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "out" / "activation_analysis"

    acts_dict: ActsDict = {
        (layer, run_type): get_acts_from_layer_type(layer, run_type, n_batches=10)
        for layer in CONSTANT_CE_RUNS
        for run_type in CONSTANT_CE_RUNS[layer]
    }

    acts_6_e2e = acts_dict[6, "e2e"]
    assert acts_6_e2e is not None
    norm_scatterplot(
        acts_6_e2e, xlim=(0, 3200), ylim=(0, 2000), out_file=out_dir / "norm_scatter.png"
    )

    norms = {
        layer: {
            run_type: get_norm_ratios(acts_dict[(layer, run_type)])
            for run_type in ["local", "e2e", "downstream"]
        }
        for layer in [2, 6, 10]
    }

    print(norms)

    # Generate LaTeX table
    print(create_latex_table(norms))

    with open(out_dir / "norm_ratios.json", "w") as f:
        json.dump(norms, f)

    cosine_sim_plot(acts_dict, out_file=out_dir / "cosine_similarity.png")
