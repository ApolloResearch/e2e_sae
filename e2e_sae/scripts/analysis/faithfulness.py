# %%
import functools
import json
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Self

import matplotlib.pyplot as plt
import requests
import torch
from fire import Fire
from tqdm import tqdm
from transformer_lens import HookedTransformer

from e2e_sae.models.transformers import SAETransformer
from e2e_sae.scripts.analysis.plot_settings import SIMILAR_CE_RUNS, SIMILAR_L0_RUNS, STYLE_MAP
from e2e_sae.settings import REPO_ROOT

# %%
OUT_DIR = REPO_ROOT / "e2e_sae/scripts/analysis" / "out/faithfulness"
DATA_FOLDER = REPO_ROOT / "e2e_sae/scripts/analysis" / "data"
device = "cuda" if torch.cuda.is_available() else "cpu"


def download_data():
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    for dataset_name in ["rc", "simple", "nounpp", "within_rc"]:
        file_name = f"{dataset_name}_train.json"
        url = f"https://raw.githubusercontent.com/saprmarks/feature-circuits/main/data/{file_name}"
        out_path = DATA_FOLDER / file_name

        if not out_path.exists():
            print(f"Downloading {url} to {out_path}")
            response = requests.get(url)
            response.raise_for_status()
            out_path.write_text(response.text)


def get_tokenizer():
    return HookedTransformer.from_pretrained("gpt2").tokenizer


tokenizer = get_tokenizer()


# %%
class DataPoint(NamedTuple):
    clean_prefix: str
    patch_prefix: str
    clean_answer: str
    patch_answer: str
    # is_plural_clean: bool
    # is_plural_patch: bool

    def from_dict(d: dict[str, str]):
        # is_plural_clean = d["case"].split("_")[0] == "plural"
        # is_plural_patch = d["case"].split("_")[1] == "plural_patch"
        return DataPoint(
            clean_prefix=d["clean_prefix"],
            patch_prefix=d["patch_prefix"],
            clean_answer=d["clean_answer"],
            patch_answer=d["patch_answer"],
            # is_plural_clean=is_plural_clean,
            # is_plural_patch=is_plural_patch,
        )


# %%
templates: dict[str, list[str]] = {
    "rc": ["the_subj", "subj_main", "that", "the_dist", "subj_dist", "verb_dist"],
    "simple": ["the_subj", "subj_main"],
    "nounpp": ["the_subj", "subj_main", "prep", "the_dist", "subj_dist"],
    "within_rc": ["the_subj", "subj_main", "that", "the_dist", "subj_dist"],
}


def batch_slices(length: int, batch_size: int) -> list[slice]:
    return [slice(start, start + batch_size) for start in range(0, length, batch_size)]


def tokenize(texts: str | list[str], add_eot: bool = True) -> torch.Tensor:
    if add_eot:
        if isinstance(texts, str):
            texts = "<|endoftext|>" + texts
        else:
            texts = ["<|endoftext|>" + text for text in texts]
    return tokenizer(texts, return_tensors="pt", padding=True)["input_ids"]


class Dataset:
    def __init__(self, data: list[DataPoint], short_name: str):
        self.data = data
        self.short_name = short_name
        self.tokens = tokenize([d.clean_prefix for d in self.data])
        self.clean_tok_ids = tokenize([d.clean_answer for d in self.data], add_eot=False)[:, 0]
        self.patch_tok_ids = tokenize([d.patch_answer for d in self.data], add_eot=False)[:, 0]
        self.input_lens = (self.tokens != tokenizer.eos_token_id).sum(dim=1) + 1  # +1 from bos

    @property
    def template(self) -> list[str]:
        return templates[self.short_name]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: slice) -> Self:
        assert isinstance(idx, slice), idx
        return self.__class__(self.data[idx], self.short_name)

    @functools.cached_property
    def masks(self) -> dict[str, torch.Tensor]:
        masks = {
            template_word: torch.zeros_like(self.tokens, dtype=torch.bool)
            for template_word in self.template
        }
        for i in range(len(self.tokens)):
            annotations = self._get_annotations(self.data[i].clean_prefix)
            for template_word, (start, end) in annotations.items():
                masks[template_word][i, start : end + 1] = True
        return masks

    def _get_annotations(self, text: str) -> dict[str, tuple[int, int]]:
        # implementation from Sam Marks
        annotations = {}
        curr_token = 1  # because of eot start token
        for template_word, word in zip(self.template, text.split(), strict=True):
            if word != "The":
                word = " " + word
            word_tok = tokenize(word, add_eot=False)
            num_tokens = word_tok.shape[1]
            span = (curr_token, curr_token + num_tokens - 1)
            curr_token += num_tokens
            annotations[template_word] = span

        return annotations

    @classmethod
    def load(cls, short_name: str, split: Literal["train", "test"]):
        path = DATA_FOLDER / f"{short_name}_{split}.json"
        with open(path) as f:
            data = [DataPoint.from_dict(json.loads(line)) for line in f]
        return cls(data, short_name)


# %%
def abl_all_but_mask(idxs: list[int]):
    mask = torch.ones(46080, dtype=torch.bool, device=device)
    if idxs:
        mask[idxs] = False
    return mask


def abl_only_mask(idxs: list[int]):
    mask = torch.zeros(46080, dtype=torch.bool, device=device)
    if idxs:
        mask[idxs] = True
    return mask


@functools.lru_cache
def get_model(run_id: str):
    return SAETransformer.from_wandb(f"gpt2/{run_id}").to(device)


@dataclass
class Experiment:
    run_id: str
    sae_pos: str
    dataset: Dataset
    batch_size: int = 1_000

    def __post_init__(self):
        self.model = get_model(self.run_id)
        self.batch_slices = batch_slices(len(self.dataset), self.batch_size)
        self.sae_acts = self.get_sae_acts()
        self.mean_acts = {
            k: self.sae_acts[m].mean(dim=0).to(device) for k, m in self.dataset.masks.items()
        }
        self.active_saes = (self.sae_acts > 0).any(dim=(0, 1)).nonzero().squeeze().tolist()

    @torch.no_grad()
    def get_sae_acts(self):
        acts_list = []
        for batch_slice in tqdm(batch_slices(len(self.dataset), self.batch_size), desc="SAE acts"):
            batch_tokens = self.dataset.tokens[batch_slice].to(device)
            _, sae_cache = self.model.forward(batch_tokens, [self.sae_pos])
            acts_list.append(sae_cache[self.sae_pos].c.cpu())

        return torch.cat(acts_list, dim=0)

    @torch.no_grad()
    def run_ablation(self, ablation_mask: torch.Tensor):
        """Return logit diff when mean-ablating the SAEs which are true in the ablation_mask"""

        hook_idx_iter = iter(batch_slices(len(self.dataset), self.batch_size))

        def hook_fn(module: Any, input: Any, output: torch.Tensor):
            idxs = next(hook_idx_iter)
            for template_name, means in self.mean_acts.items():
                template_mask = self.dataset.masks[template_name][idxs]
                output[template_mask] = torch.where(
                    ablation_mask[None, :], means[None, :], output[template_mask]
                )
            return output

        sae = self.model.saes[self.sae_pos.replace(".", "-")]
        hook_handle = sae.encoder.register_forward_hook(hook_fn)
        try:
            metrics_list = []
            for batch_slice in tqdm(self.batch_slices, desc="Ablation", disable=True):
                batch_tokens = self.dataset.tokens[batch_slice].to(device)
                logits, _ = self.model.forward(batch_tokens, [self.sae_pos])
                metric = self._get_metric(logits, batch_slice)
                metrics_list.append(metric.cpu())
        finally:
            hook_handle.remove()
        return torch.cat(metrics_list, dim=0)

    def _get_metric(self, logits: torch.Tensor, idxs: slice):
        seq_idxs = self.dataset.input_lens[idxs] - 1
        correct_logits = logits[range(len(logits)), seq_idxs, self.dataset.clean_tok_ids[idxs]]
        patch_logits = logits[range(len(logits)), seq_idxs, self.dataset.patch_tok_ids[idxs]]
        return correct_logits - patch_logits

    def get_orig_model_scores(self) -> torch.Tensor:
        metrics_list = []
        for batch_slice in tqdm(self.batch_slices, desc="Ablation", disable=True):
            batch_tokens = self.dataset.tokens[batch_slice].to(device)
            logits, _ = self.model.forward(batch_tokens, [])
            assert logits is not None
            metric = self._get_metric(logits, batch_slice)
            metrics_list.append(metric.cpu())
        metrics = torch.cat(metrics_list)
        return metrics

    @functools.cached_property
    def orig_model_score(self) -> float:
        return self.get_orig_model_scores().mean().item()

    @functools.cached_property
    def ablate_sae_err_score(self) -> float:
        return self.run_ablation(abl_only_mask([])).mean().item()

    @functools.cached_property
    def ablate_all_score(self) -> float:
        return self.run_ablation(abl_all_but_mask([])).mean().item()

    def abl_one_at_a_time(self) -> dict[int, float]:
        idxs = tqdm(self.active_saes, desc="Ablating one at a time")
        return {idx: self.run_ablation(abl_only_mask([idx])).mean().item() for idx in idxs}

    @functools.cached_property
    def m_one_at_a_time(self) -> dict[int, float]:
        path = OUT_DIR / f"{self.dataset.short_name}_ablate_one_{self.run_id}.json"
        if not path.exists():
            m_one_at_a_time = self.abl_one_at_a_time()
            with open(path, "w") as f:
                info = {
                    "no_ablation": self.orig_model_score,
                    "sae_error": self.ablate_sae_err_score,
                    "all_ablated": self.ablate_all_score,
                    "n_train": len(self.dataset),
                    "m_ablate_one": m_one_at_a_time,
                }
                json.dump(info, f)
            return m_one_at_a_time
        else:
            with open(path) as f:
                info = json.load(f)
            assert self.ablate_all_score == info["all_ablated"]
            assert self.ablate_sae_err_score == info["unablated"]
            return {int(k): v for k, v in info["m_ablate_one"].items()}

    def sorted_active_saes(self) -> list[int]:
        return sorted(
            self.active_saes,
            key=lambda x: abs(self.ablate_sae_err_score - self.m_one_at_a_time[x]),
            reverse=True,
        )

    def get_m_curve(self, xs: list[int], faithful=False) -> list[float]:
        sorted_saes = self.sorted_active_saes()
        masks = [abl_all_but_mask(sorted_saes[:n_preserve]) for n_preserve in xs]
        scores = [self.run_ablation(mask).mean().item() for mask in tqdm(masks, desc="m curve")]
        if faithful:
            return [self.score_to_faithfulness(score) for score in scores]
        return scores

    def score_to_faithfulness(self, score: float | torch.Tensor) -> float | torch.Tensor:
        # # below is Marks version, but assuming ablate_all_score=0 reduces noise imo
        # nom = score - self.ablate_all_score
        # denom = self.orig_model_score - self.ablate_all_score
        # return nom / denom
        return score / self.orig_model_score


# %%
def get_experiment(
    layer: int = 6, sae_type: str = "local", data_set: str = "rc", sim_metric: str = "l0"
):
    if sim_metric == "ce":
        run_id = SIMILAR_CE_RUNS[layer][sae_type]
    else:
        run_id = SIMILAR_L0_RUNS[layer][sae_type]
    sae_pos = f"blocks.{layer}.hook_resid_pre"
    train_data = Dataset.load(data_set, "train")[:1_000]
    return Experiment(run_id, sae_pos, train_data)


def run_and_cache_all(layer: int | None = 6):
    """For each combination we run the experiment and cache the one-at-a-time ablation results"""
    layers = [2, 6, 10] if layer is None else [layer]
    exps = [
        get_experiment(layer, sae_type, data_set)
        for layer in layers
        for sae_type in ["local", "downstream", "e2e"]
        for data_set in ["rc", "simple", "nounpp", "within_rc"]
    ]
    for exp in tqdm(exps, desc="R   unning experiments"):
        _ = exp.m_one_at_a_time


def compute_overall_faithfulness():
    experiment_scores_by_id = {}  # run_id -> data_set -> scores

    for layer in [2, 6, 10]:
        for sae_type in ["local", "downstream", "e2e"]:
            for data_set in ["rc", "simple", "nounpp", "within_rc"]:
                for sim_metric in ["l0", "ce"]:
                    print(f"Computing {layer=} {sae_type=} {data_set=} {sim_metric=}")
                    exp = get_experiment(layer, sae_type, data_set, sim_metric)
                    if exp.run_id not in experiment_scores_by_id:
                        experiment_scores_by_id[exp.run_id] = {}
                    # some run ids are used for both CE and L0 and thus we can avoid recomputing
                    if data_set not in experiment_scores_by_id[exp.run_id]:
                        experiment_scores_by_id[exp.run_id][data_set] = {
                            "orig_model": exp.orig_model_score,
                            "ablate_sae_err": exp.ablate_sae_err_score,
                            "ablate_all": exp.ablate_all_score,
                        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "experiment_scores_by_id.json", "w") as f:
        json.dump(experiment_scores_by_id, f)


def print_tables():
    with open(OUT_DIR / "experiment_scores_by_id.json") as f:
        experiment_scores_by_id = json.load(f)

    for metric in ["L0", "CE"]:
        for layer in [2, 6, 10]:
            print(f"Layer {layer} - similar {metric}")
            print("sae_type".rjust(10), "  simple  ", "nounpp  ", "rc      ", "within_rc")
            for sae_type in ["local", "e2e", "downstream"]:
                id_dict = SIMILAR_L0_RUNS if metric == "L0" else SIMILAR_CE_RUNS
                run_id = id_dict[layer][sae_type]
                print(f"{sae_type.rjust(10)}", end="   ")
                for data_set in ["simple", "nounpp", "rc", "within_rc"]:
                    scores = experiment_scores_by_id[run_id][data_set]
                    faithfulness = scores["ablate_sae_err"] / scores["orig_model"]
                    print(f"{faithfulness:.1%}".ljust(8), end=" ")
                print("")
            print("")
        print("\n\n")


def compute_faithfulness_curve(name_prefix: str = "", layer: int = 6):
    xs = list(range(100))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name_prefix}faithfulness.json"

    # Initialize empty faithfulness dict if file doesn't exist
    if not out_path.exists():
        faithfulness = {}
    else:
        with open(out_path) as f:
            faithfulness = json.load(f)

    def compute_one(layer: int, sae_type: str, data_set: str):
        print(f"Computing curve {sae_type} {data_set} for layer {layer}")
        exp = get_experiment(layer, sae_type, data_set, sim_metric="l0")
        return {
            "n_kept": xs,
            "faithfulness": exp.get_m_curve(xs, faithful=True),
        }

    for sae_type in ["local", "downstream", "e2e"]:
        if sae_type not in faithfulness:
            faithfulness[sae_type] = {}

        for data_set in ["simple", "nounpp", "rc", "within_rc"]:
            if data_set not in faithfulness[sae_type]:
                faithfulness[sae_type][data_set] = compute_one(layer, sae_type, data_set)
                # Save after each computation
                with open(out_path, "w") as f:
                    json.dump(faithfulness, f)


def plot_faithfulness_curve():
    with open(OUT_DIR / "faithfulness.json") as f:
        faithfulness = json.load(f)

    fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharey=True, sharex=True)

    for ax, dataset_shortname in zip(
        axs.flat, ["simple", "nounpp", "rc", "within_rc"], strict=True
    ):
        for sae_type in ["local", "e2e", "downstream"]:
            xs = faithfulness[sae_type][dataset_shortname]["n_kept"]
            ys = faithfulness[sae_type][dataset_shortname]["faithfulness"]
            style = STYLE_MAP[sae_type]
            ax.plot(xs, ys, color=style["color"], label=style["label"])

        ax.axhline(1, color="black", linestyle="--", alpha=0.3)

    plt.xlim(0, 100)
    plt.ylim(0, 1.2)
    # plt.xscale("log")

    axs[0, 0].set_title("Simple")
    axs[0, 1].set_title("Across Participial Phrases")
    axs[1, 0].set_title("Across Relative Clause")
    axs[1, 1].set_title("Within Relative Clause")

    fig.supxlabel("Number of SAE features preserved")
    fig.supylabel("Faithfulness")
    axs[1, 1].legend(loc="lower right")

    plt.tight_layout()

    plt.savefig(OUT_DIR / "faithfulness.png", dpi=300)
    print(f"Saved to {OUT_DIR / 'faithfulness.png'}")


def compute():
    print("Saving experiment scores by id")
    compute_overall_faithfulness()
    print("Computing faithfulness")
    compute_faithfulness_curve()


if __name__ == "__main__":
    Fire(
        {
            "download_data": download_data,
            "compute": compute,
            "tables": print_tables,
            "plot": plot_faithfulness_curve,
        }
    )
