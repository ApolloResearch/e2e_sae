# %%
import functools
import json
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Self

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from e2e_sae.models.transformers import SAETransformer
from e2e_sae.scripts.analysis.plot_settings import SIMILAR_CE_RUNS, SIMILAR_L0_RUNS, STYLE_MAP
from e2e_sae.settings import REPO_ROOT

# %%
OUT_DIR = REPO_ROOT / "e2e_sae/scripts/analysis" / "out/faithfulness"
DATA_FOLDER = REPO_ROOT / "e2e_sae/scripts/analysis" / "data"
device = "cuda" if torch.cuda.is_available() else "cpu"


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


@dataclass
class Experiment:
    run_id: str
    sae_pos: str
    dataset: Dataset
    batch_size: int = 1_000

    def __post_init__(self):
        self.model = SAETransformer.from_wandb(f"gpt2/{self.run_id}")
        self.model.to(device)
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
# def run(n_train: int = 1_000, layer: int = 6, sae_type: str = "local", data_set: str = "rc"):
#     print("Running experiment", locals())
#     run_id = SIMILAR_CE_RUNS[layer][sae_type]
#     sae_pos = f"blocks.{layer}.hook_resid_pre"
#     train_data = Dataset.load(data_set, "train")[:n_train]
#     experiment = Experiment(run_id, sae_pos, train_data)
#     print(f"None ablated: {experiment.ablate_none_score:.2f}")
#     print(f"All ablated: {experiment.ablate_all_score:.2f}")
#     experiment.m_one_at_a_time

# if __name__ == "__main__":
#     Fire(run)


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


def cache_one_at_a_time(layer: int = 6, sae_type: str = "local", data_set: str = "rc"):
    exp = get_experiment(layer, sae_type, data_set)
    exp.m_one_at_a_time


for sae_type in ["local", "downstream", "e2e"]:
    for data_set in ["rc", "simple", "nounpp", "within_rc"]:
        print(sae_type, data_set)
        cache_one_at_a_time(6, sae_type, data_set)


# local_exp = get_experiment(6, "local", "simple")
# downstream_exp = get_experiment(6, "downstream", "simple")
# e2e_exp = get_experiment(6, "e2e", "within_rc")


# %%
# local_orig = local_exp.get_orig_model_scores()
# local_abl_sae_err = local_exp.run_ablation(abl_only_mask([]))
# plt.scatter(local_orig.detach(), local_abl_sae_err, c="orange", s=2)

# downstream_orig = downstream_exp.get_orig_model_scores()
# downstream_abl_sae_err = downstream_exp.run_ablation(abl_only_mask([]))
# plt.scatter(downstream_orig.detach(), downstream_abl_sae_err, c="blue", s=2)

# plt.plot([0, 8], [0, 8], c="k")


# # %%
# def get_rms(orig: torch.Tensor, ablated: torch.Tensor) -> float:
#     return torch.sqrt(((orig - ablated) ** 2).mean()).item()


# print("Local", get_rms(local_orig, local_abl_sae_err))
# print("Downstream", get_rms(downstream_orig, downstream_abl_sae_err))


# # %%

# xs = list(range(100)) + list(range(100, 500, 5))


# def progressively_ablate_rms(experiment: Experiment) -> float:
#     rms = []
#     orig_scores = experiment.get_orig_model_scores().detach()
#     for x in tqdm(xs):
#         ablated = experiment.run_ablation(abl_all_but_mask(experiment.sorted_active_saes()[:x]))
#         rms.append(get_rms(orig_scores, ablated))
#     return rms


# local_rms = progressively_ablate_rms(local_exp)
# downstream_rms = progressively_ablate_rms(downstream_exp)

# # %%
# plt.plot(xs, local_rms, label="Local", color="orange")
# plt.plot(xs, downstream_rms, label="Downstream", color="blue")


# # %%


# # %%


# # experiments = {
# #     (sae_type, data_set): get_experiment(sae_type, data_set)
# #     for sae_type in ["local", "downstream", "e2e"]
# #     for data_set in ["rc", "simple", "nounpp", "within_rc"]
# # }

# # # %%
# # experiments_sim_ce_l6 = experiments


# # # %%


# # experiments_sim_ce_l2 = {
# #     (sae_type, data_set): get_experiment(2, sae_type, data_set)
# #     for sae_type in ["local", "downstream", "e2e"]
# #     for data_set in ["rc", "simple", "nounpp", "within_rc"]
# # }
# # %%

# datasets = {sn: Dataset.load(sn, "train")[:1_000] for sn in ["rc", "simple", "nounpp", "within_rc"]}

# experiment_scores_by_id_data = {}


# def get_scores(experiment: Experiment):
#     return {
#         "orig_model": experiment.orig_model_score,
#         "ablate_sae_err": experiment.ablate_sae_err_score,
#         "ablate_all": experiment.ablate_all_score,
#     }


# for layer in [2, 6, 10]:
#     sae_pos = f"blocks.{layer}.hook_resid_pre"

#     for sae_type in ["local", "downstream", "e2e"]:
#         sim_ce_run = SIMILAR_CE_RUNS[layer][sae_type]
#         sim_l0_run = SIMILAR_L0_RUNS[layer][sae_type]
#         for data_set in ["rc", "simple", "nounpp", "within_rc"]:
#             if (sim_ce_run, data_set) not in experiment_scores_by_id_data:
#                 experiment_scores_by_id_data[(sim_ce_run, data_set)] = get_scores(
#                     Experiment(sim_ce_run, sae_pos, datasets[data_set])
#                 )
#             if (sim_l0_run, data_set) not in experiment_scores_by_id_data:
#                 experiment_scores_by_id_data[(sim_l0_run, data_set)] = get_scores(
#                     Experiment(sim_l0_run, sae_pos, datasets[data_set])
#                 )


# # %%
# experiment_scores_by_id = {}
# for (run_id, data_set), scores in experiment_scores_by_id_data.items():
#     experiment_scores_by_id[run_id] = experiment_scores_by_id.get(run_id, {})
#     experiment_scores_by_id[run_id][data_set] = scores


# with open(OUT_DIR / "experiment_scores_by_id.json", "w") as f:
#     json.dump(experiment_scores_by_id, f)


# # %%
# for layer in [2, 6, 10]:
#     print(f"Layer {layer} - similar L0")
#     print("sae_type".rjust(10), "  simple  ", "nounpp  ", "rc      ", "within_rc")
#     for sae_type in ["local", "downstream", "e2e"]:
#         scores = experiment_scores_by_id[SIMILAR_L0_RUNS[layer][sae_type]]
#         print(f"{sae_type.rjust(10)}", end="   ")
#         for data_set in ["simple", "nounpp", "rc", "within_rc"]:
#             faithfulness = scores[data_set]["ablate_sae_err"] / scores[data_set]["orig_model"]
#             print(f"{faithfulness:.1%}".ljust(8), end=" ")
#         print("")
#     print("")

# # %%

# print("Layer 2 - similar CE")
# print("sae_type".rjust(10), "  simple  ", "nounpp  ", "rc      ", "within_rc")
# for sae_type in ["local", "downstream", "e2e"]:
#     print(f"{sae_type.rjust(10)}", end="   ")
#     for data_set in ["simple", "nounpp", "rc", "within_rc"]:
#         experiment = experiments_sim_ce_l2[(sae_type, data_set)]
#         faithfulness = experiment.ablate_sae_err_score / experiment.orig_model_score
#         print(f"{faithfulness:.1%}".ljust(8), end=" ")
#     print("")
# print("")

# # %%

# sae_type -> data_set -> Experiment
ExperimentDict = dict[str, dict[str, Experiment]]


def compute_and_save_faithfulness(exp_dict: ExperimentDict, name_prefix: str = ""):
    xs = list(range(100)) + list(range(100, 500, 5))
    faithfulness = {
        sae_type: {
            data_set: {"n_kept": xs, "faithfulness": exp.get_m_curve(xs, faithful=True)}
            for data_set, exp in data_dict.items()
        }
        for sae_type, data_dict in exp_dict.items()
    }
    with open(OUT_DIR / f"{name_prefix}faithfulness.json", "w") as f:
        json.dump(faithfulness, f)


# %%
with open(OUT_DIR / "faithfulness.json") as f:
    faithfulness_to_save = json.load(f)
    faithfulness = {
        (d["sae_type"], d["data_set"]): (d["n_kept"], d["faithfulness"])
        for d in faithfulness_to_save
    }


# %%


def plot_faithfulness_curve(exp_dict: dict[str, dict[str, Experiment]]):
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharey=True, sharex=True)

    for ax, dataset_shortname in zip(
        axs.flat, ["simple", "nounpp", "rc", "within_rc"], strict=True
    ):
        ax.set_title(dataset_shortname)
        for sae_type in ["local", "downstream", "e2e"]:
            xs, ys = faithfulness[(sae_type, dataset_shortname)]
            ax.plot(xs, ys, color=STYLE_MAP[sae_type]["color"], label=sae_type)

        ax.axhline(1, color="black", linestyle="--", alpha=0.3)

    plt.xlim(0, 200)
    plt.ylim(0.3, 1.1)
    # plt.xscale("log")

    axs[0, 0].set_ylabel("Faithfulness")
    axs[1, 0].set_ylabel("Faithfulness")
    axs[1, 0].set_xlabel("Number of SAEs features preserved")
    axs[1, 1].set_xlabel("Number of SAEs features preserved")
    plt.tight_layout()
    plt.legend()


# %%
