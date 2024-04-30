"""Run and analyze autointerp for SAEs.

NOTE: Running this script currently requires installing ApolloResearch's fork of neuron-explainer.
https://github.com/ApolloResearch/automated-interpretability

This has been updated to work with gpt4-turbo-2024-04-09 and fixes an OPENAI_API_KEY issue.
"""

import asyncio
import glob
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import statsmodels.stats.api as sms
from dotenv import load_dotenv
from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.explainer import ContextSize, TokenActivationPairExplainer
from neuron_explainer.explanations.explanations import ScoredSimulation
from neuron_explainer.explanations.few_shot_examples import FewShotExampleSet
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import (
    _simulate_and_score_sequence,
    aggregate_scored_sequence_simulations,
)
from neuron_explainer.explanations.simulator import (
    LogprobFreeExplanationTokenSimulator,
    NeuronSimulator,
)
from numpy.typing import ArrayLike
from scipy.stats import bootstrap
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

load_dotenv()  # Required before importing UncalibratedNeuronSimulator
NEURONPEDIA_DOMAIN = "https://neuronpedia.org"
POSITIVE_INF_REPLACEMENT = 9999
NEGATIVE_INF_REPLACEMENT = -9999
NAN_REPLACEMENT = 0
OTHER_INVALID_REPLACEMENT = -99999


def NanAndInfReplacer(value: str):
    """Replace NaNs and Infs in outputs."""
    replacements = {
        "-Infinity": NEGATIVE_INF_REPLACEMENT,
        "Infinity": POSITIVE_INF_REPLACEMENT,
        "NaN": NAN_REPLACEMENT,
    }
    if value in replacements:
        replaced_value = replacements[value]
        return float(replaced_value)
    else:
        return NAN_REPLACEMENT


def get_neuronpedia_feature(
    feature: int, layer: int, model: str = "gpt2-small", dataset: str = "res-jb"
) -> dict[str, Any]:
    """Fetch a feature from Neuronpedia API."""
    url = f"{NEURONPEDIA_DOMAIN}/api/feature/{model}/{layer}-{dataset}/{feature}"
    result = requests.get(url).json()
    if "index" in result:
        result["index"] = int(result["index"])
    else:
        raise Exception(f"Feature {model}@{layer}-{dataset}:{feature} does not exist.")
    return result


def test_key(api_key: str):
    """Test the validity of the Neuronpedia API key."""
    url = f"{NEURONPEDIA_DOMAIN}/api/test"
    body = {"apiKey": api_key}
    response = requests.post(url, json=body)
    if response.status_code != 200:
        raise Exception("Neuronpedia API key is not valid.")


class NeuronpediaActivation:
    """Represents an activation from Neuronpedia."""

    def __init__(self, id: str, tokens: list[str], act_values: list[float]):
        self.id = id
        self.tokens = tokens
        self.act_values = act_values


class NeuronpediaFeature:
    """Represents a feature from Neuronpedia."""

    def __init__(
        self,
        modelId: str,
        layer: int,
        dataset: str,
        feature: int,
        description: str = "",
        activations: list[NeuronpediaActivation] | None = None,
        autointerp_explanation: str = "",
        autointerp_explanation_score: float = 0.0,
    ):
        self.modelId = modelId
        self.layer = layer
        self.dataset = dataset
        self.feature = feature
        self.description = description
        self.activations = activations or []
        self.autointerp_explanation = autointerp_explanation
        self.autointerp_explanation_score = autointerp_explanation_score

    def has_activating_text(self) -> bool:
        """Check if the feature has activating text."""
        return any(max(activation.act_values) > 0 for activation in self.activations)


T = TypeVar("T")


@retry(wait=wait_random_exponential(min=1, max=500), stop=stop_after_attempt(10))
def sleep_identity(x: T) -> T:
    """Dummy function for retrying."""
    return x


@retry(wait=wait_random_exponential(min=1, max=500), stop=stop_after_attempt(10))
async def simulate_and_score(
    simulator: NeuronSimulator, activation_records: list[ActivationRecord]
) -> ScoredSimulation:
    """Score an explanation of a neuron by how well it predicts activations on the given text sequences."""
    scored_sequence_simulations = await asyncio.gather(
        *[
            sleep_identity(
                _simulate_and_score_sequence(
                    simulator,
                    activation_record,
                )
            )
            for activation_record in activation_records
        ]
    )
    return aggregate_scored_sequence_simulations(scored_sequence_simulations)


async def autointerp_neuronpedia_features(
    features: list[NeuronpediaFeature],
    openai_api_key: str,
    autointerp_retry_attempts: int = 3,
    autointerp_score_max_concurrent: int = 20,
    neuronpedia_api_key: str = "",
    do_score: bool = True,
    output_dir: str = "neuronpedia_outputs/autointerp",
    num_activations_to_use: int = 20,
    max_explanation_activation_records: int = 20,
    upload_to_neuronpedia: bool = True,
    autointerp_explainer_model_name: Literal[
        "gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview", "gpt-4-turbo-2024-04-09"
    ] = "gpt-4-1106-preview",
    autointerp_scorer_model_name: Literal[
        "gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview", "gpt-4-turbo-2024-04-09"
    ] = "gpt-3.5-turbo",
):
    """
    Autointerp Neuronpedia features.

    Args:
        features: List of NeuronpediaFeature objects.
        openai_api_key: OpenAI API key.
        autointerp_retry_attempts: Number of retry attempts for autointerp.
        autointerp_score_max_concurrent: Maximum number of concurrent requests for autointerp scoring.
        neuronpedia_api_key: Neuronpedia API key.
        do_score: Whether to score the features.
        output_dir: Output directory for saving the results.
        num_activations_to_use: Number of activations to use.
        max_explanation_activation_records: Maximum number of activation records for explanation.
        upload_to_neuronpedia: Whether to upload the results to Neuronpedia.
        autointerp_explainer_model_name: Model name for autointerp explainer.
        autointerp_scorer_model_name: Model name for autointerp scorer.

    Returns:
        None
    """
    print("\n\n")

    os.environ["OPENAI_API_KEY"] = openai_api_key

    if upload_to_neuronpedia and not neuronpedia_api_key:
        raise Exception(
            "You need to provide a Neuronpedia API key to upload the results to Neuronpedia."
        )

    test_key(neuronpedia_api_key)

    print("\n\n=== Step 1) Fetching features from Neuronpedia")
    for feature in features:
        feature_data = get_neuronpedia_feature(
            feature=feature.feature,
            layer=feature.layer,
            model=feature.modelId,
            dataset=feature.dataset,
        )

        if "modelId" not in feature_data:
            raise Exception(
                f"Feature {feature.feature} in layer {feature.layer} of model {feature.modelId} and dataset {feature.dataset} does not exist."
            )

        if "activations" not in feature_data or len(feature_data["activations"]) == 0:
            raise Exception(
                f"Feature {feature.feature} in layer {feature.layer} of model {feature.modelId} and dataset {feature.dataset} does not have activations."
            )

        activations = feature_data["activations"]
        activations_to_add = []
        for activation in activations:
            if len(activations_to_add) < num_activations_to_use:
                activations_to_add.append(
                    NeuronpediaActivation(
                        id=activation["id"],
                        tokens=activation["tokens"],
                        act_values=activation["values"],
                    )
                )
        feature.activations = activations_to_add

        if not feature.has_activating_text():
            raise Exception(
                f"Feature {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature} appears dead - it does not have activating text."
            )

    for iteration_num, feature in enumerate(features):
        start_time = datetime.now()

        print(
            f"\n========== Feature {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature} ({iteration_num + 1} of {len(features)} Features) =========="
        )
        print(
            f"\n=== Step 2) Explaining feature {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature}"
        )

        activation_records = [
            ActivationRecord(tokens=activation.tokens, activations=activation.act_values)
            for activation in feature.activations
        ]

        activation_records_explaining = activation_records[:max_explanation_activation_records]

        explainer = TokenActivationPairExplainer(
            model_name=autointerp_explainer_model_name,
            prompt_format=PromptFormat.HARMONY_V4,
            context_size=ContextSize.SIXTEEN_K,
            max_concurrent=1,
        )

        explanations = []
        for _ in range(autointerp_retry_attempts):
            try:
                explanations = await explainer.generate_explanations(
                    all_activation_records=activation_records_explaining,
                    max_activation=calculate_max_activation(activation_records_explaining),
                    num_samples=1,
                )
            except Exception as e:
                print(f"ERROR, RETRYING: {e}")
            else:
                break
        else:
            print(
                f"ERROR: Failed to explain feature {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature}"
            )

        assert len(explanations) == 1
        explanation = explanations[0].rstrip(".")
        print(f"===== {autointerp_explainer_model_name}'s explanation: {explanation}")
        feature.autointerp_explanation = explanation

        if do_score:
            print(
                f"\n=== Step 3) Scoring feature {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature}"
            )
            print("=== This can take up to 30 seconds.")

            temp_activation_records = [
                ActivationRecord(
                    tokens=[
                        token.replace("<|endoftext|>", "<|not_endoftext|>")
                        .replace(" 55", "_55")
                        .encode("ascii", errors="backslashreplace")
                        .decode("ascii")
                        for token in activation_record.tokens
                    ],
                    activations=activation_record.activations,
                )
                for activation_record in activation_records
            ]

            score = None
            scored_simulation = None
            for _ in range(autointerp_retry_attempts):
                try:
                    simulator = UncalibratedNeuronSimulator(
                        LogprobFreeExplanationTokenSimulator(
                            autointerp_scorer_model_name,
                            explanation,
                            json_mode=True,
                            max_concurrent=autointerp_score_max_concurrent,
                            few_shot_example_set=FewShotExampleSet.JL_FINE_TUNED,
                            prompt_format=PromptFormat.HARMONY_V4,
                        )
                    )
                    scored_simulation = await simulate_and_score(simulator, temp_activation_records)
                    score = scored_simulation.get_preferred_score()
                except Exception as e:
                    print(f"ERROR, RETRYING: {e}")
                else:
                    break

            if (
                score is None
                or scored_simulation is None
                or len(scored_simulation.scored_sequence_simulations) != num_activations_to_use
            ):
                print(
                    f"ERROR: Failed to score feature {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature}. Skipping it."
                )
                continue
            feature.autointerp_explanation_score = score
            print(f"===== {autointerp_scorer_model_name}'s score: {(score * 100):.0f}")

            output_file = (
                f"{output_dir}/{feature.layer}-{feature.dataset}_feature-{feature.feature}_score-"
                f"{feature.autointerp_explanation_score}_time-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
            )
            os.makedirs(output_dir, exist_ok=True)
            print(f"===== Your results will be saved to: {output_file} =====")

            output_data = json.dumps(
                {
                    "apiKey": neuronpedia_api_key,
                    "feature": {
                        "modelId": feature.modelId,
                        "layer": f"{feature.layer}-{feature.dataset}",
                        "index": feature.feature,
                        "activations": feature.activations,
                        "explanation": feature.autointerp_explanation,
                        "explanationScore": feature.autointerp_explanation_score,
                        "explanationModel": autointerp_explainer_model_name,
                        "autointerpModel": autointerp_scorer_model_name,
                        "simulatedActivations": scored_simulation.scored_sequence_simulations,
                    },
                },
                default=vars,
            )
            output_data_json = json.loads(output_data, parse_constant=NanAndInfReplacer)
            output_data_str = json.dumps(output_data)

            print(f"\n=== Step 4) Saving feature to {output_file}")
            with open(output_file, "a") as f:
                f.write(output_data_str)
                f.write("\n")

            if upload_to_neuronpedia:
                print(
                    f"\n=== Step 5) Uploading feature to Neuronpedia: {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature}"
                )
                url = f"{NEURONPEDIA_DOMAIN}/api/upload-explanation"
                body = output_data_json
                response = requests.post(url, json=body)
                if response.status_code != 200:
                    print(f"ERROR: Couldn't upload explanation to Neuronpedia: {response.text}")

        end_time = datetime.now()
        print(f"\n========== Time Spent for Feature: {end_time - start_time}\n")

    print("\n\n========== Generation and Upload Complete ==========\n\n")


def run_autointerp(
    sae_sets: list[list[str]],
    n_random_features: int,
    dict_size: int,
    feature_model_id: str,
    autointerp_explainer_model_name: Literal[
        "gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview", "gpt-4-turbo-2024-04-09"
    ],
    autointerp_scorer_model_name: Literal[
        "gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview", "gpt-4-turbo-2024-04-09"
    ],
):
    """Explain and score random features across SAEs and upload to Neuronpedia.

    Args:
        sae_sets: List of SAE sets.
        n_random_features: Number of random features to explain and score for each SAE.
        dict_size: Size of the SAE dictionary.
        feature_model_id: Model ID that the SAE is attached to.
        autointerp_explainer_model_name: Model name for autointerp explainer.
        autointerp_scorer_model_name: Model name for autointerp scorer (much more expensive).
    """
    for sae_set in tqdm(sae_sets, desc="sae sets"):
        for i in tqdm(range(n_random_features), desc="random features"):
            for sae in tqdm(sae_set, desc="sae"):
                layer = int(sae.split("-")[0])
                dataset = "-".join(sae.split("-")[1:])
                feature_exists = False
                while not feature_exists:
                    feature = random.randint(0, dict_size)
                    feature_data = get_neuronpedia_feature(
                        feature=feature,
                        layer=layer,
                        model=feature_model_id,
                        dataset=dataset,
                    )
                    if "activations" in feature_data and len(feature_data["activations"]) >= 20:
                        feature_exists = True
                        print(f"sae: {sae}, feature: {feature}")

                        features = [
                            NeuronpediaFeature(
                                modelId=feature_model_id,
                                layer=layer,
                                dataset=dataset,
                                feature=feature,
                            ),
                        ]

                        asyncio.run(
                            autointerp_neuronpedia_features(
                                features=features,
                                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                                neuronpedia_api_key=os.getenv("NEURONPEDIA_API_KEY", ""),
                                autointerp_explainer_model_name=autointerp_explainer_model_name,
                                autointerp_scorer_model_name=autointerp_scorer_model_name,
                                num_activations_to_use=20,
                                max_explanation_activation_records=5,
                            )
                        )

                        feature_url = f"https://neuronpedia.org/{feature_model_id}/{layer}-{dataset}/{feature}"
                        print(f"Your feature is at: {feature_url}")


def get_autointerp_results_df(out_dir: Path):
    autointerp_files = glob.glob(f"{out_dir}/**/*_feature-*_score-*", recursive=True)
    stats = {
        "layer": [],
        "sae": [],
        "feature": [],
        "explanationScore": [],
        "explanationModel": [],
        "autointerpModel": [],
        "explanation": [],
        "sae_type": [],
    }
    for autointerp_file in tqdm(autointerp_files, "constructing stats from json autointerp files"):
        with open(autointerp_file) as f:
            json_data = json.loads(json.load(f))
        if "feature" in json_data:
            json_data = json_data["feature"]
        stats["layer"].append(int(json_data["layer"].split("-")[0]))
        stats["sae"].append("-".join(json_data["layer"].split("-")[1:]))
        stats["feature"].append(json_data["index"])
        stats["explanationScore"].append(json_data["explanationScore"])
        stats["autointerpModel"].append(json_data["autointerpModel"])
        stats["explanationModel"].append("gpt-4-1106-preview")
        stats["explanation"].append(json_data["explanation"])
        stats["sae_type"].append(
            {"e": "e2e", "l": "local", "r": "downstream"}[json_data["layer"].split("-")[1][-1]]
        )
    df_stats = pd.DataFrame(stats)
    assert not df_stats.empty
    df_stats.dropna(inplace=True)
    return df_stats


def compare_autointerp_results(df_stats: pd.DataFrame):
    """
    Compare autointerp results across SAEs.

    Args:
        output_dir: Directory containing the autointerp output files.

    Returns:
        None
    """
    sns.set_theme(style="whitegrid", palette="pastel")
    g = sns.catplot(
        data=df_stats,
        x="sae",
        y="explanationScore",
        kind="box",
        palette="pastel",
        hue="layer",
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "10",
        },
    )
    sns.swarmplot(
        data=df_stats, x="sae", y="explanationScore", color="k", size=4, ax=g.ax, legend=False
    )
    plt.title("Quality of auto-interpretability explanations across SAEs")
    plt.ylabel("Auto-interpretability score")
    plt.xlabel("SAE")
    results_file = out_dir / "Auto-interpretability_results.png"
    plt.savefig(results_file, bbox_inches="tight")
    # plt.savefig("Auto-interpretability_results.pdf", bbox_inches="tight")
    # plt.show()
    print(f"Saved to {results_file}")

    sns.set_theme(rc={"figure.figsize": (6, 6)})
    b = sns.barplot(
        df_stats,
        x="sae",
        y="explanationScore",
        palette="pastel",
        hue="layer",
        capsize=0.3,
        legend=False,
    )
    s = sns.swarmplot(data=df_stats, x="sae", y="explanationScore", color="k", alpha=0.25, size=6)
    plt.title("Quality of auto-interpretability explanations across SAEs")
    plt.ylabel("Auto-interpretability score")
    plt.xlabel("SAE")
    plt.yticks(np.arange(0, 1, 0.1))
    bar_file = out_dir / "Auto-interpretability_results_bar.png"
    plt.savefig(bar_file, bbox_inches="tight")
    # plt.savefig("Auto-interpretability_results_bar.pdf", bbox_inches="tight")
    # plt.show()
    print(f"Saved to {bar_file}")


def compare_across_saes(df_stats: pd.DataFrame):
    # Plot the relative performance of the SAEs
    sns.set_theme(style="whitegrid", palette="pastel")
    g = sns.catplot(
        data=df_stats,
        x="sae",
        y="explanationScore",
        kind="box",
        palette="pastel",
        hue="layer",
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "5",
        },
    )
    sns.swarmplot(
        data=df_stats, x="sae", y="explanationScore", color="k", size=4, ax=g.ax, legend=False
    )
    # ax.set_title("Quality of auto-interpretability explanations across SAEs")
    # ax.set_ylabel("Auto-interpretability score")
    # ax.set_xlabel("SAE")
    result_file = out_dir / "Auto-interpretability_results.png"
    plt.savefig(result_file, bbox_inches="tight")
    plt.close()
    # plt.savefig("Auto-interpretability_results.pdf", bbox_inches="tight")
    # plt.show()
    print(result_file)

    # Plot the relative performance of the SAEs
    sns.set_theme(rc={"figure.figsize": (6, 6)})
    b = sns.barplot(
        df_stats,
        x="sae",
        y="explanationScore",
        palette="pastel",
        hue="layer",
        capsize=0.3,
        legend=False,
    )
    s = sns.swarmplot(data=df_stats, x="sae", y="explanationScore", color="k", alpha=0.25, size=6)
    plt.title("Quality of auto-interpretability explanations across SAEs")
    plt.ylabel("Auto-interpretability score")
    plt.xlabel("SAE")
    plt.yticks(np.arange(0, 1, 0.1))
    bar_file = out_dir / "Auto-interpretability_results_bar.png"
    plt.savefig(bar_file, bbox_inches="tight")
    plt.close()
    # plt.savefig("Auto-interpretability_results_bar.pdf", bbox_inches="tight")
    # plt.show()
    print(bar_file)


def bootstrapped_bar(df_stats: pd.DataFrame):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    sae_names = {
        "res_scefr-ajt": "Downstream",
        "res_sll-ajt": "CE Local",
        "res_scl-ajt": "L0 Local",
    }
    # make matplotlib histogram with ci as error bars
    for layer, ax in zip([6, 10], axs, strict=True):
        layer_data = df_stats.loc[df_stats.layer == layer]

        means, yerrs = [], [[], []]
        for sae_type in sae_names:
            sae_data = layer_data.loc[layer_data.sae == sae_type]
            scores = sae_data.explanationScore.to_numpy()
            ci = bootstrap((scores,), statistic=np.mean).confidence_interval
            means.append(scores.mean())
            yerrs[0].append(scores.mean() - ci.low)
            yerrs[1].append(ci.high - scores.mean())

        ax.bar(range(3), means, yerr=yerrs, capsize=5)
        ax.set_title(f"Layer {layer}")
        ax.set_xticks(range(3), sae_names.values())

    axs[0].set_ylabel("Mean Explanation Score")

    plt.tight_layout()
    plt.show()
    plt.savefig(out_dir / "bootstrapped_bar.png")
    print(f"Saved to {out_dir / 'bootstrapped_bar.png'}")
    plt.close()


def pair_violin_plot(df_stats: pd.DataFrame, pairs: dict[int, dict[str, str]]):
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(7, 3.5))

    for ax, layer in zip(axs, pairs.keys(), strict=True):
        sae_ids = [pairs[layer]["local"], pairs[layer]["downstream"]]
        layer_data = df_stats[df_stats["layer"] == layer]
        grouped_scores = layer_data.groupby("sae")["explanationScore"]
        mean_explanationScores = [grouped_scores.get_group(sae_id).mean() for sae_id in sae_ids]
        confidence_intervals = [
            sms.DescrStatsW(grouped_scores.get_group(sae_id)).tconfint_mean() for sae_id in sae_ids
        ]

        data = layer_data.loc[layer_data.sae.isin(sae_ids)]
        colors = {
            "local": "#f0a70a",
            "e2e": "#518c31",
            "downstream": plt.get_cmap("tab20b").colors[2],  # type: ignore[reportAttributeAccessIssue]
        }
        l = sns.violinplot(
            data=data,
            hue="sae_type",
            x=[1 if sae_type == "downstream" else 0 for sae_type in data.sae_type],
            y="explanationScore",
            palette=colors,
            native_scale=True,
            orient="v",
            ax=ax,
            inner=None,
            cut=0,
            bw_adjust=0.7,
            hue_order=["local", "downstream"],
            alpha=0.8,
            legend=False,
        )
        ax.set_xticks([0, 1], ["Local", "Downstream"])
        ax.errorbar(
            x=(0, 1),
            y=mean_explanationScores,
            yerr=[(c[1] - c[0]) / 2 for c in confidence_intervals],
            fmt="o",
            color="black",
            capsize=5,
        )
        ax.set_ylim((None, 1))
        ax.set_title(label=f"Layer {layer}")

    axs[0].set_ylabel("Auto-intepretability score")

    plt.tight_layout()
    out_file = Path(__file__).parent / "out/autointerp" / "autointerp_same_l0.png"
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Saved to {out_file}")


def bootstrap_p_value(sample_a: ArrayLike, sample_b: ArrayLike) -> float:
    """
    Computes 2 sided p-value, for null hypothesis that means are the same
    """
    sample_a, sample_b = np.asarray(sample_a), np.asarray(sample_b)
    mean_diff = np.mean(sample_a) - np.mean(sample_b)
    n_bootstraps = 100_000
    bootstrapped_diffs = []
    combined = np.concatenate([sample_a, sample_b])
    for _ in range(n_bootstraps):
        boot_a = np.random.choice(combined, len(sample_a), replace=True)
        boot_b = np.random.choice(combined, len(sample_b), replace=True)
        boot_diff = np.mean(boot_a) - np.mean(boot_b)
        bootstrapped_diffs.append(boot_diff)
    bootstrapped_diffs = np.array(bootstrapped_diffs)
    p_value = np.mean(np.abs(bootstrapped_diffs) > np.abs(mean_diff))
    return p_value


def bootstrap_mean_diff(sample_a: ArrayLike, sample_b: ArrayLike):
    def mean_diff(resample_a, resample_b):  # type: ignore
        return np.mean(resample_a) - np.mean(resample_b)

    result = bootstrap((sample_a, sample_b), n_resamples=100_000, statistic=mean_diff)
    diff = np.mean(sample_a) - np.mean(sample_b)  # type: ignore
    low, high = result.confidence_interval
    print(f"Diff {diff:.2f}, [{low:.2f},{high:.2f}]")


def compute_p_values(df: pd.DataFrame, pairs: dict[int, dict[str, str]]):
    score_groups = df.groupby(["layer", "sae"]).explanationScore
    for layer, sae_dict in pairs.items():
        sae_local, sae_downstream = sae_dict["local"], sae_dict["downstream"]
        pval = bootstrap_p_value(
            score_groups.get_group((layer, sae_downstream)).to_numpy(),
            score_groups.get_group((layer, sae_local)).to_numpy(),
        )
        print(f"L{layer}, {sae_downstream} vs {sae_local}: p={pval}")
        bootstrap_mean_diff(
            score_groups.get_group((layer, sae_downstream)).to_numpy(),
            score_groups.get_group((layer, sae_local)).to_numpy(),
        )


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "out/autointerp/"
    input_dir = out_dir  # Path("/data/apollo/autointerp")
    ## Running autointerp
    # Get runs for similar CE and similar L0 for e2e+Downstream and local
    # Note that "10-res_slefr-ajt" does not exist, we use 10-res_scefr-ajt for similar l0 too
    sae_sets = [
        ["6-res_scefr-ajt", "6-res_slefr-ajt", "6-res_sll-ajt", "6-res_scl-ajt"],
        ["10-res_scefr-ajt", "10-res_sll-ajt", "10-res_scl-ajt"],
        ["2-res_scefr-ajt", "2-res_slefr-ajt", "2-res_sll-ajt", "2-res_scl-ajt"],
    ]
    # run_autointerp(
    #     sae_sets=sae_sets,
    #     n_random_features=50,
    #     dict_size=768 * 60,
    #     feature_model_id="gpt2-small",
    #     autointerp_explainer_model_name="gpt-4-turbo-2024-04-09",
    #     autointerp_scorer_model_name="gpt-3.5-turbo",
    # )

    # TO UPDATE
    const_l0_pairs = {
        2: {"local": "res_sll-ajt", "downstream": "res_slefr-ajt"},
        6: {"local": "res_sll-ajt", "downstream": "res_scefr-ajt"},
        10: {"local": "res_sll-ajt", "downstream": "res_scefr-ajt"},
    }

    df = get_autointerp_results_df(input_dir)
    print(df.autointerpModel.unique(), df.explanationModel.unique())
    ## Analysis of autointerp results
    compare_autointerp_results(df)
    compare_across_saes(df)
    bootstrapped_bar(df)
    pair_violin_plot(df, const_l0_pairs)
    compute_p_values(df, const_l0_pairs)
