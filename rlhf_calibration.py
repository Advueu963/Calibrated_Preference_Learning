#!/usr/bin/env python3
"""
Analyze average logit difference for top rewardbench models.

Computes: mean(|chosen_score - rejected_score|) across all examples and rejected responses.
"""

import argparse
import csv
import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from pathlib import Path

from huggingface_hub import HfFileSystem, hf_hub_download
import numpy as np
import torch
from scipy.stats import kendalltau
from cal_pref.utils import calculate_binary_ece, calculate_binary_ece_general
import scienceplots

MODEL_NAME_ABBREVIATIONS = {
    "Skywork/Skywork-Reward-V2-Llama-3.1-8B": "SkyworkV2-Llama-8B",
    "Skywork/Skywork-Reward-V2-Qwen3-8B": "SkyworkV2-Qwen3-8B",
    "infly/INF-ORM-Llama3.1-70B": "INF-ORM-Llama3.1-70B",
    "allenai/Llama-3.1-70B-Instruct-RM-RB2": "Llama-3.1-70B-Instruct-RM",
    "Skywork/Skywork-Reward-V2-Qwen3-4B": "SkyworkV2-Qwen3-4B",
    "Skywork/Skywork-Reward-V2-Llama-3.2-3B": "SkyworkV2-Llama-3.2-3B",
    "LxzGordon/URM-LLaMa-3.1-8B": "URM-LLaMa-3.1-8B",
    "Skywork/Skywork-Reward-Llama-3.1-8B": "Skywork-Llama-8B",
    "allenai/Llama-3.1-8B-Instruct-RM-RB2": "Llama-3.1-8B-Instruct-RM",
    "ShikaiChen/LDL-Reward-Gemma-2-27B-v0.1": "LDL-Reward-Gemma-27B",
    "allenai/Llama-3.1-Tulu-3-70B-SFT-RM-RB2": "Llama-3.1-Tulu-70B-SFT-RM",
    "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2": "Skywork-Llama-8B-v0.2",
    "HFXM/RAMO-Llama3.1-8B": "RAMO-Llama3.1-8B",
    "Skywork/Skywork-VL-Reward-7B": "Skywork-VL-Reward-7B",
    "allenai/Llama-3.1-Tulu-3-8B-RL-RM-RB2": "Llama-3.1-Tulu-8B-RL-RM",
    "allenai/Llama-3.1-Tulu-3-8B-DPO-RM-RB2": "Llama-3.1-Tulu-8B-DPO-RM",
    "allenai/Llama-3.1-Tulu-3-8B-SFT-RM-RB2": "Llama-3.1-Tulu-8B-SFT-RM",
    "Skywork/Skywork-Reward-V2-Qwen3-1.7B": "SkyworkV2-Qwen3-1.7B",
    "Ray2333/GRM-Llama3-8B-rewardmodel-ft": "GRM-Llama3-8B-rewardmodel-ft",
}


def parse_leaderboard(csv_path: Path) -> list[tuple[int, str, str, float]]:
    """Parse leaderboard CSV and extract all models."""
    models = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = int(row[""])
            factuality = row["Factuality"]
            precise_if = row["Precise IF"]
            math = row["Math"]
            safety = row["Safety"]
            focus = row["Focus"]
            ties = row["Ties"]

            model_html = row["Model"]
            match = re.search(r">([^<]+)</a>", model_html)
            assert match is not None
            model_name = match.group(1).strip()

            model_type = row["Model Type"]
            score = float(row["Score"])
            models.append(
                (
                    rank,
                    factuality,
                    precise_if,
                    math,
                    safety,
                    focus,
                    ties,
                    model_name,
                    model_type,
                    score,
                )
            )

    return models


def find_model_file(fs: HfFileSystem, model_name: str) -> str | None:
    """Find the results file for a model."""
    base_path = "datasets/allenai/reward-bench-2-results/eval-set-scores"

    parts = model_name.split("/")
    assert len(parts) == 2

    org, name = parts
    expected_path = f"{base_path}/{org}/{name}.json"
    try:
        fs.info(expected_path)
        return expected_path
    except FileNotFoundError:
        return None


def load_scores(file_path: str, cache_dir: Path | None) -> dict:
    """Download and parse model scores."""
    parts = file_path.split("/")
    relative_path = "/".join(parts[3:])

    local_path = hf_hub_download(
        repo_id="allenai/reward-bench-2-results",
        filename=relative_path,
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    with open(local_path) as f:
        return json.load(f)


def compute_margins(scores_data: dict) -> tuple[float, float, int]:
    """
    Compute margin metrics.

    Returns:
        - mean_margin: mean(chosen - rejected_i) across all examples and rejected responses
        - margin_to_mean: mean(chosen - mean(rejected)) across examples
        - n_examples: number of examples
    """
    all_diffs = []
    all_margin_to_mean = []
    # probs_chosen_over_rejected = []
    top_1_pl_probs = []
    distribution = []
    model_chosen_alternative = []
    # Filter out examples with multiple correct responses
    scores_list = scores_data["scores"]

    for example_scores in scores_list:
        # prob_c_o_r = 0
        # Some models store scores as nested lists: [[-4.21], [-6.25], ...]
        # Others use flat format: [-4.21, -6.25, ...]
        if isinstance(example_scores[0], list):
            for s in example_scores:
                assert len(s) == 1
            example_scores = [s[0] for s in example_scores]

        chosen = example_scores[0]
        rejected = example_scores[1:]
        # Normalize for numerical stability
        chosen_pl = chosen - np.max(example_scores)
        rejected_pl = [r - np.max(example_scores) for r in rejected]

        top_1_pl_prob = np.exp(chosen_pl) / (
            np.exp(chosen_pl) + sum(np.exp(r) for r in rejected_pl)
        )
        distribution.append(
            np.array([np.exp(a) for a in [chosen_pl] + rejected_pl])
            / sum(np.exp(a) for a in [chosen_pl] + rejected_pl)
        )

        for rej in rejected:
            all_diffs.append(chosen - rej)
            # prob_c_o_r += 1 / (1 + np.exp(rej - chosen))
        # prob_c_o_r /= len(rejected)  # Now this is E[P(chosen > {rejected_1, ...})]
        # probs_chosen_over_rejected.append(prob_c_o_r)
        top_1_pl_probs.append(top_1_pl_prob)
        margin_to_mean = chosen - (sum(rejected) / len(rejected))
        all_margin_to_mean.append(margin_to_mean)

    # print(
    #     f"Prob range: {min(probs_chosen_over_rejected):.4f} to {max(probs_chosen_over_rejected):.4f}"
    # )
    print(
        f"Prob Top-1 PL range: {min(top_1_pl_probs):.4f} to {max(top_1_pl_probs):.4f}"
    )
    # Calculate ECE metrics
    # ece_probs_chosen_over_rejected = calculate_binary_ece(
    #     y_true=torch.tensor([1] * len(probs_chosen_over_rejected)),
    #     y_prob=torch.tensor(probs_chosen_over_rejected),
    #     n_bins=10,
    # )
    ece_top_1_pl = calculate_binary_ece_general(
        y_true=torch.tensor([1] * len(top_1_pl_probs)),
        y_prob=torch.tensor(top_1_pl_probs),
        # n_bins=10,
        discrepancy="abs",
        bin_spacing="linear",
    )

    entropy = np.array([-np.sum(d * np.log(d + 1e-12)) for d in distribution]) / np.log(
        len(distribution[0])
    )

    mean_margin = sum(all_diffs) / len(all_diffs)
    avg_margin_to_mean = sum(all_margin_to_mean) / len(all_margin_to_mean)
    model_correct_alternatives = scores_data["results"]

    return (
        mean_margin,
        avg_margin_to_mean,
        len(all_margin_to_mean),
        # ece_probs_chosen_over_rejected,
        ece_top_1_pl,
        entropy,
        model_correct_alternatives,
    )


def visualize_raw_ece_scores(results, top_k=10, save_folder="rlhf_ece"):
    """Scatter plot of (raw) ECE values for the best models.

    Styling is kept consistent with the ECE boxplots in `experiment_calibration.py`.

    Args:
        results: list of model result dicts.
        top_k: number of best (lowest-mean-ECE) models to plot.
        save_folder: folder for the output PNG.
    """

    if not results:
        return

    def _abbreviate_reward_model_name(model_name: str, max_len: int = 22) -> str:
        s = str(model_name)
        # RewardBench models are typically "org/name"; keep the repo part for readability.
        if "/" in s:
            s = s.rsplit("/", 1)[-1]

        # Compact a few common suffixes.
        s = re.sub(r"([-_]?instruct(ion)?)$", "-inst", s, flags=re.IGNORECASE)

        if len(s) <= max_len:
            return s

        # Middle ellipsis to preserve both prefix and suffix.
        keep_left = max(3, (max_len - 1) // 2)
        keep_right = max(12, max_len - 1 - keep_left)
        return f"{s[:keep_left]}…{s[-keep_right:]}"

    def _make_unique(labels: list[str], max_len: int = 22) -> list[str]:
        seen: dict[str, int] = {}
        out: list[str] = []
        for lab in labels:
            base = lab
            if base not in seen:
                seen[base] = 1
                out.append(base)
                continue

            seen[base] += 1
            suffix = f"~{seen[base]}"
            allowed = max(1, max_len - len(suffix))
            trimmed = base[:allowed]
            out.append(f"{trimmed}{suffix}")
        return out

    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    sorted_results = sorted(results, key=lambda x: float(x["ece_score"]))
    top_models = sorted_results[: min(top_k, len(sorted_results))]

    model_names_full = [r.get("model", "") for r in top_models]

    # Keep fixed figure size; abbreviate labels instead to avoid overlap.
    # 1) apply any manual abbreviations
    # 2) apply a consistent length cap with ellipsis
    # 3) ensure uniqueness (adds ~2, ~3, ... if needed)
    label_max_len = 14
    model_names = _make_unique(
        [
            _abbreviate_reward_model_name(
                MODEL_NAME_ABBREVIATIONS.get(n, n),
                max_len=label_max_len,
            )
            for n in model_names_full
        ],
        max_len=label_max_len,
    )

    raw_ece_scores: list[list[float]] = []
    for r in top_models:
        raw = r.get("ece_score")

        if raw is None:
            raw_ece_scores.append([])
            continue

        if isinstance(raw, (int, float, np.floating)):
            raw_vals = [float(raw)]
        else:
            raw_vals = [float(v) for v in raw]

        raw_ece_scores.append(raw_vals)

    os.makedirs(save_folder, exist_ok=True)

    plt.style.use(["science", "no-latex"])
    fig, ax = plt.subplots(figsize=(4, 3))

    # Assign distinct colors per reward model.
    unique_models = list(model_names)
    if len(unique_models) <= 8:
        model_cmap = plt.get_cmap("Dark2")
    else:
        model_cmap = plt.get_cmap("tab20b")
    model_to_color = {
        m: model_cmap(i % model_cmap.N) for i, m in enumerate(unique_models)
    }
    model_alpha = 0.90

    # Draw the mean global ECE line for reference.
    mean_line = None
    all_raw_scores = [score for scores in raw_ece_scores for score in scores]
    if all_raw_scores:
        global_mean_ece = float(np.mean(all_raw_scores))
        mean_line = ax.axhline(
            global_mean_ece,
            color="#FF7F0E",
            linestyle="--",
            linewidth=2.0,
            label=f"Mean ECE: {global_mean_ece:.2f}",
            zorder=0,
        )

    # Add individual points (scatter / strip-plot style), colored only by reward model.
    rng = np.random.default_rng(0)
    for i, ys in enumerate(raw_ece_scores, start=1):
        if not ys:
            continue
        y = np.asarray(ys, dtype=float)
        # Light x-jitter so multiple raw points (if present) remain visible.
        jitter = rng.uniform(-0.06, 0.06, size=y.shape[0])
        x = i + jitter
        model_label = model_names[i - 1] if (i - 1) < len(model_names) else None
        c = model_to_color.get(model_label, "#4C78A8")
        ax.scatter(
            x,
            y,
            s=8.5**2,
            alpha=model_alpha,
            c=[c],
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )

    # Reward-model legend.
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor=model_to_color[m],
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=m,
            alpha=model_alpha,
        )
        for m in unique_models
    ]

    # Include the mean reference line in the same legend as reward models.
    if mean_line is not None:
        model_handles.append(mean_line)

    # ax.legend(
    #     handles=model_handles,
    #     title="Reward model",
    #     frameon=True,
    #     fontsize=12,
    #     title_fontsize=13,
    #     loc="lower right",
    #     #bbox_to_anchor=(1.02, 1.0),
    #     ncols=1 if len(unique_models) <= 12 else 2,
    # ).get_frame().set_alpha(0.95)

    # Match the boxplot axis styling used in `experiment_calibration.py`.
    # ax.set_title(
    #     f"Top-{top_k} Reward Models by ECE",
    #     fontsize=16,
    #     fontweight="semibold",
    #     pad=12,
    # )
    ax.set_ylabel("ECE", fontsize=13)
    # ax.set_xlabel("Reward model", fontsize=12)
    ax.set_ylim(bottom=-0.001)
    ax.set_xticks(range(1, len(model_names) + 1), labels=model_names)
    ax.tick_params(axis="x", labelsize=8)
    # Slightly stronger rotation + right alignment avoids overlaps for long model names.
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_facecolor("#f7f9fc")
    ax.set_xlim(0.5, len(raw_ece_scores) + 0.5)
    ax.set_ylim(0.1, 0.3)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.margins(x=0.04)
    fig.tight_layout()
    fig.savefig(
        os.path.join(save_folder, f"raw_ece_scores_top_{top_k}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )


def visualize_raw_ece_scores_compact(results, top_k=10, save_folder="rlhf_ece"):
    """Compact version of `visualize_raw_ece_scores`.

    Differences vs the default plot:
      - draws an in-figure legend (reward model colors + global mean)
      - removes x-axis ticks/labels to save horizontal space

    This is useful for small figures where x tick labels would dominate the layout.
    """

    if not results:
        return

    def _abbreviate_reward_model_name(model_name: str, max_len: int = 22) -> str:
        s = str(model_name)
        if "/" in s:
            s = s.rsplit("/", 1)[-1]

        s = re.sub(r"([-_]?instruct(ion)?)$", "-inst", s, flags=re.IGNORECASE)

        if len(s) <= max_len:
            return s

        keep_left = max(3, (max_len - 1) // 2)
        keep_right = max(12, max_len - 1 - keep_left)
        return f"{s[:keep_left]}…{s[-keep_right:]}"

    def _make_unique(labels: list[str], max_len: int = 22) -> list[str]:
        seen: dict[str, int] = {}
        out: list[str] = []
        for lab in labels:
            base = lab
            if base not in seen:
                seen[base] = 1
                out.append(base)
                continue

            seen[base] += 1
            suffix = f"~{seen[base]}"
            allowed = max(1, max_len - len(suffix))
            trimmed = base[:allowed]
            out.append(f"{trimmed}{suffix}")
        return out

    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    sorted_results = sorted(results, key=lambda x: float(x["ece_score"]))
    top_models = sorted_results[: min(top_k, len(sorted_results))]

    model_names_full = [r.get("model", "") for r in top_models]
    label_max_len = 14
    model_names = _make_unique(
        [
            _abbreviate_reward_model_name(
                MODEL_NAME_ABBREVIATIONS.get(n, n),
                max_len=label_max_len,
            )
            for n in model_names_full
        ],
        max_len=label_max_len,
    )

    raw_ece_scores: list[list[float]] = []
    for r in top_models:
        raw = r.get("ece_score")

        if raw is None:
            raw_ece_scores.append([])
            continue

        if isinstance(raw, (int, float, np.floating)):
            raw_vals = [float(raw)]
        else:
            raw_vals = [float(v) for v in raw]

        raw_ece_scores.append(raw_vals)

    os.makedirs(save_folder, exist_ok=True)

    plt.style.use(["science", "no-latex"])
    fig, ax = plt.subplots(figsize=(4, 3))

    unique_models = list(model_names)
    if len(unique_models) <= 8:
        model_cmap = plt.get_cmap("Dark2")
    else:
        model_cmap = plt.get_cmap("tab20b")
    model_to_color = {
        m: model_cmap(i % model_cmap.N) for i, m in enumerate(unique_models)
    }
    model_alpha = 0.90

    mean_line = None
    all_raw_scores = [score for scores in raw_ece_scores for score in scores]
    if all_raw_scores:
        global_mean_ece = float(np.mean(all_raw_scores))
        mean_line = ax.axhline(
            global_mean_ece,
            color="#FF7F0E",
            linestyle="--",
            linewidth=2.0,
            label=f"Mean ECE: {global_mean_ece:.2f}",
            zorder=0,
        )

    rng = np.random.default_rng(0)
    for i, ys in enumerate(raw_ece_scores, start=1):
        if not ys:
            continue
        y = np.asarray(ys, dtype=float)
        jitter = rng.uniform(-0.06, 0.06, size=y.shape[0])
        x = i + jitter
        model_label = model_names[i - 1] if (i - 1) < len(model_names) else None
        c = model_to_color.get(model_label, "#4C78A8")
        ax.scatter(
            x,
            y,
            s=7.5**2,
            alpha=model_alpha,
            c=[c],
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )

    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6.2,
            markerfacecolor=model_to_color[m],
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=m,
            alpha=model_alpha,
        )
        for m in unique_models
    ]
    if mean_line is not None:
        model_handles.append(mean_line)

    leg = ax.legend(
        handles=model_handles,
        frameon=True,
        fontsize=7.0,
        loc="lower right",
        ncols=2 if len(unique_models) <= 10 else 2,
        borderpad=0.4,
        handletextpad=0.5,
        columnspacing=0.8,
        labelspacing=0.25,
    )
    leg.get_frame().set_alpha(0.95)

    ax.set_ylabel("ECE", fontsize=11)
    ax.set_ylim(bottom=-0.001)

    # Remove x ticks/labels for a more compact layout.
    ax.set_xticks([])
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_facecolor("#f7f9fc")
    ax.set_xlim(0.5, len(raw_ece_scores) + 0.5)
    ax.set_ylim(0.1, 0.3)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.margins(x=0.02)
    fig.tight_layout(pad=0)
    fig.savefig(
        os.path.join(save_folder, f"raw_ece_scores_top_{top_k}_compact.pdf"),
        dpi=300,
        bbox_inches="tight",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze average absolute logit difference for top reward models"
    )
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=Path("current-rbv2-data.csv"),
        help="Path to leaderboard CSV (click 'Download CSV' at https://huggingface.co/spaces/allenai/reward-bench)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Limit to top N models (default: all)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default="cache_reward_bench",
        help="Cache directory for downloaded files",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top/bottom models to highlight in visualizations",
    )
    args = parser.parse_args()

    with open("label_ranking_datasets.json", "r") as f:
        label_ranking_datasets = json.load(f)

    if not args.leaderboard.exists():
        print(f"Leaderboard file not found: {args.leaderboard}")
        print(
            "Download it from https://huggingface.co/spaces/allenai/reward-bench (click 'Download CSV')"
        )
        return

    print(f"Parsing models from {args.leaderboard}...")
    all_models = parse_leaderboard(args.leaderboard)

    if args.top_n is None:
        models = all_models
        label = "All"
    else:
        models = all_models[: args.top_n]
        label = f"Top {args.top_n}"

    print(f"\n{label} models ({len(models)} total):")
    for (
        rank,
        factuality,
        precise_if,
        math,
        safety,
        focus,
        ties,
        model_name,
        model_type,
        leaderboard_score,
    ) in models:
        print(
            f"  {rank:2}. {model_name} ({model_type}) - Score: {leaderboard_score:.2f}"
        )

    fs = HfFileSystem()

    print("\nDownloading scores and computing metrics...")
    results = []

    for (
        rank,
        factuality,
        precise_if,
        math,
        safety,
        focus,
        ties,
        model_name,
        model_type,
        leaderboard_score,
    ) in models:
        file_path = find_model_file(fs, model_name)

        if file_path is None:
            print(
                f"  {rank:2}. {model_name}: No score data available (likely generative model)"
            )
            continue
        # Remove models which are not trained based on Bradley-Terry
        if "QRM" in model_name:
            print(f"  {rank:2}. {model_name}: Skipping due to known score data issues")
            continue

        scores_data = load_scores(file_path, args.cache_dir)

        # Remove Ties subset
        scores_data_filtered = {}
        for key in scores_data:
            scores_data_filtered[key] = [
                v
                for i, v in enumerate(scores_data[key])
                if scores_data["subset"][i] != "Ties"
            ]
        scores_data = scores_data_filtered

        first_score = scores_data["scores"][0]
        if first_score is None or (
            isinstance(first_score, list) and first_score[0] is None
        ):
            print(f"  {rank:2}. {model_name}: No logit scores (generative model)")
            continue

        # results_per_dataset = []
        # for group, indices_to_keep in label_ranking_datasets.items():
        #     scores_data_filtered = {}
        #     for key in scores_data:
        #         scores_data_filtered[key] = [
        #             v
        #             for i, v in enumerate(scores_data[key])
        #             if scores_data["num_correct"][i] == 1
        #             and scores_data["id"][i] in indices_to_keep
        #         ]

        (
            mean_margin,
            margin_to_mean,
            n_examples,
            ece,
            entropy,
            model_correct_alternatives,
        ) = compute_margins(scores_data)

        # results_per_dataset.append(
        #     (
        #         mean_margin,
        #         margin_to_mean,
        #         n_examples,
        #         ece,
        #         entropy,
        #         model_correct_alternatives,
        #         group,
        #     )
        # )
        # Aggregate results across datasets
        # mean_margin = np.mean([r[0] for r in results_per_dataset])
        # margin_to_mean = np.mean([r[1] for r in results_per_dataset])
        # n_examples = np.sum([r[2] for r in results_per_dataset])
        # ece = np.mean([r[3] for r in results_per_dataset])
        # model_groups = [r[6] for r in results_per_dataset]
        # raw_ece = [r[3] for r in results_per_dataset]
        # entropy = np.concatenate([r[4] for r in results_per_dataset])
        # raw_entropy = [r[4] for r in results_per_dataset]
        # model_correct_alternatives = np.concatenate([r[5] for r in results_per_dataset])
        # raw_model_correct_alternatives = [r[5] for r in results_per_dataset]

        results.append(
            {
                "rank": rank,
                "factuality": factuality,
                "precise_if": precise_if,
                "math": math,
                "safety": safety,
                "focus": focus,
                "ties": ties,
                "model": model_name,
                "model_type": model_type,
                "leaderboard_score": leaderboard_score,
                "mean_margin": mean_margin,
                "margin_to_mean": margin_to_mean,
                "ece_score": ece,
                "entropy": entropy,
                "model_accuracy": np.mean(model_correct_alternatives),
                "model_correct_alternatives": model_correct_alternatives,
                # "raw_ece": raw_ece,
                # "raw_entropy": raw_entropy,
                # "raw_model_correct_alternatives": raw_model_correct_alternatives,
                "n_examples": n_examples,
                # "model_groups": model_groups,
            }
        )

        print(
            f"  {rank:2}. {model_name}: mean-margin = {mean_margin:.4f}, margin-to-mean = {margin_to_mean:.4f} , ece_score = {ece:.4f}"
        )
    # Readjust Rank based on how many models had logit scores
    results = sorted(results, key=lambda x: x["rank"])
    for idx, r in enumerate(results):
        r["original_rank"] = r["rank"]
        r["rank"] = idx + 1

    print("\n" + "=" * 90)
    print(f"RESULTS: Margin Metrics ({len(results)} models with logit scores)")
    print("=" * 90)
    print(
        f"{'Rank':<5} {'Model':<50} {'Mean-Margin':<15} {'Margin-to-Mean':<15} {'ECE Score':<15} {'ECE Rank':<10} {'Accuracy':<10}"
    )
    print("-" * 90)
    ece_probs_chosen_over_rejected_sorted = sorted(
        results, key=lambda x: x["ece_score"]
    )
    ece_ranks = {
        r["model"]: idx + 1
        for idx, r in enumerate(ece_probs_chosen_over_rejected_sorted)
    }
    for r in results:
        print(
            f"{r['rank']:<5} {r['model']:<50} {r['mean_margin']:<15.4f} {r['margin_to_mean']:<15.4f} {r['ece_score']:<15.4f} {ece_ranks[r['model']]:<10} {r['model_accuracy']:<10.4f}"
        )
    print("=" * 90)

    # Compute Kendall Tau corelation between ECE rank and Leaderboard rank
    leaderboard_ranks = [r["rank"] for r in results]
    ece_rank_list = [ece_ranks[r["model"]] for r in results]
    kendall_tau, p_value = kendalltau(leaderboard_ranks, ece_rank_list)
    print(
        f"\nKendall Tau correlation between Leaderboard rank and ECE rank: {kendall_tau:.4f} (p-value: {p_value:.4e})"
    )

    # Compute Kendall Tau correlation between ECE rank and Leaderboard rank (only top 10 models)
    top_10_models = sorted(results, key=lambda x: x["rank"])[:10]
    leaderboard_ranks_top_10 = [r["rank"] for r in top_10_models]
    ece_ranks_top_10 = [ece_ranks[r["model"]] for r in top_10_models]
    kendall_tau_top_10, p_value_top_10 = kendalltau(
        leaderboard_ranks_top_10, ece_ranks_top_10
    )
    print(
        f"Kendall Tau correlation between Leaderboard rank and ECE rank (Top 10 models): {kendall_tau_top_10:.4f} (p-value: {p_value_top_10:.4e})"
    )

    # Visualize the raw_ece scores for the top_10 ece models
    visualize_raw_ece_scores(results, top_k=10, save_folder="rlhf_ece")
    visualize_raw_ece_scores_compact(results, top_k=10, save_folder="rlhf_ece")

    # Write results to csv
    df = pd.DataFrame(results)
    output_csv = os.path.join("rlhf_ece", f"rbv2_margin_metrics_{args.k}.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nResults written to {output_csv}")

    # Print out the top-10 ranked Models based on Leaderboard Score
    print("\nTop 10 Models based on Leaderboard Score:")
    top_10_leaderboard = sorted(results, key=lambda x: x["rank"])[:10]
    top_10_leaderboard_names = [r["model"] for r in top_10_leaderboard]
    print(" > ".join(top_10_leaderboard_names))

    # Print out the top-10 ranked Models based on ECE Score
    print("\nTop 10 Models based on ECE Score:")
    top_10_ece = sorted(results, key=lambda x: x["ece_score"])[:10]
    top_10_ece_names = [r["model"] for r in top_10_ece]
    print(" > ".join(top_10_ece_names))


if __name__ == "__main__":
    main()
