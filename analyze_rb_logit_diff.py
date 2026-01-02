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

from cal_pref.utils import calculate_binary_ece


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
    ece_top_1_pl = calculate_binary_ece(
        y_true=torch.tensor([1] * len(top_1_pl_probs)),
        y_prob=torch.tensor(top_1_pl_probs),
        n_bins=10,
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


def compute_correlations(results: list[dict]) -> dict[str, tuple[float, float]]:
    # Compute Spearman correlation between ECE and Leaderboard rank, Factuality, Precise If, Math, Safety, Focus, Ties
    from scipy.stats import spearmanr

    ece_ranks_list = [r["ece_score"] for r in results]
    leaderboard_ranks_list = [r["rank"] for r in results]
    factuality_ranks_list = [float(r["factuality"]) for r in results]
    precise_if_ranks_list = [float(r["precise_if"]) for r in results]
    math_ranks_list = [float(r["math"]) for r in results]
    safety_ranks_list = [float(r["safety"]) for r in results]
    focus_ranks_list = [float(r["focus"]) for r in results]
    ties_ranks_list = [float(r["ties"]) for r in results]

    # Compute Spearman correlations
    correlations = {
        "Leaderboard": spearmanr(ece_ranks_list, leaderboard_ranks_list),
        "Factuality": spearmanr(ece_ranks_list, factuality_ranks_list),
        "Precise If": spearmanr(ece_ranks_list, precise_if_ranks_list),
        "Math": spearmanr(ece_ranks_list, math_ranks_list),
        "Safety": spearmanr(ece_ranks_list, safety_ranks_list),
        "Focus": spearmanr(ece_ranks_list, focus_ranks_list),
        "Ties": spearmanr(ece_ranks_list, ties_ranks_list),
    }
    for key, (corr, pval) in correlations.items():
        print(
            f"Spearman correlation between ECE and {key}: {corr:.4f} (p-value: {pval:.4f})"
        )
    return correlations


def visualize_correlation(correlations: dict[str, tuple[float, float]], save_folder: str = "rlhf_ece"):
    # Visualize correlations using a cleaner, more readable bar chart
    import matplotlib.pyplot as plt

    labels = list(correlations.keys())
    corr_values = np.array([correlations[label][0] for label in labels], dtype=float)

    # Sort by absolute correlation (strongest effects first)
    order = np.argsort(np.abs(corr_values))[::-1]
    labels_sorted = [labels[i] for i in order]
    corr_sorted = corr_values[order]

    # Color by sign
    colors = ["#2E86AB" if v >= 0 else "#D1495B" for v in corr_sorted]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    bars = ax.barh(
        labels_sorted, corr_sorted, color=colors, edgecolor="white", linewidth=1.0
    )
    ax.invert_yaxis()
    ax.set_xlabel("Spearman correlation (ECE vs metric)")
    ax.set_title("Correlation between ECE and RewardBench metrics")
    ax.set_xlim(-1, 1)
    ax.axvline(0, color="black", linewidth=1.0, alpha=0.8)

    # Value labels at the end of each bar
    for bar, v in zip(bars, corr_sorted):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(
            x + (0.02 if v >= 0 else -0.02),
            y,
            f"{v:+.2f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=10,
            color="#222222",
        )

    ax.grid(True, axis="x", alpha=0.35)
    ax.grid(False, axis="y")

    fig.savefig(os.path.join(save_folder, "ece_correlation_bar_chart.png"), dpi=200, bbox_inches="tight")


def visualize_accuracy_rejection_curve(results: list[dict], k: int = 10, save_folder: str = "rlhf_ece"):
    """Plot accuracy on the remaining set after rejecting highest-entropy points.

    X-axis: fraction removed (reject p% highest entropy)
    Y-axis: accuracy on remaining examples
    Curves are colored by ECE (low=red, high=blue).
    """

    if not results:
        return

    eces = np.array([float(r["ece_score"]) for r in results], dtype=float)
    vmin = float(np.min(eces))
    vmax = float(np.max(eces))
    # Avoid a degenerate norm when all ECE values are identical
    if np.isclose(vmin, vmax):
        vmin -= 1e-12
        vmax += 1e-12

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.coolwarm_r  # low ECE=red, high ECE=blue
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Highlight extremes for intuition
    sorted_by_ece = sorted(results, key=lambda x: float(x["ece_score"]))
    top_k = min(k, len(sorted_by_ece))
    bottom_k = min(k, len(sorted_by_ece))
    low_ece_models = {r["model"] for r in sorted_by_ece[:top_k]}
    high_ece_models = {r["model"] for r in sorted_by_ece[-bottom_k:]}

    highlight_red = "#D62728"  # vivid red (best calibration)
    highlight_blue = "#1F77B4"  # vivid blue (worst calibration)
    muted_alpha = 0.25

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    # Plot in order of increasing ECE so "best calibrated" curves are easy to notice
    for r in sorted_by_ece:
        entropy = np.asarray(r["entropy"], dtype=float)
        correct = np.asarray(r["model_correct_alternatives"], dtype=float)

        if entropy.shape[0] == 0:
            continue
        if entropy.shape[0] != correct.shape[0]:
            raise ValueError(
                f"Entropy and correctness length mismatch for {r.get('model', '<unknown>')}: "
                f"{entropy.shape[0]} vs {correct.shape[0]}"
            )

        # Sort by entropy descending: removing the first k removes the highest-entropy points
        order = np.argsort(entropy)[::-1]
        correct_sorted = correct[order]

        # Accuracy on remaining after removing k points = mean(correct_sorted[k:])
        # Compute efficiently with suffix sums
        suffix_correct = np.cumsum(correct_sorted[::-1])[::-1]
        remaining = np.arange(len(correct_sorted), 0, -1)
        acc_remaining = suffix_correct / remaining
        frac_removed = np.arange(len(correct_sorted)) / len(correct_sorted)

        model_name = r["model"]
        ece_value = float(r["ece_score"])
        base_color = cmap(norm(ece_value))
        color = base_color
        lw = 1.6
        alpha = muted_alpha
        zorder = 1

        if model_name in low_ece_models:
            color = highlight_red
            lw = 3.0
            alpha = 0.95
            zorder = 3
        elif model_name in high_ece_models:
            color = highlight_blue
            lw = 3.0
            alpha = 0.95
            zorder = 3

        ax.plot(
            frac_removed,
            acc_remaining,
            color=color,
            lw=lw,
            alpha=alpha,
            zorder=zorder,
        )

    ax.set_xlabel("Fraction removed (reject highest entropy first)")
    ax.set_ylabel("Accuracy on remaining examples")
    ax.set_title("Accuracy–Rejection Curve colored by calibration (ECE)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.35)

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("ECE (lower = better)")

    # Small legend explaining highlights (avoid listing all models)

    handles = [
        Line2D([0], [0], color=highlight_red, lw=3.0, label=f"Lowest {top_k} ECE"),
        Line2D([0], [0], color=highlight_blue, lw=3.0, label=f"Highest {bottom_k} ECE"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left")

    fig.savefig(
        os.path.join(save_folder, f"accuracy_rejection_curve_entropy_{k}.png"), dpi=200, bbox_inches="tight"
    )


def visualize_accuracy_rejection_curve_by_rank(results: list[dict], k: int = 10, save_folder: str = "rlhf_ece"):
    """Plot accuracy–rejection curves colored by leaderboard rank.

    X-axis: fraction removed (reject p% highest entropy)
    Y-axis: accuracy on remaining examples
    Curves are colored by `rank` (lower=better).
    """

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if not results:
        return

    ranks = np.array([int(r["rank"]) for r in results], dtype=float)
    vmin = float(np.min(ranks))
    vmax = float(np.max(ranks))
    if np.isclose(vmin, vmax):
        vmin -= 1e-12
        vmax += 1e-12

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.coolwarm_r  # low rank (better)=red, high rank=blue
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Highlight extremes for intuition
    sorted_by_rank = sorted(results, key=lambda x: int(x["rank"]))
    top_k = min(k, len(sorted_by_rank))
    bottom_k = min(k, len(sorted_by_rank))
    top_models = {r["model"] for r in sorted_by_rank[:top_k]}
    bottom_models = {r["model"] for r in sorted_by_rank[-bottom_k:]}

    highlight_red = "#D62728"  # vivid red
    highlight_blue = "#1F77B4"  # vivid blue
    muted_alpha = 0.25

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    # Plot in order of increasing rank so the best-ranked models are prominent
    for r in sorted_by_rank:
        entropy = np.asarray(r["entropy"], dtype=float)
        correct = np.asarray(r["model_correct_alternatives"], dtype=float)

        if entropy.shape[0] == 0:
            continue
        if entropy.shape[0] != correct.shape[0]:
            raise ValueError(
                f"Entropy and correctness length mismatch for {r.get('model', '<unknown>')}: "
                f"{entropy.shape[0]} vs {correct.shape[0]}"
            )

        order = np.argsort(entropy)[::-1]
        correct_sorted = correct[order]

        suffix_correct = np.cumsum(correct_sorted[::-1])[::-1]
        remaining = np.arange(len(correct_sorted), 0, -1)
        acc_remaining = suffix_correct / remaining
        frac_removed = np.arange(len(correct_sorted)) / len(correct_sorted)

        model_name = r["model"]
        rank_value = float(r["rank"])
        base_color = cmap(norm(rank_value))
        color = base_color
        lw = 1.6
        alpha = muted_alpha
        zorder = 1

        if model_name in top_models:
            color = highlight_red
            lw = 3.0
            alpha = 0.95
            zorder = 3
        elif model_name in bottom_models:
            color = highlight_blue
            lw = 3.0
            alpha = 0.95
            zorder = 3

        ax.plot(
            frac_removed,
            acc_remaining,
            color=color,
            lw=lw,
            alpha=alpha,
            zorder=zorder,
        )

    ax.set_xlabel("Fraction removed (reject highest entropy first)")
    ax.set_ylabel("Accuracy on remaining examples")
    ax.set_title("Accuracy–Rejection Curve colored by leaderboard rank")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.35)

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Leaderboard rank (lower = better)")

    # Add a tiny legend explaining the highlights (avoid listing all models)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=highlight_red, lw=3.0, label=f"Top {top_k} ranks"),
        Line2D([0], [0], color=highlight_blue, lw=3.0, label=f"Worst {bottom_k} ranks"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left")

    fig.savefig(
        os.path.join(save_folder, f"accuracy_rejection_curve_entropy_rank_colored_{k}.png"),
        dpi=200,
        bbox_inches="tight",
    )


def visualize_accuracy_rejection_curve_by_factuality(results: list[dict], k: int = 10, save_folder: str = "rlhf_ece"):
    """Plot accuracy–rejection curves colored by factuality.

    X-axis: fraction removed (reject p% highest entropy)
    Y-axis: accuracy on remaining examples
    Curves are colored by factuality (lower = better).
    """

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if not results:
        return

    # Use an error-style score so that "lower = better" consistently.
    factuality_err = 100 - np.array(
        [float(r["factuality"]) for r in results], dtype=float
    )
    vmin = float(np.min(factuality_err))
    vmax = float(np.max(factuality_err))
    if np.isclose(vmin, vmax):
        vmin -= 1e-12
        vmax += 1e-12

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.coolwarm_r  # low (better)=red, high (worse)=blue
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Highlight extremes for intuition
    factuality_err_by_model = {
        r["model"]: 100 - float(r["factuality"]) for r in results
    }
    sorted_by_factuality = sorted(
        results, key=lambda x: factuality_err_by_model[x["model"]]
    )
    top_k = min(k, len(sorted_by_factuality))
    bottom_k = min(k, len(sorted_by_factuality))
    top_models = {r["model"] for r in sorted_by_factuality[:top_k]}
    bottom_models = {r["model"] for r in sorted_by_factuality[-bottom_k:]}

    highlight_red = "#D62728"  # vivid red
    highlight_blue = "#1F77B4"  # vivid blue
    muted_alpha = 0.25

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    # Plot in order of improving factuality so the best models are prominent
    for r in sorted_by_factuality:
        entropy = np.asarray(r["entropy"], dtype=float)
        correct = np.asarray(r["model_correct_alternatives"], dtype=float)

        if entropy.shape[0] == 0:
            continue
        if entropy.shape[0] != correct.shape[0]:
            raise ValueError(
                f"Entropy and correctness length mismatch for {r.get('model', '<unknown>')}: "
                f"{entropy.shape[0]} vs {correct.shape[0]}"
            )

        order = np.argsort(entropy)[::-1]
        correct_sorted = correct[order]

        suffix_correct = np.cumsum(correct_sorted[::-1])[::-1]
        remaining = np.arange(len(correct_sorted), 0, -1)
        acc_remaining = suffix_correct / remaining
        frac_removed = np.arange(len(correct_sorted)) / len(correct_sorted)

        model_name = r["model"]
        factuality_value = float(factuality_err_by_model[model_name])
        base_color = cmap(norm(factuality_value))
        color = base_color
        lw = 1.6
        alpha = muted_alpha
        zorder = 1

        if model_name in top_models:
            color = highlight_red
            lw = 3.0
            alpha = 0.95
            zorder = 3
        elif model_name in bottom_models:
            color = highlight_blue
            lw = 3.0
            alpha = 0.95
            zorder = 3

        ax.plot(
            frac_removed,
            acc_remaining,
            color=color,
            lw=lw,
            alpha=alpha,
            zorder=zorder,
        )

    ax.set_xlabel("Fraction removed (reject highest entropy first)")
    ax.set_ylabel("Accuracy on remaining examples")
    ax.set_title("Accuracy–Rejection Curve colored by factuality rank")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.35)

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Factuality Score (lower = better)")

    # Add a tiny legend explaining the highlights (avoid listing all models)
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0], [0], color=highlight_red, lw=3.0, label=f"Lowest {top_k} factuality"
        ),
        Line2D(
            [0],
            [0],
            color=highlight_blue,
            lw=3.0,
            label=f"Highest {bottom_k} factuality",
        ),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left")

    fig.savefig(
        os.path.join(save_folder, f"accuracy_rejection_curve_entropy_factuality_colored_{k}.png"),
        dpi=200,
        bbox_inches="tight",
    )


def visualize_accuracy_rejection_curve_by_math(results: list[dict], k: int = 10, save_folder: str = "rlhf_ece"):
    """Plot accuracy–rejection curves colored by math.

    X-axis: fraction removed (reject p% highest entropy)
    Y-axis: accuracy on remaining examples
    Curves are colored by math (lower = better).
    """

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if not results:
        return

    # Use an error-style score so that "lower = better" consistently.
    math_err = 100 - np.array([float(r["math"]) for r in results], dtype=float)
    vmin = float(np.min(math_err))
    vmax = float(np.max(math_err))
    if np.isclose(vmin, vmax):
        vmin -= 1e-12
        vmax += 1e-12

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.coolwarm_r  # low (better)=red, high (worse)=blue
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Highlight extremes for intuition
    math_err_by_model = {r["model"]: 100 - float(r["math"]) for r in results}
    sorted_by_math = sorted(results, key=lambda x: math_err_by_model[x["model"]])
    top_k = min(k, len(sorted_by_math))
    bottom_k = min(k, len(sorted_by_math))
    top_models = {r["model"] for r in sorted_by_math[:top_k]}
    bottom_models = {r["model"] for r in sorted_by_math[-bottom_k:]}

    highlight_red = "#D62728"  # vivid red
    highlight_blue = "#1F77B4"  # vivid blue
    muted_alpha = 0.25

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    # Plot in order of improving math so the best models are prominent
    for r in sorted_by_math:
        entropy = np.asarray(r["entropy"], dtype=float)
        correct = np.asarray(r["model_correct_alternatives"], dtype=float)

        if entropy.shape[0] == 0:
            continue
        if entropy.shape[0] != correct.shape[0]:
            raise ValueError(
                f"Entropy and correctness length mismatch for {r.get('model', '<unknown>')}: "
                f"{entropy.shape[0]} vs {correct.shape[0]}"
            )

        order = np.argsort(entropy)[::-1]
        correct_sorted = correct[order]

        suffix_correct = np.cumsum(correct_sorted[::-1])[::-1]
        remaining = np.arange(len(correct_sorted), 0, -1)
        acc_remaining = suffix_correct / remaining
        frac_removed = np.arange(len(correct_sorted)) / len(correct_sorted)

        model_name = r["model"]
        math_value = float(math_err_by_model[model_name])
        base_color = cmap(norm(math_value))
        color = base_color
        lw = 1.6
        alpha = muted_alpha
        zorder = 1

        if model_name in top_models:
            color = highlight_red
            lw = 3.0
            alpha = 0.95
            zorder = 3
        elif model_name in bottom_models:
            color = highlight_blue
            lw = 3.0
            alpha = 0.95
            zorder = 3

        ax.plot(
            frac_removed,
            acc_remaining,
            color=color,
            lw=lw,
            alpha=alpha,
            zorder=zorder,
        )

    ax.set_xlabel("Fraction removed (reject highest entropy first)")
    ax.set_ylabel("Accuracy on remaining examples")
    ax.set_title("Accuracy–Rejection Curve colored by math rank")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.35)

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Math Score (lower = better)")

    # Add a tiny legend explaining the highlights (avoid listing all models)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=highlight_red, lw=3.0, label=f"Lowest {top_k} math"),
        Line2D(
            [0], [0], color=highlight_blue, lw=3.0, label=f"Highest {bottom_k} math"
        ),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left")

    fig.savefig(
        os.path.join(save_folder, f"accuracy_rejection_curve_entropy_math_colored_{k}.png"),
        dpi=200,
        bbox_inches="tight",
    )


def visualize_accuracy_rejection_curve_by_safety(results: list[dict], k: int = 10, save_folder: str = "rlhf_ece"):
    """Plot accuracy–rejection curves colored by safety.

    X-axis: fraction removed (reject p% highest entropy)
    Y-axis: accuracy on remaining examples
    Curves are colored by safety (lower = better).
    """

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if not results:
        return

    safety_err = 100 - np.array([float(r["safety"]) for r in results], dtype=float)
    vmin = float(np.min(safety_err))
    vmax = float(np.max(safety_err))
    if np.isclose(vmin, vmax):
        vmin -= 1e-12
        vmax += 1e-12

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.coolwarm_r  # low (better)=red, high (worse)=blue
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    safety_err_by_model = {r["model"]: 100 - float(r["safety"]) for r in results}
    sorted_by_safety = sorted(results, key=lambda x: safety_err_by_model[x["model"]])
    top_k = min(k, len(sorted_by_safety))
    bottom_k = min(k, len(sorted_by_safety))
    top_models = {r["model"] for r in sorted_by_safety[:top_k]}
    bottom_models = {r["model"] for r in sorted_by_safety[-bottom_k:]}

    highlight_red = "#D62728"
    highlight_blue = "#1F77B4"
    muted_alpha = 0.25

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    for r in sorted_by_safety:
        entropy = np.asarray(r["entropy"], dtype=float)
        correct = np.asarray(r["model_correct_alternatives"], dtype=float)

        if entropy.shape[0] == 0:
            continue
        if entropy.shape[0] != correct.shape[0]:
            raise ValueError(
                f"Entropy and correctness length mismatch for {r.get('model', '<unknown>')}: "
                f"{entropy.shape[0]} vs {correct.shape[0]}"
            )

        order = np.argsort(entropy)[::-1]
        correct_sorted = correct[order]

        suffix_correct = np.cumsum(correct_sorted[::-1])[::-1]
        remaining = np.arange(len(correct_sorted), 0, -1)
        acc_remaining = suffix_correct / remaining
        frac_removed = np.arange(len(correct_sorted)) / len(correct_sorted)

        model_name = r["model"]
        safety_value = float(safety_err_by_model[model_name])
        color = cmap(norm(safety_value))
        lw = 1.6
        alpha = muted_alpha
        zorder = 1

        if model_name in top_models:
            color = highlight_red
            lw = 3.0
            alpha = 0.95
            zorder = 3
        elif model_name in bottom_models:
            color = highlight_blue
            lw = 3.0
            alpha = 0.95
            zorder = 3

        ax.plot(
            frac_removed, acc_remaining, color=color, lw=lw, alpha=alpha, zorder=zorder
        )

    ax.set_xlabel("Fraction removed (reject highest entropy first)")
    ax.set_ylabel("Accuracy on remaining examples")
    ax.set_title("Accuracy–Rejection Curve colored by safety")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.35)

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Safety Score (lower = better)")

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=highlight_red, lw=3.0, label=f"Lowest {top_k} safety"),
        Line2D(
            [0], [0], color=highlight_blue, lw=3.0, label=f"Highest {bottom_k} safety"
        ),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left")

    fig.savefig(
        os.path.join(save_folder, f"accuracy_rejection_curve_entropy_safety_colored_{k}.png"),
        dpi=200,
        bbox_inches="tight",
    )


def visualize_accuracy_rejection_curve_by_focus(results: list[dict], k: int = 10, save_folder: str = "rlhf_ece"):
    """Plot accuracy–rejection curves colored by focus.

    X-axis: fraction removed (reject p% highest entropy)
    Y-axis: accuracy on remaining examples
    Curves are colored by focus (lower = better).
    """

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if not results:
        return

    focus_err = 100 - np.array([float(r["focus"]) for r in results], dtype=float)
    vmin = float(np.min(focus_err))
    vmax = float(np.max(focus_err))
    if np.isclose(vmin, vmax):
        vmin -= 1e-12
        vmax += 1e-12

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.coolwarm_r
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    focus_err_by_model = {r["model"]: 100 - float(r["focus"]) for r in results}
    sorted_by_focus = sorted(results, key=lambda x: focus_err_by_model[x["model"]])
    top_k = min(k, len(sorted_by_focus))
    bottom_k = min(k, len(sorted_by_focus))
    top_models = {r["model"] for r in sorted_by_focus[:top_k]}
    bottom_models = {r["model"] for r in sorted_by_focus[-bottom_k:]}

    highlight_red = "#D62728"
    highlight_blue = "#1F77B4"
    muted_alpha = 0.25

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    for r in sorted_by_focus:
        entropy = np.asarray(r["entropy"], dtype=float)
        correct = np.asarray(r["model_correct_alternatives"], dtype=float)

        if entropy.shape[0] == 0:
            continue
        if entropy.shape[0] != correct.shape[0]:
            raise ValueError(
                f"Entropy and correctness length mismatch for {r.get('model', '<unknown>')}: "
                f"{entropy.shape[0]} vs {correct.shape[0]}"
            )

        order = np.argsort(entropy)[::-1]
        correct_sorted = correct[order]

        suffix_correct = np.cumsum(correct_sorted[::-1])[::-1]
        remaining = np.arange(len(correct_sorted), 0, -1)
        acc_remaining = suffix_correct / remaining
        frac_removed = np.arange(len(correct_sorted)) / len(correct_sorted)

        model_name = r["model"]
        focus_value = float(focus_err_by_model[model_name])
        color = cmap(norm(focus_value))
        lw = 1.6
        alpha = muted_alpha
        zorder = 1

        if model_name in top_models:
            color = highlight_red
            lw = 3.0
            alpha = 0.95
            zorder = 3
        elif model_name in bottom_models:
            color = highlight_blue
            lw = 3.0
            alpha = 0.95
            zorder = 3

        ax.plot(
            frac_removed, acc_remaining, color=color, lw=lw, alpha=alpha, zorder=zorder
        )

    ax.set_xlabel("Fraction removed (reject highest entropy first)")
    ax.set_ylabel("Accuracy on remaining examples")
    ax.set_title("Accuracy–Rejection Curve colored by focus")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.35)

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Focus Score (lower = better)")

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=highlight_red, lw=3.0, label=f"Lowest {top_k} focus"),
        Line2D(
            [0], [0], color=highlight_blue, lw=3.0, label=f"Highest {bottom_k} focus"
        ),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left")

    fig.savefig(
        os.path.join(save_folder, f"accuracy_rejection_curve_entropy_focus_colored_{k}.png"),
        dpi=200,
        bbox_inches="tight",
    )


def visualize_accuracy_rejection_curve_by_ties(results: list[dict], k: int = 10, save_folder: str = "rlhf_ece"):
    """Plot accuracy–rejection curves colored by ties.

    X-axis: fraction removed (reject p% highest entropy)
    Y-axis: accuracy on remaining examples
    Curves are colored by ties (lower = better).
    """

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if not results:
        return

    ties_err = 100 - np.array([float(r["ties"]) for r in results], dtype=float)
    vmin = float(np.min(ties_err))
    vmax = float(np.max(ties_err))
    if np.isclose(vmin, vmax):
        vmin -= 1e-12
        vmax += 1e-12

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.coolwarm_r
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    ties_err_by_model = {r["model"]: 100 - float(r["ties"]) for r in results}
    sorted_by_ties = sorted(results, key=lambda x: ties_err_by_model[x["model"]])
    top_k = min(k, len(sorted_by_ties))
    bottom_k = min(k, len(sorted_by_ties))
    top_models = {r["model"] for r in sorted_by_ties[:top_k]}
    bottom_models = {r["model"] for r in sorted_by_ties[-bottom_k:]}

    highlight_red = "#D62728"
    highlight_blue = "#1F77B4"
    muted_alpha = 0.25

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    for r in sorted_by_ties:
        entropy = np.asarray(r["entropy"], dtype=float)
        correct = np.asarray(r["model_correct_alternatives"], dtype=float)

        if entropy.shape[0] == 0:
            continue
        if entropy.shape[0] != correct.shape[0]:
            raise ValueError(
                f"Entropy and correctness length mismatch for {r.get('model', '<unknown>')}: "
                f"{entropy.shape[0]} vs {correct.shape[0]}"
            )

        order = np.argsort(entropy)[::-1]
        correct_sorted = correct[order]

        suffix_correct = np.cumsum(correct_sorted[::-1])[::-1]
        remaining = np.arange(len(correct_sorted), 0, -1)
        acc_remaining = suffix_correct / remaining
        frac_removed = np.arange(len(correct_sorted)) / len(correct_sorted)

        model_name = r["model"]
        ties_value = float(ties_err_by_model[model_name])
        color = cmap(norm(ties_value))
        lw = 1.6
        alpha = muted_alpha
        zorder = 1

        if model_name in top_models:
            color = highlight_red
            lw = 3.0
            alpha = 0.95
            zorder = 3
        elif model_name in bottom_models:
            color = highlight_blue
            lw = 3.0
            alpha = 0.95
            zorder = 3

        ax.plot(
            frac_removed, acc_remaining, color=color, lw=lw, alpha=alpha, zorder=zorder
        )

    ax.set_xlabel("Fraction removed (reject highest entropy first)")
    ax.set_ylabel("Accuracy on remaining examples")
    ax.set_title("Accuracy–Rejection Curve colored by ties")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.35)

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Ties Score (lower = better)")

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=highlight_red, lw=3.0, label=f"Lowest {top_k} ties"),
        Line2D(
            [0], [0], color=highlight_blue, lw=3.0, label=f"Highest {bottom_k} ties"
        ),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left")

    fig.savefig(
        os.path.join(save_folder, f"accuracy_rejection_curve_entropy_ties_colored_{k}.png"),
        dpi=200,
        bbox_inches="tight",
    )


def _accuracy_rejection_curve(entropy: np.ndarray, correct: np.ndarray):
    """Compute accuracy–rejection curve arrays for a single model.

    Returns:
        frac_removed: shape (n,)
        acc_remaining: shape (n,)
    """
    entropy = np.asarray(entropy, dtype=float)
    correct = np.asarray(correct, dtype=float)

    if entropy.shape[0] == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if entropy.shape[0] != correct.shape[0]:
        raise ValueError(
            f"Entropy and correctness length mismatch: {entropy.shape[0]} vs {correct.shape[0]}"
        )

    order = np.argsort(entropy)[::-1]
    correct_sorted = correct[order]

    suffix_correct = np.cumsum(correct_sorted[::-1])[::-1]
    remaining = np.arange(len(correct_sorted), 0, -1)
    acc_remaining = suffix_correct / remaining
    frac_removed = np.arange(len(correct_sorted)) / len(correct_sorted)
    return frac_removed, acc_remaining


def _interp_curve_to_grid(
    frac_removed: np.ndarray, acc_remaining: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Interpolate an accuracy–rejection curve onto a shared grid."""
    frac_removed = np.asarray(frac_removed, dtype=float)
    acc_remaining = np.asarray(acc_remaining, dtype=float)
    grid = np.asarray(grid, dtype=float)

    if frac_removed.size == 0:
        return np.full_like(grid, np.nan, dtype=float)

    # Ensure increasing x for interpolation.
    order = np.argsort(frac_removed)
    x = frac_removed[order]
    y = acc_remaining[order]

    # Clamp ends to avoid NaNs outside range.
    return np.interp(grid, x, y, left=y[0], right=y[-1])


def quantify_grouping_quality(
    results: list[dict],
    k: int = 10,
    grid_size: int = 101,
    output_csv: str = "accuracy_rejection_grouping_quality.csv",
    output_plot: str = "accuracy_rejection_curve_group_variance_topk.png",
    output_side_by_side_plot: str = "accuracy_rejection_curve_group_variance_and_separation_topk.png",
):
    """Quantify how well different metrics group accuracy–rejection curves.

    For each metric, forms top-k (best) and bottom-k (worst) groups, then computes:
      - within-group variance vs rejection fraction (and its mean over the grid)
      - between-group separation (AUC of |mean_top(p) - mean_bottom(p)| over p∈[0,1])
      - a simple quality ratio: separation / (avg within-var + eps)

        Saves:
            - a CSV summary
            - a variance curve plot (top-k group)
            - a separation curve plot (|mean_top(p) - mean_bottom(p)|)
            - a side-by-side figure showing variance and separation
    """

    if not results:
        return

    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")

    grid_size = int(grid_size)
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")

    grid = np.linspace(0.0, 1.0, grid_size)

    # Precompute curve for each model on a shared grid.
    curves_by_model: dict[str, np.ndarray] = {}
    for r in results:
        model = r.get("model")
        if model is None:
            continue
        frac, acc = _accuracy_rejection_curve(
            r["entropy"], r["model_correct_alternatives"]
        )
        curves_by_model[model] = _interp_curve_to_grid(frac, acc, grid)

    def _metric_value(r: dict, metric_key: str) -> float:
        if metric_key == "ECE":
            return float(r["ece_score"])
        if metric_key == "Leaderboard":
            return float(r["rank"])
        # RewardBench category scores appear to be "higher is better"; convert to an error-like score.
        if metric_key == "Factuality":
            return 100.0 - float(r["factuality"])
        if metric_key == "Math":
            return 100.0 - float(r["math"])
        if metric_key == "Safety":
            return 100.0 - float(r["safety"])
        if metric_key == "Focus":
            return 100.0 - float(r["focus"])
        if metric_key == "Ties":
            return 100.0 - float(r["ties"])
        raise KeyError(metric_key)

    metrics = ["ECE", "Leaderboard", "Factuality", "Math", "Safety", "Focus", "Ties"]
    eps = 1e-12

    summary_rows = []
    var_curves_top: dict[str, np.ndarray] = {}
    sep_curves: dict[str, np.ndarray] = {}

    for metric in metrics:
        sortable = []
        for r in results:
            model = r.get("model")
            if model not in curves_by_model:
                continue
            curve = curves_by_model[model]
            if np.all(np.isnan(curve)):
                continue
            try:
                val = _metric_value(r, metric)
            except Exception:
                continue
            sortable.append((val, model))

        if len(sortable) == 0:
            continue

        sortable.sort(key=lambda x: x[0])  # lower is better
        k_eff = min(k, len(sortable))
        top_models = [m for _, m in sortable[:k_eff]]
        bottom_models = [m for _, m in sortable[-k_eff:]]

        top_mat = np.vstack([curves_by_model[m] for m in top_models])
        bottom_mat = np.vstack([curves_by_model[m] for m in bottom_models])

        # Variance across models at each grid point.
        # If k_eff == 1, variance is 0 (np.var handles this).
        var_top = np.nanvar(top_mat, axis=0)
        var_bottom = np.nanvar(bottom_mat, axis=0)
        var_curves_top[metric] = var_top

        mean_var_top = float(np.nanmean(var_top))
        mean_var_bottom = float(np.nanmean(var_bottom))

        mean_top = np.nanmean(top_mat, axis=0)
        mean_bottom = np.nanmean(bottom_mat, axis=0)
        sep_curve = np.abs(mean_top - mean_bottom)
        sep_curves[metric] = sep_curve
        # Area between the two mean curves (absolute difference integrated over rejection fraction).
        separation = float(np.trapezoid(sep_curve, grid))

        avg_within = 0.5 * (mean_var_top + mean_var_bottom)
        quality = separation / (avg_within + eps)

        summary_rows.append(
            {
                "metric": metric,
                "k": k_eff,
                "mean_within_var_topk": mean_var_top,
                "mean_within_var_bottomk": mean_var_bottom,
                "auc_abs_diff_mean_curves": separation,
                "quality_ratio": quality,
            }
        )

    if summary_rows:
        dfq = pd.DataFrame(summary_rows).sort_values("quality_ratio", ascending=False)
        dfq.to_csv(output_csv, index=False)
        print(f"\nGrouping-quality summary written to {output_csv}")
        print(dfq.to_string(index=False))
        print("Latex table format:")
        print(dfq.to_latex(index=False, float_format="%.4f"))

    # Plot variance-vs-fraction-removed for top-k groups across metrics.
    if var_curves_top:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)

        ece_color = "#D1495B"  # standout (distinct from the blue family)
        non_ece = [m for m in metrics if m != "ECE"]
        blue_shades = plt.cm.Blues(np.linspace(0.25, 0.90, max(len(non_ece), 1)))
        non_ece_colors = {m: blue_shades[i] for i, m in enumerate(non_ece)}

        for metric in metrics:
            if metric not in var_curves_top:
                continue
            is_ece = metric == "ECE"
            ax.plot(
                grid,
                var_curves_top[metric],
                lw=3.2 if is_ece else 2.2,
                alpha=1.0 if is_ece else 0.9,
                color=(ece_color if is_ece else non_ece_colors.get(metric, "#4C78A8")),
                label=metric,
            )

        ax.set_xlabel("Fraction removed (reject highest entropy first)")
        ax.set_ylabel("Within-group variance of accuracy (top-k)")
        ax.set_title(f"Top-{k} curve tightness by metric")
        ax.set_xlim(0, 1)
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")
        ax.legend(frameon=False, ncols=3, loc="upper left")
        fig.savefig(output_plot, dpi=200, bbox_inches="tight")
        print(f"Variance plot written to {output_plot}")

    # Plot separation curve (pointwise gap between best and worst mean curves).
    if sep_curves:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)

        ece_color = "#D1495B"
        non_ece = [m for m in metrics if m != "ECE"]
        blue_shades = plt.cm.Blues(np.linspace(0.25, 0.90, max(len(non_ece), 1)))
        non_ece_colors = {m: blue_shades[i] for i, m in enumerate(non_ece)}

        for metric in metrics:
            if metric not in sep_curves:
                continue
            is_ece = metric == "ECE"
            ax.plot(
                grid,
                sep_curves[metric],
                lw=3.2 if is_ece else 2.2,
                alpha=1.0 if is_ece else 0.9,
                color=(ece_color if is_ece else non_ece_colors.get(metric, "#4C78A8")),
                label=metric,
            )

        ax.set_xlabel("Fraction removed (reject highest entropy first)")
        ax.set_ylabel("Pointwise separation  |Δ mean accuracy|")
        ax.set_title(f"Top vs bottom separation by metric (k={k})")
        ax.set_xlim(0, 1)
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")
        ax.legend(frameon=False, ncols=3, loc="upper left")
        separation_plot = output_plot.replace("variance", "separation")
        fig.savefig(separation_plot, dpi=200, bbox_inches="tight")
        print(f"Separation plot written to {separation_plot}")

    # Side-by-side figure for quick comparison (variance vs separation).
    if var_curves_top and sep_curves:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, (ax_var, ax_sep) = plt.subplots(
            1, 2, figsize=(15.5, 6.0), constrained_layout=True
        )

        ece_color = "#D1495B"
        non_ece = [m for m in metrics if m != "ECE"]
        blue_shades = plt.cm.Blues(np.linspace(0.25, 0.90, max(len(non_ece), 1)))
        non_ece_colors = {m: blue_shades[i] for i, m in enumerate(non_ece)}

        for metric in metrics:
            if metric in var_curves_top:
                is_ece = metric == "ECE"
                ax_var.plot(
                    grid,
                    var_curves_top[metric],
                    lw=3.2 if is_ece else 2.2,
                    alpha=1.0 if is_ece else 0.9,
                    color=(
                        ece_color if is_ece else non_ece_colors.get(metric, "#4C78A8")
                    ),
                    label=metric,
                )
            if metric in sep_curves:
                is_ece = metric == "ECE"
                ax_sep.plot(
                    grid,
                    sep_curves[metric],
                    lw=3.2 if is_ece else 2.2,
                    alpha=1.0 if is_ece else 0.9,
                    color=(
                        ece_color if is_ece else non_ece_colors.get(metric, "#4C78A8")
                    ),
                    label=metric,
                )

        ax_var.set_xlabel("Fraction removed")
        ax_var.set_ylabel("Variance (top-k)")
        ax_var.set_title("Tightness")
        ax_var.set_xlim(0, 1)
        ax_var.grid(True, axis="y", alpha=0.25)
        ax_var.grid(False, axis="x")

        ax_sep.set_xlabel("Fraction removed")
        ax_sep.set_ylabel("|Δ mean|")
        ax_sep.set_title("Separation")
        ax_sep.set_xlim(0, 1)
        ax_sep.grid(True, axis="y", alpha=0.25)
        ax_sep.grid(False, axis="x")

        # Shared legend below both plots for readability.
        handles, labels = ax_sep.get_legend_handles_labels()
        if handles:
            # Reserve a bit more bottom space so the legend isn't squished.
            fig.subplots_adjust(bottom=0.20)
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.2),
                ncols=4,
                frameon=False,
            )

        fig.savefig(output_side_by_side_plot, dpi=200, bbox_inches="tight")
        print(f"Side-by-side plot written to {output_side_by_side_plot}")


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

    with open('label_ranking_datasets.json', 'r') as f:
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
            print(
                f"  {rank:2}. {model_name}: Skipping due to known score data issues"
            )
            continue

        scores_data = load_scores(file_path, args.cache_dir)

        first_score = scores_data["scores"][0]
        if first_score is None or (
            isinstance(first_score, list) and first_score[0] is None
        ):
            print(f"  {rank:2}. {model_name}: No logit scores (generative model)")
            continue

        results_per_dataset = []
        for indices_to_keep in label_ranking_datasets.values():
            scores_data_filtered = {}
            for key in scores_data:
                scores_data_filtered[key] = [
                    v
                    for i, v in enumerate(scores_data[key])
                    if scores_data["num_correct"][i] == 1
                    and scores_data["id"][i] in indices_to_keep
                ]

            (
                mean_margin,
                margin_to_mean,
                n_examples,
                ece,
                entropy,
                model_correct_alternatives,
            ) = compute_margins(scores_data_filtered)
            results_per_dataset.append((
                mean_margin,
                margin_to_mean,
                n_examples,
                ece,
                entropy,
                model_correct_alternatives,
            ))
        # Aggregate results across datasets
        mean_margin = np.mean([r[0] for r in results_per_dataset])
        margin_to_mean = np.mean([r[1] for r in results_per_dataset])
        n_examples = np.sum([r[2] for r in results_per_dataset])
        ece = np.mean([r[3] for r in results_per_dataset])
        raw_ece = [r[3] for r in results_per_dataset]
        entropy = np.concatenate([r[4] for r in results_per_dataset])
        raw_entropy =[r[4] for r in results_per_dataset]
        model_correct_alternatives = np.concatenate([r[5] for r in results_per_dataset])
        raw_model_correct_alternatives = [r[5] for r in results_per_dataset]

        

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
                "raw_ece": raw_ece,
                "raw_entropy": raw_entropy,
                "raw_model_correct_alternatives": raw_model_correct_alternatives,
                "n_examples": n_examples,
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

    # Compute and visualize correlations
    save_folder = "rlhf_ece"
    os.makedirs(save_folder, exist_ok=True)
    correlations = compute_correlations(results)
    visualize_correlation(correlations, save_folder=save_folder)

    # Visualize accuracy–rejection curve (reject highest entropy first), colored by ECE
    visualize_accuracy_rejection_curve(results, k=args.k, save_folder=save_folder)

    # Visualize accuracy–rejection curve colored by leaderboard rank
    visualize_accuracy_rejection_curve_by_rank(results, k=args.k, save_folder=save_folder)

    # Visualize accuracy–rejection curve colored by factuality
    visualize_accuracy_rejection_curve_by_factuality(results, k=args.k, save_folder=save_folder)

    # Visualize accuracy–rejection curve colored by math
    visualize_accuracy_rejection_curve_by_math(results, k=args.k, save_folder=save_folder)

    # Visualize accuracy–rejection curve colored by safety
    visualize_accuracy_rejection_curve_by_safety(results, k=args.k, save_folder=save_folder)

    # Visualize accuracy–rejection curve colored by focus
    visualize_accuracy_rejection_curve_by_focus(results, k=args.k, save_folder=save_folder)

    # Visualize accuracy–rejection curve colored by ties
    visualize_accuracy_rejection_curve_by_ties(results, k=args.k, save_folder=save_folder)

    # Quantify how well each metric groups accuracy–rejection curves
    quantify_grouping_quality(
        results,
        k=args.k,
        grid_size=101,
        output_csv=os.path.join(save_folder, f"accuracy_rejection_grouping_quality_k{args.k}.csv"),
        output_plot=os.path.join(save_folder, f"accuracy_rejection_curve_group_variance_topk_k{args.k}.png"),
        output_side_by_side_plot=os.path.join(save_folder, f"accuracy_rejection_curve_group_variance_and_separation_topk_k{args.k}.png"),
    )

    # Write results to csv
    df = pd.DataFrame(results)
    output_csv = os.path.join(save_folder, f"rbv2_margin_metrics_{args.k}.csv")
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
