#!/usr/bin/env python3
"""
Analyze average logit difference for top rewardbench models.

Computes: mean(|chosen_score - rejected_score|) across all examples and rejected responses.
"""

import argparse
import csv
import json
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


def visualize_correlation(correlations: dict[str, tuple[float, float]]):
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

    fig.savefig("ece_correlation_bar_chart.png", dpi=200, bbox_inches="tight")


def visualize_accuracy_rejection_curve(results: list[dict], k: int = 10):
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
        f"accuracy_rejection_curve_entropy_{k}.png", dpi=200, bbox_inches="tight"
    )


def visualize_accuracy_rejection_curve_by_rank(results: list[dict], k: int = 10):
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
        f"accuracy_rejection_curve_entropy_rank_colored_{k}.png",
        dpi=200,
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
    args = parser.parse_args()

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
        # if model_name == "nicolinho/QRM-Gemma-2-27B":
        #     print(
        #         f"  {rank:2}. {model_name}: Skipping due to known score data issues"
        #     )
        #     continue

        scores_data = load_scores(file_path, args.cache_dir)

        first_score = scores_data["scores"][0]
        if first_score is None or (
            isinstance(first_score, list) and first_score[0] is None
        ):
            print(f"  {rank:2}. {model_name}: No logit scores (generative model)")
            continue

        scores_data_filtered = {}
        for key in scores_data:
            scores_data_filtered[key] = [
                v
                for i, v in enumerate(scores_data[key])
                if scores_data["num_correct"][i] == 1
            ]
        scores_data = scores_data_filtered
        (
            mean_margin,
            margin_to_mean,
            n_examples,
            ece,
            entropy,
            model_correct_alternatives,
        ) = compute_margins(scores_data)

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
        f"{'Rank':<5} {'Model':<50} {'Mean-Margin':<15} {'Margin-to-Mean':<15} {'ECE Score':<15} {'ECE Rank':<10}"
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
            f"{r['rank']:<5} {r['model']:<50} {r['mean_margin']:<15.4f} {r['margin_to_mean']:<15.4f} {r['ece_score']:<15.4f} {ece_ranks[r['model']]:<10}"
        )

    print("=" * 90)

    # Compute and visualize correlations
    correlations = compute_correlations(results)
    visualize_correlation(correlations)

    # Visualize accuracy–rejection curve (reject highest entropy first), colored by ECE
    visualize_accuracy_rejection_curve(results, k=20)

    # Visualize accuracy–rejection curve colored by leaderboard rank
    visualize_accuracy_rejection_curve_by_rank(results, k=20)

    # Write results to csv
    df = pd.DataFrame(results)
    output_csv = "rbv2_margin_metrics.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nResults written to {output_csv}")


if __name__ == "__main__":
    main()
