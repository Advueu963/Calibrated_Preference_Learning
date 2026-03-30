import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scienceplots
plt.style.use("science")


_MODEL_ABBREV = {
    "MallowsModel":    "MM",
    "PlackettLuce":    "PL",
    "PlackettLuceRPC": "PL-RPC",
    "PreferenceModel": "RC",   # renamed to RankingModel
    "RPC":             "RPC",
}
_MODEL_FULLNAME = {
    "MM":     "MallowsModel",
    "PL":     "PlackettLuce",
    "PL-RPC": "PlackettLuceRPC",
    "RC":     "RankingModel",
    "RPC":    "RPC",
}


def _build_model_palette(df):
    """Return a consistent {model: color} dict keyed on sorted model names."""
    models = sorted(df["model"].unique())
    colors = sns.color_palette("Set2", n_colors=len(models))
    return dict(zip(models, colors))


def _legend_label(abbrev):
    full = _MODEL_FULLNAME.get(abbrev, abbrev)
    return f"{abbrev} ({full})" if full != abbrev else abbrev


def _plot_multi_k_boxplots(df, k_values, fig_title, ylabel, save_path):
    """
    Multi-panel boxplot: one panel per k value, models on x-axis, independent y-axes.
    A single figure-level legend is placed below the panels.
    """
    df = df.copy()
    df["model"] = df["model"].map(_MODEL_ABBREV).fillna(df["model"])

    model_palette = _build_model_palette(df)
    model_order = sorted(df["model"].unique())
    n_k = len(k_values)

    fig, axes = plt.subplots(1, n_k, figsize=(2.5 * n_k, 4), sharey=False)
    if n_k == 1:
        axes = [axes]

    for i, (k, ax) in enumerate(zip(k_values, axes)):
        # Drop rows with invalid ECE (e.g. RPC at k > 2)
        k_df = df[(df["k"] == k) & (df["ece"] >= 0)]
        valid_order = [m for m in model_order if m in k_df["model"].values]

        sns.boxplot(
            data=k_df,
            x="model",
            y="ece",
            hue="model",
            order=valid_order,
            palette=model_palette,
            dodge=False,
            legend=False,
            ax=ax,
            linewidth=1.0,
            width=0.55,
            fliersize=0,
        )
        sns.stripplot(
            data=k_df,
            x="model",
            y="ece",
            hue="model",
            order=valid_order,
            palette=model_palette,
            dodge=False,
            legend=False,
            ax=ax,
            alpha=0.6,
            size=5,
            linewidth=0.3,
            edgecolor="white",
        )

        ax.set_title(f"$k = {k}$", fontsize=13, fontweight="semibold", pad=8)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if i == 0 else "", fontsize=11)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", labelsize=9, rotation=35)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.margins(x=0.1)

    # Single shared legend below all subplots
    legend_handles = [
        mpatches.Patch(color=model_palette[m], label=_legend_label(m))
        for m in model_order
    ]
    fig.legend(
        handles=legend_handles,
        title="Model",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(len(legend_handles), 5),
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        framealpha=0.9,
    )

    fig.suptitle(fig_title, fontsize=15, fontweight="semibold")
    fig.tight_layout(rect=[0, 0.1, 1, 0.96])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {save_path}")


def _style_lineplot_axis(
    ax, title, xlabel, ylabel, x_ticks, y_upper=None, y_lower=-0.001
):
    """Apply consistent styling to ECE line plots."""
    ax.set_title(title, fontsize=16, fontweight="semibold", pad=12)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_facecolor("#f7f9fc")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xticks(x_ticks)
    ax.set_ylim(y_lower, y_upper)
    ax.margins(x=0.04)


def visualize_ece_results(
    dataset_name,
    k_values_sub_k,
    k_values_top_k,
    sub_df,
    top_df,
    sub_full_rank_df,
    top_full_rank_df,
    proportion_of_considered_rankings_in_ece,
    save_folder="results/",
    rank_weighting="uniform",
    discrepancy="abs",
    bin_spacing="linear",
    log_scale=False,
):
    prop = round(proportion_of_considered_rankings_in_ece, 2)
    base = f"{dataset_name}_{prop}_{rank_weighting}_{discrepancy}_{bin_spacing}"
    full_rank_base = f"{dataset_name}_{prop}_full_rank_{rank_weighting}_{discrepancy}_{bin_spacing}"

    print("Visualizing Sub-k Rank-wise ECE...")
    _plot_multi_k_boxplots(
        df=sub_df,
        k_values=k_values_sub_k,
        fig_title=f"Sub-$k$ ECE per model — {dataset_name}",
        ylabel="Sub-$k$ ECE",
        save_path=f"{save_folder}subk_ece_grouped_{base}.pdf",
    )

    print("Visualizing Sub-k Full-Rank ECE...")
    _plot_multi_k_boxplots(
        df=sub_full_rank_df,
        k_values=k_values_sub_k,
        fig_title=f"Sub-$k$ ECE per model (Full Rank) — {dataset_name}",
        ylabel="Sub-$k$ ECE (Full Rank)",
        save_path=f"{save_folder}subk_ece_grouped_{full_rank_base}.pdf",
    )

    print("Visualizing Top-k Rank-wise ECE...")
    _plot_multi_k_boxplots(
        df=top_df,
        k_values=k_values_top_k,
        fig_title=f"Top-$k$ ECE per model — {dataset_name}",
        ylabel="Top-$k$ ECE",
        save_path=f"{save_folder}topk_ece_grouped_{base}.pdf",
    )

    print("Visualizing Top-k Full-Rank ECE...")
    _plot_multi_k_boxplots(
        df=top_full_rank_df,
        k_values=k_values_top_k,
        fig_title=f"Top-$k$ ECE per model (Full Rank) — {dataset_name}",
        ylabel="Top-$k$ ECE (Full Rank)",
        save_path=f"{save_folder}topk_ece_grouped_{full_rank_base}.pdf",
    )

    print("Visualizing ECE vs k with Error Bars...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharey=False)
    fig.patch.set_facecolor("white")
    sns.lineplot(
        data=sub_df,
        x="k",
        y="ece",
        hue="model",
        marker="o",
        errorbar=("sd"),
        palette="Dark2",
        linestyle="--",
        linewidth=2.4,
        markersize=7,
        markeredgewidth=0,
        ax=axes[0, 0],
        legend=False,
    )
    _style_lineplot_axis(
        axes[0, 0], "Sub-k ECE vs k", "k", "ECE", k_values_sub_k, y_upper=None
    )

    sns.lineplot(
        data=top_df,
        x="k",
        y="ece",
        hue="model",
        marker="o",
        errorbar=("sd"),
        palette="Dark2",
        linestyle="--",
        linewidth=2.4,
        markersize=7,
        markeredgewidth=0,
        ax=axes[0, 1],
        legend=False,
    )
    _style_lineplot_axis(
        axes[0, 1], "Top-k ECE vs k", "k", "ECE", k_values_top_k, y_upper=None
    )

    sns.lineplot(
        data=sub_full_rank_df,
        x="k",
        y="ece",
        hue="model",
        marker="o",
        errorbar=("sd"),
        palette="Dark2",
        linestyle="--",
        linewidth=2.4,
        markersize=7,
        markeredgewidth=0,
        ax=axes[1, 0],
        legend=True,
    )
    _style_lineplot_axis(
        axes[1, 0],
        "Sub-k ECE vs k (Full Rank)",
        "k",
        "ECE",
        k_values_sub_k,
        y_upper=None,
    )

    sns.lineplot(
        data=top_full_rank_df,
        x="k",
        y="ece",
        hue="model",
        marker="o",
        errorbar=("sd"),
        palette="Dark2",
        linestyle="--",
        linewidth=2.4,
        markersize=7,
        markeredgewidth=0,
        ax=axes[1, 1],
        legend=False,
    )
    _style_lineplot_axis(
        axes[1, 1],
        "Top-k ECE vs k (Full Rank)",
        "k",
        "ECE",
        k_values_top_k,
        y_upper=None,
        y_lower=None,
    )

    handles, labels = axes[1, 0].get_legend_handles_labels()
    legend_top = axes[1, 0].legend(
        handles,
        labels,
        title="Model",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    legend_top.get_frame().set_alpha(0.9)

    fig.tight_layout()
    fig.savefig(f"{save_folder}ece_vs_k_errorbars_{base}.pdf", dpi=300)
    plt.close(fig)
    print(f"  → Saved {save_folder}ece_vs_k_errorbars_{base}.pdf")


# Hard coding the proportion used to save the .csv files through experiment.calibration
dataset_to_proportion = {
    "political": 1.0,
    "movies": 0.0,
}

if __name__ == "__main__":
    ## Load the movies dataset ECE results
    dataset_name = "movies"
    k_values_sub_k = [2, 3, 4, 5]
    k_values_top_k = [1, 2, 3, 4]
    rank_weighting = "95_prob_mass"
    discrepancy = "abs"
    bin_spacing = "linear"
    proportion_of_considered_rankings = dataset_to_proportion[dataset_name]
    save_folder = f"results/{dataset_name}/"
    sub_df = pd.read_csv(
        f"{save_folder}subk_ece_results_{dataset_name}_{round(proportion_of_considered_rankings, 2)}_{rank_weighting}_{discrepancy}_{bin_spacing}.csv"
    )
    top_df = pd.read_csv(
        f"{save_folder}topk_ece_results_{dataset_name}_{round(proportion_of_considered_rankings, 2)}_{rank_weighting}_{discrepancy}_{bin_spacing}.csv"
    )
    sub_full_rank_df = pd.read_csv(
        f"{save_folder}subk_full_rank_ece_results_{dataset_name}_{round(proportion_of_considered_rankings, 2)}_{rank_weighting}_{discrepancy}_{bin_spacing}.csv",
    )
    top_full_rank_df = pd.read_csv(
        f"{save_folder}topk_full_rank_ece_results_{dataset_name}_{round(proportion_of_considered_rankings, 2)}_{rank_weighting}_{discrepancy}_{bin_spacing}.csv",
    )
    ## Filter the dataframes to only consider the desired k values
    sub_df = sub_df[sub_df["k"].isin(k_values_sub_k)]
    top_df = top_df[top_df["k"].isin(k_values_top_k)]
    sub_full_rank_df = sub_full_rank_df[sub_full_rank_df["k"].isin(k_values_sub_k)]
    top_full_rank_df = top_full_rank_df[top_full_rank_df["k"].isin(k_values_top_k)]

    ## Visualize ECE results for different k values
    save_folder = "results/improved_plots/"
    visualize_ece_results(
        dataset_name=dataset_name,
        k_values_sub_k=k_values_sub_k,
        k_values_top_k=k_values_top_k,
        sub_df=sub_df,
        top_df=top_df,
        sub_full_rank_df=sub_full_rank_df,
        top_full_rank_df=top_full_rank_df,
        proportion_of_considered_rankings_in_ece=proportion_of_considered_rankings,
        rank_weighting=rank_weighting,
        discrepancy=discrepancy,
        bin_spacing=bin_spacing,
        save_folder=save_folder,
        log_scale=False,
    )
