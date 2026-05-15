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
    colors = sns.color_palette("Dark2", n_colors=len(models))
    return dict(zip(models, colors))


def _abbrev_models(df):
    df = df.copy()
    df["model"] = df["model"].map(_MODEL_ABBREV).fillna(df["model"])
    return df


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
    k_values = [
        k for k in k_values if not df[(df["k"] == k) & (df["ece"] >= 0)].empty
    ]
    if not k_values:
        return
    n_k = len(k_values)

    fig, axes = plt.subplots(1, n_k, figsize=(3.0 * n_k, 4), sharey=False)
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
            linewidth=0.9,
            width=0.6,
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
            alpha=0.45,
            size=3.5,
            linewidth=0.2,
            edgecolor="white",
        )

        ax.set_title(f"$k={k}$", fontsize=12, fontweight="semibold", pad=8)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if i == 0 else "", fontsize=11)
        data_max = k_df["ece"].max()
        ax.set_ylim(bottom=0, top=min(1.01, data_max * 1.15 + 0.02))
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
        bbox_to_anchor=(0.5, -0.06),
        ncol=min(len(legend_handles), 5),
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        framealpha=0.9,
    )

    fig.suptitle(fig_title, fontsize=14, fontweight="semibold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {save_path}")


def _plot_lineplots(
    sub_df, top_df, sub_full_rank_df, top_full_rank_df,
    k_values_sub_k, k_values_top_k,
    fig_title, save_path,
):
    """
    2x2 line-plot grid (sub-k, top-k x standard, full-rank).
    One line per model with a +/-SD error band. Styling matches the
    comparison line plots in visualise_ece_comparison.py.
    """
    datasets = [
        (sub_df,           k_values_sub_k, "Sub-$k$ ECE vs $k$"),
        (top_df,           k_values_top_k, "Top-$k$ ECE vs $k$"),
        (sub_full_rank_df, k_values_sub_k, "Sub-$k$ ECE (Full Rank) vs $k$"),
        (top_full_rank_df, k_values_top_k, "Top-$k$ ECE (Full Rank) vs $k$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharey=False)
    fig.patch.set_facecolor("white")
    ax_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    model_palette = None  # set on first dataset

    for ax, (df, k_vals, subtitle) in zip(ax_flat, datasets):
        d = _abbrev_models(df)

        k_vals = [
            k for k in k_vals
            if not d[(d["k"] == k) & (d["ece"] >= 0)].empty
        ]

        all_models = sorted(d["model"].unique())
        if model_palette is None:
            colors = sns.color_palette("Dark2", n_colors=len(all_models))
            model_palette = dict(zip(all_models, colors))

        for model in all_models:
            color = model_palette.get(model)
            if color is None:
                continue
            mdf = d[(d["model"] == model) & (d["ece"] >= 0)]
            if mdf.empty:
                continue
            means = mdf.groupby("k")["ece"].mean().reindex(k_vals)
            sds = mdf.groupby("k")["ece"].std().reindex(k_vals)
            ax.plot(
                k_vals, means.values,
                color=color, linestyle="-",
                linewidth=2.2, marker="o", markersize=6,
                markeredgewidth=0,
                label=model,
            )
            ax.fill_between(
                k_vals,
                (means - sds).values,
                (means + sds).values,
                color=color, alpha=0.10,
            )

        ax.set_title(subtitle, fontsize=12, fontweight="semibold", pad=8)
        ax.set_xlabel("$k$", fontsize=12)
        ax.set_ylabel("ECE", fontsize=12)
        if k_vals:
            ax.set_xticks(k_vals)
        data_max = d[d["ece"] >= 0]["ece"].max()
        ax.set_ylim(bottom=0, top=min(1.01, data_max * 1.15 + 0.02))
        ax.set_facecolor("#f7f9fc")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="both", labelsize=10)
        ax.margins(x=0.04)

    model_handles = [
        mpatches.Patch(color=model_palette[m], label=_legend_label(m))
        for m in sorted(model_palette)
    ]
    fig.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=len(model_handles),
        fontsize=10,
        frameon=True,
        framealpha=0.9,
    )

    fig.suptitle(fig_title, fontsize=15, fontweight="semibold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {save_path}")


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
    _plot_lineplots(
        sub_df=sub_df,
        top_df=top_df,
        sub_full_rank_df=sub_full_rank_df,
        top_full_rank_df=top_full_rank_df,
        k_values_sub_k=k_values_sub_k,
        k_values_top_k=k_values_top_k,
        fig_title=f"ECE vs $k$ — {dataset_name}",
        save_path=f"{save_folder}ece_vs_k_errorbars_{base}.pdf",
    )


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
