import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use("science")


def _style_boxplot_axis(ax, title, ylabel, xlabel="", y_upper=None):
    """Apply consistent styling to single-axis ECE boxplots."""
    ax.set_title(title, fontsize=16, fontweight="semibold", pad=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylim(-0.001, y_upper)
    ax.tick_params(axis="x", labelsize=11, rotation=10)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_facecolor("#f7f9fc")
    ax.margins(x=0.04)


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
    print("Visualizing Sub-k Rank-wise ECE...")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=sub_df,
        x="k_label",
        y="ece",
        hue="model",
        palette="Set2",
        dodge=True,
        ax=ax,
        linewidth=1.2,
        width=0.55,
        fliersize=0,
        log_scale=log_scale,
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    sns.stripplot(
        data=sub_df,
        x="k_label",
        y="ece",
        hue="model",
        palette="Set2",
        dodge=True,
        ax=ax,
        alpha=0.55,
        size=4.5,
        linewidth=0.3,
        edgecolor="white",
    )

    legend = ax.legend(
        by_label.values(),
        by_label.keys(),
        title="model",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    legend.get_frame().set_alpha(0.9)
    _style_boxplot_axis(
        ax, f"Sub-k ECE per model on {dataset_name}", "Sub-k ECE", y_upper=1.001
    )
    fig.tight_layout()
    fig.savefig(
        f"{save_folder}subk_ece_grouped_{dataset_name}_{round(proportion_of_considered_rankings_in_ece, 2)}_{rank_weighting}_{discrepancy}_{bin_spacing}.png",
        dpi=300,
    )

    print("Visualizing Sub-k Full-Rank ECE...")
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    sns.boxplot(
        data=sub_full_rank_df,
        x="k_label",
        y="ece",
        hue="model",
        palette="Set2",
        dodge=True,
        ax=ax,
        linewidth=1.2,
        width=0.55,
        fliersize=0,
        log_scale=log_scale,
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    sns.stripplot(
        data=sub_full_rank_df,
        x="k_label",
        y="ece",
        hue="model",
        palette="Set2",
        dodge=True,
        ax=ax,
        alpha=0.55,
        size=4.5,
        linewidth=0.3,
        edgecolor="white",
        log_scale=log_scale,
    )

    legend = ax.legend(
        by_label.values(),
        by_label.keys(),
        title="model",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    legend.get_frame().set_alpha(0.9)
    _style_boxplot_axis(
        ax,
        f"Sub-k ECE per model on {dataset_name} (Full Rank)",
        "Sub-k ECE (Full Rank)",
    )
    fig.tight_layout()
    fig.savefig(
        f"{save_folder}subk_ece_grouped_{dataset_name}_{round(proportion_of_considered_rankings_in_ece, 2)}_full_rank_{rank_weighting}_{discrepancy}_{bin_spacing}.png",
        dpi=300,
    )

    print("Visualizing Top-k Rank-wise ECE...")
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    sns.boxplot(
        data=top_df,
        x="k_label",
        y="ece",
        hue="model",
        palette="Set2",
        dodge=True,
        ax=ax,
        linewidth=1.2,
        width=0.55,
        fliersize=0,
        log_scale=log_scale,
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    sns.stripplot(
        data=top_df,
        x="k_label",
        y="ece",
        hue="model",
        palette="Set2",
        dodge=True,
        ax=ax,
        alpha=0.55,
        size=4.5,
        linewidth=0.3,
        edgecolor="white",
        log_scale=log_scale,
    )

    legend = ax.legend(
        by_label.values(),
        by_label.keys(),
        title="model",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    legend.get_frame().set_alpha(0.9)
    _style_boxplot_axis(
        ax, f"Top-k ECE per model on {dataset_name}", "Top-k ECE", y_upper=1.001
    )
    fig.tight_layout(pad=2.0)
    fig.savefig(
        f"{save_folder}topk_ece_grouped_{dataset_name}_{round(proportion_of_considered_rankings_in_ece, 2)}_{rank_weighting}_{discrepancy}_{bin_spacing}.png",
        dpi=300,
    )

    print("Visualizing Top-k Full-Rank ECE...")
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")

    sns.boxplot(
        data=top_full_rank_df,
        x="k_label",
        y="ece",
        hue="model",
        palette="Set2",
        dodge=True,
        ax=ax,
        linewidth=1.2,
        width=0.55,
        fliersize=0,
        log_scale=log_scale,
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    sns.stripplot(
        data=top_full_rank_df,
        x="k_label",
        y="ece",
        hue="model",
        palette="Set2",
        dodge=True,
        ax=ax,
        alpha=0.55,
        size=4.5,
        linewidth=0.3,
        edgecolor="white",
        log_scale=log_scale,
    )

    legend = ax.legend(
        by_label.values(),
        by_label.keys(),
        title="model",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    legend.get_frame().set_alpha(0.9)
    _style_boxplot_axis(
        ax,
        f"Top-k ECE per model on {dataset_name} (Full Rank)",
        "Top-k ECE (Full Rank)",
    )
    fig.tight_layout()
    fig.savefig(
        f"{save_folder}topk_ece_grouped_{dataset_name}_{round(proportion_of_considered_rankings_in_ece, 2)}_full_rank_{rank_weighting}_{discrepancy}_{bin_spacing}.png",
        dpi=300,
    )

    print("Visualizing Sub-k Rankwise ECE vs k with Error Bars...")
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

    print("Visualizing Top-k Rankwise ECE vs k with Error Bars...")
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
    print("Visualizing Sub-k Full-Rank Rankwise ECE vs k with Error Bars...")
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
    print("Visualizing Top-k Full-Rank Rankwise ECE vs k with Error Bars...")
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

    # handles, labels = axes[1, 1].get_legend_handles_labels()
    # legend_bottom = axes[1, 1].legend(
    #     handles,
    #     labels,
    #     title="Model",
    #     frameon=True,
    #     fontsize=11,
    #     title_fontsize=12,
    #     loc="upper left",
    #     bbox_to_anchor=(1.02, 1),
    # )
    # legend_bottom.get_frame().set_alpha(0.9)
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
    fig.savefig(
        f"{save_folder}ece_vs_k_errorbars_{dataset_name}_{round(proportion_of_considered_rankings_in_ece, 2)}_{rank_weighting}_{discrepancy}_{bin_spacing}.png",
        dpi=300,
    )


# Hard coding the proportion used to save the .csv files through experiment.calibration
dataset_to_proportion = {
    "political": 1.0,
    "glass": 1.0,
    "authorship": 1.0,
    "iris": 1.0,
    # "libras": 0.0,
    "segment": 1.0,
    "vehicle": 1.0,
    "vowel": 0.0,
    "wine": 1.0,
    "yeast": 0.0,
    "movies": 0.0,
}

if __name__ == "__main__":
    ## Load the movies dataset ECE results
    for (
        dataset_name,
        proportion_of_considered_rankings,
    ) in dataset_to_proportion.items():

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
        
        
        k_values_sub_k = sub_df["k"].unique().tolist() #[2,3,4,5]
        k_values_top_k = top_df["k"].unique().tolist() # [1,2,3,4]
        ## Filter the dataframes to only consider the desired k values
        # sub_df = sub_df[sub_df["k"].isin(k_values_sub_k)]
        # top_df = top_df[top_df["k"].isin(k_values_top_k)]
        # sub_full_rank_df = sub_full_rank_df[sub_full_rank_df["k"].isin(k_values_sub_k)]
        # top_full_rank_df = top_full_rank_df[top_full_rank_df["k"].isin(k_values_top_k)]

        ## Visualize ECE results for different k values
        save_folder = f"results/improved_plots/{dataset_name}/"
        os.makedirs(save_folder, exist_ok=True)
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
