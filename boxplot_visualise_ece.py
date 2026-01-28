import os
import matplotlib
import matplotlib.ticker as mticker
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import numpy as np

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


def plot_fancy_boxplot_battery(
    dataset_name,
    df,
    proportion_of_considered_rankings_in_ece,
    save_folder="results/",
    rank_weighting="uniform",
    discrepancy="abs",
    bin_spacing="linear",
    subplot_title="Sub-k ECE per model on {}",
    save_name_suffix="",
    log_scale=False,
):
    # Lay out subplots in a grid instead of a single long row.
    # "Four per row" keeps figures readable; rows grow as needed.
    max_cols = 4
    k_values = sorted(df["k"].unique().tolist())
    n_plots = len(k_values)
    ncols = min(max_cols, n_plots) if n_plots > 0 else 1
    nrows = int(np.ceil(n_plots / ncols)) if n_plots > 0 else 1

    fig_w = ncols * 2.8
    fig_h = nrows * 2.8
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
    axes = np.atleast_2d(axes)

    for plot_id, k in enumerate(k_values):
        values = df[df["k"] == k]
        row, col = divmod(plot_id, ncols)
        ax = axes[row, col]

        plot_df = values.groupby("model")["ece"].apply(list)
        plot_df.sort_index(inplace=True)
        # Remove rows with all -1 ECE values
        plot_df = plot_df[~plot_df.apply(lambda x: all(v == -1 for v in x))]

        bp = ax.boxplot(
            plot_df,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8, "color": "black"},
            capprops={"linewidth": 0.8, "color": "black"},
        )
        for patch, model in zip(bp["boxes"], plot_df.index):
            patch.set_facecolor(MODEL_TO_COLORS.get(model, "#FFFFFF"))
            patch.set_alpha(0.7)

        ax.set_title(f"k={k}", fontsize=14, fontweight="semibold")
        ax.set_xticklabels(
            [MODEL_ABBREVIATIONS.get(label, label) for label in plot_df.index],
            rotation=45,
        )
        ax.xaxis.set_minor_locator(plt.NullLocator())

        # Robust y-limits
        if len(plot_df.index) > 0:
            min_ece = plot_df.apply(min).min()
            max_ece = plot_df.apply(max).max()
            if log_scale:
                min_ece = max(min_ece, 1e-12)
                ax.set_ylim(bottom=min_ece * 0.8, top=max_ece * 1.2)
            else:
                ax.set_ylim(bottom=max(min_ece * 0.8, 0.0), top=max_ece * 1.2)

        if log_scale:
            ax.set_yscale("log")

            # Keep log ticks readable: label only decades as 10^x.
            ax.yaxis.set_major_locator(
                mticker.LogLocator(base=10.0, subs=(1.0,), numticks=12)
            )
            ax.yaxis.set_major_formatter(
                mticker.LogFormatterMathtext(base=10.0, labelOnlyBase=True)
            )
            ax.yaxis.set_minor_locator(
                mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12)
            )
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        else:
            # Linear scale with scientific offset (×10^x) shown at the top.
            ax.set_yscale("linear")
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
            scalar_fmt = mticker.ScalarFormatter(useMathText=True)
            scalar_fmt.set_scientific(True)
            scalar_fmt.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(scalar_fmt)
            ax.yaxis.get_offset_text().set_size(10)
            ax.yaxis.set_minor_locator(mticker.NullLocator())

        # Force y tick labels on every subplot (not just left-most).
        ax.tick_params(axis="y", which="both", labelleft=True)

    # Hide any unused axes if the grid isn't full
    for empty_id in range(n_plots, nrows * ncols):
        row, col = divmod(empty_id, ncols)
        axes[row, col].axis("off")

    fig.suptitle(
        subplot_title.format(dataset_name), fontsize=16, fontweight="semibold", y=1.02
    )
    fig.text(
        0,
        0.5,
        (save_name_suffix.capitalize() + " ECE") if save_name_suffix else "ECE",
        va="center",
        rotation="vertical",
        fontsize=13,
    )
    fig.text(0.5, 0, "Model", ha="center", fontsize=13)

    # Leave space for the suptitle and axis labels
    fig.tight_layout(pad=1.2, rect=[0.02, 0.02, 1, 0.96])
    fig.savefig(
        f"{save_folder}{save_name_suffix}_ece_better_{dataset_name}_{round(proportion_of_considered_rankings_in_ece, 2)}_{rank_weighting}_{discrepancy}_{bin_spacing}.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    # Save Legend separately
    fig_legend = plt.figure(figsize=(2, 2))
    ax_legend = fig_legend.add_subplot(111)
    handles = [
        matplotlib.patches.Patch(color=color, label=model)
        for model, color in MODEL_TO_COLORS.items()
    ]
    legend = ax_legend.legend(
        handles,
        MODEL_TO_COLORS.keys(),
        title="Model",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
        loc="center",
    )
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor("none")
    ax_legend.axis("off")
    fig_legend.savefig(
        f"{save_folder}{save_name_suffix}_ece_legend_{dataset_name}_{round(proportion_of_considered_rankings_in_ece, 2)}_{rank_weighting}_{discrepancy}_{bin_spacing}.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    plt.close(fig_legend)


def visualize_ece_better(
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

    print("Visualizing Sub-k Rank-wise ECE with improved boxplots...")
    plot_fancy_boxplot_battery(
        dataset_name=dataset_name,
        df=sub_df,
        proportion_of_considered_rankings_in_ece=proportion_of_considered_rankings_in_ece,
        save_folder=save_folder,
        rank_weighting=rank_weighting,
        discrepancy=discrepancy,
        bin_spacing=bin_spacing,
        subplot_title="Sub-k ECE per model on {}",
        save_name_suffix="Sub-k",
        log_scale=log_scale,
    )

    print("Visualizing Sub-k Full-Rank ECE with improved boxplots...")
    plot_fancy_boxplot_battery(
        dataset_name=dataset_name,
        df=sub_full_rank_df,
        proportion_of_considered_rankings_in_ece=proportion_of_considered_rankings_in_ece,
        save_folder=save_folder,
        rank_weighting=rank_weighting,
        discrepancy=discrepancy,
        bin_spacing=bin_spacing,
        subplot_title="Sub-k ECE per model on {} (Full Rank)",
        save_name_suffix="Sub-k_full_rank",
        log_scale=log_scale,
    )

    print("Visualizing Top-k Rank-wise ECE with improved boxplots...")
    plot_fancy_boxplot_battery(
        dataset_name=dataset_name,
        df=top_df,
        proportion_of_considered_rankings_in_ece=proportion_of_considered_rankings_in_ece,
        save_folder=save_folder,
        rank_weighting=rank_weighting,
        discrepancy=discrepancy,
        bin_spacing=bin_spacing,
        subplot_title="Top-k ECE per model on {}",
        save_name_suffix="Top-k",
        log_scale=log_scale,
    )

    print("Visualizing Top-k Full-Rank ECE with improved boxplots...")
    plot_fancy_boxplot_battery(
        dataset_name=dataset_name,
        df=top_full_rank_df,
        proportion_of_considered_rankings_in_ece=proportion_of_considered_rankings_in_ece,
        save_folder=save_folder,
        rank_weighting=rank_weighting,
        discrepancy=discrepancy,
        bin_spacing=bin_spacing,
        subplot_title="Top-k ECE per model on {} (Full Rank)",
        save_name_suffix="Top-k_full_rank",
        log_scale=log_scale,
    )


def _plot_battery_into_gridspec(
    fig,
    parent_gs,
    df,
    group_title,
    *,
    max_cols=4,
    k_min=None,
    k_max=None,
    log_scale=False,
    show_xlabels=True,
):
    """Render the per-k boxplot 'battery' into a provided GridSpec cell."""

    if k_min is not None:
        df = df[df["k"] >= k_min]
    if k_max is not None:
        df = df[df["k"] <= k_max]

    k_values = sorted(df["k"].unique().tolist())
    n_plots = len(k_values)
    ncols = min(max_cols, n_plots) if n_plots > 0 else 1
    nrows = int(np.ceil(n_plots / ncols)) if n_plots > 0 else 1
    subgs = parent_gs.subgridspec(nrows=nrows, ncols=ncols, wspace=0.25, hspace=0.35)

    for plot_id in range(nrows * ncols):
        r, c = divmod(plot_id, ncols)
        ax = fig.add_subplot(subgs[r, c])

        if plot_id >= n_plots:
            ax.axis("off")
            continue

        k = k_values[plot_id]
        values = df[df["k"] == k]
        plot_df = values.groupby("model")["ece"].apply(list)
        plot_df.sort_index(inplace=True)
        plot_df = plot_df[~plot_df.apply(lambda x: all(v == -1 for v in x))]

        if len(plot_df.index) == 0:
            ax.set_title(f"{group_title}\n(k={k})" if plot_id == 0 else f"k={k}")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
            ax.axis("off")
            continue

        bp = ax.boxplot(
            plot_df,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.2},
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8, "color": "black"},
            capprops={"linewidth": 0.8, "color": "black"},
        )
        for patch, model in zip(bp["boxes"], plot_df.index):
            patch.set_facecolor(MODEL_TO_COLORS.get(model, "#FFFFFF"))
            patch.set_alpha(0.7)

        ax.set_title(f"{group_title}\n(k={k})" if plot_id == 0 else f"k={k}")

        # Only show x tick labels on the last row of this battery grid.
        show_labels_here = bool(show_xlabels) and (r == nrows - 1)
        if show_labels_here:
            ax.set_xticklabels(
                [MODEL_ABBREVIATIONS.get(label, label) for label in plot_df.index],
                rotation=45,
            )
            ax.tick_params(axis="x", which="both", labelbottom=True)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", labelbottom=False)
        ax.xaxis.set_minor_locator(plt.NullLocator())

        min_ece = plot_df.apply(min).min()
        max_ece = plot_df.apply(max).max()

        if log_scale:
            min_ece = max(min_ece, 1e-12)
            ax.set_ylim(bottom=min_ece * 0.8, top=max_ece * 1.2)
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(
                mticker.LogLocator(base=10.0, subs=(1.0,), numticks=12)
            )
            ax.yaxis.set_major_formatter(
                mticker.LogFormatterMathtext(base=10.0, labelOnlyBase=True)
            )
            ax.yaxis.set_minor_locator(
                mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12)
            )
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        else:
            ax.set_ylim(bottom=max(min_ece * 0.8, 0.0), top=max_ece * 1.2)
            ax.set_yscale("linear")
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
            scalar_fmt = mticker.ScalarFormatter(useMathText=True)
            scalar_fmt.set_scientific(True)
            scalar_fmt.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(scalar_fmt)
            ax.yaxis.get_offset_text().set_size(9)
            ax.yaxis.set_minor_locator(mticker.NullLocator())

        ax.tick_params(axis="y", which="both", labelleft=True)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_facecolor("#f7f9fc")


def plot_all_calibrations_summary_for_dataset(
    dataset_name,
    *,
    sub_df,
    top_df,
    sub_full_rank_df,
    top_full_rank_df,
    proportion_of_considered_rankings_in_ece,
    save_folder="results/improved_plots/",
    rank_weighting="95_prob_mass",
    discrepancy="abs",
    bin_spacing="linear",
    sub_k_min=2,
    sub_k_max=9,
    top_k_min=1,
    top_k_max=8,
    log_scale=False,
):
    """Create one figure per dataset showing Sub-k / Top-k / Full-rank variants."""

    os.makedirs(save_folder, exist_ok=True)
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(nrows=2, ncols=2, wspace=0.15, hspace=0.25)

    _plot_battery_into_gridspec(
        fig,
        gs[0, 0],
        sub_df,
        "Sub-k",
        max_cols=4,
        k_min=sub_k_min,
        k_max=sub_k_max,
        log_scale=log_scale,
        show_xlabels=False,
    )
    _plot_battery_into_gridspec(
        fig,
        gs[0, 1],
        top_df,
        "Top-k",
        max_cols=4,
        k_min=top_k_min,
        k_max=top_k_max,
        log_scale=log_scale,
        show_xlabels=False,
    )
    _plot_battery_into_gridspec(
        fig,
        gs[1, 0],
        sub_full_rank_df,
        "Sub-k (Full Rank)",
        max_cols=4,
        k_min=sub_k_min,
        k_max=sub_k_max,
        log_scale=log_scale,
        show_xlabels=True,
    )
    _plot_battery_into_gridspec(
        fig,
        gs[1, 1],
        top_full_rank_df,
        "Top-k (Full Rank)",
        max_cols=4,
        k_min=top_k_min,
        k_max=top_k_max,
        log_scale=log_scale,
        show_xlabels=True,
    )

    fig.suptitle(
        f"ECE calibrations on {dataset_name} (sub-k {sub_k_min}..{sub_k_max}, top-k {top_k_min}..{top_k_max})",
        fontsize=16,
        fontweight="semibold",
        y=0.99,
    )

    out_path = (
        f"{save_folder}all_calibrations_{dataset_name}_{round(proportion_of_considered_rankings_in_ece, 2)}_"
        f"{rank_weighting}_{discrepancy}_{bin_spacing}_sub{sub_k_min}-{sub_k_max}_top{top_k_min}-{top_k_max}.pdf"
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_political_movies_2x4_summary(
    dataset_to_proportion,
    save_folder="results/improved_plots/political_movies/",
    rank_weighting="95_prob_mass",
    discrepancy="abs",
    bin_spacing="linear",
    log_scale=False,
):
    """Create a single 2x4 figure for political and movies.

    Layout:
      - Rows: political, then movies
      - Columns: Sub-2, Sub-3, Top-2, Top-3
    """

    datasets = ["political", "movies"]
    panels = [
        ("Sub-2", "sub", 2),
        ("Sub-3", "sub", 3),
        ("Top-2", "top", 2),
        ("Top-3", "top", 3),
    ]

    os.makedirs(save_folder, exist_ok=True)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 3), sharey=False)
    fig.patch.set_facecolor("white")

    def _plot_single_panel(ax, values_df, title, set_xlabels=True):
        plot_df = values_df.groupby("model")["ece"].apply(list)
        plot_df.sort_index(inplace=True)
        plot_df = plot_df[~plot_df.apply(lambda x: all(v == -1 for v in x))]

        if len(plot_df.index) == 0:
            ax.set_title(title, fontsize=13, fontweight="semibold")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
            ax.axis("off")
            return

        bp = ax.boxplot(
            plot_df,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8, "color": "black"},
            capprops={"linewidth": 0.8, "color": "black"},
        )
        for patch, model in zip(bp["boxes"], plot_df.index):
            patch.set_facecolor(MODEL_TO_COLORS.get(model, "#FFFFFF"))
            patch.set_alpha(0.7)
        if set_xlabels:
            # ax.set_title(title, fontsize=13, fontweight="semibold")
            ax.set_xticklabels(
                [MODEL_ABBREVIATIONS.get(label, label) for label in plot_df.index],
                rotation=45,
            )
            ax.xaxis.set_minor_locator(plt.NullLocator())
        else:
            ax.set_title(title, fontsize=13, fontweight="semibold")
            ax.set_xticklabels([])
            ax.xaxis.set_minor_locator(plt.NullLocator())

        min_ece = plot_df.apply(min).min()
        max_ece = plot_df.apply(max).max()
        if log_scale:
            min_ece = max(min_ece, 1e-12)
            ax.set_ylim(bottom=min_ece * 0.8, top=max_ece * 1.2)
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(
                mticker.LogLocator(base=10.0, subs=(1.0,), numticks=12)
            )
            ax.yaxis.set_major_formatter(
                mticker.LogFormatterMathtext(base=10.0, labelOnlyBase=True)
            )
            ax.yaxis.set_minor_locator(
                mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12)
            )
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        else:
            ax.set_ylim(bottom=max(min_ece * 0.8, 0.0), top=max_ece * 1.2)
            ax.set_yscale("linear")
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
            scalar_fmt = mticker.ScalarFormatter(useMathText=True)
            scalar_fmt.set_scientific(True)
            scalar_fmt.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(scalar_fmt)
            ax.yaxis.get_offset_text().set_size(10)
            ax.yaxis.set_minor_locator(mticker.NullLocator())

        ax.tick_params(axis="y", which="both", labelleft=True)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_facecolor("#f7f9fc")

    for row, dataset_name in enumerate(datasets):
        prop = dataset_to_proportion[dataset_name]
        base_folder = f"results/{dataset_name}/"

        sub_df = pd.read_csv(
            f"{base_folder}subk_ece_results_{dataset_name}_{round(prop, 2)}_{rank_weighting}_{discrepancy}_{bin_spacing}.csv"
        )
        top_df = pd.read_csv(
            f"{base_folder}topk_ece_results_{dataset_name}_{round(prop, 2)}_{rank_weighting}_{discrepancy}_{bin_spacing}.csv"
        )

        sub_df["model"] = sub_df["model"].replace(RENAME_MODELS)
        top_df["model"] = top_df["model"].replace(RENAME_MODELS)

        for col, (panel_title, panel_kind, k) in enumerate(panels):
            ax = axes[row, col]
            df = sub_df if panel_kind == "sub" else top_df
            values_df = df[df["k"] == k]
            _plot_single_panel(ax, values_df, panel_title, set_xlabels=(row == 1))

        # Row label on the left-most subplot
        axes[row, 0].set_ylabel(f"{dataset_name.capitalize()}\nECE", fontsize=12)

    # fig.suptitle(
    #     "ECE Summary: political vs movies", fontsize=16, fontweight="semibold", y=1.02
    # )
    # fig.tight_layout(pad=1.0, rect=[0.02, 0.02, 1, 0.94])
    fig.tight_layout()
    fig.savefig(
        f"{save_folder}political_movies_2x4_sub2_sub3_top2_top3_{rank_weighting}_{discrepancy}_{bin_spacing}.pdf",
        dpi=300,
    )

    # Save a shared legend once
    fig_legend = plt.figure(figsize=(2.2, 2.2))
    ax_legend = fig_legend.add_subplot(111)
    handles = [
        matplotlib.patches.Patch(color=color, label=model)
        for model, color in MODEL_TO_COLORS.items()
    ]
    legend = ax_legend.legend(
        handles,
        MODEL_TO_COLORS.keys(),
        title="Model",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
        loc="center",
    )
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor("none")
    ax_legend.axis("off")
    fig_legend.savefig(
        f"{save_folder}political_movies_legend.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    plt.close(fig_legend)


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
MODEL_ABBREVIATIONS = {
    "PlackettLuce": "PL",
    "MallowsModel": "MM",
    "RankClassifier": "RC",
    "PlackettLuceRPC": " PL-RPC",
    "RPC": "RPC",
}
MODEL_TO_COLORS = {
    "PlackettLuce": "#0C45C7",
    "MallowsModel": "#FC69EA",
    "RankClassifier": "#F37748",
    "PlackettLuceRPC": "#5AA9E6",
    "RPC": "#04E762",
}
RENAME_MODELS = {
    "PreferenceModel": "RankClassifier",
}
if __name__ == "__main__":
    RUN_ALL_DATASETS = True
    RUN_POLITICAL_MOVIES_SUMMARY = True
    RUN_ALL_CALIBRATIONS_SUMMARY = True

    if RUN_ALL_DATASETS:
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
            try:
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
            except FileNotFoundError as e:
                print(f"Skipping {dataset_name}: missing CSV ({e.filename})")
                continue

            ## Rename models for better visualization
            sub_df["model"] = sub_df["model"].replace(RENAME_MODELS)
            top_df["model"] = top_df["model"].replace(RENAME_MODELS)
            sub_full_rank_df["model"] = sub_full_rank_df["model"].replace(RENAME_MODELS)
            top_full_rank_df["model"] = top_full_rank_df["model"].replace(RENAME_MODELS)

            # Restrict k ranges for plotting:
            # - Sub-k: k = 2..9
            # - Top-k: k = 1..8
            sub_df = sub_df[sub_df["k"].between(2, 9)]
            sub_full_rank_df = sub_full_rank_df[sub_full_rank_df["k"].between(2, 9)]
            top_df = top_df[top_df["k"].between(1, 8)]
            top_full_rank_df = top_full_rank_df[top_full_rank_df["k"].between(1, 8)]

            k_values_sub_k = sub_df["k"].unique().tolist()  # [2,3,4,5]
            k_values_top_k = top_df["k"].unique().tolist()  # [1,2,3,4]

            ## Visualize ECE results for different k values
            save_folder = f"results/improved_plots/{dataset_name}/"
            os.makedirs(save_folder, exist_ok=True)

            visualize_ece_better(
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

            if RUN_ALL_CALIBRATIONS_SUMMARY:
                plot_all_calibrations_summary_for_dataset(
                    dataset_name=dataset_name,
                    sub_df=sub_df,
                    top_df=top_df,
                    sub_full_rank_df=sub_full_rank_df,
                    top_full_rank_df=top_full_rank_df,
                    proportion_of_considered_rankings_in_ece=proportion_of_considered_rankings,
                    save_folder=f"results/improved_plots/{dataset_name}/",
                    rank_weighting=rank_weighting,
                    discrepancy=discrepancy,
                    bin_spacing=bin_spacing,
                    sub_k_min=2,
                    sub_k_max=9,
                    top_k_min=1,
                    top_k_max=8,
                    log_scale=False,
                )

    if RUN_POLITICAL_MOVIES_SUMMARY:
        plot_political_movies_2x4_summary(
            dataset_to_proportion=dataset_to_proportion,
            save_folder="results/improved_plots/political_movies/",
            rank_weighting="95_prob_mass",
            discrepancy="abs",
            bin_spacing="linear",
            log_scale=False,
        )
