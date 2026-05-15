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
    "PreferenceModel": "RC",
    "RankingModel":    "RC",
    "RPC":             "RPC",
}

# Consistent two-color palette for the two discrepancy types
_DISC_PALETTE = {"Abs": "#4c72b0", "Jeff": "#dd8452"}
_DISC_LABELS  = {"abs": "Abs", "jeff": "Jeff"}


def _abbrev_models(df):
    df = df.copy()
    df["model"] = df["model"].map(_MODEL_ABBREV).fillna(df["model"])
    return df


def _compute_kendall_tau_per_k(df_abs, df_jeff, k_values, tie_threshold=0.01):
    """
    For each k, compute Kendall's tau between abs and jeff mean ECE rankings.
    Values are bucketed to tie_threshold before comparison so that near-zero
    ECE values at high k are treated as tied rather than creating noisy ranks.
    Returns {k: tau} with nan when fewer than 2 models are available.
    """
    from scipy.stats import kendalltau
    results = {}
    for k in k_values:
        a = df_abs[(df_abs["k"] == k) & (df_abs["ece"] >= 0)]
        j = df_jeff[(df_jeff["k"] == k) & (df_jeff["ece"] >= 0)]

        a_mean = a.groupby("model")["ece"].mean()
        j_mean = j.groupby("model")["ece"].mean()

        common = list(a_mean.index.intersection(j_mean.index))
        if len(common) < 2:
            results[k] = float("nan")
            continue

        # Bucket to tie_threshold so near-identical values become exact ties
        a_bucketed = (a_mean[common] / tie_threshold).round()
        j_bucketed = (j_mean[common] / tie_threshold).round()

        tau, _ = kendalltau(a_bucketed.values, j_bucketed.values)
        results[k] = tau
    return results


def _tau_str(tau):
    """
    Maps Kendall's tau to a human-readable ranking-agreement label.
      tau = 1.0  → "equivalent"
      tau > 1/3  → "near-equivalent"
      tau ≤ 1/3  → "divergent"
    """
    if pd.isna(tau):
        return ""
    if tau == 1.0:
        label = "equivalent"
    elif tau > 1 / 3:
        label = "near-equivalent"
    else:
        label = "divergent"
    return f"{label} ($\\tau={tau:.2f}$)"


def _plot_comparison_boxplots(df_abs, df_jeff, k_values, fig_title, ylabel, save_path):
    """
    Multi-panel boxplot (one panel per k).
    Each panel shows Abs and Jeff side-by-side per model.
    Spearman rho of model ECE rankings is embedded in the panel title.
    Legend uses abbreviations only.
    """
    df_abs  = _abbrev_models(df_abs)
    df_jeff = _abbrev_models(df_jeff)

    taus = _compute_kendall_tau_per_k(df_abs, df_jeff, k_values)

    df_abs["discrepancy"]  = "Abs"
    df_jeff["discrepancy"] = "Jeff"
    df_all = pd.concat([df_abs, df_jeff], ignore_index=True)

    model_order = sorted(df_all["model"].unique())
    k_values = [k for k in k_values if not df_all[(df_all["k"] == k) & (df_all["ece"] >= 0)].empty]
    if not k_values:
        return
    n_k = len(k_values)

    fig, axes = plt.subplots(1, n_k, figsize=(3.0 * n_k, 4), sharey=False)
    if n_k == 1:
        axes = [axes]

    for i, (k, ax) in enumerate(zip(k_values, axes)):
        k_df = df_all[(df_all["k"] == k) & (df_all["ece"] >= 0)]
        valid_models = [m for m in model_order if m in k_df["model"].values]

        sns.boxplot(
            data=k_df,
            x="model",
            y="ece",
            hue="discrepancy",
            order=valid_models,
            hue_order=["Abs", "Jeff"],
            palette=_DISC_PALETTE,
            dodge=True,
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
            hue="discrepancy",
            order=valid_models,
            hue_order=["Abs", "Jeff"],
            palette=_DISC_PALETTE,
            dodge=True,
            legend=False,
            ax=ax,
            alpha=0.45,
            size=3.5,
            linewidth=0.2,
            edgecolor="white",
        )

        tau_str = _tau_str(taus.get(k, float("nan")))
        title_suffix = f",\\ {tau_str}" if tau_str else ""
        ax.set_title(
            f"$k={k}${title_suffix}",
            fontsize=12,
            fontweight="semibold",
            pad=8,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if i == 0 else "", fontsize=11)
        data_max = k_df["ece"].max()
        ax.set_ylim(bottom=0, top=min(1.01, data_max * 1.15 + 0.02))
        ax.tick_params(axis="x", labelsize=9, rotation=35)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.margins(x=0.1)

    legend_handles = [
        mpatches.Patch(color=_DISC_PALETTE[d], label=d) for d in ["Abs", "Jeff"]
    ]
    fig.legend(
        handles=legend_handles,
        title="Discrepancy",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=2,
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


def _plot_comparison_lineplots(
    sub_abs, top_abs, sub_abs_fr, top_abs_fr,
    sub_jeff, top_jeff, sub_jeff_fr, top_jeff_fr,
    k_values_sub_k, k_values_top_k,
    fig_title, save_path,
):
    """
    2x2 line-plot grid (sub-k, top-k × standard, full-rank).
    Each subplot overlays Abs (solid) and Jeff (dashed) lines per model.
    Legend uses model abbreviations only; line style distinguishes discrepancy.
    Per-k Spearman rho values are annotated in the subplot title as a list.
    """
    datasets = [
        (sub_abs,    sub_jeff,    k_values_sub_k, "Sub-$k$ ECE vs $k$"),
        (top_abs,    top_jeff,    k_values_top_k, "Top-$k$ ECE vs $k$"),
        (sub_abs_fr, sub_jeff_fr, k_values_sub_k, "Sub-$k$ ECE (Full Rank) vs $k$"),
        (top_abs_fr, top_jeff_fr, k_values_top_k, "Top-$k$ ECE (Full Rank) vs $k$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharey=False)
    fig.patch.set_facecolor("white")
    ax_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    model_palette = None  # set on first dataset

    for ax, (df_a, df_j, k_vals, subtitle) in zip(ax_flat, datasets):
        da = _abbrev_models(df_a)
        dj = _abbrev_models(df_j)

        # Only plot k values present in both datasets
        available_k = [
            k for k in k_vals
            if not da[(da["k"] == k) & (da["ece"] >= 0)].empty
            or not dj[(dj["k"] == k) & (dj["ece"] >= 0)].empty
        ]
        k_vals = available_k

        all_models = sorted(set(da["model"].unique()) | set(dj["model"].unique()))
        if model_palette is None:
            colors = sns.color_palette("Dark2", n_colors=len(all_models))
            model_palette = dict(zip(all_models, colors))

        # Compute per-k Kendall tau annotation
        taus = _compute_kendall_tau_per_k(da, dj, k_vals)
        tau_parts = [
            f"$k={k}$: {_tau_str(taus[k])}"
            for k in k_vals
            if not pd.isna(taus.get(k, float("nan"))) and _tau_str(taus[k])
        ]
        rho_annotation = ",\\ ".join(tau_parts)

        for model in all_models:
            color = model_palette[model]

            for df, ls in [(da, "-"), (dj, "--")]:
                mdf = df[(df["model"] == model) & (df["ece"] >= 0)]
                if mdf.empty:
                    continue
                means = mdf.groupby("k")["ece"].mean().reindex(k_vals)
                sds   = mdf.groupby("k")["ece"].std().reindex(k_vals)
                ax.plot(
                    k_vals, means.values,
                    color=color, linestyle=ls,
                    linewidth=2.2, marker="o", markersize=6,
                    markeredgewidth=0,
                    label=model if ls == "-" else "_nolegend_",
                )
                ax.fill_between(
                    k_vals,
                    (means - sds).values,
                    (means + sds).values,
                    color=color, alpha=0.10,
                )

        title_text = f"{subtitle}\n{rho_annotation}" if rho_annotation else subtitle
        ax.set_title(title_text, fontsize=12, fontweight="semibold", pad=8)
        ax.set_xlabel("$k$", fontsize=12)
        ax.set_ylabel("ECE", fontsize=12)
        ax.set_xticks(k_vals)
        data_max = max(
            da[da["ece"] >= 0]["ece"].max(),
            dj[dj["ece"] >= 0]["ece"].max(),
        )
        ax.set_ylim(bottom=0, top=min(1.01, data_max * 1.15 + 0.02))
        ax.set_facecolor("#f7f9fc")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="both", labelsize=10)
        ax.margins(x=0.04)

    # Model legend (abbreviations only)
    model_handles = [
        mpatches.Patch(color=model_palette[m], label=m) for m in sorted(model_palette)
    ]
    # Discrepancy legend (line style)
    disc_handles = [
        plt.Line2D([0], [0], color="gray", linestyle="-",  linewidth=2, label="Abs"),
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label="Jeff"),
    ]
    all_handles = model_handles + [mpatches.Patch(color="none", label="")] + disc_handles

    fig.legend(
        handles=all_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=len(model_handles) + 3,
        fontsize=10,
        frameon=True,
        framealpha=0.9,
    )

    fig.suptitle(fig_title, fontsize=15, fontweight="semibold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {save_path}")


def visualize_ece_comparison(
    dataset_name,
    k_values_sub_k,
    k_values_top_k,
    sub_abs_df,
    top_abs_df,
    sub_abs_full_rank_df,
    top_abs_full_rank_df,
    sub_jeff_df,
    top_jeff_df,
    sub_jeff_full_rank_df,
    top_jeff_full_rank_df,
    proportion_of_considered_rankings_in_ece,
    save_folder="results/",
    rank_weighting="uniform",
    bin_spacing="linear",
):
    prop  = round(proportion_of_considered_rankings_in_ece, 2)
    base  = f"{dataset_name}_{prop}_{rank_weighting}_abs_vs_jeff_{bin_spacing}"

    print("Visualizing Sub-k ECE comparison (Abs vs Jeff)...")
    _plot_comparison_boxplots(
        df_abs=sub_abs_df,
        df_jeff=sub_jeff_df,
        k_values=k_values_sub_k,
        fig_title=f"Sub-$k$ ECE — Abs vs Jeff — {dataset_name}",
        ylabel="Sub-$k$ ECE",
        save_path=f"{save_folder}subk_ece_comparison_{base}.pdf",
    )

    print("Visualizing Sub-k Full-Rank ECE comparison (Abs vs Jeff)...")
    _plot_comparison_boxplots(
        df_abs=sub_abs_full_rank_df,
        df_jeff=sub_jeff_full_rank_df,
        k_values=k_values_sub_k,
        fig_title=f"Sub-$k$ ECE (Full Rank) — Abs vs Jeff — {dataset_name}",
        ylabel="Sub-$k$ ECE (Full Rank)",
        save_path=f"{save_folder}subk_ece_comparison_full_rank_{base}.pdf",
    )

    print("Visualizing Top-k ECE comparison (Abs vs Jeff)...")
    _plot_comparison_boxplots(
        df_abs=top_abs_df,
        df_jeff=top_jeff_df,
        k_values=k_values_top_k,
        fig_title=f"Top-$k$ ECE — Abs vs Jeff — {dataset_name}",
        ylabel="Top-$k$ ECE",
        save_path=f"{save_folder}topk_ece_comparison_{base}.pdf",
    )

    print("Visualizing Top-k Full-Rank ECE comparison (Abs vs Jeff)...")
    _plot_comparison_boxplots(
        df_abs=top_abs_full_rank_df,
        df_jeff=top_jeff_full_rank_df,
        k_values=k_values_top_k,
        fig_title=f"Top-$k$ ECE (Full Rank) — Abs vs Jeff — {dataset_name}",
        ylabel="Top-$k$ ECE (Full Rank)",
        save_path=f"{save_folder}topk_ece_comparison_full_rank_{base}.pdf",
    )

    print("Visualizing ECE vs k comparison (Abs vs Jeff) with error bands...")
    _plot_comparison_lineplots(
        sub_abs=sub_abs_df,
        top_abs=top_abs_df,
        sub_abs_fr=sub_abs_full_rank_df,
        top_abs_fr=top_abs_full_rank_df,
        sub_jeff=sub_jeff_df,
        top_jeff=top_jeff_df,
        sub_jeff_fr=sub_jeff_full_rank_df,
        top_jeff_fr=top_jeff_full_rank_df,
        k_values_sub_k=k_values_sub_k,
        k_values_top_k=k_values_top_k,
        fig_title=f"ECE vs $k$ — Abs vs Jeff — {dataset_name}",
        save_path=f"{save_folder}ece_vs_k_comparison_{base}.pdf",
    )


# ---------------------------------------------------------------------------
# Hard-coded proportions (matching experiment.calibration CSV naming)
# ---------------------------------------------------------------------------
dataset_to_proportion = {
    "political": 1.0,
    "movies":    0.0,
    "authorship": 1.0
}

if __name__ == "__main__":
    dataset_name   = "political"  # "political", "movies", or "authorship"
    k_values_sub_k = [2, 3, 4, 5]
    k_values_top_k = [1, 2, 3, 4]
    rank_weighting = "95_prob_mass"
    bin_spacing    = "linear"
    proportion     = dataset_to_proportion[dataset_name]

    data_folder = f"results/{dataset_name}/"
    save_folder = "results/improved_plots/"

    def _load(prefix, discrepancy, rank_weighting=rank_weighting, bin_spacing=bin_spacing):
        return pd.read_csv(
            f"{data_folder}{prefix}_{dataset_name}_{round(proportion, 2)}"
            f"_{rank_weighting}_{discrepancy}_{bin_spacing}.csv"
        )

    sub_abs_df          = _load("subk_ece_results",           "abs", rank_weighting=rank_weighting, bin_spacing=bin_spacing)
    top_abs_df          = _load("topk_ece_results",           "abs", rank_weighting=rank_weighting, bin_spacing=bin_spacing)
    sub_abs_fr_df       = _load("subk_full_rank_ece_results", "abs", rank_weighting=rank_weighting, bin_spacing=bin_spacing)
    top_abs_fr_df       = _load("topk_full_rank_ece_results", "abs", rank_weighting=rank_weighting, bin_spacing=bin_spacing)

    sub_jeff_df         = _load("subk_ece_results",           "jeff", rank_weighting="uniform", bin_spacing=bin_spacing)
    top_jeff_df         = _load("topk_ece_results",           "jeff", rank_weighting="uniform", bin_spacing=bin_spacing)
    sub_jeff_fr_df      = _load("subk_full_rank_ece_results", "jeff", rank_weighting="uniform", bin_spacing=bin_spacing)
    top_jeff_fr_df      = _load("topk_full_rank_ece_results", "jeff", rank_weighting="uniform", bin_spacing=bin_spacing)

    # Filter to the desired k values
    sub_abs_df     = sub_abs_df    [sub_abs_df    ["k"].isin(k_values_sub_k)]
    top_abs_df     = top_abs_df    [top_abs_df    ["k"].isin(k_values_top_k)]
    sub_abs_fr_df  = sub_abs_fr_df [sub_abs_fr_df ["k"].isin(k_values_sub_k)]
    top_abs_fr_df  = top_abs_fr_df [top_abs_fr_df ["k"].isin(k_values_top_k)]

    sub_jeff_df    = sub_jeff_df   [sub_jeff_df   ["k"].isin(k_values_sub_k)]
    top_jeff_df    = top_jeff_df   [top_jeff_df   ["k"].isin(k_values_top_k)]
    sub_jeff_fr_df = sub_jeff_fr_df[sub_jeff_fr_df["k"].isin(k_values_sub_k)]
    top_jeff_fr_df = top_jeff_fr_df[top_jeff_fr_df["k"].isin(k_values_top_k)]

    visualize_ece_comparison(
        dataset_name=dataset_name,
        k_values_sub_k=k_values_sub_k,
        k_values_top_k=k_values_top_k,
        sub_abs_df=sub_abs_df,
        top_abs_df=top_abs_df,
        sub_abs_full_rank_df=sub_abs_fr_df,
        top_abs_full_rank_df=top_abs_fr_df,
        sub_jeff_df=sub_jeff_df,
        top_jeff_df=top_jeff_df,
        sub_jeff_full_rank_df=sub_jeff_fr_df,
        top_jeff_full_rank_df=top_jeff_fr_df,
        proportion_of_considered_rankings_in_ece=proportion,
        rank_weighting=rank_weighting,
        bin_spacing=bin_spacing,
        save_folder=save_folder,
    )
