"""Batch-generate improved ECE plots for every dataset/config present in
``results/`` by reusing :func:`visualise_ece.visualize_ece_results`.

For each ``results/<dataset>/`` it discovers every
``subk_ece_results_<dataset>_<prop>_<rw>_<disc>_<bin>.csv`` file (together with
its ``topk`` / ``*_full_rank`` siblings), derives the sub-k / top-k ``k`` values
directly from the data, and writes the figures to ``results/improved_plots/``.

Usage:
    uv run python make_improved_plots.py                # all datasets
    uv run python make_improved_plots.py political iris  # only these
"""

import os
import sys

import pandas as pd

from visualise_ece import visualize_ece_results

RESULTS_DIR = "results"
SAVE_FOLDER = "results/improved_plots/"

# Known token vocabularies — used to unambiguously split the config suffix
# (rank weightings such as "95_prob_mass" / "top_10" themselves contain "_").
RANK_WEIGHTINGS = [
    "95_prob_mass",
    "top_10",
    "most_confident",
    "pred_mass",
    "prevalence",
    "uniform",
]
DISCREPANCIES = ["log_ratio", "rel_p", "rel_q", "jeff", "abs", "kl"]
BIN_SPACINGS = ["linear", "log"]


def _parse_config(dataset, stem):
    """Split ``<dataset>_<prop>_<rw>_<disc>_<bin>`` into its components."""
    assert stem.startswith(dataset + "_"), stem
    rest = stem[len(dataset) + 1 :]
    prop_str, rest = rest.split("_", 1)
    for bin_spacing in BIN_SPACINGS:
        for disc in DISCREPANCIES:
            for rw in RANK_WEIGHTINGS:
                if rest == f"{rw}_{disc}_{bin_spacing}":
                    return float(prop_str), rw, disc, bin_spacing
    raise ValueError(f"Unrecognised config suffix: {stem!r}")


def _process_dataset(dataset):
    ds_dir = os.path.join(RESULTS_DIR, dataset)
    subk_files = sorted(
        f
        for f in os.listdir(ds_dir)
        if f.startswith("subk_ece_results_") and f.endswith(".csv")
    )
    if not subk_files:
        print(f"[{dataset}] no subk_ece_results CSVs — skipped")
        return 0

    made = 0
    for subk_name in subk_files:
        stem = subk_name[len("subk_ece_results_") : -len(".csv")]
        try:
            prop, rw, disc, bin_spacing = _parse_config(dataset, stem)
        except ValueError as e:
            print(f"[{dataset}] {e} — skipped")
            continue

        paths = {
            "sub": os.path.join(ds_dir, f"subk_ece_results_{stem}.csv"),
            "top": os.path.join(ds_dir, f"topk_ece_results_{stem}.csv"),
            "sub_fr": os.path.join(ds_dir, f"subk_full_rank_ece_results_{stem}.csv"),
            "top_fr": os.path.join(ds_dir, f"topk_full_rank_ece_results_{stem}.csv"),
        }
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            print(f"[{dataset}] {stem}: missing {missing} — skipped")
            continue

        sub_df = pd.read_csv(paths["sub"])
        top_df = pd.read_csv(paths["top"])
        sub_fr_df = pd.read_csv(paths["sub_fr"])
        top_fr_df = pd.read_csv(paths["top_fr"])

        if sub_df.empty or top_df.empty:
            print(f"[{dataset}] {stem}: empty CSV — skipped")
            continue

        k_values_sub_k = sorted(int(k) for k in sub_df["k"].unique())
        k_values_top_k = sorted(int(k) for k in top_df["k"].unique())

        print(
            f"[{dataset}] config prop={prop} rw={rw} disc={disc} bin={bin_spacing} "
            f"| sub-k={k_values_sub_k} top-k={k_values_top_k}"
        )
        try:
            visualize_ece_results(
                dataset_name=dataset,
                k_values_sub_k=k_values_sub_k,
                k_values_top_k=k_values_top_k,
                sub_df=sub_df,
                top_df=top_df,
                sub_full_rank_df=sub_fr_df,
                top_full_rank_df=top_fr_df,
                proportion_of_considered_rankings_in_ece=prop,
                rank_weighting=rw,
                discrepancy=disc,
                bin_spacing=bin_spacing,
                save_folder=SAVE_FOLDER,
                log_scale=False,
            )
            made += 1
        except Exception as e:  # keep going across configs
            print(f"[{dataset}] {stem}: FAILED ({e}) — skipped")
    return made


def main(argv):
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    if argv:
        datasets = argv
    else:
        datasets = sorted(
            d
            for d in os.listdir(RESULTS_DIR)
            if os.path.isdir(os.path.join(RESULTS_DIR, d)) and d != "improved_plots"
        )

    total = 0
    for ds in datasets:
        total += _process_dataset(ds)
    print(f"\nDone. Generated improved plots for {total} config(s) in {SAVE_FOLDER}")


if __name__ == "__main__":
    main(sys.argv[1:])
