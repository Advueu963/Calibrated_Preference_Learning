import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import torch
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
from .preference_models import PlackettLuceModelWeights, MallowsModel

########################################
## Synthetic Data Generation Function #
######################################


def synthetic_data(rng, name, num_samples=1000, num_features=5, num_items=3):
    possible_rankings = list(itertools.permutations(range(1, num_items + 1)))
    if name == "synthetic_equal_probs":
        probs_of_rankings = np.array(
            [1 / len(possible_rankings)] * len(possible_rankings)
        )
    elif name == "synthetic_skewed_probs":
        probs_of_rankings = np.array(
            [0.4, 0.2, 0.0, 0.2, 0.0, 0.2] + [0.0] * (len(possible_rankings) - 6)
        )
    elif name == "synthetic_pl_probs":
        weights = np.array([[3, 2, 1]])
        pl_model = PlackettLuceModelWeights(weights, n_items=num_items)
        probs_of_rankings = np.array(
            [
                pl_model.predict_proba_ranking(None, torch.tensor(ranking)).item()
                for ranking in possible_rankings
            ]
        )
        rank_to_prob = {
            ranking: prob for ranking, prob in zip(possible_rankings, probs_of_rankings)
        }
        # print("Ranking to Probability Mapping: ", rank_to_prob)
        # normalize numerical issues
        probs_of_rankings = probs_of_rankings / np.sum(probs_of_rankings)
    elif name == "synthetic_mallows_probs":
        reference_ranking = torch.tensor([1, 2, 3])
        dispersion = 0.5
        mallows_model = MallowsModel(
            reference_ranking=reference_ranking, dispersion=dispersion
        )
        probs_of_rankings = np.array(
            [
                mallows_model.predict_proba_ranking(
                    np.array([1]), torch.tensor(ranking)
                ).item()
                for ranking in possible_rankings
            ]
        )
        # normalize numerical issues
        probs_of_rankings = probs_of_rankings / np.sum(probs_of_rankings)
    elif name == "synthetic_random_probs":
        random_probs = rng.random(len(possible_rankings))
        probs_of_rankings = random_probs / random_probs.sum()
    else:
        raise ValueError("Invalid synthetic dataset name.")
    relation_probs_ranks = {
        ranking: prob for ranking, prob in zip(possible_rankings, probs_of_rankings)
    }
    # print("Probability Distribution: ", relation_probs_ranks)
    # print("True probabilities of rankings: ", probs_of_rankings)

    # print("Sum of true probabilities of rankings: ", np.sum(probs_of_rankings))
    X = np.ones((num_samples, num_features))
    y = rng.choice(len(possible_rankings), size=num_samples, p=probs_of_rankings)
    # Return labels as ranks-per-item (inverse permutation), consistent with real datasets.
    sampled_orders = np.array([possible_rankings[i] for i in y], dtype=np.int64)
    n_items = sampled_orders.shape[1]
    sampled_ranks = np.empty_like(sampled_orders)
    sampled_ranks[np.arange(sampled_orders.shape[0])[:, None], sampled_orders - 1] = (
        np.arange(1, n_items + 1, dtype=np.int64)[None, :]
    )
    y = sampled_ranks
    return X, y, probs_of_rankings


def visualize_per_class_probs(
    possible_rankings,
    y_probs_pl,
    y_probs_mallows,
    y_probs_baseline,
    y_probs_pref,
    y_true_probs,
    dataset_name,
):
    plt.figure(figsize=(12, 7))
    bar_width = 0.13
    x = np.arange(len(possible_rankings))
    labels = [">".join([str(x) for x in ranking]) for ranking in possible_rankings]
    means = [
        np.mean(y_probs_pl, axis=0),
        # np.mean(y_probs_pl_brier, axis=0),
        np.mean(y_probs_pref, axis=0),
        np.mean(y_probs_mallows, axis=0),
        np.mean(y_probs_baseline, axis=0),
        y_true_probs,
    ]
    model_labels = [
        "PL",
        "Preference Model",
        "Mallows",
        "RPC_PL",
        "True Probabilities",
    ]
    colors = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#2ca02c",  # green
    ]
    for i, (mean, label, color) in enumerate(zip(means, model_labels, colors)):
        plt.bar(
            x + i * bar_width,
            mean,
            width=bar_width,
            label=label,
            color=color,
            edgecolor="black",
        )
    # Show all difference texts above the tallest bar in each group, stacked vertically and aligned
    true_idx = len(means) - 1
    n_models = true_idx
    for j in range(len(possible_rankings)):
        # Find the top of the tallest bar in this group
        max_height = max([means[m][j] for m in range(len(means))])
        y_start = max_height + 0.03  # start a bit above the tallest bar
        for k, model_idx in enumerate(range(n_models)):
            model_val = means[model_idx][j]
            true_val = means[true_idx][j]
            diff = model_val - true_val
            plt.text(
                x[j] + (n_models - 1) / 2 * bar_width,  # center above group
                y_start + k * 0.04,  # stack vertically
                f"{model_labels[model_idx]}: {diff:+.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=colors[model_idx],
                fontweight="bold",
            )

    plt.xticks(x + (len(means) - 1) * bar_width / 2, labels, rotation=90)
    plt.xlabel("Rankings")
    plt.ylabel("Predicted Probability")
    plt.title("Predicted Ranking Probabilities on Synthetic Data")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"predicted_ranking_probabilities_{dataset_name}.png")
    # plt.show()


def load_lr_data(dataset_name):
    if dataset_name not in [
        "authorship",
        "glass",
        "iris",
        "letter",
        "libras",
        "movies",
        "pendigits",
        "segment",
        "vehicle",
        "vowel",
        "wine",
        "yeast",
        "political",
    ]:
        raise ValueError(
            "Invalid dataset name. Choose from 'authorship', 'glass', 'iris', 'letter', 'libras', 'movies', 'pendigits', 'segment', 'vehicle', 'vowel', 'wine', 'yeast'."
        )

    if dataset_name == "political":
        dataFrame = pd.read_csv(f"src/cal_pref/data/political.csv")

        # Split in Features and Targets
        X, Y = dataFrame.iloc[:, :-6], dataFrame.iloc[:, -6:]

        # Extract the numpy arrays
        X_data = X.values
        Y_data = Y.values

        # Build Numerical Preprocessor
        numerical_cols = [
            i
            for i, colname in enumerate(X.columns)
            if X[colname].dtype in ["int64", "float64"]
        ]
        numerical_transformer = Pipeline(
            steps=[
                ("imputer_nan", SimpleImputer(strategy="most_frequent")),
                ("impute_977", SimpleImputer(missing_values=977, strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Build Categorical Preprocessor
        categorical_cols = [
            i for i, colname in enumerate(X.columns) if X[colname].dtype == "object"
        ]
        categorical_transformer = Pipeline(
            steps=[
                ("imputer_nan", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", numerical_transformer, numerical_cols),
                ("categorical", categorical_transformer, categorical_cols),
            ]
        )
        X = preprocessor.fit_transform(X)
        y = Y_data

    else:
        data = pd.read_csv(f"src/cal_pref/data/{dataset_name}.csv")
        target_columns = [col for col in data.columns if col.startswith("L")]
        X = data.drop(columns=target_columns).values
        y = data[target_columns].values
    return X, y


def kendal_distance(y_true, y_pred, normalize=True):
    n = y_true.shape[1]
    total_pairs = n * (n - 1) / 2
    discordant_pairs = 0

    for i, j in itertools.combinations(range(n), 2):
        true_order = y_true[:, i] - y_true[:, j]
        pred_order = y_pred[:, i] - y_pred[:, j]
        discordant_pairs += torch.sum((true_order * pred_order) < 0).item()

    if normalize:
        return discordant_pairs / (y_true.shape[0] * total_pairs)
    else:
        return discordant_pairs


#######################################
############# ECE helper ##############
#######################################


def _coerce_ranking_probabilities(
    y_pred_proba,
    *,
    n_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[tuple[int, ...]], torch.Tensor]:
    """Normalize predicted ranking probabilities into a (ranking_list, prob_matrix) pair.

    Supported inputs:
    Ranking keys are interpreted as *orderings* (best->worst item IDs).
    Keys may be full rankings of length n_items (e.g. (2,1,3)) or partial
    orderings (e.g. (2,1) meaning 2 > 1).

    Supported inputs:
    - dict: ordering_tuple -> length-n_samples probs (list/np/torch)
    - list[dict]: length n_samples, each dict maps ordering_tuple -> scalar prob for that sample
    """

    def _as_prob_vec(v) -> torch.Tensor:
        t = torch.as_tensor(v, device=device, dtype=dtype)
        if t.ndim == 0:
            t = t.expand(n_samples)
        else:
            t = t.reshape(-1)
        if t.numel() != n_samples:
            raise ValueError(
                f"Probability vector must have length n_samples={n_samples}, got {t.numel()}"
            )
        return t

    # Case 1: per-sample list of dicts.
    if (
        isinstance(y_pred_proba, list)
        and (len(y_pred_proba) == n_samples)
        and all(isinstance(d, dict) for d in y_pred_proba)
    ):
        ranking_set: set[tuple[int, ...]] = set()
        for d in y_pred_proba:
            for r in d.keys():
                ranking_set.add(tuple(r))
        ranking_list = list(ranking_set)
        R = len(ranking_list)
        if R == 0:
            return [], torch.zeros((0, n_samples), device=device, dtype=dtype)
        idx = {r: i for i, r in enumerate(ranking_list)}
        prob_matrix = torch.zeros((R, n_samples), device=device, dtype=dtype)
        for s, d in enumerate(y_pred_proba):
            for r, p in d.items():
                prob_matrix[idx[tuple(r)], s] = float(p)
        return ranking_list, prob_matrix

    # Case 2: dict-of-vectors.
    if isinstance(y_pred_proba, dict):
        ranking_list = [tuple(r) for r in y_pred_proba.keys()]
        if len(ranking_list) == 0:
            return [], torch.zeros((0, n_samples), device=device, dtype=dtype)
        prob_matrix = torch.stack(
            [_as_prob_vec(y_pred_proba[r]) for r in y_pred_proba], dim=0
        )
        return ranking_list, prob_matrix

    # Last resort: try to coerce into dict-of-vectors.
    try:
        y_pred_proba = dict(y_pred_proba)
    except Exception as e:
        raise TypeError(
            "y_pred_proba must be a dict ranking->probs or a list of per-sample dicts"
        ) from e
    return _coerce_ranking_probabilities(
        y_pred_proba, n_samples=n_samples, device=device, dtype=dtype
    )


def filter_rankings_by_occurrence(
    y_true, y_pred_proba, full_order_true, mode="95_prob_mass"
) -> tuple[torch.Tensor, dict]:
    """
    Filter out rankings from y_pred_proba and y_true which do not occur frequently enough in y_true.
    Parameters
    ----------
    y_true : torch.Tensor
        True rankings of shape (n_samples, n_items).
    y_pred_proba : dict {ranking_tuple: torch.Tensor}
        Predicted probabilities for each ranking.
    full_order_true : torch.Tensor
        Full orderings (best->worst) of shape (n_samples, n_items).

    Returns:
    -------
    y_true_filtered : torch.Tensor
        Filtered true rankings.
    y_pred_proba_filtered : dict {ranking_tuple: torch.Tensor}
        Filtered predicted probabilities for each ranking.
    """
    # Consider only those rankings which together account for 95% of the ranking occurences
    ranking_occurences = {}
    for order in full_order_true:
        ranking_tuple = tuple(order.tolist())
        if ranking_tuple not in ranking_occurences:
            ranking_occurences[ranking_tuple] = 0
        ranking_occurences[ranking_tuple] += 1
    total_occurences = sum(ranking_occurences.values())
    sorted_rankings = sorted(
        ranking_occurences.items(), key=lambda x: x[1], reverse=True
    )
    if mode == "95_prob_mass":
        cum_occurences = 0
        selected_rankings = []
        for ranking, count in sorted_rankings:
            selected_rankings.append(ranking)
            cum_occurences += count
            if cum_occurences / total_occurences >= 0.95:
                break
    elif mode == "top_10":
        selected_rankings = [ranking for ranking, count in sorted_rankings[:10]]
    else:
        raise ValueError("Invalid mode for filtering rankings.")
    # print("Selected", round(len(selected_rankings) / len(ranking_occurences), 3)*100, "% of rankings covering (goal was 95%).")

    mask = torch.tensor(
        [tuple(order.tolist()) in selected_rankings for order in full_order_true]
    )
    # Filter out y_true and y_pred_proba to only include selected rankings
    # This will also remove some samples from y_true
    y_pred_proba = {
        ranking: prob[mask]
        for ranking, prob in y_pred_proba.items()
        if _check_ranking_in_selected_rankings(ranking, selected_rankings)
    }
    y_true = y_true[mask]
    # print(
    #     "Filtered y_true to only include selected rankings. New shape:",
    #     y_true.shape,
    # )
    # print(
    #     "Filtered y_pred_proba to only include selected rankings. New size:",
    #     len(list(y_pred_proba.values())[0]),
    # )

    return y_true, y_pred_proba


###################################
## Core ECE Computation Function #
#################################
def calculate_binary_ece_general(
    y_true,
    y_prob,
    discrepancy="abs",  # "abs" | "rel_p" | "rel_q" | "log_ratio" | "kl" | "jeff"
    eps=1e-12,
    bin_spacing="linear",  # "linear" | "log"
):
    n = y_true.shape[0]
    if bin_spacing == "log":
        # Log spaced bins
        bins = torch.logspace(np.log10(eps), 0.0, steps=11)
        bins[0] = 0.0
        bins[-1] = 1.0
    else:
        # Linear spaced bins
        bins = torch.linspace(0.0, 1.0, steps=11)
    idx = torch.bucketize(y_prob, bins, right=True) - 1
    idx = idx.clamp(0, len(bins) - 2)  # n_bins intervals

    ece = 0.0
    for b in range(len(bins) - 1):
        mask = idx == b
        cnt = mask.sum().item()
        if cnt == 0:
            continue

        p_hat = (y_true[mask].float().mean()).item()
        q_hat = (y_prob[mask].float().mean()).item()

        if discrepancy == "abs":
            d = abs(p_hat - q_hat)
        elif discrepancy == "rel_p":
            d = abs(p_hat - q_hat) / (p_hat + eps)
        elif discrepancy == "rel_q":
            d = abs(p_hat - q_hat) / (q_hat + eps)
        elif discrepancy == "log_ratio":
            d = abs(np.log((p_hat + eps) / (q_hat + eps)))
        elif discrepancy == "kl":
            # Bernoulli KL(p||q)
            p = min(max(p_hat, eps), 1 - eps)
            q = min(max(q_hat, eps), 1 - eps)
            d = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
        elif discrepancy == "jeff":
            # Jeffreys divergence
            p = (y_true[mask].float().sum() + 1 / 2) / (
                len(y_true[mask]) + 1
            )  # Jeffrey smoothing for numerical stability
            q = min(max(q_hat, eps), 1 - eps)
            kl_pq = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
            kl_qp = q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))
            d = kl_pq + kl_qp
            d = 1 - np.exp(
                -d
            )  # scale to [0,1], where 1 is maximum divergence and 0 is no divergence
        else:
            raise ValueError(discrepancy)

        ece += (cnt / n) * d

    return float(ece)


def _parse_tace_rank_weighting_spec(spec: str) -> tuple[float, int]:
    """Parse rank_weighting spec for TACE.

    Supported forms:
    - "tace" -> threshold=0.01, n_bins=10
    - "tace@<threshold>" -> n_bins=10
    - "tace@<threshold>@<n_bins>"

    Returns:
        (threshold, n_bins)
    """

    if spec == "tace":
        return 0.01, 10
    if not spec.startswith("tace@"):
        raise ValueError(
            "Invalid TACE spec. Use 'tace' or 'tace@<threshold>' or 'tace@<threshold>@<n_bins>'."
        )
    parts = spec.split("@")
    if len(parts) not in (2, 3):
        raise ValueError(
            "Invalid TACE spec. Use 'tace' or 'tace@<threshold>' or 'tace@<threshold>@<n_bins>'."
        )
    try:
        threshold = float(parts[1])
    except Exception as e:
        raise ValueError(f"Invalid TACE threshold: {parts[1]!r}") from e
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("TACE threshold must be in [0, 1].")

    n_bins = 10
    if len(parts) == 3:
        try:
            n_bins = int(parts[2])
        except Exception as e:
            raise ValueError(f"Invalid TACE n_bins: {parts[2]!r}") from e
    if n_bins <= 0:
        raise ValueError("TACE n_bins must be a positive integer.")
    return threshold, n_bins


def _parse_tva_rank_weighting_spec(spec: str) -> tuple[float, int]:
    """Parse rank_weighting spec for TvA calibration.

    Supported forms:
    - "tva" -> threshold=0.0, n_bins=10
    - "tva@<n_bins>" -> threshold=0.0
    - "tva@<threshold>@<n_bins>"

    Returns:
        (threshold, n_bins)

    Notes:
        TvA here refers to a *single* binary calibration problem per sample:
        correctness of the model's top prediction vs its confidence.
        We still use equal-frequency bins (TACE-style binning) for stability.
    """

    if spec == "tva":
        return 0.0, 10
    if not spec.startswith("tva@"):
        raise ValueError(
            "Invalid TvA spec. Use 'tva' or 'tva@<n_bins>' or 'tva@<threshold>@<n_bins>'."
        )
    parts = spec.split("@")
    if len(parts) == 2:
        # tva@<n_bins>
        try:
            n_bins = int(parts[1])
        except Exception as e:
            raise ValueError(f"Invalid TvA n_bins: {parts[1]!r}") from e
        if n_bins <= 0:
            raise ValueError("TvA n_bins must be a positive integer.")
        return 0.0, n_bins
    if len(parts) == 3:
        # tva@<threshold>@<n_bins>
        try:
            threshold = float(parts[1])
        except Exception as e:
            raise ValueError(f"Invalid TvA threshold: {parts[1]!r}") from e
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("TvA threshold must be in [0, 1].")
        try:
            n_bins = int(parts[2])
        except Exception as e:
            raise ValueError(f"Invalid TvA n_bins: {parts[2]!r}") from e
        if n_bins <= 0:
            raise ValueError("TvA n_bins must be a positive integer.")
        return threshold, n_bins
    raise ValueError(
        "Invalid TvA spec. Use 'tva' or 'tva@<n_bins>' or 'tva@<threshold>@<n_bins>'."
    )


def _parse_topl_tace_rank_weighting_spec(spec: str) -> tuple[int, float, int, bool]:
    """Parse rank_weighting spec for top-L truncated distribution calibration.

    Supported forms:
    - "topl_tace@<L>" -> threshold=0.01, n_bins=10, include_true=True
    - "topl_tace@<L>@<threshold>@<n_bins>" -> include_true=True
    - "topl_tace@<L>@<threshold>@<n_bins>@<include_true>" where include_true is 0/1

    Returns:
        (L, threshold, n_bins, include_true)
    """

    if not spec.startswith("topl_tace@"):
        raise ValueError(
            "Invalid top-L TACE spec. Use 'topl_tace@<L>' or 'topl_tace@<L>@<threshold>@<n_bins>' or add '@<include_true>'."
        )
    parts = spec.split("@")
    if len(parts) not in (2, 4, 5):
        raise ValueError(
            "Invalid top-L TACE spec. Use 'topl_tace@<L>' or 'topl_tace@<L>@<threshold>@<n_bins>' or add '@<include_true>'."
        )
    try:
        L = int(parts[1])
    except Exception as e:
        raise ValueError(f"Invalid top-L value: {parts[1]!r}") from e
    if L <= 0:
        raise ValueError("top-L must be a positive integer.")

    threshold = 0.01
    n_bins = 10
    include_true = True
    if len(parts) >= 4:
        try:
            threshold = float(parts[2])
        except Exception as e:
            raise ValueError(f"Invalid top-L TACE threshold: {parts[2]!r}") from e
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("top-L TACE threshold must be in [0, 1].")
        try:
            n_bins = int(parts[3])
        except Exception as e:
            raise ValueError(f"Invalid top-L TACE n_bins: {parts[3]!r}") from e
        if n_bins <= 0:
            raise ValueError("top-L TACE n_bins must be a positive integer.")
    if len(parts) == 5:
        v = parts[4].strip().lower()
        if v in ("1", "true", "yes"):
            include_true = True
        elif v in ("0", "false", "no"):
            include_true = False
        else:
            raise ValueError("top-L include_true must be 0/1 (or true/false).")
    return L, threshold, n_bins, include_true


def _parse_filter_topl_spec(spec: str) -> tuple[int, bool]:
    """Parse filtering spec for per-sample top-L truncation + renormalization.

    Supported forms:
    - "filter_topl@<L>" -> include_true=False
    - "filter_topl@<L>@<include_true>" where include_true is 0/1 or true/false

    Returns:
        (L, include_true)
    """

    if not spec.startswith("filter_topl@"):
        raise ValueError(
            "Invalid filter_topl spec. Use 'filter_topl@<L>' or 'filter_topl@<L>@<include_true>'."
        )
    parts = spec.split("@")
    if len(parts) not in (2, 3):
        raise ValueError(
            "Invalid filter_topl spec. Use 'filter_topl@<L>' or 'filter_topl@<L>@<include_true>'."
        )
    try:
        L = int(parts[1])
    except Exception as e:
        raise ValueError(f"Invalid filter_topl L: {parts[1]!r}") from e
    if L <= 0:
        raise ValueError("filter_topl L must be a positive integer.")

    include_true = False
    if len(parts) == 3:
        v = parts[2].strip().lower()
        if v in ("1", "true", "yes"):
            include_true = True
        elif v in ("0", "false", "no"):
            include_true = False
        else:
            raise ValueError("filter_topl include_true must be 0/1 (or true/false).")
    return L, include_true


def _filter_pred_proba_topl(
    y_pred_proba,
    *,
    n_samples: int,
    device: torch.device,
    dtype: torch.dtype,
    top_l: int,
    include_true_full_orders: list[tuple[int, ...]] | None,
) -> dict[tuple[int, ...], torch.Tensor]:
    """Filter a full-ranking distribution to the per-sample top-L support and renormalize.

    Input can be a dict-of-vectors or list-of-dicts; output is a dict-of-vectors.
    """

    ranking_list, prob_matrix = _coerce_ranking_probabilities(
        y_pred_proba, n_samples=n_samples, device=device, dtype=dtype
    )
    R = len(ranking_list)
    if R == 0:
        return {}

    probs = prob_matrix.clone()  # (R, n_samples)
    key_to_idx = {tuple(r): i for i, r in enumerate(ranking_list)}

    k_keep = min(int(top_l), R)
    for s in range(n_samples):
        col = probs[:, s]
        if k_keep < R:
            top_idx = torch.topk(col, k_keep, largest=True, sorted=False).indices
            keep_mask = torch.zeros((R,), device=device, dtype=torch.bool)
            keep_mask[top_idx] = True
        else:
            keep_mask = torch.ones((R,), device=device, dtype=torch.bool)

        if include_true_full_orders is not None:
            true_key = include_true_full_orders[s]
            j = key_to_idx.get(true_key, None)
            if j is not None:
                keep_mask[j] = True

        col = col * keep_mask.to(dtype=dtype)
        denom = float(col.sum().item())
        if denom > 0.0:
            col = col / denom
        probs[:, s] = col

    return {tuple(r): probs[i] for i, r in enumerate(ranking_list)}


def _aggregate_event_distribution(
    sample_pred: dict,
    *,
    event: str,
    k: int,
    item_set: tuple[int, ...] | None = None,
) -> dict[tuple[int, ...], float]:
    """Aggregate a per-sample ranking distribution into an event distribution.

    Args:
        sample_pred: dict mapping ranking key -> prob for a single sample.
        event: "topk" or "subk".
        k: event length.
        item_set: required when event=="subk".
    """

    out: dict[tuple[int, ...], float] = {}
    if not isinstance(sample_pred, dict):
        sample_pred = dict(sample_pred)
    for r_key, p in sample_pred.items():
        r = tuple(r_key)
        if len(r) < k:
            continue
        if event == "topk":
            key = r[:k]
        elif event == "subk":
            if item_set is None:
                raise ValueError("item_set is required for subk aggregation")
            key = tuple([int(x) for x in r if int(x) in item_set])
            if len(key) != k:
                continue
        else:
            raise ValueError(f"Unknown event: {event!r}")

        out[key] = out.get(key, 0.0) + float(p)
    return out


def _truncate_and_renormalize_distribution(
    dist: dict[tuple[int, ...], float],
    *,
    top_l: int,
    include_key: tuple[int, ...] | None,
) -> dict[tuple[int, ...], float]:
    """Keep only the top-L keys (by prob), optionally forcing one key to be included, then renormalize."""

    if top_l <= 0:
        raise ValueError("top_l must be positive")

    if len(dist) == 0:
        if include_key is None:
            return {}
        return {include_key: 1.0}

    # Sort by probability mass descending.
    items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    kept = dict(items[:top_l])

    if include_key is not None and include_key not in kept:
        kept[include_key] = float(dist.get(include_key, 0.0))

    total = float(sum(kept.values()))
    if total <= 0.0:
        # Degenerate: fall back to a point mass on include_key if provided, else return original.
        if include_key is not None:
            return {include_key: 1.0}
        return dist
    return {k: v / total for k, v in kept.items()}


def thresholded_adaptive_calibration_error_torch(
    probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_bins: int = 10,
    threshold: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Thresholded Adaptive Calibration Error (TACE-style) for multiclass probs.

    Implementation used in this repo:
    - For each class c, keep samples with p_c > threshold.
    - Equal-frequency bin those retained samples by p_c.
    - Compute weighted |acc - conf| within each bin (weights by bin frequency).
    - Average across classes that have at least one retained sample.

    Notes:
        - Any sample with a negative label is ignored (used as a sentinel in some helpers).
    """

    if probs.ndim != 2:
        raise ValueError("probs must have shape (N, K)")
    if labels.ndim != 1:
        raise ValueError("labels must have shape (N,)")
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("probs and labels must have the same number of samples")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")

    device = probs.device
    dtype = probs.dtype
    N, K = probs.shape

    valid = labels >= 0
    if torch.any(~valid):
        probs = probs[valid]
        labels = labels[valid]
        N = probs.shape[0]
        if N == 0:
            return {"ece": torch.tensor(0.0, device=device, dtype=dtype)}

    per_class_ece: list[torch.Tensor] = []
    for c in range(K):
        p_c = probs[:, c]
        keep = p_c > float(threshold)
        n_keep = int(keep.sum().item())
        if n_keep == 0:
            continue

        p = p_c[keep]
        y = (labels[keep] == c).to(dtype=dtype)

        sorted_p, order = torch.sort(p)
        sorted_y = y[order]

        b = min(n_bins, n_keep)
        edges = torch.linspace(0, n_keep, steps=b + 1, device=device)
        edges = torch.round(edges).to(torch.long)
        edges[0] = 0
        edges[-1] = n_keep

        ece_c = torch.tensor(0.0, device=device, dtype=dtype)
        for bi in range(b):
            start = int(edges[bi].item())
            end = int(edges[bi + 1].item())
            if end <= start:
                continue
            p_bin = sorted_p[start:end]
            y_bin = sorted_y[start:end]
            conf = p_bin.mean()
            acc = y_bin.mean()
            weight = (end - start) / float(n_keep)
            ece_c = ece_c + weight * torch.abs(acc - conf)

        per_class_ece.append(ece_c)

    if len(per_class_ece) == 0:
        return {"ece": torch.tensor(0.0, device=device, dtype=dtype)}
    return {"ece": torch.stack(per_class_ece).mean()}


# def thresholded_adaptive_calibration_error_torch(
#     probs: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     n_bins: int = 10,
#     threshold: float = 0.01,
# ) -> dict[str, torch.Tensor]:
#     """Thresholded Adaptive Calibration Error (TACE-style) for multiclass probs.

#     Motivation for this codebase: when the number of classes is factorial in n_items,
#     most class probabilities are extremely small. Thresholding removes near-zero
#     predictions that otherwise dominate binning artifacts, and adaptive (equal-sample)
#     binning reduces empty-bin issues versus equal-width bins.

#     This follows the standard per-class decomposition:
#     - For each class c, keep samples with p_c > threshold.
#     - Bin remaining samples by quantiles of p_c into ~equal-size bins.
#     - Compute an ECE-style weighted absolute gap |acc - conf| per bin.
#     - Average across classes that have at least one retained sample.

#     Args:
#         probs: Tensor of shape (N, K) with predicted probabilities.
#         labels: Tensor of shape (N,) with integer class labels in [0, K-1].
#         n_bins: Number of equal-frequency bins per class.
#         threshold: Probability threshold for retaining predictions per class.

#     Returns:
#         dict with key "ece" and a scalar torch.Tensor.
#     """

#     if probs.ndim != 2:
#         raise ValueError("probs must have shape (N, K)")
#     if labels.ndim != 1:
#         raise ValueError("labels must have shape (N,)")
#     if probs.shape[0] != labels.shape[0]:
#         raise ValueError("probs and labels must have the same number of samples")
#     if n_bins <= 0:
#         raise ValueError("n_bins must be positive")
#     if threshold < 0.0 or threshold > 1.0:
#         raise ValueError("threshold must be in [0, 1]")

#     device = probs.device
#     dtype = probs.dtype
#     N, K = probs.shape

#     # Ignore any sentinel/invalid labels (some helper tensors use -2 for 'no match').
#     valid = labels >= 0
#     if torch.any(~valid):
#         probs = probs[valid]
#         labels = labels[valid]
#         N = probs.shape[0]
#         if N == 0:
#             return {"ece": torch.tensor(0.0, device=device, dtype=dtype)}

#     per_class_ece = []
#     for c in range(K):
#         p_c = probs[:, c]
#         keep = p_c > float(threshold)
#         n_keep = int(keep.sum().item())
#         if n_keep == 0:
#             continue

#         p = p_c[keep]
#         y = (labels[keep] == c).to(dtype=dtype)

#         # Sort by confidence for equal-frequency bins.
#         sorted_p, order = torch.sort(p)
#         sorted_y = y[order]

#         # Use at most one sample per bin; for tiny n_keep, reduce bin count.
#         b = min(n_bins, n_keep)
#         # Bin edges in index space.
#         edges = torch.linspace(0, n_keep, steps=b + 1, device=device)
#         edges = torch.round(edges).to(torch.long)
#         # Ensure monotonic and last edge is exactly n_keep.
#         edges[0] = 0
#         edges[-1] = n_keep

#         ece_c = torch.tensor(0.0, device=device, dtype=dtype)
#         for bi in range(b):
#             start = int(edges[bi].item())
#             end = int(edges[bi + 1].item())
#             if end <= start:
#                 continue
#             p_bin = sorted_p[start:end]
#             y_bin = sorted_y[start:end]
#             conf = p_bin.mean()
#             acc = y_bin.mean()
#             weight = (end - start) / float(n_keep)
#             ece_c = ece_c + weight * torch.abs(acc - conf)

#         per_class_ece.append(ece_c)

#     if len(per_class_ece) == 0:
#         return {"ece": torch.tensor(0.0, device=device, dtype=dtype)}
#     return {"ece": torch.stack(per_class_ece).mean()}


def thresholded_adaptive_binary_ece_torch(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    *,
    n_bins: int = 10,
    threshold: float = 0.01,
    debug: bool = False,
) -> torch.Tensor:
    """Binary TACE-style ECE: threshold + equal-frequency binning.

    This mirrors the TACE idea for a single "positive class" event:
    keep samples where the model assigns non-trivial mass to the event
    (y_prob > threshold), then compute an ECE with equal-sample bins.

    Args:
        y_true: (N,) tensor with {0,1} labels (or float in [0,1]).
        y_prob: (N,) tensor with predicted probabilities in [0,1].
        n_bins: number of equal-frequency bins.
        threshold: ignore samples with y_prob <= threshold.

    Returns:
        Scalar tensor ECE.
    """
    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("y_true and y_prob must be 1D")
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob must have the same length")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")

    device = y_prob.device
    dtype = y_prob.dtype

    keep = y_prob > float(threshold)
    n_keep = int(keep.sum().item())
    if n_keep == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    p = y_prob[keep].to(dtype=dtype)
    y = y_true[keep].to(dtype=dtype)

    sorted_p, order = torch.sort(p)
    sorted_y = y[order]

    b = min(n_bins, n_keep)
    edges = torch.linspace(0, n_keep, steps=b + 1, device=device)
    edges = torch.round(edges).to(torch.long)
    edges[0] = 0
    edges[-1] = n_keep

    ece = torch.tensor(0.0, device=device, dtype=dtype)
    for bi in range(b):
        start = int(edges[bi].item())
        end = int(edges[bi + 1].item())
        if end <= start:
            continue
        p_bin = sorted_p[start:end]
        y_bin = sorted_y[start:end]
        conf = p_bin.mean()
        acc = y_bin.mean()
        weight = (end - start) / float(n_keep)
        ece = ece + weight * torch.abs(acc - conf)
        if debug:
            print(
                f"Bin {bi}: weight={weight}, acc={acc}, conf={conf}, contrib={weight * torch.abs(acc - conf)}"
            )
    return ece


def calculate_binary_ece(
    y_true: torch.Tensor, y_prob: torch.Tensor, n_bins=10
) -> float:
    """Calculates the ECE for binary classification.

    Args:
        y_true (torch.Tensor): The true binary labels. Shape (n_samples,).
        y_prob (torch.Tensor): The predicted probabilities for the positive class. Shape (n_samples,).
        equal_frequency_bins (bool, optional): Whether to use equal frequency bins. Defaults to False.
        bin_size (int, optional): The number of bins to use. Defaults to 10.
    Returns:
        float: The ECE score.
    """
    n_instances_total = y_true.shape[0]
    # Log spaced
    # edges = torch.logspace(
    #     np.log10(y_prob.min().item()), 0.0, steps=n_bins + 1
    # )
    # edges[0] = 0.0
    # edges[-1] = 1.0
    # Linear spaced
    edges = torch.linspace(0.0, 1.0, steps=n_bins + 1)
    idx = torch.bucketize(y_prob, edges, right=True) - 1
    bin_indices = idx.clamp(0, n_bins - 1)

    # print("Y_PROB MIN MAX: ", y_prob.min().item(), y_prob.max().item())
    # print("Non Empty Bins: ", torch.unique(bin_indices))
    ECE = 0.0
    for bin_idx in range(n_bins):
        bin_mask = bin_indices == bin_idx
        freq_true_in_bin = (
            (y_true[bin_mask]).sum().float() / (bin_mask.sum().float() + 1e-6)
        ).item()
        mean_prob_in_bin = (
            torch.mean(y_prob[bin_mask]).item() if torch.sum(bin_mask) > 0 else 0.0
        )
        # print("Bin:", bin_idx, " Count in bin:", torch.sum(bin_mask).item(), " Freq true in bin:", freq_true_in_bin, " Mean prob in bin:", mean_prob_in_bin)
        count_in_bin = torch.sum(bin_mask).item()
        ECE += (count_in_bin / n_instances_total) * abs(
            freq_true_in_bin - mean_prob_in_bin
        )
    return ECE


def bin_probability_simplex(probs: torch.Tensor, n_bins: int = 10):
    """
    Bins probability vectors on a simplex (K=2 or K=3 classes).

    Parameters
    ----------
    probs : (N, K) tensor
        Predicted probability vectors.
    resolution : int
        Number of bins per axis.

    Returns
    -------
    bin_indices : (N, K-1) tensor of ints
        Bin coordinates for each sample.
    bins : list of tuples
        All valid bin coordinates.
    """
    device = probs.device
    N, K = probs.shape
    step = 1.0 / n_bins

    if K not in (2, 3):
        raise ValueError("Only K=2 or K=3 supported for simplex binning.")

    #
    bins = []
    if K == 2:
        for i in range(n_bins):
            bins.append((i,))
    else:  # K=3
        for i in range(n_bins + 1):
            for j in range(n_bins + 1 - i):
                bins.append((i, j))

    # ---- Compute bin indices for each sample ----
    if K == 2:
        # Only one coordinate matters: p1
        i = torch.clamp((probs[:, 0] / step).long(), max=n_bins - 1)
        bin_indices = torch.stack([i], dim=1)

    else:
        # Two coordinates: p1, p2
        i = torch.clamp((probs[:, 0] / step).long(), max=n_bins)
        j = torch.clamp((probs[:, 1] / step).long(), max=n_bins)

        # Project back into triangular simplex: i + j <= resolution
        overflow = i + j > n_bins
        # If overflow, reduce whichever coordinate is larger
        reduce_i = overflow & (probs[:, 0] >= probs[:, 1])
        reduce_j = overflow & (probs[:, 1] > probs[:, 0])

        i = i.clone()
        j = j.clone()
        i[reduce_i] = n_bins - j[reduce_i]
        j[reduce_j] = n_bins - i[reduce_j]

        bin_indices = torch.stack([i, j], dim=1)

    return bin_indices, bins


def strong_calibration_error_torch(
    probs: torch.Tensor, labels: torch.Tensor, resolution: int = 10
) -> dict[str, float]:
    """
    Computes strong (joint) calibration error using simplex binning in PyTorch.

    Parameters
    ----------
    probs : (N, K) tensor
        Model probability predictions.
    labels : (N,) tensor of ints
        True class labels.
    resolution : int
        Number of bins per axis.

    Returns
    -------
    dict with key "ece" and value the strong calibration error.
    """
    device = probs.device
    N, K = probs.shape

    # 1. Bin each sample
    bin_indices, bins = bin_probability_simplex(probs, resolution)

    # convert bins to a dict mapping bin -> index list
    # (bins is small; iterating in Python is fine)
    # create mapping to bin IDs
    bin_to_id = {b: i for i, b in enumerate(bins)}

    # convert torch bin_indices to a 1D list of bin IDs
    flat_bin_ids = []
    if K == 2:
        for bi in bin_indices[:, 0].tolist():
            flat_bin_ids.append(bin_to_id[(bi,)])
    else:
        for bi, bj in bin_indices.tolist():
            flat_bin_ids.append(bin_to_id[(bi, bj)])
    flat_bin_ids = torch.tensor(flat_bin_ids, device=device, dtype=torch.long)

    # 2. Group by bin ID using scatter
    num_bins = len(bins)

    # bin_counts[b] = number of samples in bin b
    bin_counts = torch.bincount(flat_bin_ids, minlength=num_bins)

    # Sum predicted vectors per bin
    sum_probs = torch.zeros((num_bins, K), device=device)
    sum_probs.index_add_(0, flat_bin_ids, probs)

    # Sum one-hot labels per bin
    onehot = torch.nn.functional.one_hot(labels, num_classes=K).float()
    sum_labels = torch.zeros((num_bins, K), device=device)
    sum_labels.index_add_(0, flat_bin_ids, onehot)

    # 3. Calculate calibration error per bin
    nonempty = bin_counts > 0
    counts = bin_counts[nonempty].unsqueeze(1).float()

    # average predicted distribution in bin
    pred_avg = sum_probs[nonempty] / counts

    # empirical distribution
    emp_avg = sum_labels[nonempty] / counts

    # L1 distance per bin
    l1_dist = torch.abs(pred_avg - emp_avg).sum(dim=1)

    # Weight by bin proportion (ECE-style)
    weights = bin_counts[nonempty].float() / N

    strong_error = (weights * l1_dist).sum()

    return {"ece": strong_error}


def dirichlet_kernel_log(
    f_j: torch.Tensor, alphas: torch.Tensor, eps=1e-8
) -> torch.Tensor:
    """
    Compute log kDir(f_j, f_i) where:
        - f_j are the "evaluation points" (j) of shape (n_samples, n_classes)
        - alphas are the kernel parameters determined by f_i: alpha_i_k = f_i_k / h + 1 of shape (n_samples, n_classes)
    We return a (n_samples, n_classes) tensor of log kernel values.
    Implementation uses:
      log k = lgamma(sum_k alpha_ik) - sum_k lgamma(alpha_ik) + sum_k (alpha_ik - 1) * log(f_jk)

    While the code in the paper is written to sum over i!= j, here we compute the full matrix including i=j.
    Later on, when using this kernel matrix, we can mask out the diagonal if needed.

    Args:
        k_targets (torch.Tensor): The target points f_j of shape (n_samples, n_classes)
        alphas (torch.Tensor): The Dirichlet parameters alpha_i of shape (I, K)
        eps (float, optional): Small value to avoid log(0). Defaults to 1e-8.
    Returns:
        torch.Tensor: The log kernel values of shape (n_samples, I)
    """
    # shapes
    n_samples, n_classes = f_j.shape
    # compute log f_jk
    log_fj = torch.log(f_j.clamp(min=eps))  # (n_samples,n_classes)
    # pieces:
    # The Fractional term infront of the product over k:
    sum_alpha = alphas.sum(dim=1)  # (n_samples,)
    term1 = torch.lgamma(sum_alpha)  # (n_samples,)
    term2 = torch.lgamma(alphas).sum(dim=1)  # (n_samples,)
    # term3: (alpha_ik -1) * log(f_jk) summed over k: results (n_samples, n_samples)
    # Expand to broadcast
    # (n_samples, 1, n_classes) * (1, n_samples, n_classes) -> (n_samples, n_samples, n_classes) then sum over n_classes
    aj = (alphas - 1.0).unsqueeze(0)  # (1,n_samples,n_classes)
    logfj = log_fj.unsqueeze(1)  # (n_samples,1,n_classes)
    term3 = (aj * logfj).sum(dim=2)  # (n_samples, n_samples)
    logk = term1.unsqueeze(0) - term2.unsqueeze(0) + term3  # (n_samples,n_samples)
    return logk


def strong_calibration_error_dirichlet_kernel(
    probs: torch.Tensor,
    labels: torch.Tensor,
    h: float = 0.1,
    p_norm: float = 1.0,
) -> dict[str, float]:
    """
    Computes strong (joint) calibration error using Dirichlet kernel method in PyTorch.

    Parameters
    ----------
    probs : (N, K) tensor
        Model probability predictions.
    labels : (N,) tensor of ints
        True class labels.
    h : float
        Bandwidth parameter for the Dirichlet kernel.

    Returns
    -------
    dict with key "ece" and value the strong calibration error.
    """
    device = probs.device
    dtype = probs.dtype
    K = probs.shape[1]
    n = probs.shape[0]
    EPS = 1e-8
    # Build one-hot labels (n, K)
    y_oh = torch.nn.functional.one_hot(labels, num_classes=K).to(dtype=dtype)

    # alphas for each i: alpha_i_k = f_i_k / h + 1
    alphas = probs / float(h) + 1.0  # (n,K)

    # We'll compute log kDir for all pairs (j,i)
    # logk: (n, n)
    logk = dirichlet_kernel_log(probs, alphas)  # (J=n, I=n)
    # mask self terms
    idx = torch.arange(n, device=device)
    logk[idx, idx] = -float("inf")  # so that exp -> 0

    # Compute denominator terms using log-sum-exp trick
    max_row, _ = torch.max(logk, dim=1, keepdim=True)  # (n,1)
    stable = logk - max_row
    weights = torch.exp(stable)  # (n,n)
    denom = weights.sum(dim=1, keepdim=True)  # (n,1)
    denom = denom + EPS

    # Compute Numerator terms
    numerator = torch.matmul(weights, y_oh)  # (n, K)

    # Compute fraction
    cond_est = numerator / denom  # (n,K)

    # compute ||cond_est - probs||_p^p per row
    dif = cond_est - probs
    dp_dif = torch.norm(dif, p=p_norm, dim=1)  # (n,)
    strong_error = dp_dif.mean()
    return {"ece": strong_error}


###########################################
## Sub-k Full-Rank Calibration Functions #
#########################################


def construct_sub_k_full_rank_tensors(
    possible_sub_k_rankings: list[list[int]],
    y_true: torch.Tensor,
    y_pred_proba: list[dict[tuple[int], float]],
    ranking_to_idx: dict[tuple[int], int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constructs the multi-class classification tensors for the underlying calibration

    Args:
        possible_sub_k_rankings (list[list[int]]): The possible sub-rankings to consider
        y_true (torch.Tensor): True labels as ranks-per-item, shape (n_samples, n_items)
        y_pred_proba: Predicted ranking distribution(s), keyed by *orderings* (best->worst)
        ranking_to_idx (dict[tuple[int], int]): Mapping from sub-ranking to index in the multi-class tensor
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The constructed tensors
    """
    # NOTE: In this project, y_true is ranks-per-item (inverse permutation):
    # y_true[s, j] is the rank (1=best) of item (j+1) for sample s.
    # A sub-ranking [a,b,c] holds iff rank(a) < rank(b) < rank(c).

    device = y_true.device
    dtype = torch.float32
    n_samples, n_items = y_true.shape

    if len(possible_sub_k_rankings) == 0:
        raise ValueError("possible_sub_k_rankings must be non-empty")

    # Normalize to list[list[int]] (permutations() yields tuples).
    possible_sub_k_rankings = [list(r) for r in possible_sub_k_rankings]

    k = len(possible_sub_k_rankings[0])
    n_classes = len(possible_sub_k_rankings)

    # Ensure we use the same class ordering as `ranking_to_idx`.
    if set(ranking_to_idx.keys()) == {tuple(r) for r in possible_sub_k_rankings}:
        ordered_by_idx = [None] * n_classes
        for r, idx in ranking_to_idx.items():
            ordered_by_idx[idx] = list(r)
        possible_sub_k_rankings = ordered_by_idx

    # Validate class shapes & that all classes are permutations of the same item set.
    item_set = list(possible_sub_k_rankings[0])
    if len(item_set) != k or len(set(item_set)) != k:
        raise ValueError("Invalid sub-k ranking length")
    base_set = set(item_set)
    for r in possible_sub_k_rankings:
        if len(r) != k or set(r) != base_set:
            raise ValueError(
                "possible_sub_k_rankings must be permutations of the same item set"
            )

    all_sub = torch.tensor(
        possible_sub_k_rankings, device=device, dtype=torch.long
    )  # (C,k)
    item_set_t = torch.tensor(item_set, device=device, dtype=torch.long)  # (k,)
    item_idx = item_set_t - 1  # 0-based item indices

    # ---- True multi-class labels (N,) ----
    ranks_subset = y_true.index_select(1, item_idx)  # (N,k)
    order_idx = torch.argsort(ranks_subset, dim=1)  # (N,k)
    item_ids = item_set_t.unsqueeze(0).expand(n_samples, k)  # (N,k) item IDs
    sub_order = torch.gather(item_ids, 1, order_idx)  # (N,k) best->worst

    match = (sub_order.unsqueeze(1) == all_sub.unsqueeze(0)).all(dim=2)  # (N,C)
    match_count = match.sum(dim=1)
    if torch.any(match_count > 1):
        raise ValueError(
            "possible_sub_k_rankings contains duplicate classes (a sample matched >1 class)."
        )

    y_true_sub = torch.full((n_samples,), -2, device=device, dtype=torch.long)
    has_match = match_count == 1
    if torch.any(has_match):
        y_true_sub[has_match] = (
            match[has_match].to(torch.int64).argmax(dim=1).to(dtype=torch.long)
        )

    # ---- Predicted probabilities distribution (N,C) ----
    ranking_list, prob_matrix = _coerce_ranking_probabilities(
        y_pred_proba, n_samples=n_samples, device=device, dtype=dtype
    )

    if len(ranking_list) == 0:
        y_prob_sub = torch.zeros((n_samples, n_classes), device=device, dtype=dtype)
        return y_true_sub, y_prob_sub

    full_order = torch.tensor(ranking_list, device=device, dtype=torch.long)  # (R,L)
    if full_order.ndim != 2:
        raise ValueError("Ranking keys in y_pred_proba must be sequences")

    R, L = full_order.shape
    if full_order.numel() > 0:
        if full_order.min().item() < 1 or full_order.max().item() > n_items:
            raise ValueError(
                "Ranking keys in y_pred_proba must use item IDs in [1, n_items]"
            )

    # Positions of each item in each (possibly partial) ranking.
    # Missing items get a large sentinel position.
    sentinel = n_items + 1
    pos = torch.full((R, n_items), sentinel, device=device, dtype=torch.long)
    pos.scatter_(
        1,
        full_order - 1,
        torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(R, L),
    )  # (R,n_items)

    present = torch.zeros((R, n_items), device=device, dtype=torch.bool)
    present.scatter_(
        1,
        full_order - 1,
        torch.ones((R, L), device=device, dtype=torch.bool),
    )  # (R,n_items)

    sub_pos = pos.index_select(1, item_idx)  # (R,k)
    order_idx_r = torch.argsort(sub_pos, dim=1)  # (R,k)
    item_ids_r = item_set_t.unsqueeze(0).expand(R, k)  # (R,k)
    sub_order_r = torch.gather(item_ids_r, 1, order_idx_r)  # (R,k)

    present_sub = present.index_select(1, item_idx)  # (R,k)
    has_all_items = present_sub.all(dim=1)  # (R,)

    match_r = (sub_order_r.unsqueeze(1) == all_sub.unsqueeze(0)).all(dim=2)  # (R,C)
    match_r = match_r & has_all_items.unsqueeze(1)
    match_r_count = match_r.sum(dim=1)
    if torch.any(match_r_count > 1):
        raise ValueError(
            "possible_sub_k_rankings contains duplicate classes (a ranking key matched >1 class)."
        )

    class_idx_r = torch.full((R,), -1, device=device, dtype=torch.long)
    has_match_r = match_r_count == 1
    if torch.any(has_match_r):
        class_idx_r[has_match_r] = (
            match_r[has_match_r].to(torch.int64).argmax(dim=1).to(dtype=torch.long)
        )

    keep = class_idx_r >= 0
    y_prob = torch.zeros((n_classes, n_samples), device=device, dtype=dtype)
    if torch.any(keep):
        y_prob.index_add_(0, class_idx_r[keep], prob_matrix[keep])
    y_prob_sub = y_prob.transpose(0, 1).contiguous()  # (N,C)
    return y_true_sub, y_prob_sub


def calculate_sub_k_full_rank_calibration(
    items: list[int],
    y_true: torch.Tensor,
    y_pred_proba: list[dict[tuple[int], float]],
    k=2,
    mode="binning",
    h=1,
    p_norm=1.0,
    rank_weighting: str = None,
):
    """This method calucates the sub_k calibration as definined in our work.
    For this it constructs all rankings of `items` which are of length `k` and then aggregates `y_pred_proba` accordingly.

    Args:
        items (list[int]): The number of items to consider
        y_true (torch.Tensor): The true rankings. Shape (n_samples, n_items)
        y_pred_proba (list[dict[tuple[int], float]]): The predicted probabilities for each ranking
        k (int, optional): The length of the sub-rankings to consider. Defaults to 2.
        mode (str, optional): The mode to use. Defaults to "binning". Options are "binning" and "kernel".
        h (float, optional): The bandwidth parameter for the kernel mode. Defaults to 1.
        p_norm (float, optional): The p-norm to use for the kernel mode. Defaults to 1.0.

    Returns:
        dict: The ECE per sub-ranking and the total ECE
    """
    from itertools import permutations, combinations

    # y_true is ranks-per-item; derive full best->worst orderings when needed.
    full_order_true = torch.argsort(y_true, dim=1) + 1

    if rank_weighting == "95_prob_mass":
        y_true, y_pred_proba = filter_rankings_by_occurrence(
            y_true, y_pred_proba, full_order_true, mode="95_prob_mass"
        )
    elif rank_weighting == "top_10":
        y_true, y_pred_proba = filter_rankings_by_occurrence(
            y_true, y_pred_proba, full_order_true, mode="top_10"
        )

    possible_items_sets = list(combinations(items, k))

    sub_k_full_rank_ece = {}

    for i in tqdm(range(len(possible_items_sets)), desc="Sub-k Full-Rank ECE"):
        item_set = possible_items_sets[i]
        if y_true.shape[1] >= 8:
            # This would be too computationally expensive. We restrict the permutations to only those which are present in y_true
            unique_sub_k_rankings = set()
            for true_ranking in full_order_true:
                # NOTE: y_true is ranks-per-item, so convert to full order first.
                sub_k_ranking = tuple(
                    [int(item) for item in true_ranking if item in item_set]
                )
                unique_sub_k_rankings.add(sub_k_ranking)
            possible_sub_rankings = list(unique_sub_k_rankings)
        else:
            possible_sub_rankings = list(permutations(item_set, k))
        rankings_to_idx = {
            ranking: idx for idx, ranking in enumerate(possible_sub_rankings)
        }
        y_true_sub, y_prob_sub = construct_sub_k_full_rank_tensors(
            possible_sub_rankings, y_true, y_pred_proba, rankings_to_idx
        )

        if mode == "binning":
            ece_sub_ranking = strong_calibration_error_torch(
                y_prob_sub, y_true_sub, resolution=10
            )
        elif mode == "kernel":
            ece_sub_ranking = strong_calibration_error_dirichlet_kernel(
                y_prob_sub, y_true_sub, h=h, p_norm=p_norm
            )
        else:
            raise ValueError("Invalid mode. Options are 'binning' and 'kernel'.")

        sub_k_full_rank_ece[item_set] = ece_sub_ranking["ece"]
    total_ece = np.mean(list(sub_k_full_rank_ece.values()))
    return {"sub_k_full_rank_ece": sub_k_full_rank_ece, "total_ece": total_ece}


###########################################
## Top-k Full-Rank Calibration Functions #
#########################################
def construct_top_k_full_rank_tensors(
    possible_top_k_rankings: list[list[int]],
    y_true: torch.Tensor,
    y_pred_proba: list[dict[tuple[int], float]],
    ranking_to_idx: dict[tuple[int], int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constructs the multi-class classification tensors for the underlying calibration
        Be aware that it might be the case that not all possible top-k rankings are present in y_true.
        Therefore some classes might not be represented in y_true_top_k, which can lead to issues when calculating calibration errors.
        We recommend to filter those instanceses out before calculating the calibration error.
    Args:
        possible_top_k_rankings (list[list[int]]): The possible top-k rankings to consider
        y_true (torch.Tensor): The true rankings. Shape (n_samples, n_items)
        y_pred_proba (list[dict[tuple[int], float]]): The predicted probabilities for each ranking
        ranking_to_idx (dict[tuple[int], int]): Mapping from ranking to index
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The constructed tensors
    """
    # NOTE: In this project, y_true is ranks-per-item (inverse permutation):
    # y_true[s, j] is the rank (1=best) of item (j+1) for sample s.
    # A top-k ranking [a,b,c] holds iff the best k items are a,b,c in that order.

    device = y_true.device
    dtype = torch.float32
    n_samples, n_items = y_true.shape

    if len(possible_top_k_rankings) == 0:
        raise ValueError("possible_top_k_rankings must be non-empty")

    # Normalize to list[list[int]] (permutations() yields tuples).
    possible_top_k_rankings = [list(r) for r in possible_top_k_rankings]

    k = len(possible_top_k_rankings[0])
    n_classes = len(possible_top_k_rankings)

    # Ensure we use the same class ordering as `ranking_to_idx`.
    if set(ranking_to_idx.keys()) == {tuple(r) for r in possible_top_k_rankings}:
        ordered_by_idx = [None] * n_classes
        for r, idx in ranking_to_idx.items():
            ordered_by_idx[idx] = list(r)
        possible_top_k_rankings = ordered_by_idx

    all_top = torch.tensor(
        possible_top_k_rankings, device=device, dtype=torch.long
    )  # (C,k)

    # ---- True multi-class labels (N,) ----
    full_order_true = torch.argsort(y_true, dim=1) + 1  # (N,n_items) best->worst
    top_k_true = full_order_true[:, :k]  # (N,k)

    match = (top_k_true.unsqueeze(1) == all_top.unsqueeze(0)).all(dim=2)  # (N,C)
    match_count = match.sum(dim=1)
    if torch.any(match_count > 1):
        raise ValueError(
            "possible_top_k_rankings contains duplicate classes (a sample matched >1 class)."
        )

    y_true_top_k = torch.full((n_samples,), -2, device=device, dtype=torch.long)
    has_match = match_count == 1
    if torch.any(has_match):
        y_true_top_k[has_match] = (
            match[has_match].to(torch.int64).argmax(dim=1).to(dtype=torch.long)
        )

    # ---- Predicted probabilities distribution (N,C) ----
    ranking_list, prob_matrix = _coerce_ranking_probabilities(
        y_pred_proba, n_samples=n_samples, device=device, dtype=dtype
    )

    if len(ranking_list) == 0:
        y_prob_top_k = torch.zeros((n_samples, n_classes), device=device, dtype=dtype)
        return y_true_top_k, y_prob_top_k

    full_order = torch.tensor(ranking_list, device=device, dtype=torch.long)  # (R,L)
    if full_order.ndim != 2:
        raise ValueError("Ranking keys in y_pred_proba must be sequences")

    R, L = full_order.shape
    # Check item ID range
    if full_order.numel() > 0:
        if full_order.min().item() < 1 or full_order.max().item() > n_items:
            raise ValueError(
                "Ranking keys in y_pred_proba must use item IDs in [1, n_items]"
            )

    # Predicted distribution keys are orderings. They may be full rankings (L=n_items)
    # or already-aggregated prefixes (L>=k). In both cases, top-k is a prefix event.
    if L < k:
        raise ValueError(
            "Ranking keys in y_pred_proba must have length >= k to extract top-k rankings."
        )

    match_r = (full_order[:, :k].unsqueeze(1) == all_top.unsqueeze(0)).all(
        dim=2
    )  # (R,C)
    match_r_count = match_r.sum(dim=1)
    if torch.any(match_r_count > 1):
        raise ValueError(
            "possible_top_k_rankings contains duplicate classes (a ranking key matched >1 class)."
        )

    class_idx_r = torch.full(
        (full_order.shape[0],), -1, device=device, dtype=torch.long
    )
    has_match_r = match_r_count == 1
    if torch.any(has_match_r):
        class_idx_r[has_match_r] = (
            match_r[has_match_r].to(torch.int64).argmax(dim=1).to(dtype=torch.long)
        )

    keep = class_idx_r >= 0
    y_prob = torch.zeros((n_classes, n_samples), device=device, dtype=dtype)
    if torch.any(keep):
        y_prob.index_add_(0, class_idx_r[keep], prob_matrix[keep])
    y_prob_top_k = y_prob.transpose(0, 1).contiguous()  # (N,C)
    return y_true_top_k, y_prob_top_k


def calculate_top_k_full_rank_calibration(
    items: list[int],
    y_true: torch.Tensor,
    y_pred_proba: list[dict[tuple[int], float]],
    k=2,
    mode="binning",
    h=1,
    p_norm=1.0,
    rank_weighting: str = None,
):
    """This method calucates the top_k full-rank calibration as definined in our work.
    For this it constructs all top-k rankings of `items` which are of length `k` and then aggregates `y_pred_proba` accordingly.

    Args:
        items (list[int]): The number of items to consider.
        y_true (torch.Tensor): The true rankings. Shape (n_samples, n_items).
        y_pred_proba (list[dict[tuple[int], float]]): The predicted probabilities for each ranking.
        k (int, optional): The length of the top-k rankings to consider. Defaults to 2.
        mode (str, optional): The mode to use. Defaults to "binning". Options are "binning" and "kernel".
        h (float, optional): The bandwidth parameter for the kernel mode. Defaults to 1.
        p_norm (float, optional): The p-norm to use for the kernel mode. Defaults to 1.0.
    Returns:
        dict: The ECE per top-k ranking and the total ECE.
    """
    from itertools import permutations, combinations

    # y_true is ranks-per-item; derive full best->worst orderings when needed.
    full_order_true = torch.argsort(y_true, dim=1) + 1

    if rank_weighting == "95_prob_mass":
        y_true, y_pred_proba = filter_rankings_by_occurrence(
            y_true, y_pred_proba, full_order_true, mode="95_prob_mass"
        )
    elif rank_weighting == "top_10":
        y_true, y_pred_proba = filter_rankings_by_occurrence(
            y_true, y_pred_proba, full_order_true, mode="top_10"
        )

    if y_true.shape[1] >= 8:
        # This would be too computationally expensive. We restrict the permutations to only those which are present in y_true
        unique_top_k_rankings = set()
        # NOTE: y_true is ranks-per-item, so convert to full order first.
        for i in range(full_order_true.shape[0]):
            top_k_ranking = tuple(full_order_true[i, :k].tolist())
            unique_top_k_rankings.add(top_k_ranking)
        possible_top_k_rankings = list(unique_top_k_rankings)
    else:
        possible_top_k_rankings = list(permutations(items, k))

    rankings_to_idx = {
        ranking: idx for idx, ranking in enumerate(possible_top_k_rankings)
    }
    y_true_top_k, y_prob_top_k = construct_top_k_full_rank_tensors(
        possible_top_k_rankings, y_true, y_pred_proba, rankings_to_idx
    )

    if mode == "binning":
        ece_top_ranking = strong_calibration_error_torch(
            y_prob_top_k, y_true_top_k, resolution=10
        )
    elif mode == "kernel":
        ece_top_ranking = strong_calibration_error_dirichlet_kernel(
            y_prob_top_k, y_true_top_k, h=h, p_norm=p_norm
        )
    else:
        raise ValueError("Invalid mode. Options are 'binning' and 'kernel'.")

    return {"total_ece": ece_top_ranking["ece"]}


################################
## Sub-k Calibration Functions #
################################


def check_sub_k_in_ranking(sub_ranking, full_ranking) -> bool:
    """This method checks whether the sub_ranking is contained in the full_ranking.

    Args:
        sub_ranking (list[int]): The sub-ranking to check
        full_ranking (list[int]): The full ranking to check against

    Returns:
        bool: True if the sub_ranking is contained in the full_ranking, False otherwise
    """
    try:
        positions_of_sub_ranking = [full_ranking.index(item) for item in sub_ranking]
    except ValueError:
        # print("Item from sub-ranking not found in full ranking. Obviously False.")
        return False
    return positions_of_sub_ranking == sorted(positions_of_sub_ranking)


def construct_sub_k_tensors(
    sub_k_ranking: list[int],
    y_true: torch.Tensor,
    y_pred_proba: list[dict[tuple[int], float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constructs the binary classification tensors for the underlying calibration

    Args:
        sub_k_ranking (list[int]): The sub-ranking to consider
        y_true (torch.Tensor): The true rankings. Shape (n_samples, n_items)
        y_pred_proba (list[dict[tuple[int], float]]): The predicted probabilities for each ranking
    """
    # NOTE: In this project, y_true is ranks-per-item (inverse permutation):
    # y_true[s, j] is the rank (1=best) of item (j+1) for sample s.
    # A sub-ranking [a,b,c] holds iff rank(a) < rank(b) < rank(c).

    device = y_true.device
    dtype = torch.float32
    n_samples, n_items = y_true.shape

    # ---- True binary labels: does sample satisfy the sub-ordering? ----
    item_idx = torch.tensor(sub_k_ranking, device=device, dtype=torch.long) - 1
    ranks = y_true.index_select(1, item_idx)  # (n_samples, k)
    y_true_sub = (
        (ranks[:, :-1] < ranks[:, 1:]).all(dim=1).to(dtype=dtype)
    )  # (n_samples,)

    # ---- Predicted probabilities aggregated over rankings containing the sub-ordering ----
    # y_pred_proba is expected to be a dict: ranking_tuple -> probs_per_sample
    # where ranking_tuple is an ordering (best->worst) of item IDs in {1..n_items}.
    if not isinstance(y_pred_proba, dict):
        # Backwards compatibility for older type hints.
        try:
            y_pred_proba = dict(y_pred_proba)
        except Exception as e:
            raise TypeError(
                "y_pred_proba must be a dict mapping ranking -> probs"
            ) from e

    if len(y_pred_proba) == 0:
        y_prob_sub = torch.zeros((n_samples,), device=device, dtype=dtype)
        return y_true_sub, y_prob_sub

    ranking_list = list(y_pred_proba.keys())

    # If ranking keys are not uniform full rankings, fall back to a safe matcher:
    # a key contributes iff it contains all items in sub_k_ranking in the right order.
    key_lengths = {len(tuple(r)) for r in ranking_list}
    if len(key_lengths) != 1 or (next(iter(key_lengths)) != n_items):
        k = len(sub_k_ranking)
        mask_vals = []
        for r in ranking_list:
            r = tuple(r)
            if len(r) < k:
                mask_vals.append(0.0)
                continue
            # If any required item is missing, cannot match.
            try:
                positions = [r.index(it) for it in sub_k_ranking]
            except ValueError:
                mask_vals.append(0.0)
                continue
            mask_vals.append(float(positions == sorted(positions)))

        prob_matrix = torch.stack(
            [
                torch.as_tensor(y_pred_proba[r], device=device, dtype=dtype)
                for r in ranking_list
            ],
            dim=0,
        )
        mask = torch.as_tensor(mask_vals, device=device, dtype=dtype)
        y_prob_sub = torch.matmul(mask, prob_matrix)
        return y_true_sub, y_prob_sub

    # Stack into (R, n_samples)
    prob_matrix = torch.stack(
        [
            torch.as_tensor(y_pred_proba[r], device=device, dtype=dtype)
            for r in ranking_list
        ],
        dim=0,
    )

    # Build position matrix pos[r, item-1] = position (0=best) of item in ranking r.

    order = torch.tensor(ranking_list, device=device, dtype=torch.long)  # (R, n_items)

    R = order.shape[0]
    pos = torch.empty((R, n_items), device=device, dtype=torch.long)
    pos.scatter_(
        1,
        order - 1,
        torch.arange(n_items, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(R, n_items),
    )

    # Mask rankings that satisfy pos(a) < pos(b) < ...
    sub_pos = pos.index_select(1, item_idx)  # (R, k)
    mask = (sub_pos[:, :-1] < sub_pos[:, 1:]).all(dim=1).to(dtype=dtype)  # (R,)

    # Sum probs across the selected rankings: (R,) @ (R, n_samples) -> (n_samples,)
    y_prob_sub = torch.matmul(mask, prob_matrix)
    assert len(y_prob_sub) == y_true_sub.shape[0]
    return y_true_sub, y_prob_sub


def _check_ranking_in_selected_rankings(
    ranking: tuple[int], selected_rankings: list[tuple[int]]
) -> bool:
    """This method checks whether the ranking is contained in the selected_rankings.
    There exist the special case of RPC which does not output full rankings but only partial rankings.
    In this case, we consider a ranking to be contained in the selected_rankings if the selected ranking is consistent with the full ranking.
    Args:
        ranking (tuple[int]): The ranking to check
        selected_rankings (list[tuple[int]]): The selected rankings to check against

    Returns:
        bool: True if the ranking is contained in the selected_rankings, False otherwise
    """
    if len(ranking) != len(selected_rankings[0]):
        # Partial ranking case
        for selected_ranking in selected_rankings:
            if check_sub_k_in_ranking(list(ranking), list(selected_ranking)):
                return True
        return False
    return ranking in selected_rankings


def calculate_sub_k_calibration(
    items: list[int],
    y_true: torch.Tensor,
    y_pred_proba: dict[tuple[int], float],
    k=2,
    rank_weighting="uniform",
    bin_spacing="linear",
    discrepancy="abs",
    ece_method: str | None = None,
    filter_mode: str | None = None,
    agg_weighting: str | None = None,
):
    """This method calucates the sub_k calibration as definined in our work.
    For this it constructs all rankings of `items` which are of length `k` and then aggregates `y_pred_proba` accordingly.

    Args:
        items (list[int]): The number of items to consider
        y_true (torch.Tensor): The true rankings. Shape (n_samples, n_items)
        y_pred_proba (list[dict[tuple[int], float]]): The predicted probabilities for each ranking
        k (int, optional): The length of the sub-rankings to consider. Defaults to 2.
        rank_weighting (str, optional): The method to weight the ECE values. Defaults to "uniform".

    Returns:
        dict: The ECE per sub-ranking and the total ECE
    """
    from itertools import combinations, permutations

    # --- Backwards compatibility: interpret legacy `rank_weighting` if split args are not provided ---
    if ece_method is None and filter_mode is None and agg_weighting is None:
        if isinstance(rank_weighting, str) and rank_weighting.startswith(
            ("tace", "tva")
        ):
            ece_method = rank_weighting
            agg_weighting = "uniform"
        elif rank_weighting in ("95_prob_mass", "top_10"):
            filter_mode = rank_weighting
            agg_weighting = "uniform"
        else:
            agg_weighting = rank_weighting
    else:
        if agg_weighting is None:
            if rank_weighting in (
                "uniform",
                "prevalence",
                "pred_mass",
            ):
                agg_weighting = rank_weighting
            else:
                agg_weighting = "uniform"

    tace_params: tuple[float, int] | None = None
    tva_params: tuple[float, int] | None = None
    if isinstance(ece_method, str) and ece_method.startswith("tace"):
        tace_params = _parse_tace_rank_weighting_spec(ece_method)
    if isinstance(ece_method, str) and ece_method.startswith("tva"):
        tva_params = _parse_tva_rank_weighting_spec(ece_method)

    # y_true is ranks-per-item; derive full best->worst orderings when needed.
    full_order_true = torch.argsort(y_true, dim=1) + 1

    if filter_mode == "95_prob_mass":
        y_true, y_pred_proba = filter_rankings_by_occurrence(
            y_true, y_pred_proba, full_order_true, mode="95_prob_mass"
        )
    elif filter_mode == "top_10":
        y_true, y_pred_proba = filter_rankings_by_occurrence(
            y_true, y_pred_proba, full_order_true, mode="top_10"
        )
    elif isinstance(filter_mode, str) and filter_mode.startswith("filter_topl"):
        L, include_true = _parse_filter_topl_spec(filter_mode)
        include_orders = (
            [tuple(x.tolist()) for x in full_order_true] if include_true else None
        )
        y_pred_proba = _filter_pred_proba_topl(
            y_pred_proba,
            n_samples=y_true.shape[0],
            device=y_true.device,
            dtype=torch.float32,
            top_l=L,
            include_true_full_orders=include_orders,
        )

    if y_true.shape[1] >= 8:
        # We restrict the permutations to only those which are present in y_true
        unique_sub_rankings = set()
        for true_order in full_order_true:
            for item_combination in combinations(items, k):
                sub_ranking = tuple(
                    item for item in true_order.tolist() if item in item_combination
                )
                if len(sub_ranking) == k:
                    unique_sub_rankings.add(sub_ranking)
        possible_sub_rankings = list(unique_sub_rankings)
    else:
        possible_sub_rankings = list(permutations(items, k))

    # TvA mode for sub-k: pick the single most confident sub-ranking event per sample
    # (events are not mutually exclusive, so this is a multilabel-to-binary reduction).
    if tva_params is not None:
        tva_threshold, tva_bins = tva_params
        device = y_true.device
        dtype = torch.float32
        n_samples = y_true.shape[0]

        best_conf = torch.zeros((n_samples,), device=device, dtype=dtype)
        best_correct = torch.zeros((n_samples,), device=device, dtype=dtype)

        for sub_ranking in tqdm(possible_sub_rankings, desc="Calculating Sub-k TvA"):
            y_true_sub, y_prob_sub = construct_sub_k_tensors(
                list(sub_ranking), y_true, y_pred_proba
            )
            y_true_sub = y_true_sub.to(dtype=dtype)
            y_prob_sub = y_prob_sub.to(dtype=dtype)

            better = y_prob_sub > best_conf
            best_conf = torch.where(better, y_prob_sub, best_conf)
            best_correct = torch.where(better, y_true_sub, best_correct)

        total_ece = float(
            thresholded_adaptive_binary_ece_torch(
                best_correct,
                best_conf,
                n_bins=tva_bins,
                threshold=tva_threshold,
            ).item()
        )
        return {"sub_rankings_ece": [], "total_ece": total_ece}

    sub_rankings_ece = []
    for i in tqdm(range(len(possible_sub_rankings)), desc="Calculating Sub-k ECE"):
        sub_ranking = possible_sub_rankings[i]
        # Construct the binary classification tensors
        y_true_sub, y_prob_sub = construct_sub_k_tensors(
            list(sub_ranking), y_true, y_pred_proba
        )
        # Calculate the ECE for this sub-ranking
        if tace_params is not None:
            threshold, n_bins = tace_params
            ece_sub_ranking = float(
                thresholded_adaptive_binary_ece_torch(
                    y_true_sub, y_prob_sub, n_bins=n_bins, threshold=threshold
                ).item()
            )
        else:
            ece_sub_ranking = calculate_binary_ece_general(
                y_true_sub,
                y_prob_sub,
                discrepancy=discrepancy,
                eps=1e-12,
                bin_spacing=bin_spacing,
            )
        weight_prev = float(y_true_sub.mean().item())
        weight_pred = float(y_prob_sub.mean().item())

        sub_rankings_ece.append(
            {
                "sub_ranking": sub_ranking,
                "ece": ece_sub_ranking,
                "weight_prevalence": weight_prev,
                "weight_pred_mass": weight_pred,
            }
        )
    # Normalize the weights
    total_weight_prev = sum(r["weight_prevalence"] for r in sub_rankings_ece)
    total_weight_pred = sum(r["weight_pred_mass"] for r in sub_rankings_ece)
    for r in sub_rankings_ece:
        r["weight_prevalence"] /= total_weight_prev
        r["weight_pred_mass"] /= total_weight_pred

    if tva_params is not None:
        # TvA mode for sub-k: pick the single most confident sub-ranking event per sample
        # (events are not mutually exclusive, so this is a multilabel-to-binary reduction).
        tva_threshold, tva_bins = tva_params
        device = y_true.device
        dtype = torch.float32
        n_samples = y_true.shape[0]

        best_conf = torch.zeros((n_samples,), device=device, dtype=dtype)
        best_correct = torch.zeros((n_samples,), device=device, dtype=dtype)

        for sub_ranking in tqdm(possible_sub_rankings, desc="Calculating Sub-k TvA"):
            y_true_sub, y_prob_sub = construct_sub_k_tensors(
                list(sub_ranking), y_true, y_pred_proba
            )
            y_true_sub = y_true_sub.to(dtype=dtype)
            y_prob_sub = y_prob_sub.to(dtype=dtype)

            better = y_prob_sub > best_conf
            best_conf = torch.where(better, y_prob_sub, best_conf)
            best_correct = torch.where(better, y_true_sub, best_correct)

        total_ece = float(
            thresholded_adaptive_binary_ece_torch(
                best_correct,
                best_conf,
                n_bins=tva_bins,
                threshold=tva_threshold,
            ).item()
        )
        return {"sub_rankings_ece": [], "total_ece": total_ece}

    if tace_params is not None:
        # TACE path: rank_weighting encodes the ECE method, so we default to uniform
        # aggregation across sub-rankings (consistent with previous "uniform").
        total_ece = np.mean([r["ece"] for r in sub_rankings_ece])
    elif agg_weighting in ["uniform", "95_prob_mass", "top_10"]:
        total_ece = np.mean([r["ece"] for r in sub_rankings_ece])
    elif agg_weighting == "prevalence":
        total_ece = np.sum(
            [r["ece"] * r["weight_prevalence"] for r in sub_rankings_ece]
        )
    elif agg_weighting == "pred_mass":
        total_ece = np.sum([r["ece"] * r["weight_pred_mass"] for r in sub_rankings_ece])
    elif agg_weighting == "max":
        total_ece = np.max([r["ece"] for r in sub_rankings_ece])
    elif agg_weighting == "sum":
        total_ece = np.sum([r["ece"] for r in sub_rankings_ece])
    else:
        raise ValueError(agg_weighting)
    return {"sub_rankings_ece": sub_rankings_ece, "total_ece": total_ece}


################################
## Top-k Calibration Functions #
################################


def check_top_k_in_ranking(top_k_ranking, full_ranking) -> bool:
    """This method checks whether the top_k_ranking matches the full_ranking in the first k positions.

    Args:
        top_k_ranking (list[int]): The top-k ranking to check
        full_ranking (list[int]): The full ranking to check against
    Returns:
        bool: True if the top_k_ranking matches the full_ranking in the first k positions, False otherwise
    """
    if not isinstance(top_k_ranking, list):
        top_k_ranking = list(top_k_ranking)
    if not isinstance(full_ranking, list):
        full_ranking = list(full_ranking)

    return top_k_ranking == full_ranking[0 : len(top_k_ranking)]


def construct_top_k_tensors(
    top_k_ranking: list[int],
    y_true: torch.Tensor,
    y_pred_proba: list[dict[list[int], float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constructs the binary classification tensors for the underlying calibration

    Args:
        top_k_ranking (list[int]): The top-k ranking to consider
        y_true (torch.Tensor): The true rankings. Shape (n_samples, n_items)
        y_pred_proba (list[dict[list[int], float]]): The predicted probabilities for each ranking
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The true labels and predicted probabilities for the top-k ranking
    """
    # NOTE: In this project, y_true is ranks-per-item (inverse permutation):
    # y_true[s, j] is the rank (1=best) of item (j+1) for sample s.
    # A top-k ranking [a,b,c] holds iff the best k items are a,b,c in that order.

    device = y_true.device
    dtype = torch.float32
    n_samples, n_items = y_true.shape

    # Construct the binary y_true_top_k tensor
    item_idx = torch.tensor(top_k_ranking, device=device, dtype=torch.long) - 1
    ranks = y_true.index_select(1, item_idx)  # (n_samples, k)
    k = len(top_k_ranking)
    target_ranks = (
        torch.arange(1, k + 1, device=device, dtype=y_true.dtype)
        .unsqueeze(0)
        .expand(n_samples, k)
    )
    y_true_top = (ranks == target_ranks).all(dim=1).to(dtype=dtype)  # (n_samples,)

    # ---- Predicted probabilities aggregated over rankings containing the sub-ordering ----
    # y_pred_proba is expected to be a dict: ranking_tuple -> probs_per_sample
    # where ranking_tuple is an ordering (best->worst) of item IDs in {1..n_items}.
    if not isinstance(y_pred_proba, dict):
        # Backwards compatibility for older type hints.
        try:
            y_pred_proba = dict(y_pred_proba)
        except Exception as e:
            raise TypeError(
                "y_pred_proba must be a dict mapping ranking -> probs"
            ) from e

    if len(y_pred_proba) == 0:
        y_prob_top = torch.zeros((n_samples,), device=device, dtype=dtype)
        return y_true_top, y_prob_top

    ranking_list = list(y_pred_proba.keys())

    # Distribution keys are orderings (best->worst). They may be:
    # - full rankings of length n_items: (a, b, c, ...)
    # - already-aggregated top-k prefixes of length k: (a, b, ...)
    # We support both without changing semantics.

    key_lens = {len(tuple(r)) for r in ranking_list}
    has_full = n_items in key_lens

    if has_full:
        keys = [r for r in ranking_list if len(tuple(r)) == n_items]
        prob_matrix = torch.stack(
            [
                torch.as_tensor(y_pred_proba[r], device=device, dtype=dtype)
                for r in keys
            ],
            dim=0,
        )
        order = torch.tensor(keys, device=device, dtype=torch.long)  # (R, n_items)
        R = order.shape[0]
        top = (
            torch.tensor(top_k_ranking, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(R, k)
        )
        mask = (order[:, :k] == top).all(dim=1).to(dtype=dtype)  # (R,)
        y_prob_top = torch.matmul(mask, prob_matrix)
        return y_true_top, y_prob_top

    # No full rankings available: fall back to interpreting keys as top-prefixes.
    # If a key is shorter than k, it can't specify top-k.
    keys = [r for r in ranking_list if len(tuple(r)) >= k]
    if len(keys) == 0:
        y_prob_top = torch.zeros((n_samples,), device=device, dtype=dtype)
        return y_true_top, y_prob_top

    prob_matrix = torch.stack(
        [torch.as_tensor(y_pred_proba[r], device=device, dtype=dtype) for r in keys],
        dim=0,
    )
    order = torch.tensor(keys, device=device, dtype=torch.long)  # (R, L)
    R = order.shape[0]
    top = (
        torch.tensor(top_k_ranking, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(R, k)
    )
    mask = (order[:, :k] == top).all(dim=1).to(dtype=dtype)
    y_prob_top = torch.matmul(mask, prob_matrix)
    return y_true_top, y_prob_top


def calculate_top_k_calibration(
    items: list[int],
    y_true: torch.Tensor,
    y_pred_proba: dict[list[int], float],
    k=2,
    rank_weighting="uniform",
    bin_spacing="linear",
    discrepancy="abs",
    ece_method: str | None = None,
    filter_mode: str | None = None,
    agg_weighting: str | None = None,
):
    """This method calucates the top_k calibration as definined in our work.
    For this it constructs all rankings of `items` which are of length `k` and then aggregates `y_pred_proba` accordingly.

    Args:
        items (list[int]): The number of items to consider
        y_true (torch.Tensor): The true rankings. Shape (n_samples, n_items)
        y_pred_proba (list[dict[list[int], float]]): The predicted probabilities for each ranking.
        k (int, optional): The length of the top-k rankings to consider. Defaults to 2.
        rank_weighting (str, optional): The method to weight the ECE values. Defaults to "uniform".
        Options are "uniform", "prevalence", and "pred_mass".
    Returns:
        dict: The ECE per top-k ranking and the total ECE
    """
    from itertools import combinations, permutations

    # --- Backwards compatibility: interpret legacy `rank_weighting` if split args are not provided ---
    if ece_method is None and filter_mode is None and agg_weighting is None:
        if isinstance(rank_weighting, str) and rank_weighting.startswith(
            ("tace", "tva", "topl_tace")
        ):
            ece_method = rank_weighting
            agg_weighting = "uniform"
        elif rank_weighting in ("95_prob_mass", "top_10"):
            filter_mode = rank_weighting
            agg_weighting = "uniform"
        else:
            agg_weighting = rank_weighting
    else:
        if agg_weighting is None:
            if rank_weighting in (
                "uniform",
                "prevalence",
                "pred_mass",
                "most_confident",
            ):
                agg_weighting = rank_weighting
            else:
                agg_weighting = "uniform"

    tace_params: tuple[float, int] | None = None
    tva_params: tuple[float, int] | None = None
    topl_tace_params: tuple[int, float, int, bool] | None = None
    if isinstance(ece_method, str) and ece_method.startswith("tace"):
        tace_params = _parse_tace_rank_weighting_spec(ece_method)
    if isinstance(ece_method, str) and ece_method.startswith("tva"):
        tva_params = _parse_tva_rank_weighting_spec(ece_method)
    if isinstance(ece_method, str) and ece_method.startswith("topl_tace"):
        topl_tace_params = _parse_topl_tace_rank_weighting_spec(ece_method)

    # y_true is ranks-per-item; derive full best->worst orderings when needed.
    full_order_true = torch.argsort(y_true, dim=1) + 1

    if filter_mode == "95_prob_mass":
        y_true, y_pred_proba = filter_rankings_by_occurrence(
            y_true, y_pred_proba, full_order_true, mode="95_prob_mass"
        )
    elif filter_mode == "top_10":
        y_true, y_pred_proba = filter_rankings_by_occurrence(
            y_true, y_pred_proba, full_order_true, mode="top_10"
        )
    elif isinstance(filter_mode, str) and filter_mode.startswith("filter_topl"):
        L, include_true = _parse_filter_topl_spec(filter_mode)
        include_orders = (
            [tuple(x.tolist()) for x in full_order_true] if include_true else None
        )
        y_pred_proba = _filter_pred_proba_topl(
            y_pred_proba,
            n_samples=y_true.shape[0],
            device=y_true.device,
            dtype=torch.float32,
            top_l=L,
            include_true_full_orders=include_orders,
        )

    if y_true.shape[1] >= 8:
        # We restrict the permutations to only those which are present in y_true
        unique_top_k_rankings = set()
        for true_order in full_order_true:
            top_k_ranking = tuple(true_order.tolist()[:k])
            unique_top_k_rankings.add(top_k_ranking)
        possible_top_k_rankings = list(unique_top_k_rankings)
    else:
        possible_top_k_rankings = list(permutations(items, k))

    # TvA mode: per sample, take the most confident predicted top-k ranking and
    # evaluate calibration of correctness vs confidence.
    if tva_params is not None:
        tva_threshold, tva_bins = tva_params
        device = y_true.device
        dtype = torch.float32
        n_samples = y_true.shape[0]

        full_order_true = torch.argsort(y_true, dim=1) + 1
        true_keys = [tuple(full_order_true[s, :k].tolist()) for s in range(n_samples)]
        true_to_idx = {tuple(r): i for i, r in enumerate(possible_top_k_rankings)}
        true_idx = torch.full((n_samples,), -1, device=device, dtype=torch.long)
        for s in range(n_samples):
            j = true_to_idx.get(true_keys[s], None)
            if j is not None:
                true_idx[s] = int(j)

        best_conf = torch.zeros((n_samples,), device=device, dtype=dtype)
        best_idx = torch.full((n_samples,), -1, device=device, dtype=torch.long)

        for j, top_k_ranking in enumerate(
            tqdm(possible_top_k_rankings, desc="Calculating Top-k TvA")
        ):
            _, y_prob_top_k = construct_top_k_tensors(
                list(top_k_ranking), y_true, y_pred_proba
            )
            y_prob_top_k = y_prob_top_k.to(dtype=dtype)
            better = y_prob_top_k > best_conf
            best_conf = torch.where(better, y_prob_top_k, best_conf)
            best_idx = torch.where(
                better,
                torch.tensor(j, device=device, dtype=torch.long),
                best_idx,
            )

        correct = (best_idx == true_idx).to(dtype=dtype)
        ece = float(
            thresholded_adaptive_binary_ece_torch(
                correct, best_conf, n_bins=tva_bins, threshold=tva_threshold
            ).item()
        )
        return {"top_k_rankings_ece": [], "total_ece": ece}

    # Top-L truncated+renormalized distribution calibration (multiclass TACE).
    # Aggregate predicted mass to top-k prefixes, keep top-L per sample, renormalize,
    # then compute multiclass TACE.
    if topl_tace_params is not None:
        top_l, tace_threshold, tace_bins, include_true = topl_tace_params
        device = y_true.device
        dtype = torch.float32
        n_samples = y_true.shape[0]

        full_order_true = torch.argsort(y_true, dim=1) + 1
        true_keys = [tuple(full_order_true[s, :k].tolist()) for s in range(n_samples)]

        # Build per-sample distributions over top-k prefixes.
        ranking_list, prob_matrix = _coerce_ranking_probabilities(
            y_pred_proba, n_samples=n_samples, device=device, dtype=dtype
        )
        per_sample: list[dict[tuple[int, ...], float]] = [{} for _ in range(n_samples)]
        for i_r, r in enumerate(ranking_list):
            r = tuple(r)
            if len(r) < k:
                continue
            key = tuple(r[:k])
            probs_r = prob_matrix[i_r]  # (n_samples,)
            for s in range(n_samples):
                v = float(probs_r[s].item())
                if v == 0.0:
                    continue
                per_sample[s][key] = per_sample[s].get(key, 0.0) + v

        truncated = []
        for s in range(n_samples):
            inc = true_keys[s] if include_true else None
            truncated.append(
                _truncate_and_renormalize_distribution(
                    per_sample[s], top_l=top_l, include_key=inc
                )
            )

        top_list, top_prob_matrix = _coerce_ranking_probabilities(
            truncated, n_samples=n_samples, device=device, dtype=dtype
        )
        idx = {tuple(r): i for i, r in enumerate(top_list)}
        labels = torch.full((n_samples,), -2, device=device, dtype=torch.long)
        for s in range(n_samples):
            j = idx.get(true_keys[s], None)
            if j is not None:
                labels[s] = int(j)

        probs = top_prob_matrix.transpose(0, 1).contiguous()  # (N,K)
        ece = float(
            thresholded_adaptive_calibration_error_torch(
                probs, labels, n_bins=tace_bins, threshold=tace_threshold
            )["ece"].item()
        )
        return {"top_k_rankings_ece": [], "total_ece": ece}

    top_k_rankings_ece = []
    for top_k_ranking in possible_top_k_rankings:
        # Construct the binary classification tensors
        y_true_top_k, y_prob_top_k = construct_top_k_tensors(
            list(top_k_ranking), y_true, y_pred_proba
        )
        # Calculate the ECE for this top-k ranking
        if tace_params is not None:
            threshold, n_bins = tace_params
            ece_top_k_ranking = float(
                thresholded_adaptive_binary_ece_torch(
                    y_true_top_k, y_prob_top_k, n_bins=n_bins, threshold=threshold
                ).item()
            )
        else:
            ece_top_k_ranking = calculate_binary_ece_general(
                y_true_top_k,
                y_prob_top_k,
                discrepancy=discrepancy,
                eps=1e-12,
                bin_spacing=bin_spacing,
            )

        weights_prev = float(y_true_top_k.mean().item())
        weights_pred = float(y_prob_top_k.mean().item())
        top_k_rankings_ece.append(
            {
                "top_k_ranking": top_k_ranking,
                "ece": ece_top_k_ranking,
                "weight_prevalence": weights_prev,
                "weight_pred_mass": weights_pred,
            }
        )
    total_weight_prev = sum(r["weight_prevalence"] for r in top_k_rankings_ece)
    total_weight_pred = sum(r["weight_pred_mass"] for r in top_k_rankings_ece)
    for r in top_k_rankings_ece:
        r["weight_prevalence"] /= total_weight_prev
        r["weight_pred_mass"] /= total_weight_pred

    if tace_params is not None:
        # TACE path: rank_weighting encodes the ECE method, so we default to uniform
        # aggregation across top-k rankings.
        total_ece = np.mean([r["ece"] for r in top_k_rankings_ece])
    elif agg_weighting in ["uniform", "95_prob_mass", "top_10"]:
        total_ece = np.mean([r["ece"] for r in top_k_rankings_ece])
    elif agg_weighting == "prevalence":
        total_ece = np.sum(
            [r["ece"] * r["weight_prevalence"] for r in top_k_rankings_ece]
        )
    elif agg_weighting == "pred_mass":
        total_ece = np.sum(
            [r["ece"] * r["weight_pred_mass"] for r in top_k_rankings_ece]
        )
    elif agg_weighting == "max":
        total_ece = np.max([r["ece"] for r in top_k_rankings_ece])
    elif agg_weighting == "most_confident":
        # Sum only the sub-ranking with 5 the highest predicted mass
        sorted_ece = list(
            sorted(
                top_k_rankings_ece, key=lambda x: x["weight_pred_mass"], reverse=True
            )
        )
        total_ece = sum(r["ece"] for r in sorted_ece[:5]) / 5.0
    elif agg_weighting == "sum":
        total_ece = np.sum([r["ece"] for r in top_k_rankings_ece])
    else:
        raise ValueError(agg_weighting)

    return {"top_k_rankings_ece": top_k_rankings_ece, "total_ece": total_ece}


#####################################
## Bradley-Terry to Plackett-Luce  #
###################################
def from_bradley_terry_to_placet_luce_old(rng, pair_order_matrices, n_iterations=20):
    """Compute Plackett-Luce weights using Zermelo Algorithm based on Bradley Terry weights.

    Args:
        rng (_type_): random Number generator used for inizialisation of the algorithm.
        pair_order_matrices (np.ndarray): The pair order matrix of shape (n_samples, n_items, n_items), where pair_order_matices[s,i,j] the probability of item i being preferred over j for sample s.
        n_iterations (int, optional): The number of iterations for the approximation algorithm. Defaults to 20.

    Returns:
        np.ndarray: The plackett-luce weights of shape (n_samples, n_items)
    """
    eps = 1e-10
    placket_luce_weights = (
        rng.random((pair_order_matrices.shape[0], pair_order_matrices.shape[1])) + eps
    )  # ensure positive init

    n_samples, n_items, _ = pair_order_matrices.shape
    for i_sample in range(n_samples):
        scores = placket_luce_weights[i_sample, :].copy()

        for _ in range(n_iterations):
            for i_weight in range(n_items):
                a_i = pair_order_matrices[i_sample, i_weight, :]  # a_ij
                a_j = pair_order_matrices[i_sample, :, i_weight]  # a_ji

                # exclude diagonal j == i_weight
                mask = np.ones(n_items, dtype=bool)
                mask[i_weight] = False

                wins_i = a_i[mask].sum()
                games_ij = a_i[mask] + a_j[mask]  # a_ij + a_ji

                denom = (games_ij / (scores[i_weight] + scores[mask] + eps)).sum()

                scores[i_weight] = wins_i / (denom + eps)

            # normalize to keep scale stable (PL weights are defined up to a constant)
            s = scores.sum()
            if s > 0:
                scores = scores / s

        placket_luce_weights[i_sample, :] = scores

    return placket_luce_weights


def from_bradley_terry_to_placket_luce_simple(
    rng, pair_order_matrices, n_iterations=20
):
    """Compute Plackett-Luce weights using (improved) Zermelo Algorithm based on Bradley Terry weights.
    This function is the slowest variant, but it is the most straightforward to understand.

    Args:
        rng (_type_): random Number generator used for inizialisation of the algorithm.
        pair_order_matrices (np.ndarray): The pair order matrix of shape (n_samples, n_items, n_items), where pair_order_matices[s,i,j] the probability of item i being preferred over j for sample s.
        n_iterations (int, optional): The number of iterations for the approximation algorithm. Defaults to 20.

    Returns:
        np.ndarray: The plackett-luce weights of shape (n_samples, n_items)
    """
    placket_luce_weights = rng.random(
        (pair_order_matrices.shape[0], pair_order_matrices.shape[1])
    )

    n_samples, n_items, _ = pair_order_matrices.shape
    for i_sample in range(n_samples):
        scores = placket_luce_weights[i_sample, :]
        for _ in range(n_iterations):
            for i_weight in range(n_items):
                a_i = pair_order_matrices[i_sample, i_weight, :]  # a_ij
                a_j = pair_order_matrices[i_sample, :, i_weight]  # a_ji

                # exclude diagonal j == i_weight
                mask = np.ones(n_items, dtype=bool)
                mask[i_weight] = False

                nominator = (
                    a_i[mask] * scores[mask] / (scores[i_weight] + scores[mask] + 1e-10)
                )
                denominator = a_j[mask] / (scores[i_weight] + scores[mask] + 1e-10)
                scores[i_weight] = nominator.sum() / (denominator.sum() + 1e-10)
            # normalize to keep scale stable (PL weights are defined up to a constant)
            s = scores.sum()
            if s > 0:
                scores = scores / s
        placket_luce_weights[i_sample, :] = scores
    return placket_luce_weights


def from_bradley_terry_to_placket_luce_vectorized(
    rng, pair_order_matrices, n_iterations=20
):
    """Vectorized variant of `from_bradley_terry_to_placket_luce_simple`.

    Matches the update:
        w_i <- sum_j a_ij * w_j / (w_i + w_j)   /   sum_j a_ji / (w_i + w_j)
    (excluding j=i), applied independently per sample, with normalization each iteration.
    """
    eps = 1e-10
    n_samples, n_items, _ = pair_order_matrices.shape

    # Ensure float ndarray and don't mutate caller's array.
    A = np.asarray(pair_order_matrices, dtype=float).copy()  # (S, I, J)

    # Exclude diagonal comparisons.
    diag = np.arange(n_items)
    A[:, diag, diag] = 0.0

    # Initialise positive scores.
    placket_luce_weights = rng.random((n_samples, n_items)) + eps
    s0 = placket_luce_weights.sum(axis=1, keepdims=True)
    placket_luce_weights = np.where(
        s0 > 0.0,
        placket_luce_weights / s0,
        np.ones_like(placket_luce_weights) / n_items,
    )

    A_T = np.swapaxes(A, 1, 2)  # (S, I, J) where A_T[:, i, j] = a_ji

    for _ in range(n_iterations):
        # denom_mat[s, i, j] = w_si + w_sj
        denom_mat = (
            placket_luce_weights[:, :, None] + placket_luce_weights[:, None, :] + eps
        )

        numerator = (A * placket_luce_weights[:, None, :] / denom_mat).sum(axis=2)
        denominator = (A_T / denom_mat).sum(axis=2)

        placket_luce_weights = numerator / (denominator + eps)

        # Normalize per sample for stability.
        s = placket_luce_weights.sum(axis=1, keepdims=True)
        placket_luce_weights = np.where(
            s > 0.0,
            placket_luce_weights / s,
            np.ones_like(placket_luce_weights) / float(n_items),
        )

    return placket_luce_weights


def from_bradley_terry_to_placket_luce_map(rng, pair_order_matrices, n_iterations=20):
    """MAP-style variant of the Zermelo update, vectorized.

    This matches the math of the loop-based implementation:

      score_term_i = 1 / (w_i + 1)
      upper_i = score_term_i + sum_j a_ij * w_i / (w_i + w_j)
      lower_i = score_term_i + sum_j a_ji       / (w_i + w_j)
      w_i <- upper_i / lower_i

    (excluding i=j via zero diagonal), applied independently per sample.
    """
    eps = 1e-10
    n_samples, n_items, _ = pair_order_matrices.shape

    A = np.asarray(pair_order_matrices, dtype=float).copy()  # (S, I, J)
    diag = np.arange(n_items)
    A[:, diag, diag] = 0.0
    A_T = np.swapaxes(A, 1, 2)

    placket_luce_weights = rng.random((n_samples, n_items)) + eps

    for _ in range(n_iterations):
        # denom_mat[s, i, j] = w_si + w_sj
        denom_mat = (
            placket_luce_weights[:, :, None] + placket_luce_weights[:, None, :] + eps
        )

        score_term = 1.0 / (placket_luce_weights + 1.0)

        # upper_sum[s, i] = sum_j a_ij * w_i / (w_i + w_j)
        upper_sum = (A * placket_luce_weights[:, :, None] / denom_mat).sum(axis=2)
        # lower_sum[s, i] = sum_j a_ji / (w_i + w_j)
        lower_sum = (A_T / denom_mat).sum(axis=2)

        upper_term = score_term + upper_sum
        lower_term = score_term + lower_sum

        placket_luce_weights = upper_term / (lower_term + eps)

    return placket_luce_weights
