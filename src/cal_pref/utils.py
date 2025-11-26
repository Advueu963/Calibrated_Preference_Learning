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
    y = np.array([possible_rankings[i] for i in y])
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


###################################
## Core ECE Computation Function #
#################################


def calculate_binary_ece(y_true, y_prob, equal_frequency_bins=False, n_bins=10):
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
    probs_range = y_prob.max() - y_prob.min()
    if equal_frequency_bins and probs_range > 0:
        sorted_probs, _ = torch.sort(y_prob)
        bins = [
            sorted_probs[int(i * n_instances_total / n_bins)].item()
            for i in range(n_bins - 1)
        ] + [sorted_probs[-1].item() + 1e-6]
        bins = torch.tensor(bins)
    else:
        bins = torch.linspace(0, 1, n_bins)
    if len(bins) <= 1:
        bin_indices = torch.zeros_like(y_prob, dtype=torch.long)
    else:
        bin_indices = torch.bucketize(y_prob, bins)

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
        y_true (torch.Tensor): The true rankings. Shape (n_samples, n_items)
        y_pred_proba (list[dict[tuple[int], float]]): The predicted probabilities for each ranking
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The constructed tensors
    """
    # Construct the Multi-class tensors
    y_true_sub = np.ones((y_true.shape[0])) * (-2)
    y_prob_sub = np.zeros((y_true.shape[0], len(ranking_to_idx)))
    # Loop over the different instances
    for i, true_ranking in enumerate(y_true):
        # Check which sub-ranking this instance contains
        for sub_k_ranking in possible_sub_k_rankings:
            if check_sub_k_in_ranking(sub_k_ranking, true_ranking.tolist()):
                y_true_sub[i] = ranking_to_idx[tuple(sub_k_ranking)]

        # Aggregate the predicted probabilities for this instance
        for ranking, prob in y_pred_proba.items():
            for sub_k_ranking in possible_sub_k_rankings:
                if check_sub_k_in_ranking(sub_k_ranking, list(ranking)):
                    y_prob_sub[i, ranking_to_idx[sub_k_ranking]] += prob[i]
    return torch.tensor(y_true_sub, dtype=torch.long), torch.tensor(
        y_prob_sub, dtype=torch.float32
    )


def calculate_sub_k_full_rank_calibration(
    items: list[int],
    y_true: torch.Tensor,
    y_pred_proba: list[dict[tuple[int], float]],
    k=2,
    mode="binning",
    h=1,
    p_norm=1.0,
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

    possible_items_sets = list(combinations(items, k))

    sub_k_full_rank_ece = {}

    for item_set in possible_items_sets:
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
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The constructed tensors
    """
    # Construct the Multi-class tensors
    y_true_top_k = np.ones((y_true.shape[0])) * (-2)
    y_prob_top_k = np.zeros((y_true.shape[0], len(ranking_to_idx)))
    # Loop over the different instances
    for i, true_ranking in enumerate(y_true):
        # Check which top-k ranking this instance contains
        for top_k_ranking in possible_top_k_rankings:
            if check_top_k_in_ranking(top_k_ranking, tuple(true_ranking.tolist())):
                y_true_top_k[i] = ranking_to_idx[tuple(top_k_ranking)]

        # Aggregate the predicted probabilities for this instance
        for ranking, prob in y_pred_proba.items():
            for top_k_ranking in possible_top_k_rankings:
                if check_top_k_in_ranking(top_k_ranking, list(ranking)):
                    y_prob_top_k[i, ranking_to_idx[top_k_ranking]] += prob[i]
    return torch.tensor(y_true_top_k, dtype=torch.long), torch.tensor(
        y_prob_top_k, dtype=torch.float32
    )


def calculate_top_k_full_rank_calibration(
    items: list[int],
    y_true: torch.Tensor,
    y_pred_proba: list[dict[tuple[int], float]],
    k=2,
    mode="binning",
    h=1,
    p_norm=1.0,
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

    possible_top_rankings = list(permutations(items, k))

    top_k_full_rank_ece = {}

    rankings_to_idx = {
        ranking: idx for idx, ranking in enumerate(possible_top_rankings)
    }
    y_true_top_k, y_prob_top_k = construct_top_k_full_rank_tensors(
        possible_top_rankings, y_true, y_pred_proba, rankings_to_idx
    )

    if mode == "binning":
        ece_top_ranking = strong_calibration_error_torch(
            y_prob_top_k, y_true_top_k, resolution=10
        )
    elif mode == "kernel":
        ece_top_ranking = strong_calibration_error_dirichlet_kernel(
            y_prob_top_k, y_true_top_k, h=h, p_norm=p_norm
        )

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
    positions_of_sub_ranking = [full_ranking.index(item) for item in sub_ranking]
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
    y_true_sub = np.zeros(y_true.shape[0])
    y_prob_sub = np.zeros(y_true.shape[0])
    # Loop over the different instances
    for i, true_ranking in enumerate(y_true):
        # Check if this instance contains the sub_k_ranking
        if check_sub_k_in_ranking(sub_k_ranking, true_ranking.tolist()):
            y_true_sub[i] = 1.0

        # Aggregate the predicted probabilities for this instance
        for ranking, prob in y_pred_proba.items():
            if check_sub_k_in_ranking(sub_k_ranking, list(ranking)):
                y_prob_sub[i] += prob[i]
    return torch.tensor(y_true_sub), torch.tensor(y_prob_sub, dtype=torch.float32)


def calculate_sub_k_calibration(
    items: list[int], y_true: torch.Tensor, y_pred_proba: dict[tuple[int], float], k=2
):
    """This method calucates the sub_k calibration as definined in our work.
    For this it constructs all rankings of `items` which are of length `k` and then aggregates `y_pred_proba` accordingly.

    Args:
        items (list[int]): The number of items to consider
        y_true (torch.Tensor): The true rankings. Shape (n_samples, n_items)
        y_pred_proba (list[dict[tuple[int], float]]): The predicted probabilities for each ranking
        k (int, optional): The length of the sub-rankings to consider. Defaults to 2.

    Returns:
        dict: The ECE per sub-ranking and the total ECE
    """
    from itertools import combinations, permutations

    possible_sub_rankings = list(permutations(items, k))
    for sub_ranking in possible_sub_rankings:
        # Construct the binary classification tensors
        y_true_sub, y_prob_sub = construct_sub_k_tensors(
            list(sub_ranking), y_true, y_pred_proba
        )
        # Calculate the ECE for this sub-ranking
        ece_sub_ranking = calculate_binary_ece(y_true_sub, y_prob_sub)
        # print("Finished sub-ranking:", sub_ranking, " ECE:", ece_sub_ranking)
        if "sub_rankings_ece" not in locals():
            sub_rankings_ece = [{"sub_ranking": sub_ranking, "ece": ece_sub_ranking}]
        else:
            sub_rankings_ece.append(
                {"sub_ranking": sub_ranking, "ece": ece_sub_ranking}
            )
    total_ece = np.mean([r["ece"] for r in sub_rankings_ece])
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
    y_true_top_k = np.zeros(y_true.shape[0])
    y_prob_top_k = np.zeros(y_true.shape[0])
    # Loop over the different instances
    for i, true_ranking in enumerate(y_true):
        # Check if this instance contains the top_k_ranking
        if check_top_k_in_ranking(top_k_ranking, true_ranking.tolist()):
            y_true_top_k[i] = 1.0

        # Aggregate the predicted probabilities for this instance
        for ranking, prob in y_pred_proba.items():
            if check_top_k_in_ranking(top_k_ranking, list(ranking)):
                y_prob_top_k[i] += prob[i]
    return torch.tensor(y_true_top_k), torch.tensor(y_prob_top_k, dtype=torch.float32)


def calculate_top_k_calibration(
    items: list[int], y_true: torch.Tensor, y_pred_proba: dict[list[int], float], k=2
):
    """This method calucates the top_k calibration as definined in our work.
    For this it constructs all rankings of `items` which are of length `k` and then aggregates `y_pred_proba` accordingly.

    Args:
        items (list[int]): The number of items to consider
        y_true (torch.Tensor): The true rankings. Shape (n_samples, n_items)
        y_pred_proba (list[dict[list[int], float]]): The predicted probabilities for each ranking.
        k (int, optional): The length of the top-k rankings to consider. Defaults to 2.
    Returns:
        dict: The ECE per top-k ranking and the total ECE
    """
    from itertools import combinations, permutations

    possible_top_k_rankings = list(permutations(items, k))
    for top_k_ranking in possible_top_k_rankings:
        # Construct the binary classification tensors
        y_true_top_k, y_prob_top_k = construct_top_k_tensors(
            list(top_k_ranking), y_true, y_pred_proba
        )
        # Calculate the ECE for this top-k ranking
        ece_top_k_ranking = calculate_binary_ece(y_true_top_k, y_prob_top_k)
        if "top_k_rankings_ece" not in locals():
            top_k_rankings_ece = [
                {"top_k_ranking": top_k_ranking, "ece": ece_top_k_ranking}
            ]
        else:
            top_k_rankings_ece.append(
                {"top_k_ranking": top_k_ranking, "ece": ece_top_k_ranking}
            )
    total_ece = np.mean([r["ece"] for r in top_k_rankings_ece])
    return {"top_k_rankings_ece": top_k_rankings_ece, "total_ece": total_ece}


#####################################
## Bradley-Terry to Plackett-Luce  #
###################################
def from_bradley_terry_to_placet_luce_old(rng, pair_order_matrices, n_iterations=20):
    placket_luce_weights = rng.random(
        (pair_order_matrices.shape[0], pair_order_matrices.shape[1])
    )

    n_samples, n_items, _ = pair_order_matrices.shape
    for i_sample in range(n_samples):
        scores = placket_luce_weights[i_sample, :]
        for _ in range(n_iterations):
            for i_weight in range(n_items):
                bradley_win_i = pair_order_matrices[
                    i_sample, i_weight, :
                ]  # shape = (n_items,)
                nominator = bradley_win_i
                denominator = pair_order_matrices[
                    i_sample, i_weight, :
                ] + pair_order_matrices[i_sample, :, i_weight] / (
                    scores[i_weight] + scores + 1e-10
                )
                scores[i_weight] = (nominator / (denominator.sum() + 1e-10)).sum()
        placket_luce_weights[i_sample, :] = scores / scores.sum()
    return placket_luce_weights


def from_bradley_terry_to_placket_luce_simple(
    rng, pair_order_matrices, n_iterations=20
):
    placket_luce_weights = rng.random(
        (pair_order_matrices.shape[0], pair_order_matrices.shape[1])
    )

    n_samples, n_items, _ = pair_order_matrices.shape
    for i_sample in range(n_samples):
        scores = placket_luce_weights[i_sample, :]
        for _ in range(n_iterations):
            for i_weight in range(n_items):
                bradley_win_i = pair_order_matrices[
                    i_sample, i_weight, :
                ]  # shape = (n_items,)
                # Calculate the Upper part of the fraction
                upper_nominator = bradley_win_i * scores[i_weight]  # shape = (n_items,)
                upper_denominator = scores[i_weight] + scores + 1e-10

                # Compute the lower part of the fraction and update the weights
                bradley_lose_i = pair_order_matrices[
                    i_sample, :, i_weight
                ]  # shape = (n_items,)
                lower_nominator = bradley_lose_i
                lower_denominator = upper_denominator

                scores[i_weight] = (upper_nominator / upper_denominator).sum() / (
                    lower_nominator / lower_denominator + 1e-10
                ).sum()
        placket_luce_weights[i_sample, :] = scores
    return placket_luce_weights


def from_bradley_terry_to_placket_luce_vectorized(
    rng, pair_order_matrices, n_iterations=20
):
    placket_luce_weights = rng.random(
        (pair_order_matrices.shape[0], pair_order_matrices.shape[1])
    )

    n_samples, n_items, _ = pair_order_matrices.shape
    for _ in range(n_iterations):
        for item in range(n_items):
            bradley_win_i = pair_order_matrices[
                :, item, :
            ]  # shape = (n_samples, n_items)
            upper_nominator = (
                bradley_win_i * placket_luce_weights[:, item][:, np.newaxis]
            )  # shape = (n_samples, n_items)
            upper_denominator = (
                placket_luce_weights[:, item][:, np.newaxis]
                + placket_luce_weights
                + 1e-10
            )

            bradley_lose_i = pair_order_matrices[
                :, :, item
            ]  # shape = (n_samples, n_items)
            lower_nominator = bradley_lose_i
            lower_denominator = upper_denominator

            placket_luce_weights[:, item] = (upper_nominator / upper_denominator).sum(
                axis=1
            ) / (lower_nominator / lower_denominator + 1e-10).sum(axis=1)
    return placket_luce_weights


def from_bradley_terry_to_placket_luce_map(rng, pair_order_matrices, n_iterations=20):
    placket_luce_weights = rng.random(
        (pair_order_matrices.shape[0], pair_order_matrices.shape[1])
    )

    n_samples, n_items, _ = pair_order_matrices.shape
    for _ in range(n_iterations):
        for item in range(n_items):
            score_term = 1.0 / (placket_luce_weights[:, item] + 1)
            bradley_win_i = pair_order_matrices[
                :, item, :
            ]  # shape = (n_samples, n_items)
            upper_nominator = (
                bradley_win_i * placket_luce_weights[:, item][:, np.newaxis]
            )  # shape = (n_samples, n_items)
            upper_denominator = (
                placket_luce_weights[:, item][:, np.newaxis]
                + placket_luce_weights
                + 1e-10
            )

            bradley_lose_i = pair_order_matrices[
                :, :, item
            ]  # shape = (n_samples, n_items)
            lower_nominator = bradley_lose_i
            lower_denominator = upper_denominator

            upper_term = score_term + (upper_nominator / upper_denominator).sum(axis=1)
            lower_term = score_term + (lower_nominator / lower_denominator + 1e-10).sum(
                axis=1
            )
            placket_luce_weights[:, item] = upper_term / lower_term
    return placket_luce_weights
