from functools import partial
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
import itertools

from tqdm import tqdm
from cal_pref.utils import (
    from_bradley_terry_to_placket_luce_vectorized,
    from_bradley_terry_to_placket_luce_simple,
    from_bradley_terry_to_placket_luce_map,
)
from sklr.pairwise import PairwisePartialLabelRanker, PairwiseLabelRanker
from sklr.metrics import tau_score
from math import factorial
from cal_pref.preference_models import (
    PreferenceModel,
    PlackettLuceModel,
    MallowsModel,
    PlackettLuceModelWeights,
)
from cal_pref.preference_losses import (
    BrierLoss,
    BrierPreferenceLoss,
    LogLossPreferenceLoss,
    PlackettLuceLoss,
    PlackettLuceBrierPreferenceLoss,
)
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from cal_pref.utils import (
    load_lr_data,
    synthetic_data,
    calculate_sub_k_calibration,
    calculate_top_k_calibration,
    calculate_sub_k_full_rank_calibration,
    calculate_top_k_full_rank_calibration,
)

parser = argparse.ArgumentParser(
    description="Preference Learning Calibration Experiment"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="synthetic_pl_probs",
)
parser.add_argument(
    "--rank_weighting",
    type=str,
    default="uniform",
    help='Rank weighting scheme for ECE calculation. Options are "uniform", "prevalence", "pred_mass", and "most_confident".',
)
parser.add_argument(
    "--discrepancy",
    type=str,
    default="abs",
    help='Discrepancy measure for ECE calculation. Options are "abs", "jeff", "log_ratio", "rel_p", "rel_q", "kl".',
)
parser.add_argument(
    "--bin_spacing",
    type=str,
    default="linear",
    help='Bin spacing for ECE calculation. Options are "linear" and "log".',
)
args = parser.parse_args()


def get_preference_models(
    input_dim, n_items, hidden_dims, output_dim, y, constant_value=0.0
):
    """Get preference models for the given dimensions.

    Args:
        input_dim (int): The dimension of the input features.
        n_items (int): The number of items to rank.
        hidden_dims (list): The dimensions of the hidden layers.
        output_dim (int): The amount of unique rankings in the training set.
        y (np.array): All rankings in the dataset. Used to determine the most occurrent ranking for Mallows model.
    Returns:
        models_optimizer_criterion (dict): A dictionary containing the models, their optimizers and criterions.
    """
    # Preference Model with Brier Loss. Has as many outputs as there are items times positions
    preference_model = PreferenceModel(
        input_dim,
        n_items,
        hidden_dims,
        output_dim,
        torch.tensor(np.unique(y, axis=0)),
        constant_value=constant_value,
    )
    # criterion = BrierPreferenceLoss(maximal_t_list_size=n_items**n_items)
    # criterion = LogLossPreferenceLoss(maximal_t_list_size=factorial(n_items))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(preference_model.parameters(), lr=0.001)

    # Plackett-Luce Model with standard PL Loss
    placket_luce_model = PlackettLuceModel(input_dim, hidden_dims, y.shape[1])
    placket_criterion = PlackettLuceLoss()
    placket_optimizer = torch.optim.Adam(placket_luce_model.parameters(), lr=0.001)

    # Plackett-Luce Model with Brier Loss
    placket_luce_model_brier = PlackettLuceModel(input_dim, hidden_dims, y.shape[1])
    placket_brier_criterion = PlackettLuceBrierPreferenceLoss()
    placket_brier_optimizer = torch.optim.Adam(
        placket_luce_model_brier.parameters(), lr=0.001
    )

    # Mallows Model
    mallows_model = None

    # RPC Baseline with Calibrated Decision Tree
    estimator = DecisionTreeClassifier()
    estimator = CalibratedClassifierCV(
        estimator=estimator, cv=2, method="sigmoid", n_jobs=1
    )
    baseline_estimator = PairwiseLabelRanker(
        estimator=estimator, n_jobs=int(os.environ.get("OMP_NUM_THREADS", 1))
    )

    models_optimizer_criterion = {
        "PreferenceModel": (preference_model, criterion, optimizer),
        "PlackettLuceModel": (placket_luce_model, placket_criterion, placket_optimizer),
        "PlackettLuceModelBrier": (
            placket_luce_model_brier,
            placket_brier_criterion,
            placket_brier_optimizer,
        ),
        "MallowsModel": (mallows_model, None, None),
        "RPC_PL": (baseline_estimator, None, None),
    }
    return models_optimizer_criterion


def train_plackett_luce_model(
    plackett_luce_model,
    plackett_criterion,
    plackett_optimizer,
    X_train_tensor,
    y_train_tensor,
    num_epochs,
    batch_size,
):
    """Train the Plackett-Luce model.

    Args:
        placket_luce_model (nn.Module): Plackett-Luce model to be trained.
        placket_criterion (nn.Module): Loss function for the Plackett-Luce model.
        placket_optimizer (torch.optim.Optimizer): Optimizer for the Plackett-Luce model.
        X_train_tensor (torch.Tensor): Training data features.
        y_train_tensor (torch.Tensor): Training data labels.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    """
    ###### Training Loop #######
    generator = torch.Generator().manual_seed(42)
    for epoch in tqdm(range(num_epochs), desc="Training Plackett-Luce Model"):
        X_batch_indices = torch.randperm(X_train_tensor.size(0), generator=generator)
        for i in range(0, X_train_tensor.size(0), batch_size):
            batch_indices = X_batch_indices[i : i + batch_size]
            X_batch = X_train_tensor[batch_indices]
            y_batch = y_train_tensor[batch_indices]

            plackett_luce_model.train()
            plackett_optimizer.zero_grad()
            logits = plackett_luce_model(X_batch)

            loss = plackett_criterion(y_batch, logits, plackett_luce_model)
            # MSE between predicted distribution and empirical distribution

            loss.backward()

            plackett_optimizer.step()
            # print(f"Epoch {epoch}, Batch {i//batch_size}, Loss: {loss.item()}")


def train_preference_model(
    preference_model,
    criterion,
    optimizer,
    X_train_tensor,
    y_train_tensor,
    num_epochs,
    batch_size,
):
    """Train the Preference model.

    Args:
        preference_model (nn.Module): Preference model to be trained.
        criterion (nn.Module): Loss function for the Preference model.
        optimizer (torch.optim.Optimizer): Optimizer for the Preference model.
        X_train_tensor (torch.Tensor): Training data features.
        y_train_tensor (torch.Tensor): Training data labels.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    """
    ###### Training Loop #######
    generator = torch.Generator().manual_seed(42)
    for epoch in tqdm(range(num_epochs), desc="Training Preference Model"):
        X_batch_indices = torch.randperm(X_train_tensor.size(0), generator=generator)
        for i in range(0, X_train_tensor.size(0), batch_size):
            batch_indices = X_batch_indices[i : i + batch_size]
            X_batch = X_train_tensor[batch_indices]
            y_batch = y_train_tensor[batch_indices]

            preference_model.train()
            optimizer.zero_grad()
            logits = preference_model(X_batch).float()
            # probs = torch.softmax(logits, dim=1)
            y_batch_idx = torch.tensor(
                [preference_model.idx_rankings[tuple(r.tolist())] for r in y_batch],
                device=logits.device,
            ).long()
            # loss_pref = criterion(logits, y_batch_idx)
            loss_pref = criterion(logits, y_batch_idx)
            loss_pref.backward()
            optimizer.step()


def train_placket_luce_model_brier(
    placket_luce_model_brier,
    placket_brier_criterion,
    placket_brier_optimizer,
    X_train_tensor,
    y_train_tensor,
    num_epochs,
    batch_size,
):
    """Train the Plackett-Luce model with Brier loss.

    Args:
        placket_luce_model_brier (nn.Module): Plackett-Luce model to be trained.
        placket_brier_criterion (nn.Module): Brier loss function for the Plackett-Luce model.
        placket_brier_optimizer (torch.optim.Optimizer): Optimizer for the Plackett-Luce model.
        X_train_tensor (torch.Tensor): Training data features.
        y_train_tensor (torch.Tensor): Training data labels.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    """
    ###### Training Loop #######
    generator = torch.Generator().manual_seed(42)
    for epoch in range(num_epochs):
        X_batch_indices = torch.randperm(X_train_tensor.size(0), generator=generator)
        for i in range(0, X_train_tensor.size(0), batch_size):
            batch_indices = X_batch_indices[i : i + batch_size]
            X_batch = X_train_tensor[batch_indices]
            y_batch = y_train_tensor[batch_indices]

            placket_luce_model_brier.train()
            placket_brier_optimizer.zero_grad()
            logits_brier = placket_luce_model_brier(X_batch)
            y_train_pred_brier = placket_luce_model_brier.predict(X_batch)
            loss_brier = placket_brier_criterion(
                y_batch, y_train_pred_brier, logits_brier, placket_luce_model_brier
            )
            loss_brier.backward()
            placket_brier_optimizer.step()


def train_placket_luce_rpc_model(
    baseline_estimator, X_train, y_train, X_test, n_items, method_rpc_pl="vectorized"
):
    """Train a Plackett-Luce model with a rank-based loss function.

    Args:
        baseline_estimator (nn.Module): Calibrated RPC model to be trained.
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data labels.
        X_test (np.ndarray): Test data features.
        n_items (int): Number of items to rank.
    """
    baseline_estimator = baseline_estimator.fit(X_train, y_train)
    print(
        "Tau Score of RPC Baseline on Train Set: ",
        tau_score(y_train, baseline_estimator.predict(X_train)),
    )
    baseline_estimator_matrix = baseline_estimator.get_pairwise_matrix(X_test)
    print("Running RPC to PL conversion...")
    match method_rpc_pl:
        case "vectorized":
            placket_luce_weights = from_bradley_terry_to_placket_luce_vectorized(
                np.random.default_rng(42),
                baseline_estimator_matrix,
                n_iterations=N_ITERATIONS,
            )
        case "simple":
            placket_luce_weights = from_bradley_terry_to_placket_luce_simple(
                np.random.default_rng(42),
                baseline_estimator_matrix,
                n_iterations=N_ITERATIONS,
            )
        case "map":
            placket_luce_weights = from_bradley_terry_to_placket_luce_map(
                np.random.default_rng(42),
                baseline_estimator_matrix,
                n_iterations=N_ITERATIONS,
            )
        case _:
            raise ValueError("Invalid method for RPC Plackett-Luce model.")

    # Tes Case setting the weights to explicit nmnber

    placket_luce_model_baseline = PlackettLuceModelWeights(
        placket_luce_weights, n_items=n_items
    )
    return placket_luce_model_baseline, baseline_estimator_matrix


def evaluate_placket_luce_model(
    placket_luce_model, placket_criterion, X_test_tensor, y_test_tensor
):
    """Evaluate the Plackett-Luce model.
    Args:
        placket_luce_model (nn.Module): Plackett-Luce model to be evaluated.
        placket_criterion (nn.Module): Loss function for the Plackett-Luce model.
        X_test_tensor (torch.Tensor): Test data features.
        y_test_tensor (torch.Tensor): Test data labels.
    Returns:
        kendal_dist (float): Kendall's tau distance on the test set.
        test_loss (float): Test loss value.
    """

    placket_luce_model.eval()

    with torch.no_grad():

        y_test_pred = placket_luce_model.predict(X_test_tensor)
        logits = placket_luce_model(X_test_tensor)
        test_loss = placket_criterion(y_test_tensor, logits, placket_luce_model)
        print(f"Test PL Loss: {test_loss.item()}")

        kendal_dist = tau_score(y_test_tensor, y_test_pred)

    return kendal_dist, test_loss.item()


def evaluate_preference_model(
    preference_model, preference_criterion, X_test_tensor, y_test_tensor
):
    """Evaluate the Preference model.
    Args:
        preference_model (nn.Module): Preference model to be evaluated.
        preference_criterion (nn.Module): Loss function for the Preference model.
        X_test_tensor (torch.Tensor): Test data features.
        y_test_tensor (torch.Tensor): Test data labels.
    Returns:
        kendal_dist (float): Kendall's tau distance on the test set.
        test_loss (float): Test loss value.
    """

    preference_model.eval()

    with torch.no_grad():

        logits = preference_model(X_test_tensor)
        y_test_idx = torch.tensor(
            [preference_model.idx_rankings[tuple(r.tolist())] for r in y_test_tensor],
            device=logits.device,
        ).long()
        y_test_pred = preference_model.predict(X_test_tensor)
        # test_loss = preference_criterion(
        #     logits,
        #     y_test_idx,
        # )
        test_loss = preference_criterion(
            torch.softmax(logits, dim=1),
            y_test_idx,
        )
        # print(f"Test Preference Model Loss: {test_loss.item()}")

        kendal_dist = tau_score(y_test_tensor, y_test_pred)

    return kendal_dist, test_loss.item()


def evaluate_placket_luce_brier_model(
    placket_luce_model_brier, placket_brier_criterion, X_test_tensor, y_test_tensor
):
    """Evaluate the Plackett-Luce model with Brier loss.
    Args:
        placket_luce_model_brier (nn.Module): Plackett-Luce model to be evaluated.
        placket_brier_criterion (nn.Module): Brier loss function for the Plackett-Luce model.
        X_test_tensor (torch.Tensor): Test data features.
        y_test_tensor (torch.Tensor): Test data labels.
    Returns:
        kendal_dist (float): Kendall's tau distance on the test set.
        test_loss (float): Test loss value.
    """

    placket_luce_model_brier.eval()

    with torch.no_grad():

        y_test_pred_brier = placket_luce_model_brier.predict(X_test_tensor)
        logits_brier = placket_luce_model_brier(X_test_tensor)
        test_loss_brier = placket_brier_criterion(
            y_test_tensor, y_test_pred_brier, logits_brier, placket_luce_model_brier
        )
        print(f"Test PL Brier Loss: {test_loss_brier.item()}")

        kendal_dist = tau_score(y_test_tensor, y_test_pred_brier)

    return kendal_dist, test_loss_brier.item()


def evaluate_mallows_model(
    mallows_model, mallows_criterion, X_test_tensor, y_test_tensor
):
    """Evaluate the Mallows model.

    Args:
        mallows_model (nn.Module): Mallows model to be evaluated.
        mallows_criterion (nn.Module): Loss function for the Mallows model.
        X_test_tensor (torch.Tensor): Test data features.
        y_test_tensor (torch.Tensor): Test data labels.

    Returns:
        kendal_dist (float): Kendall's tau distance on the test set.
        test_loss (float): Test loss value.
    """
    with torch.no_grad():
        y_test_pred = mallows_model.predict(X_test_tensor)
        kendal_dist = tau_score(y_test_tensor, y_test_pred)
        # print(f"Kendal Distance on Test Set (Mallows): {kendal_dist}")

    return kendal_dist, None


def evaluate_placket_luce_rpc_model(
    placket_luce_model_baseline, criterion, X_test_tensor, y_test_tensor
):
    """Evaluate a Plackett-Luce model with a rank-based loss function.
    Args:
        placket_luce_model_baseline (nn.Module): Plackett-Luce model, whose weights stem from a RPC model, to be evaluated.
        criterion (nn.Module): Not used, only for compatibility.
        X_test_tensor (torch.Tensor): Test data features.
        y_test_tensor (torch.Tensor): Test data labels.
    Returns:
        kendal_dist (float): Kendall's tau distance on the test set.
    """

    placket_luce_model_baseline.eval()

    with torch.no_grad():

        y_baseline_pred = placket_luce_model_baseline.predict(X_test_tensor)

        kendal_dist = tau_score(y_test_tensor, y_baseline_pred)
        # print(f"Kendal Distance on Test Set (PL RPC): {kendal_dist}")

    return kendal_dist, None


def evaluate_calibrated_rpc_model(
    baseline_estimator, criterion, X_test_tensor, y_test_tensor
):
    """Evaluate a calibrated RPC model.
    Args:
        baseline_estimator (nn.Module): Calibrated RPC model to be evaluated.
        criterion (nn.Module): Not used, only for compatibility.
        X_test_tensor (torch.Tensor): Test data features.
        y_test_tensor (torch.Tensor): Test data labels.
    Returns:
        kendal_dist (float): Kendall's tau distance on the test set.
    """
    with torch.no_grad():

        y_baseline_pred = baseline_estimator.predict(X_test_tensor)

        kendal_dist = tau_score(y_test_tensor, y_baseline_pred)
        # print(f"Kendal Distance on Test Set (Calibrated RPC): {kendal_dist}")

    return kendal_dist, None


def evaluate_kendal_models(
    models,
    criterions,
    model_names,
    evaluate_functions,
    X_test_tensor,
    y_test_tensor,
):
    """Evaluate all models and return their Kendall's tau distances and test losses.

    Args:
        placket_luce_model (nn.Module): Plackett-Luce model to be evaluated.
        placket_criterion (nn.Module): Loss function for the Plackett-Luce model.
        preference_model (nn.Module): Preference model to be evaluated.
        criterion (nn.Module): Loss function for the Preference model.
        placket_luce_model_brier (nn.Module): Plackett-Luce model with Brier loss to be evaluated.
        placket_brier_criterion (nn.Module): Brier loss function for the Plackett-Luce model.
        placket_luce_model_baseline (nn.Module): Plackett-Luce model with rank-based loss to be evaluated.
        X_test_tensor (torch.Tensor): Test data features.
        y_test_tensor (torch.Tensor): Test data labels.
    Returns:
        results (dict): A dictionary containing the Kendall's tau distances and test losses for all models.
    """
    results = {}

    for model, criterion, model_name, eval_func in zip(
        models, criterions, model_names, evaluate_functions
    ):
        kendal_dist, test_loss = eval_func(
            model, criterion, X_test_tensor, y_test_tensor
        )
        results[model_name] = (kendal_dist, test_loss)

    return results


def evaluate_placket_luce_rpc_vs_placket_luce(
    placket_luce_model,
    placket_luce_model_baseline,
    X_test_tensor,
    possible_rankings,
    y_true_probs,
    baseline_estimator_matrix,
    placket_luce_weights_vectorized,
    placket_luce_weights_simple,
    placket_luce_weights_map,
):
    """Evaluate the Plackett-Luce model with respect to the baseline model, printing the learned weights and pairwise probabilities.
       This method only looks at the first sample in the test set.

    Args:
        placket_luce_model (nn.Module): Plackett-Luce model to be evaluated.
        placket_luce_model_baseline (nn.Module): Baseline Plackett-Luce model.
        X_test_tensor (torch.Tensor): Test data features.
        possible_rankings (list): List of possible rankings.
        y_true_probs (list): True probabilities of the rankings.
        baseline_estimator_matrix (np.ndarray): Baseline estimator matrix.
        placket_luce_weights_vectorized (np.ndarray): Vectorized weights of the Plackett-Luce model.
        placket_luce_weights_simple (np.ndarray): Simple weights of the Plackett-Luce model.
        placket_luce_weights_map (np.ndarray): Map weights of the Plackett-Luce model.
    """
    ranks_to_probs = {
        ranking: prob for ranking, prob in zip(possible_rankings, y_true_probs)
    }
    print("True probabilities of rankings: ", ranks_to_probs)
    print("Sum of true probabilities of rankings: ", np.sum(y_true_probs))
    pairwise_probs = {
        (1, 2): sum(
            [
                prob
                for ranking, prob in ranks_to_probs.items()
                if ranking.index(1) < ranking.index(2)
            ]
        ),
        (1, 3): sum(
            [
                prob
                for ranking, prob in ranks_to_probs.items()
                if ranking.index(1) < ranking.index(3)
            ]
        ),
        (2, 3): sum(
            [
                prob
                for ranking, prob in ranks_to_probs.items()
                if ranking.index(2) < ranking.index(3)
            ]
        ),
    }
    pairwise_probs[(2, 1)] = 1 - pairwise_probs[(1, 2)]
    pairwise_probs[(3, 1)] = 1 - pairwise_probs[(1, 3)]
    pairwise_probs[(3, 2)] = 1 - pairwise_probs[(2, 3)]
    print("BASE MATRIX FROM RPC: ", baseline_estimator_matrix[0])
    print("True pairwise probabilities: ", pairwise_probs)
    rpc_pairwise_probs = {
        (1, 2): baseline_estimator_matrix[0, 0, 1],
        (1, 3): baseline_estimator_matrix[0, 0, 2],
        (2, 3): baseline_estimator_matrix[0, 1, 2],
        (2, 1): baseline_estimator_matrix[0, 1, 0],
        (3, 1): baseline_estimator_matrix[0, 2, 0],
        (3, 2): baseline_estimator_matrix[0, 2, 1],
    }
    diff_rpc_true = {
        k: rpc_pairwise_probs[k] - pairwise_probs[k] for k in pairwise_probs.keys()
    }
    print(
        "Difference RPC - True pairwise probabilities: ",
        sum(diff_rpc_true.values()) / len(diff_rpc_true),
        diff_rpc_true,
    )
    print("RPC estimated pairwise probabilities: ", rpc_pairwise_probs)
    logit_weights = placket_luce_model(X_test_tensor[0:1, :]).detach().numpy()
    print(
        "Learned Weights of PL Model: ",
        np.exp(logit_weights),
        " without exp ",
        logit_weights,
    )

    print("VECTORIZED BT TO PL WEIGHTS: ", placket_luce_weights_vectorized)
    print("SIMPLE BT TO PL WEIGHTS: ", placket_luce_weights_simple)
    print("MAP BT TO PL WEIGHTS: ", placket_luce_weights_map)
    print(
        "Learned Weights of PL (RPC) Model: ",
        placket_luce_model_baseline.weights,
    )


def visualize_tau_ece_rankwise_ece(
    res_eces, res_tau_dist, model_names, res_ranking_wise_eces, dataset_name, T
):
    """Visualize ECE and Kendall's Tau scores.
        The res_rankwise_eces is the rank-wise ECE for (fixed) T.

    Args:
        res_eces (np.ndarray): Description of the ECE results. Shape (n_folds, n_models).
        res_tau_dist (np.ndarray): Description of the Kendall's Tau distribution. Shape (n_folds, n_models).
        model_names (list): List of model names. Length n_models.
        res_ranking_wise_eces (np.ndarray): Description of the rank-wise ECE results. Shape (n_folds, n_models).
        dataset_name (str): Name of the dataset.
        T (int): Description of T.
    """
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        res_eces,
        tick_labels=model_names,
    )
    plt.ylabel("Class-wise ECE")
    plt.title("Class-wise ECE across 5 folds")
    plt.grid(axis="y")
    plt.savefig(f"ece_boxplot_restricted_{dataset_name}.png")
    # plt.show()

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        res_tau_dist,
        tick_labels=model_names,
    )
    plt.ylabel("Kendall's Tau Score")
    plt.title("Kendall's Tau Score across 5 folds")
    plt.grid(axis="y")
    plt.savefig(f"kendall_tau_boxplot_restricted_{dataset_name}.png")
    # plt.show()

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        res_ranking_wise_eces,
        tick_labels=model_names,
    )
    plt.ylabel(f"Rank-wise ECE (T={T})")
    plt.title(f"Rank-wise ECE (T={T}) across 5 folds")
    plt.grid(axis="y")
    plt.savefig(f"rankwise_ece_boxplot_restricted_{dataset_name}.png")
    # plt.show()


def get_probs_for_rankings(possible_rankings, models, model_names, X_test_tensor):
    """Get predicted probabilities for all possible rankings from the models.

    Args:
        possible_rankings (list): List of possible rankings.
        models (list[nn.Module]): List of model instances.
        model_names (list): List of model names.
        X_test_tensor (torch.Tensor): Test data features.
    Returns:
        results (dict): A dictionary containing the predicted probabilities for all possible rankings from each model.
    """

    results = {model_name: [] for model_name in model_names}
    with torch.no_grad():
        for model, model_name in zip(models, model_names):
            for ranking in possible_rankings:
                prob_pl = model.predict_proba_ranking(
                    X_test_tensor, torch.tensor(ranking)
                ).numpy()
                results[model_name].append(prob_pl)

            results[model_name] = np.array(
                results[model_name]
            ).T  # shape = (n_samples, n_rankings)

    return results


def get_pairwise_probs(
    models, model_names, X_test_tensor, possible_rankings, y_true_probs, n_items
):
    """Get predicted pairwise probabilities from the models.

    Args:
        models (list[nn.Module]): List of model instances.
        model_names (list): List of model names.
        X_test_tensor (torch.Tensor): Test data features.
        possible_rankings (list): List of possible rankings.
        y_true_probs (list): True probabilities of the rankings.
        n_items (int): Number of items to rank.
    Returns:
        results (dict): A dictionary containing the predicted pairwise probabilities from each model and the true probabilities.

    """
    ranking_probabilities = get_probs_for_rankings(
        possible_rankings, models, model_names, X_test_tensor
    )
    possible_pairs = list(itertools.combinations(range(1, n_items + 1), 2))
    results = {model_name: {} for model_name in model_names}
    for model_name in model_names:
        results[model_name] = {
            pair: np.sum(
                prob
                for ranking, prob in zip(
                    possible_rankings,
                    np.mean(ranking_probabilities[model_name], axis=0),
                )
                if ranking.index(pair[0]) < ranking.index(pair[1])
            )
            for pair in possible_pairs
        }

    results["True"] = {
        pair: sum(
            prob
            for ranking, prob in zip(possible_rankings, y_true_probs)
            if ranking.index(pair[0]) < ranking.index(pair[1])
        )
        for pair in possible_pairs
    }
    return results


def _ranks_to_order_np(ranks: np.ndarray) -> np.ndarray:
    """Convert ranks-per-item (N, n_items) to best->worst orderings (N, n_items)."""
    return np.argsort(ranks, axis=1) + 1


def evaluate_calibration_rankwise_sub_k_top_k(
    distributions,
    model_names,
    y_test_tensor,
    possible_k_sub_k,
    possible_k_top_k,
    rankwise_sub_k_eces,
    rankwise_top_k_eces,
    rank_weighting="uniform",
    discrepancy="abs",
    bin_spacing="linear",
):
    """Evaluate rank-wise calibration for sub-k and top-k.

    Args:
        distributions (list[dict[tuple[int,...], float]]): List of predicted distributions from the models.
        model_names (list): List of model names.
        y_test_tensor (torch.Tensor): Test data labels.
        possible_k_sub_k (list): List of k values for sub-k evaluation.
        possible_k_top_k (list): List of k values for top-k evaluation.
        rank_weighting (str): Rank weighting scheme used in ECE calculation.
        Options are "uniform", "prevalence", and "pred_mass".
    Returns:
        rankwise_sub_k_eces (dict): Dictionary containing rank-wise sub-k ECEs for each model and k.
        rankwise_top_k_eces (dict): Dictionary containing rank-wise top-k ECEs for each model and k.
    """
    items = list(range(1, y_test_tensor.shape[1] + 1))
    for distribution, model_name in zip(distributions, model_names):
        for k in possible_k_sub_k:
            if model_name == "RPC":
                if k > 2:
                    ece_sub_k = {"total_ece": -1}
                else:
                    ece_sub_k = calculate_sub_k_calibration(
                        items=items,
                        y_true=y_test_tensor,
                        y_pred_proba=distribution,
                        k=k,
                        rank_weighting=rank_weighting,
                        discrepancy=discrepancy,
                        bin_spacing=bin_spacing,
                    )
            else:
                # For all other models we just simply calculate the sub-k calibration
                print("ECE SUB K CALCULATION FOR MODEL: ", model_name, " K: ", k)
                ece_sub_k = calculate_sub_k_calibration(
                    items=items,
                    y_true=y_test_tensor,
                    y_pred_proba=distribution,
                    k=k,
                    rank_weighting=rank_weighting,
                    discrepancy=discrepancy,
                    bin_spacing=bin_spacing,
                )
            rankwise_sub_k_eces[k][-1].append(ece_sub_k["total_ece"])

        for k in possible_k_top_k:
            if model_name == "RPC":
                ece_top_k = {"total_ece": -1}
            else:
                ece_top_k = calculate_top_k_calibration(
                    items=items,
                    y_true=y_test_tensor,
                    y_pred_proba=distribution,
                    k=k,
                    rank_weighting=rank_weighting,
                    discrepancy=discrepancy,
                    bin_spacing=bin_spacing,
                )
            rankwise_top_k_eces[k][-1].append(ece_top_k["total_ece"])
    for k in possible_k_sub_k:
        rankwise_sub_k_eces[k].append([])
    for k in possible_k_top_k:
        rankwise_top_k_eces[k].append([])
    return rankwise_sub_k_eces, rankwise_top_k_eces


def evaluate_calibration_full_rank_sub_k_top_k(
    distributions,
    model_names,
    y_test_tensor,
    possible_k_sub_k,
    possible_k_top_k,
    rankwise_full_rank_sub_k_eces,
    rankwise_full_rank_top_k_eces,
    h=1,
    p_norm=1,
    rank_weighting="uniform",
):
    """Evaluate full-rank calibration for sub-k and top-k.

    Args:
        distributions (list[dict[tuple[int,...], float]]): List of predicted distributions from the models.
        model_names (list): List of model names.
        y_test_tensor (torch.Tensor): Test data labels.
        possible_k_sub_k (list): List of k values for sub-k evaluation.
        possible_k_top_k (list): List of k values for top-k evaluation.
        h (float): Bandwidth parameter for kernel calibration.
        p_norm (int): Norm to use for distance calculation.
    Returns:
        rankwise_full_rank_sub_k_eces (dict): Dictionary containing full-rank sub-k ECEs for each model and k.
        rankwise_full_rank_top_k_eces (dict): Dictionary containing full-rank top-k ECEs for each model and k.
    """
    items = list(range(1, y_test_tensor.shape[1] + 1))
    for distribution, model_name in zip(distributions, model_names):
        for k in possible_k_sub_k:
            if model_name == "RPC":
                if k > 2:
                    ece_sub_k = {"total_ece": -1}
                else:
                    ece_sub_k = calculate_sub_k_full_rank_calibration(
                        items=items,
                        y_true=y_test_tensor,
                        y_pred_proba=distribution,
                        k=k,
                        mode="kernel",
                        h=h,
                        p_norm=p_norm,
                        rank_weighting=rank_weighting,
                    )
            else:
                # For all other models we just simply calculate the sub-k calibration
                ece_sub_k = calculate_sub_k_full_rank_calibration(
                    items=items,
                    y_true=y_test_tensor,
                    y_pred_proba=distribution,
                    k=k,
                    mode="kernel",
                    h=h,
                    p_norm=p_norm,
                    rank_weighting=rank_weighting,
                )
            rankwise_full_rank_sub_k_eces[k][-1].append(ece_sub_k["total_ece"])

        for k in possible_k_top_k:
            if model_name == "RPC":
                ece_top_k = {"total_ece": -1}
            else:
                ece_top_k = calculate_top_k_full_rank_calibration(
                    items=items,
                    y_true=y_test_tensor,
                    y_pred_proba=distribution,
                    k=k,
                    mode="kernel",
                    h=h,
                    p_norm=p_norm,
                    rank_weighting=rank_weighting,
                )
            rankwise_full_rank_top_k_eces[k][-1].append(ece_top_k["total_ece"])
    for k in possible_k_sub_k:
        rankwise_full_rank_sub_k_eces[k].append([])
    for k in possible_k_top_k:
        rankwise_full_rank_top_k_eces[k].append([])
    return rankwise_full_rank_sub_k_eces, rankwise_full_rank_top_k_eces


def construct_possible_rankings(n_items):
    """Construct all possible rankings for n_items.

    Args:
        n_items (int): Number of items to rank.
    Returns:
        possible_rankings (list): List of all possible rankings.
    """
    items = list(range(1, n_items + 1))
    possible_rankings = list(itertools.permutations(items))
    possible_rankings = [list(ranking) for ranking in possible_rankings]
    return possible_rankings


def create_ece_reports(
    POSSIBLE_K_SUB_K,
    POSSIBLE_K_TOP_K,
    rankwise_sub_k_eces,
    rankwise_top_k_eces,
    rankwise_full_rank_sub_k_eces,
    rankwise_full_rank_top_k_eces,
    model_names,
):
    sub_k_matrix = np.stack(
        [np.asarray(rankwise_sub_k_eces[k], dtype=float) for k in POSSIBLE_K_SUB_K],
        axis=1,
    )
    sub_k_matrix = np.transpose(sub_k_matrix, (2, 1, 0))

    top_k_matrix = np.stack(
        [np.asarray(rankwise_top_k_eces[k], dtype=float) for k in POSSIBLE_K_TOP_K],
        axis=1,
    )
    top_k_matrix = np.transpose(top_k_matrix, (2, 1, 0))

    sub_k_full_rank_matrix = np.stack(
        [
            np.asarray(rankwise_full_rank_sub_k_eces[k], dtype=float)
            for k in POSSIBLE_K_SUB_K
        ],
        axis=1,
    )
    sub_k_full_rank_matrix = np.transpose(sub_k_full_rank_matrix, (2, 1, 0))

    top_k_full_rank_matrix = np.stack(
        [
            np.asarray(rankwise_full_rank_top_k_eces[k], dtype=float)
            for k in POSSIBLE_K_TOP_K
        ],
        axis=1,
    )
    top_k_full_rank_matrix = np.transpose(top_k_full_rank_matrix, (2, 1, 0))

    sns.set_theme(style="whitegrid", context="talk")

    sub_records = []
    for m, model_name in enumerate(model_names):
        for j, k in enumerate(POSSIBLE_K_SUB_K):
            for fold_idx in range(sub_k_matrix.shape[2]):
                sub_records.append(
                    {
                        "model": model_name,
                        "k": POSSIBLE_K_SUB_K[j],
                        "fold": fold_idx,
                        "ece": sub_k_matrix[m, j, fold_idx],
                    }
                )
    sub_df = pd.DataFrame(sub_records)
    sub_df["k_label"] = sub_df["k"].astype(str)

    top_records = []
    for m, model_name in enumerate(model_names):
        for j, k in enumerate(POSSIBLE_K_TOP_K):
            for fold_idx in range(top_k_matrix.shape[2]):
                if model_name != "RPC":
                    top_records.append(
                        {
                            "model": model_name,
                            "k": POSSIBLE_K_TOP_K[j],
                            "fold": fold_idx,
                            "ece": top_k_matrix[m, j, fold_idx],
                        }
                    )
    top_df = pd.DataFrame(top_records)
    top_df["k_label"] = top_df["k"].astype(str)

    sub_full_rank_records = []
    for m, model_name in enumerate(model_names):
        for j, k in enumerate(POSSIBLE_K_SUB_K):
            for fold_idx in range(sub_k_full_rank_matrix.shape[2]):
                sub_full_rank_records.append(
                    {
                        "model": model_name,
                        "k": POSSIBLE_K_SUB_K[j],
                        "fold": fold_idx,
                        "ece": sub_k_full_rank_matrix[m, j, fold_idx],
                    }
                )
    sub_full_rank_df = pd.DataFrame(sub_full_rank_records)
    sub_full_rank_df["k_label"] = sub_full_rank_df["k"].astype(str)

    top_full_rank_records = []
    for m, model_name in enumerate(model_names):
        for j, k in enumerate(POSSIBLE_K_TOP_K):
            for fold_idx in range(top_k_full_rank_matrix.shape[2]):
                if model_name != "RPC":
                    top_full_rank_records.append(
                        {
                            "model": model_name,
                            "k": POSSIBLE_K_TOP_K[j],
                            "fold": fold_idx,
                            "ece": top_k_full_rank_matrix[m, j, fold_idx],
                        }
                    )
    top_full_rank_df = pd.DataFrame(top_full_rank_records)
    top_full_rank_df["k_label"] = top_full_rank_df["k"].astype(str)
    return sub_df, top_df, sub_full_rank_df, top_full_rank_df


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


def _style_lineplot_axis(ax, title, xlabel, ylabel, x_ticks, y_upper=None):
    """Apply consistent styling to ECE line plots."""
    ax.set_title(title, fontsize=16, fontweight="semibold", pad=12)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_ylim(-0.001, y_upper)
    ax.set_facecolor("#f7f9fc")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xticks(x_ticks)
    ax.set_ylim(0, None)
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
    fig.patch.set_facecolor("white")
    sns.boxplot(
        data=sub_df,
        x="model",
        y="ece",
        hue="k_label",
        palette="Set2",
        dodge=True,
        ax=ax,
        linewidth=1.2,
        width=0.55,
        fliersize=0,
        log_scale=log_scale,
    )
    sns.stripplot(
        data=sub_df,
        x="model",
        y="ece",
        hue="k_label",
        palette="Set2",
        dodge=True,
        ax=ax,
        alpha=0.55,
        size=4.5,
        linewidth=0.3,
        edgecolor="white",
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(
        by_label.values(),
        by_label.keys(),
        title="k",
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
        f"{save_folder}subk_ece_grouped_{dataset_name}_{round(proportion_of_considered_rankings_in_ece, 2)}_{rank_weighting}_{discrepancy}_{bin_spacing}.png"
    )

    print("Visualizing Sub-k Full-Rank ECE...")
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    sns.boxplot(
        data=sub_full_rank_df,
        x="model",
        y="ece",
        hue="k_label",
        palette="Set2",
        dodge=True,
        ax=ax,
        linewidth=1.2,
        width=0.55,
        fliersize=0,
        log_scale=log_scale,
    )
    sns.stripplot(
        data=sub_full_rank_df,
        x="model",
        y="ece",
        hue="k_label",
        palette="Set2",
        dodge=True,
        ax=ax,
        alpha=0.55,
        size=4.5,
        linewidth=0.3,
        edgecolor="white",
        log_scale=log_scale,
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(
        by_label.values(),
        by_label.keys(),
        title="k",
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
        f"{save_folder}subk_ece_grouped_{dataset_name}_{round(proportion_of_considered_rankings_in_ece, 2)}_full_rank_{rank_weighting}_{discrepancy}_{bin_spacing}.png"
    )

    print("Visualizing Top-k Rank-wise ECE...")
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    sns.boxplot(
        data=top_df,
        x="model",
        y="ece",
        hue="k_label",
        palette="Set2",
        dodge=True,
        ax=ax,
        linewidth=1.2,
        width=0.55,
        fliersize=0,
        log_scale=log_scale,
    )
    sns.stripplot(
        data=top_df,
        x="model",
        y="ece",
        hue="k_label",
        palette="Set2",
        dodge=True,
        ax=ax,
        alpha=0.55,
        size=4.5,
        linewidth=0.3,
        edgecolor="white",
        log_scale=log_scale,
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(
        by_label.values(),
        by_label.keys(),
        title="k",
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
        f"{save_folder}topk_ece_grouped_{dataset_name}_{round(proportion_of_considered_rankings_in_ece, 2)}_{rank_weighting}_{discrepancy}_{bin_spacing}.png"
    )

    print("Visualizing Top-k Full-Rank ECE...")
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    sns.boxplot(
        data=top_full_rank_df,
        x="model",
        y="ece",
        hue="k_label",
        palette="Set2",
        dodge=True,
        ax=ax,
        linewidth=1.2,
        width=0.55,
        fliersize=0,
        log_scale=log_scale,
    )
    sns.stripplot(
        data=top_full_rank_df,
        x="model",
        y="ece",
        hue="k_label",
        palette="Set2",
        dodge=True,
        ax=ax,
        alpha=0.55,
        size=4.5,
        linewidth=0.3,
        edgecolor="white",
        log_scale=log_scale,
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(
        by_label.values(),
        by_label.keys(),
        title="k",
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
        f"{save_folder}topk_ece_grouped_{dataset_name}_{round(proportion_of_considered_rankings_in_ece, 2)}_full_rank_{rank_weighting}_{discrepancy}_{bin_spacing}.png"
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
    )
    _style_lineplot_axis(
        axes[0, 0], "Sub-k ECE vs k", "k", "ECE", k_values_sub_k, y_upper=1.001
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
    )
    _style_lineplot_axis(
        axes[0, 1], "Top-k ECE vs k", "k", "ECE", k_values_top_k, y_upper=1.001
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
    )
    _style_lineplot_axis(
        axes[1, 0],
        "Sub-k ECE vs k (Full Rank)",
        "k",
        "ECE",
        k_values_sub_k,
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
    )
    _style_lineplot_axis(
        axes[1, 1],
        "Top-k ECE vs k (Full Rank)",
        "k",
        "ECE",
        k_values_top_k,
    )

    handles, labels = axes[1, 1].get_legend_handles_labels()
    legend_bottom = axes[1, 1].legend(
        handles,
        labels,
        title="Model",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    legend_bottom.get_frame().set_alpha(0.9)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    legend_top = axes[0, 1].legend(
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
        f"{save_folder}ece_vs_k_errorbars_{dataset_name}_{round(proportion_of_considered_rankings_in_ece, 2)}_{rank_weighting}_{discrepancy}_{bin_spacing}.png"
    )


def calibrate_preference_model(
    preference_model, preference_criterion, X_test_tensor, y_test_tensor
):
    """Calibrate the preference model using temperature scaling."""
    temperature = nn.Parameter(torch.ones(1) * 1.0)

    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        logits = preference_model(X_test_tensor)
        scaled_logits = logits / temperature
        y_batch_idx = torch.tensor(
            [preference_model.idx_rankings[tuple(r.tolist())] for r in y_test_tensor],
            device=logits.device,
        ).long()
        loss = preference_criterion(scaled_logits, y_batch_idx)
        print("Calibration loss: ", loss.item())
        loss.backward()
        return loss

    optimizer.step(eval)
    print(f"Optimal temperature: {temperature.item():.4f}")
    preference_model.temperature = temperature.item()
    return preference_model


if __name__ == "__main__":
    ###### Configurations ######
    N_ITERATIONS = 100
    N_SAMPLES = 10_000

    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)
    num_epochs = 50
    batch_size = 64
    dataset_name =  "iris" #args.dataset
    RANK_WEIGHTING = (
        args.rank_weighting
    )  # "95_prob_mass"  # Options: "uniform", "prevalence", "pred_mass", "top_10"
    DISCREPANCY = (
        args.discrepancy
    )  # "abs"  # Options: "abs", "jeff", "log_ratio", "rel_p", "rel_q", "kl"
    BIN_SPACING = args.bin_spacing  # "linear"  # Options: "linear", "log"

    if dataset_name.startswith("synthetic"):
        X, y, y_true_probs = synthetic_data(
            rng, dataset_name, num_samples=N_SAMPLES, num_features=2, num_items=3
        )
    else:
        X, y = load_lr_data(dataset_name)
    # Model Architecture and Training
    input_dim = X.shape[1]
    hidden_dims = [64, 32]
    output_dim = np.unique(y, axis=0).shape[0]  # number of unique rankings
    n_items = y.shape[1]
    POSSIBLE_K_SUB_K = list(range(1, n_items + 1))[1:]
    POSSIBLE_K_TOP_K = list(range(1, n_items + 1))[:-1]

    print(
        f"Loaded dataset '{dataset_name}' with {X.shape[0]} samples and {X.shape[1]} features."
    )

    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    kf.get_n_splits(X)

    rankwise_sub_k_eces = {k: [[]] for k in POSSIBLE_K_SUB_K}
    rankwise_top_k_eces = {k: [[]] for k in POSSIBLE_K_TOP_K}
    rankwise_full_rank_sub_k_eces = {k: [[]] for k in POSSIBLE_K_SUB_K}
    rankwise_full_rank_top_k_eces = {k: [[]] for k in POSSIBLE_K_TOP_K}

    res_tau_dist = []

    # ####### Ranking Predictions (make sure the models have well-defined probabilities) #######
    if dataset_name not in [
        "movies",
        "letter",
        "libras",
        "vowel",
        "pendigit",
        "yeast",
    ]:  # For very large ranking spaces, we only consider the observed rankings in the test set
        possible_rankings = construct_possible_rankings(y.shape[1])
    else:
        # y is ranks-per-item; convert observed rankings to order tuples.
        unique_ranks = np.unique(y, axis=0)
        possible_rankings = _ranks_to_order_np(unique_ranks)
    proportion_of_considered_rankings_in_ece = len(possible_rankings) / factorial(
        y.shape[1]
    )
    print(
        f"Calucating ECE on {proportion_of_considered_rankings_in_ece} possible rankings."
    )
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        train_index, cal_index = (
            train_index[: int(0.8 * len(train_index))],
            train_index[int(0.8 * len(train_index)) :],
        )
        #### Get Models, Optimizers and Criterions ####
        models_optimizer_criterion = get_preference_models(
            input_dim, n_items, hidden_dims, output_dim, y, constant_value=-20.0
        )
        preference_model, preference_criterion, preference_optimizer = (
            models_optimizer_criterion["PreferenceModel"]
        )
        plackett_luce_model, plackett_criterion, plackett_optimizer = (
            models_optimizer_criterion["PlackettLuceModel"]
        )
        _mallows_model_entry, _, _ = models_optimizer_criterion["MallowsModel"]
        baseline_estimator, _, _ = models_optimizer_criterion["RPC_PL"]

        print(f"Fold {fold + 1}/{n_folds}")
        X_train, X_cal, X_test = X[train_index], X[cal_index], X[test_index]
        y_train, y_cal, y_test = y[train_index], y[cal_index], y[test_index]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_cal_tensor = torch.tensor(X_cal, dtype=torch.float32)
        y_cal_tensor = torch.tensor(y_cal, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        #### Training Loop ####
        print("Training Plackett Luce Model...")
        train_plackett_luce_model(
            plackett_luce_model=plackett_luce_model,
            plackett_criterion=plackett_criterion,
            plackett_optimizer=plackett_optimizer,
            X_train_tensor=X_train_tensor,
            y_train_tensor=y_train_tensor,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        print("Training Preference Model...")
        train_preference_model(
            preference_model=preference_model,
            criterion=preference_criterion,
            optimizer=preference_optimizer,
            X_train_tensor=X_train_tensor,
            y_train_tensor=y_train_tensor,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        print("Training Plackett Luce Model via RPC...")
        placket_luce_model_baseline, baseline_estimator_matrix = (
            train_placket_luce_rpc_model(
                baseline_estimator,
                X_train,
                y_train,
                X_test,
                n_items,
                method_rpc_pl="vectorized",
            )
        )
        print("Fitting Mallows Model...")
        mallows_model_fold = MallowsModel.fit_from_data(
            y_train_tensor, distance_metric="kendall"
        )
        #### Calibration Loop via Temperature Scaling ####

        print("Calibrating Preference Model...")
        preference_model = calibrate_preference_model(
            preference_model,
            preference_criterion,
            X_cal_tensor,
            y_cal_tensor,
        )
        #### Evaluate Models ####
        print("Evaluating Models...")
        results = evaluate_kendal_models(
            models=[
                plackett_luce_model,
                mallows_model_fold,
                preference_model,
                placket_luce_model_baseline,
                baseline_estimator,
            ],
            criterions=[
                plackett_criterion,
                None,
                preference_criterion,
                None,
                None,
            ],
            model_names=[
                "PlackettLuce",
                "MallowsModel",
                "PreferenceModel",
                "PlackettLuceRPC",
                "RPC",
            ],
            evaluate_functions=[
                evaluate_placket_luce_model,
                evaluate_mallows_model,
                evaluate_preference_model,
                evaluate_placket_luce_rpc_model,
                evaluate_calibrated_rpc_model,
            ],
            X_test_tensor=X_test_tensor,
            y_test_tensor=y_test_tensor,
        )
        ##### Print Kendall's Tau Results #####
        print("> Kendall's Tau Results:")
        for model_name in results.keys():
            print(
                f"{model_name}: Kendall's Tau = {results[model_name][0]}, Test Loss = {results[model_name][1]}"
            )
        print("\n")
        res_tau_dist.append(
            (
                results["PlackettLuce"][0],
                results["MallowsModel"][0],
                results["PreferenceModel"][0],
                results["PlackettLuceRPC"][0],
            )
        )

        ####### ECE Evaluation #######
        if dataset_name in [
            "movies",
            "letter",
            "libras",
            "vowel",
            "pendigit",
            "yeast",
        ]:  # For very large ranking spaces, we only consider the observed rankings in the test set
            restricted_rankings = [tuple(r) for r in possible_rankings]
        else:
            restricted_rankings = None
        distribution_pl = plackett_luce_model.predict_ranking_distribution(
            X_test_tensor, restricted_rankings=restricted_rankings
        )
        distribution_mallows = mallows_model_fold.predict_ranking_distribution(
            X_test_tensor,
            restricted_rankings=restricted_rankings,
        )
        distribution_pref = preference_model.predict_ranking_distribution(
            X_test_tensor, restricted_rankings=restricted_rankings
        )
        distribution_rpc_pl = placket_luce_model_baseline.predict_ranking_distribution(
            X_test_tensor,
            restricted_rankings=restricted_rankings,
        )
        # Create the distribution of rpc. We only have access to P(i beats j).

        distribution_rpc = {
            (i, j): baseline_estimator_matrix[:, i - 1, j - 1]
            for i in range(1, n_items + 1)
            for j in range(i + 1, n_items + 1)
        }
        dist_rpc = distribution_rpc.copy()
        for rank, prob in dist_rpc.items():
            reversed_rank = tuple(reversed(rank))
            distribution_rpc[reversed_rank] = 1.0 - prob
        print("Evaluating Rank-wise Sub-k and Top-k ECE...")
        evaluate_calibration_rankwise_sub_k_top_k(
            distributions=[
                distribution_pl,
                distribution_mallows,
                distribution_pref,
                distribution_rpc_pl,
                distribution_rpc,
            ],
            model_names=[
                "PlackettLuce",
                "MallowsModel",
                "PreferenceModel",
                "PlackettLuceRPC",
                "RPC",
            ],
            y_test_tensor=y_test_tensor,
            possible_k_sub_k=POSSIBLE_K_SUB_K,
            possible_k_top_k=POSSIBLE_K_TOP_K,
            rankwise_sub_k_eces=rankwise_sub_k_eces,
            rankwise_top_k_eces=rankwise_top_k_eces,
            rank_weighting=RANK_WEIGHTING,
            discrepancy=DISCREPANCY,
            bin_spacing=BIN_SPACING,
        )
        print("Evaluating Full-Rank Sub-k and Top-k ECE...")
        evaluate_calibration_full_rank_sub_k_top_k(
            distributions=[
                distribution_pl,
                distribution_mallows,
                distribution_pref,
                distribution_rpc_pl,
                distribution_rpc,
            ],
            model_names=[
                "PlackettLuce",
                "MallowsModel",
                "PreferenceModel",
                "PlackettLuceRPC",
                "RPC",
            ],
            y_test_tensor=y_test_tensor,
            possible_k_sub_k=POSSIBLE_K_SUB_K,
            possible_k_top_k=POSSIBLE_K_TOP_K,
            rankwise_full_rank_sub_k_eces=rankwise_full_rank_sub_k_eces,
            rankwise_full_rank_top_k_eces=rankwise_full_rank_top_k_eces,
            h=1,
            p_norm=1,
            rank_weighting=RANK_WEIGHTING,
        )

        print(f"Completed fold {fold + 1}/{n_folds}\n")

    # Remove any trailing empty entries introduced during accumulation
    for k in POSSIBLE_K_SUB_K:
        rankwise_sub_k_eces[k] = [entry for entry in rankwise_sub_k_eces[k] if entry]
        rankwise_full_rank_sub_k_eces[k] = [
            entry for entry in rankwise_full_rank_sub_k_eces[k] if entry
        ]
    for k in POSSIBLE_K_TOP_K:
        rankwise_top_k_eces[k] = [entry for entry in rankwise_top_k_eces[k] if entry]
        rankwise_full_rank_top_k_eces[k] = [
            entry for entry in rankwise_full_rank_top_k_eces[k] if entry
        ]

    #### Prepare Data for Visualization ####
    model_names = [
        "PlackettLuce",
        "MallowsModel",
        "PreferenceModel",
        "PlackettLuceRPC",
        "RPC",
    ]

    k_values_sub_k = np.array(POSSIBLE_K_SUB_K)
    k_values_top_k = np.array(POSSIBLE_K_TOP_K)
    # reshape to (n_models, n_k, n_folds)
    sub_df, top_df, sub_full_rank_df, top_full_rank_df = create_ece_reports(
        POSSIBLE_K_SUB_K,
        POSSIBLE_K_TOP_K,
        rankwise_sub_k_eces,
        rankwise_top_k_eces,
        rankwise_full_rank_sub_k_eces,
        rankwise_full_rank_top_k_eces,
        model_names,
    )
    save_folder = f"results/{dataset_name}/"
    os.makedirs(save_folder, exist_ok=True)
    visualize_ece_results(
        dataset_name,
        k_values_sub_k,
        k_values_top_k,
        sub_df,
        top_df,
        sub_full_rank_df,
        top_full_rank_df,
        proportion_of_considered_rankings_in_ece=proportion_of_considered_rankings_in_ece,
        save_folder=save_folder,
        rank_weighting=RANK_WEIGHTING,
        discrepancy=DISCREPANCY,
        bin_spacing=BIN_SPACING,
        log_scale=False,
    )

    # Save the ECE results to CSV files
    sub_df.to_csv(
        f"{save_folder}subk_ece_results_{dataset_name}_{round(proportion_of_considered_rankings_in_ece, 2)}_{RANK_WEIGHTING}_{DISCREPANCY}_{BIN_SPACING}.csv",
        index=False,
    )
    top_df.to_csv(
        f"{save_folder}topk_ece_results_{dataset_name}_{round(proportion_of_considered_rankings_in_ece, 2)}_{RANK_WEIGHTING}_{DISCREPANCY}_{BIN_SPACING}.csv",
        index=False,
    )
    sub_full_rank_df.to_csv(
        f"{save_folder}subk_full_rank_ece_results_{dataset_name}_{round(proportion_of_considered_rankings_in_ece, 2)}_{RANK_WEIGHTING}_{DISCREPANCY}_{BIN_SPACING}.csv",
        index=False,
    )
    top_full_rank_df.to_csv(
        f"{save_folder}topk_full_rank_ece_results_{dataset_name}_{round(proportion_of_considered_rankings_in_ece, 2)}_{RANK_WEIGHTING}_{DISCREPANCY}_{BIN_SPACING}.csv",
        index=False,
    )
