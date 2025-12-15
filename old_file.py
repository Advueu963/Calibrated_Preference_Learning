from functools import partial
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
import itertools
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
    BrierPreferenceLoss,
    LogLossPreferenceLoss,
    PlackettLuceLoss,
    PlackettLuceBrierPreferenceLoss,
)
import matplotlib.pyplot as plt
import seaborn as sns

from cal_pref.utils import (
    load_lr_data,
    synthetic_data,
    calculate_sub_k_calibration,
    calculate_top_k_calibration,
    calculate_sub_k_full_rank_calibration,
    calculate_top_k_full_rank_calibration,
)


def get_preference_models(
    input_dim, n_items, hidden_dims, output_dim, y, constant_value=-10.0
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
    criterion = torch.nn.CrossEntropyLoss()
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

    # Mallows Model for reference
    unique_rankings = np.unique(y, axis=0)
    most_occurrent_ranking = unique_rankings[
        np.argmax([np.sum((y == ranking).all(axis=1)) for ranking in unique_rankings])
    ]
    print("Most occurrent ranking in training set: ", most_occurrent_ranking)
    mallows_model = MallowsModel.fit_from_data(
        torch.tensor(y, dtype=torch.long), distance_metric="kendall"
    )

    # RPC Baseline with Logistic Regression
    estimator = CalibratedClassifierCV(
        estimator=LogisticRegression(), cv=5, method="isotonic"
    )
    baseline_estimator = PairwiseLabelRanker(estimator=estimator, n_jobs=-1)

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


def train_placket_luce_model(
    placket_luce_model,
    placket_criterion,
    placket_optimizer,
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
    for epoch in range(num_epochs):
        X_batch_indices = torch.randperm(X_train_tensor.size(0), generator=generator)
        for i in range(0, X_train_tensor.size(0), batch_size):
            batch_indices = X_batch_indices[i : i + batch_size]
            X_batch = X_train_tensor[batch_indices]
            y_batch = y_train_tensor[batch_indices]

            placket_luce_model.train()
            placket_optimizer.zero_grad()
            logits = placket_luce_model(X_batch)
            loss = placket_criterion(y_batch, logits, placket_luce_model)
            loss.backward()
            placket_optimizer.step()


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
    for epoch in range(num_epochs):
        X_batch_indices = torch.randperm(X_train_tensor.size(0), generator=generator)
        for i in range(0, X_train_tensor.size(0), batch_size):
            batch_indices = X_batch_indices[i : i + batch_size]
            X_batch = X_train_tensor[batch_indices]
            y_batch = y_train_tensor[batch_indices]

            preference_model.train()
            optimizer.zero_grad()
            logits = preference_model(X_batch).float()
            y_batch_idx = torch.tensor(
                [preference_model.idx_rankings[tuple(r.tolist())] for r in y_batch],
                device=logits.device,
            ).long()
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
    X_train, y_train, X_test, n_items, method_rpc_pl="map"
):
    """Train a Plackett-Luce model with a rank-based loss function.

    Args:
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data labels.
        X_test (np.ndarray): Test data features.
        n_items (int): Number of items to rank.
    """
    baseline_estimator.fit(X_train, y_train)
    baseline_estimator_matrix = baseline_estimator.get_pairwise_matrix(X_test)

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
        test_loss = preference_criterion(
            logits,
            y_test_idx,
        )
        print(f"Test Preference Model Loss: {test_loss.item()}")

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
        print(f"Kendal Distance on Test Set (Mallows): {kendal_dist}")

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
        print(f"Kendal Distance on Test Set (PL RPC): {kendal_dist}")

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


if __name__ == "__main__":
    ###### Configurations ######
    N_ITERATIONS = 100
    N_SAMPLES = 10_000

    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)
    num_epochs = 50
    batch_size = 128
    dataset_name = "vehicle"

    X, y = load_lr_data(dataset_name)
    # Model Architecture and Training
    input_dim = X.shape[1]
    hidden_dims = [64, 32]
    output_dim = np.unique(y, axis=0).shape[0]  # number of unique rankings
    n_items = y.shape[1]
    POSSIBLE_K_SUB_K = [i for i in range(2, n_items + 1)]
    POSSIBLE_K_TOP_K = [i for i in range(1, n_items)]

    print(
        f"Loaded dataset '{dataset_name}' with {X.shape[0]} samples and {X.shape[1]} features."
    )

    #### Get Models, Optimizers and Criterions ####
    models_optimizer_criterion = get_preference_models(
        input_dim, n_items, hidden_dims, output_dim, y, constant_value=-10.0
    )
    preference_model, preference_criterion, preference_optimizer = (
        models_optimizer_criterion["PreferenceModel"]
    )
    placket_luce_model, placket_criterion, placket_optimizer = (
        models_optimizer_criterion["PlackettLuceModel"]
    )
    placket_luce_model_brier, placket_brier_criterion, placket_brier_optimizer = (
        models_optimizer_criterion["PlackettLuceModelBrier"]
    )
    _mallows_model_entry, _, _ = models_optimizer_criterion["MallowsModel"]
    baseline_estimator, _, _ = models_optimizer_criterion["RPC_PL"]

    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    kf.get_n_splits(X)

    rankwise_sub_k_eces = {k: [] for k in POSSIBLE_K_SUB_K}
    rankwise_top_k_eces = {k: [] for k in POSSIBLE_K_TOP_K}
    rankwise_full_rank_sub_k_eces = {k: [] for k in POSSIBLE_K_SUB_K}
    rankwise_full_rank_top_k_eces = {k: [] for k in POSSIBLE_K_TOP_K}

    res_tau_dist = []
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{n_folds}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        #### Training Loop ####
        train_placket_luce_model(
            placket_luce_model=placket_luce_model,
            placket_criterion=placket_criterion,
            placket_optimizer=placket_optimizer,
            X_train_tensor=X_train_tensor,
            y_train_tensor=y_train_tensor,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        train_preference_model(
            preference_model=preference_model,
            criterion=preference_criterion,
            optimizer=preference_optimizer,
            X_train_tensor=X_train_tensor,
            y_train_tensor=y_train_tensor,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        train_placket_luce_model_brier(
            placket_luce_model_brier=placket_luce_model_brier,
            placket_brier_criterion=placket_brier_criterion,
            placket_brier_optimizer=placket_brier_optimizer,
            X_train_tensor=X_train_tensor,
            y_train_tensor=y_train_tensor,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

        placket_luce_model_baseline, baseline_estimator_matrix = (
            train_placket_luce_rpc_model(
                X_train, y_train, X_test, n_items, method_rpc_pl="map"
            )
        )

        mallows_model_fold = MallowsModel.fit_from_data(
            y_train_tensor, distance_metric="kendall"
        )

        #### Evaluate Models ####
        results = evaluate_kendal_models(
            models=[
                placket_luce_model,
                mallows_model_fold,
                preference_model,
                placket_luce_model_baseline,
            ],
            criterions=[
                placket_criterion,
                None,
                preference_criterion,
                None,
            ],
            model_names=[
                "PlackettLuce",
                "MallowsModel",
                "PreferenceModel",
                "PlackettLuceRPC",
            ],
            evaluate_functions=[
                evaluate_placket_luce_model,
                evaluate_mallows_model,
                evaluate_preference_model,
                evaluate_placket_luce_rpc_model,
            ],
            X_test_tensor=X_test_tensor,
            y_test_tensor=y_test_tensor,
        )
        res_tau_dist.append(
            (
                results["PlackettLuce"][0],
                results["MallowsModel"][0],
                results["PreferenceModel"][0],
                results["PlackettLuceRPC"][0],
            )
        )

        # ####### Ranking Predictions (make sure the models have well-defined probabilities) #######
        #possible_rankings = list(itertools.permutations(range(1, y.shape[1] + 1)))
        possible_rankings = np.unique(y, axis=0).tolist()
        print("Number of possible rankings: ", len(possible_rankings) / factorial(n_items))
        # print("Possible Rankings: ", possible_rankings)

        ####### Class-wise ECE Calibration #######
        print("Calculating Class-wise ECE...")
        distribution_pl = placket_luce_model.predict_ranking_distribution(X_test_tensor)
        distribution_mallows = mallows_model_fold.predict_ranking_distribution(
            X_test_tensor
        )
        distribution_pref = preference_model.predict_ranking_distribution(X_test_tensor)
        distribution_rpc_pl = placket_luce_model_baseline.predict_ranking_distribution(
            X_test_tensor
        )
        distribution_rpc = {
            (i, j): baseline_estimator_matrix[:, i - 1, j - 1]
            for i in range(1, n_items + 1)
            for j in range(i, n_items + 1)
        }
        for k in POSSIBLE_K_SUB_K:
            ece_pl = calculate_sub_k_calibration(
                items=list(range(1, n_items + 1)),
                y_true=y_test,
                y_pred_proba=distribution_pl,
                k=k,
            )
            ece_mallows = calculate_sub_k_calibration(
                items=list(range(1, n_items + 1)),
                y_true=y_test,
                y_pred_proba=distribution_mallows,
                k=k,
            )
            ece_pref = calculate_sub_k_calibration(
                items=list(range(1, n_items + 1)),
                y_true=y_test,
                y_pred_proba=distribution_pref,
                k=k,
            )
            ece_rpc_pl = calculate_sub_k_calibration(
                items=list(range(1, n_items + 1)),
                y_true=y_test,
                y_pred_proba=distribution_rpc_pl,
                k=k,
            )
            if k == 2:
                ece_rpc = calculate_sub_k_calibration(
                    items=list(range(1, n_items + 1)),
                    y_true=y_test,
                    y_pred_proba=distribution_rpc,
                    k=2,
                )

            rankwise_sub_k_eces[k].append(
                [
                    ece_pl["total_ece"],
                    ece_mallows["total_ece"],
                    ece_pref["total_ece"],
                    ece_rpc_pl["total_ece"],
                ]
            )

        print("Calculating Full-Rank Sub-k ECE...")
        for k in POSSIBLE_K_SUB_K:
            ece_pl_full = calculate_sub_k_full_rank_calibration(
                items=list(range(1, n_items + 1)),
                y_true=y_test,
                y_pred_proba=distribution_pl,
                k=k,
                mode="kernel",  # if k in [1,3] else "binning",
            )
            ece_mallows_full = calculate_sub_k_full_rank_calibration(
                items=list(range(1, n_items + 1)),
                y_true=y_test,
                y_pred_proba=distribution_mallows,
                k=k,
                mode="kernel",  # if k  in [1,3] else "binning",
            )
            ece_pref_full = calculate_sub_k_full_rank_calibration(
                items=list(range(1, n_items + 1)),
                y_true=y_test,
                y_pred_proba=distribution_pref,
                k=k,
                mode="kernel",  # if k  in [1,3] else "binning",
            )
            ece_rpc_pl_full = calculate_sub_k_full_rank_calibration(
                items=list(range(1, n_items + 1)),
                y_true=y_test,
                y_pred_proba=distribution_rpc_pl,
                k=k,
                mode="kernel",  # if k in [1,3] else "binning",
            )

            rankwise_full_rank_sub_k_eces[k].append(
                [
                    ece_pl_full["total_ece"],
                    ece_mallows_full["total_ece"],
                    ece_pref_full["total_ece"],
                    ece_rpc_pl_full["total_ece"],
                ]
            )

        print("Calculating Top-k ECE...")
        for k in POSSIBLE_K_TOP_K:
            ece_pl = calculate_top_k_calibration(
                items=list(range(1, n_items + 1)),
                y_true=y_test,
                y_pred_proba=distribution_pl,
                k=k,
            )
            ece_mallows = calculate_top_k_calibration(
                items=list(range(1, n_items + 1)),
                y_true=y_test,
                y_pred_proba=distribution_mallows,
                k=k,
            )
            ece_pref = calculate_top_k_calibration(
                items=list(range(1, n_items + 1)),
                y_true=y_test,
                y_pred_proba=distribution_pref,
                k=k,
            )
            ece_rpc_pl = calculate_top_k_calibration(
                items=list(range(1, n_items + 1)),
                y_true=y_test,
                y_pred_proba=distribution_rpc_pl,
                k=k,
            )

            rankwise_top_k_eces[k].append(
                [
                    ece_pl["total_ece"],
                    ece_mallows["total_ece"],
                    ece_pref["total_ece"],
                    ece_rpc_pl["total_ece"],
                ]
            )

        print("Calculating Full-Rank Top-k ECE...")
        for k in POSSIBLE_K_TOP_K:
            ece_pl_full = calculate_top_k_full_rank_calibration(
                items=list(range(1, n_items + 1)),
                y_true=y_test,
                y_pred_proba=distribution_pl,
                k=k,
                mode="kernel",  # if k in [2,3] else "binning",
            )
            ece_mallows_full = calculate_top_k_full_rank_calibration(
                items=list(range(1, n_items + 1)),
                y_true=y_test,
                y_pred_proba=distribution_mallows,
                k=k,
                mode="kernel",  # if k  in [2,3] else "binning",
            )
            ece_pref_full = calculate_top_k_full_rank_calibration(
                items=list(range(1, n_items + 1)),
                y_true=y_test,
                y_pred_proba=distribution_pref,
                k=k,
                mode="kernel",  # if k in [2,3] else "binning",
            )
            ece_rpc_pl_full = calculate_top_k_full_rank_calibration(
                items=list(range(1, n_items + 1)),
                y_true=y_test,
                y_pred_proba=distribution_rpc_pl,
                k=k,
                mode="kernel",  # if k in [2,3] else "binning",
            )

            rankwise_full_rank_top_k_eces[k].append(
                [
                    ece_pl_full["total_ece"],
                    ece_mallows_full["total_ece"],
                    ece_pref_full["total_ece"],
                    ece_rpc_pl_full["total_ece"],
                ]
            )
        print(f"Completed fold {fold + 1}/{n_folds}\n")

    #### Prepare Data for Visualization ####
    model_names = ["PlackettLuce", "MallowsModel", "PreferenceModel", "PlackettLuceRPC"]

    k_values_sub_k = np.array(POSSIBLE_K_SUB_K)
    k_values_top_k = np.array(POSSIBLE_K_TOP_K)
    # reshape to (n_models, n_k, n_folds)
    sub_k_matrix = np.stack(
        [np.asarray(rankwise_sub_k_eces[k]) for k in POSSIBLE_K_SUB_K], axis=1
    )
    sub_k_matrix = np.transpose(sub_k_matrix, (2, 1, 0))

    top_k_matrix = np.stack(
        [np.asarray(rankwise_top_k_eces[k]) for k in POSSIBLE_K_TOP_K], axis=1
    )
    top_k_matrix = np.transpose(top_k_matrix, (2, 1, 0))

    sub_k_full_rank_matrix = np.stack(
        [np.asarray(rankwise_full_rank_sub_k_eces[k]) for k in POSSIBLE_K_SUB_K], axis=1
    )
    sub_k_full_rank_matrix = np.transpose(sub_k_full_rank_matrix, (2, 1, 0))

    top_k_full_rank_matrix = np.stack(
        [np.asarray(rankwise_full_rank_top_k_eces[k]) for k in POSSIBLE_K_TOP_K], axis=1
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

    print("Visualizing Sub-k Rank-wise ECE...")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=sub_df,
        x="model",
        y="ece",
        hue="k_label",
        palette="Set2",
        dodge=True,
        ax=ax,
        linewidth=1.5,
    )
    sns.stripplot(
        data=sub_df,
        x="model",
        y="ece",
        hue="k_label",
        palette="Set2",
        dodge=True,
        ax=ax,
        alpha=0.35,
        size=4,
        linewidth=0,
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="k", frameon=True)
    ax.set_xlabel("")
    ax.set_ylabel("Sub-k ECE")
    ax.set_title(f"Sub-k ECE per model on {dataset_name}")
    ax.margins(x=0.05)
    fig.tight_layout()
    fig.savefig(f"subk_ece_grouped_{dataset_name}.png")

    print("Visualizing Sub-k Full-Rank ECE...")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=sub_full_rank_df,
        x="model",
        y="ece",
        hue="k_label",
        palette="Set2",
        dodge=True,
        ax=ax,
        linewidth=1.5,
    )
    sns.stripplot(
        data=sub_full_rank_df,
        x="model",
        y="ece",
        hue="k_label",
        palette="Set2",
        dodge=True,
        ax=ax,
        alpha=0.35,
        size=4,
        linewidth=0,
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="k", frameon=True)
    ax.set_xlabel("")
    ax.set_ylabel("Sub-k ECE (Full Rank)")
    ax.set_title(f"Sub-k ECE per model on {dataset_name} (Full Rank)")
    ax.margins(x=0.05)
    fig.tight_layout()
    fig.savefig(f"subk_ece_grouped_{dataset_name}_full_rank.png")

    print("Visualizing Top-k Rank-wise ECE...")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=top_df,
        x="model",
        y="ece",
        hue="k_label",
        palette="Set2",
        dodge=True,
        ax=ax,
        linewidth=1.5,
    )
    sns.stripplot(
        data=top_df,
        x="model",
        y="ece",
        hue="k_label",
        palette="Set2",
        dodge=True,
        ax=ax,
        alpha=0.35,
        size=4,
        linewidth=0,
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="k", frameon=True)
    ax.set_xlabel("")
    ax.set_ylabel("Top-k ECE")
    ax.set_title(f"Top-k ECE per model on {dataset_name}")
    ax.margins(x=0.05)
    fig.tight_layout()
    fig.savefig(f"topk_ece_grouped_{dataset_name}.png")

    print("Visualizing Top-k Full-Rank ECE...")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=top_full_rank_df,
        x="model",
        y="ece",
        hue="k_label",
        palette="Set2",
        dodge=True,
        ax=ax,
        linewidth=1.5,
    )
    sns.stripplot(
        data=top_full_rank_df,
        x="model",
        y="ece",
        hue="k_label",
        palette="Set2",
        dodge=True,
        ax=ax,
        alpha=0.35,
        size=4,
        linewidth=0,
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="k", frameon=True)
    ax.set_xlabel("")
    ax.set_ylabel("Top-k ECE (Full Rank)")
    ax.set_title(f"Top-k ECE per model on {dataset_name} (Full Rank)")
    ax.margins(x=0.05)
    fig.tight_layout()
    fig.savefig(f"topk_ece_grouped_{dataset_name}_full_rank.png")

    print("Visualizing Sub-k Rankwise ECE vs k with Error Bars...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    sns.lineplot(
        data=sub_df,
        x="k",
        y="ece",
        hue="model",
        marker="o",
        errorbar=("sd"),
        palette="Dark2",
        linestyle="--",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Sub-k ECE vs k")
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("ECE")
    axes[0, 0].set_xticks(k_values_sub_k)
    axes[0, 0].margins(x=0.05)

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
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Top-k ECE vs k")
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("ECE")
    axes[0, 1].set_xticks(k_values_top_k)
    axes[0, 1].margins(x=0.05)
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
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Sub-k ECE vs k (Full Rank)")
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("ECE")
    axes[1, 0].set_xticks(k_values_sub_k)
    axes[1, 0].margins(x=0.05)
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
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Top-k ECE vs k (Full Rank)")
    axes[1, 1].set_xlabel("k")
    axes[1, 1].set_ylabel("ECE")
    axes[1, 1].set_xticks(k_values_top_k)
    axes[1, 1].margins(x=0.05)

    handles, labels = axes[1, 1].get_legend_handles_labels()
    axes[1, 1].legend(handles, labels, title="Model", frameon=True)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    axes[0, 1].legend(handles, labels, title="Model", frameon=True)

    fig.tight_layout()
    fig.savefig(f"ece_vs_k_errorbars_{dataset_name}.png")

    # Save the ECE results to CSV files
    sub_df.to_csv(f"subk_ece_results_{dataset_name}.csv", index=False)
    top_df.to_csv(f"topk_ece_results_{dataset_name}.csv", index=False)
    sub_full_rank_df.to_csv(f"subk_full_rank_ece_results_{dataset_name}.csv", index=False)
    top_full_rank_df.to_csv(f"topk_full_rank_ece_results_{dataset_name}.csv", index=False)