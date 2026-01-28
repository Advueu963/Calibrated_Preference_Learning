import torch
import numpy as np
from sklr.metrics import tau_score


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
