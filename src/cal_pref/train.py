import torch
import numpy as np
from tqdm import tqdm
from cal_pref.utils import (
    from_bradley_terry_to_placket_luce_vectorized,
    from_bradley_terry_to_placket_luce_simple,
    from_bradley_terry_to_placket_luce_map,
)
from sklr.metrics import tau_score
from cal_pref.preference_models import (
    PlackettLuceModelWeights,
)


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
    # In glass it seems that most rankings are quite similiar, so the Calibration Method fails. Artificially we add some noise
    x_noise = np.random.normal(0, 0.1, X_train.shape[1])
    y_noise = np.array(range(y_train.shape[-1], 0, -1))
    X_train = np.concatenate([X_train, x_noise.reshape(1, -1)], axis=0)
    y_train = np.concatenate([y_train, y_noise.reshape(1, -1)], axis=0)

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
