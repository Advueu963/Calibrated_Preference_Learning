from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
from sklr.pairwise import PairwisePartialLabelRanker

from cal_pref.preference_models import PreferenceModel, PlackettLuceModel
from cal_pref.preference_losses import BrierPreferenceLoss, PlackettLuceLoss, PlackettLuceBrierPreferenceLoss

def factorial(n):
    return torch.arange(1, n + 1).prod()


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
    ]:
        raise ValueError(
            "Invalid dataset name. Choose from 'authorship', 'glass', 'iris', 'letter', 'libras', 'movies', 'pendigits', 'segment', 'vehicle', 'vowel', 'wine', 'yeast'."
        )

    data = pd.read_csv(f"data/LR_DATA/{dataset_name}.csv")
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

def get_classwise_ece(possible_rankings, X_test_tensor, y_test_tensor, rank_prob_func, equal_frequency_bins=True, bin_size=10):
    ECE_classwise = 0.0
    for i, ranking in enumerate(possible_rankings):
        #print(f"Ranking {i + 1}: {ranking}")
        n_instances_total = y_test_tensor.shape[0]
        mask = (y_test_tensor == torch.tensor(ranking)).all(dim=-1)
        y_pred_rank_probs = rank_prob_func(X_test_tensor, torch.tensor(ranking))

        bin_size = 10
        probs_range = y_pred_rank_probs.max() - y_pred_rank_probs.min()
        # print(f"Prob range for ranking {ranking}: {y_pred_rank_probs.min().item()} - {y_pred_rank_probs.max().item()} (range: {probs_range.item()})")
        if equal_frequency_bins and probs_range > 0:
            sorted_probs, _ = torch.sort(y_pred_rank_probs)
            bins = [
                sorted_probs[int(i * n_instances_total / bin_size)].item()
                for i in range(bin_size)
            ] + [sorted_probs[-1].item() + 1e-6]
            bins = torch.tensor(bins)
        elif probs_range > 0:
            bins = torch.linspace(0, 1, bin_size + 1)
        else:
            bins = torch.linspace(
                y_pred_rank_probs.min().item(), y_pred_rank_probs.max().item() + 1e-6, 1
            )
            bin_size = 1
        # print(f"Bins for ranking {ranking}: {bins}")
        if len(bins) <= 1:
            bin_indices = torch.zeros_like(y_pred_rank_probs, dtype=torch.long)
        else:
            bin_indices = torch.bucketize(y_pred_rank_probs, bins) - 1

        ECE_ranking = 0.0
        for bin_idx in range(bin_size):
            bin_mask = bin_indices == bin_idx
            freq_true_rank_in_bin = torch.mean((mask & bin_mask).float()).item()
            mean_prob_in_bin = (
                torch.mean(y_pred_rank_probs[bin_mask]).item()
                if torch.sum(bin_mask) > 0
                else 0.0
            )
            count_in_bin = torch.sum(bin_mask).item()
            ECE_ranking += (count_in_bin / n_instances_total) * abs(
                freq_true_rank_in_bin - mean_prob_in_bin
            )
        ECE_classwise += ECE_ranking / len(possible_rankings)
    return ECE_classwise

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    dataset_name = "iris"
    X, y = load_lr_data(dataset_name)
    print(
        f"Loaded dataset '{dataset_name}' with {X.shape[0]} samples and {X.shape[1]} features."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    baseline_estimator = PairwisePartialLabelRanker(estimator=DecisionTreeClassifier(), n_jobs=-1)
    baseline_estimator.fit(X_train, y_train)
    y_baseline_pred = baseline_estimator.predict(X_test)

    input_dim = X.shape[1]
    hidden_dims = [64, 32]
    output_dim = y.shape[1] ** 2

    preference_model = PreferenceModel(input_dim, hidden_dims, output_dim)
    criterion = BrierPreferenceLoss()
    optimizer = torch.optim.Adam(preference_model.parameters(), lr=0.001)

    placket_luce_model = PlackettLuceModel(input_dim, hidden_dims, y.shape[1])
    placket_criterion = PlackettLuceLoss()
    placket_optimizer = torch.optim.Adam(placket_luce_model.parameters(), lr=0.001)
    
    placket_luce_model_bier = PlackettLuceModel(input_dim, hidden_dims, y.shape[1])
    placket_brier_criterion = PlackettLuceBrierPreferenceLoss()
    placket_brier_optimizer = torch.optim.Adam(placket_luce_model_bier.parameters(), lr=0.001)

    num_epochs = 100
    batch_size = 16

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # test_example = X_test_tensor[0:1]
    # y_pred = preference_model.predict(test_example)
    # y_pred_probs = preference_model.predict_proba(test_example)
    # print("Test Example Prediction: ", y_pred)
    # print("Test Example Prediction Probabilities: ", y_pred_probs)
    # loss = criterion(y_test_tensor, y_pred, y_pred_probs)
    # print("Test Example Loss: ", loss.item())

    torch.manual_seed(42)
    for epoch in range(num_epochs):
        X_batch_indices = torch.randperm(X_train_tensor.size(0))
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
           # print(f"Epoch {epoch + 1}/{num_epochs}, PL Loss: {loss.item()}")
            
            
            placket_luce_model_bier.train()
            placket_brier_optimizer.zero_grad()
            logits_brier = placket_luce_model_bier(X_batch)
            y_train_pred_brier = placket_luce_model_bier.predict(X_batch)
            loss_brier = placket_brier_criterion(y_batch,y_train_pred_brier, logits_brier, placket_luce_model_bier)
            loss_brier.backward()
            placket_brier_optimizer.step()
           # print(f"Epoch {epoch + 1}/{num_epochs}, PL Brier Loss: {loss_brier.item()}")
        
        
    

    placket_luce_model.eval()
    with torch.no_grad():
        y_test_pred = placket_luce_model.predict(X_test_tensor)
        logits = placket_luce_model(X_test_tensor)
        test_loss = placket_criterion(y_test_tensor, logits, placket_luce_model)
        print(f"Test PL Loss: {test_loss.item()}")
        
        
        y_test_pred_brier = placket_luce_model_bier.predict(X_test_tensor)
        logits_brier = placket_luce_model_bier(X_test_tensor)
        test_loss_brier = placket_brier_criterion(y_test_tensor,y_test_pred_brier, logits_brier, placket_luce_model_bier)
        print(f"Test PL Brier Loss: {test_loss_brier.item()}")

        kendal_dist = kendal_distance(y_test_tensor, y_test_pred)
        kendal_dist_brier = kendal_distance(y_test_tensor, y_test_pred_brier)
        print(f"Kendal Distance on Test Set (PL): {kendal_dist}")
        print(f"Kendal Distance on Test Set (PL Brier): {kendal_dist_brier}")
        print(f"Kendal Distance of Baseline on Test Set: {kendal_distance(y_test_tensor, torch.tensor(y_baseline_pred, dtype=torch.long))}")

    possible_rankings = torch.unique(y_test_tensor, dim=0).tolist() # Compute only for the rankings present in the test set
    print("Possible Rankings: ", possible_rankings)
    print("Number of Possible Rankings: ", len(possible_rankings))
    ece_pl = get_classwise_ece(possible_rankings, X_test_tensor, y_test_tensor, placket_luce_model.predict_proba_ranking)
    print(f"Class-wise ECE (PL): {ece_pl}")
    ece_pl_brier = get_classwise_ece(possible_rankings, X_test_tensor, y_test_tensor, placket_luce_model_bier.predict_proba_ranking)
    print(f"Class-wise ECE (PL Brier): {ece_pl_brier}")
    ece_prefence_model = get_classwise_ece(possible_rankings, X_test_tensor, y_test_tensor, preference_model.get_rank_prob)
    print(f"Class-wise ECE (Preference Model): {ece_prefence_model}")



    # for epoch in range(num_epochs):
    #     preference_model.train()
    #     optimizer.zero_grad()
    #     y_pred = preference_model.predict(X_train_tensor)
    #     y_pred_probs = preference_model.predict_proba(X_train_tensor)
    #     loss = criterion(y_train_tensor, y_pred, y_pred_probs)
    #     loss.backward()
    #     optimizer.step()
    #     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # preference_model.eval()
    # with torch.no_grad():
    #     y_test_pred = preference_model.predict(X_test_tensor)
    #     y_pred_probs = preference_model.predict_proba(X_test_tensor)
    #     test_loss = criterion(y_test_tensor, y_test_pred, y_pred_probs)
    #     print(f"Test Loss: {test_loss.item()}")

    #     kendal_dist = kendal_distance(y_test_tensor, y_test_pred)
    #     print(f"Kendal Distance on Test Set: {kendal_dist}")
    #     print(f"Kendal Distance of Baseline on Test Set: {kendal_distance(y_test_tensor, torch.tensor(y_baseline_pred, dtype=torch.long))}")

    # Multiclass binning calibration
    # possible_rankings = list(itertools.permutations(range(1, y.shape[1] + 1)))
    # possible_rankings = torch.unique(y_test_tensor, dim=0).tolist() # Compute only for the rankings present in the test set
    # print("Possible Rankings: ", possible_rankings)
    # print("Number of Possible Rankings: ", len(possible_rankings))
    # ECE_classwise = 0.0
    # for i, ranking in enumerate(possible_rankings):
    #     print(f"Ranking {i + 1}: {ranking}")
    #     n_instances_total = y_test_tensor.shape[0]
    #     mask = (y_test_tensor == torch.tensor(ranking)).all(dim=-1)
    #     y_pred_rank_probs = preference_model.get_rank_prob(X_test_tensor, torch.tensor(ranking))

    #     bin_size = 10
    #     probs_range = y_pred_rank_probs.max() - y_pred_rank_probs.min()
    #     #print(f"Prob range for ranking {ranking}: {y_pred_rank_probs.min().item()} - {y_pred_rank_probs.max().item()} (range: {probs_range.item()})")
    #     equal_frequency_bins = True  # Set to True for equal-frequency bins, False for equal-width bins
    #     if equal_frequency_bins and probs_range > 0:
    #         sorted_probs, _ = torch.sort(y_pred_rank_probs)
    #         bins = [sorted_probs[int(i * n_instances_total / bin_size)].item() for i in range(bin_size)] + [sorted_probs[-1].item() + 1e-6]
    #         bins = torch.tensor(bins)
    #     elif probs_range > 0:
    #         bins = torch.linspace(0, 1, bin_size + 1)
    #     else:
    #         bins = torch.linspace(y_pred_rank_probs.min().item(), y_pred_rank_probs.max().item() + 1e-6, 1)
    #         bin_size = 1
    #     #print(f"Bins for ranking {ranking}: {bins}")
    #     if len(bins) <= 1:
    #         bin_indices = torch.zeros_like(y_pred_rank_probs, dtype=torch.long)
    #     else:
    #         bin_indices = torch.bucketize(y_pred_rank_probs, bins) - 1

    #     mean_pred_prob = torch.zeros(bin_size)
    #     true_freq = torch.zeros(bin_size)
    #     counts = torch.zeros(bin_size)
    #     ECE_ranking = 0.0
    #     for bin_idx in range(bin_size):
    #         bin_mask = (bin_indices == bin_idx)
    #         freq_true_rank_in_bin = torch.mean((mask & bin_mask).float()).item()
    #         mean_prob_in_bin = torch.mean(y_pred_rank_probs[bin_mask]).item() if torch.sum(bin_mask) > 0 else 0.0
    #         count_in_bin = torch.sum(bin_mask).item()
    #         ECE_ranking += (count_in_bin / n_instances_total) * abs(freq_true_rank_in_bin - mean_prob_in_bin)
    #     print(f"ECE for ranking {ranking}: {ECE_ranking}")
    #     ECE_classwise += ECE_ranking / len(possible_rankings)
    # print(f"Class-wise ECE: {ECE_classwise}")
