from functools import partial
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import itertools
from cal_pref.utils import from_bradley_terry_to_placket_luce_map
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

from cal_pref.utils import (
    load_lr_data,
    synthetic_data,
    get_classwise_ece,
    get_rankwise_ece,
    visualize_per_class_probs,
)


def calculate_ece(
    ece_func,
    placket_luce_model,
    mallows_model,
    preference_model,
    X_test_tensor,
    y_test_tensor,
    placket_luce_model_baseline,
    possible_rankings,
):
    """Calculate the ECE for different models.

    Args:
        ece_func (callable): A function to compute the ECE.
        preference_model (nn.Module): The preference model.
        placket_luce_model (nn.Module): The Plackett-Luce model.
        placket_luce_model_brier (nn.Module): The Plackett-Luce model with Brier loss.
        mallows_model (nn.Module): The Mallows model.
        X_test_tensor (torch.Tensor): The test input features.
        y_test_tensor (torch.Tensor): The test target rankings.
        placket_luce_model_baseline (nn.Module): The baseline Plackett-Luce model.
        possible_rankings (list): The list of possible rankings.

    Returns:
        tuple: ECE scores for each model.
    """
    print("Number of Possible Rankings: ", len(possible_rankings))
    ece_pl = ece_func(
        possible_rankings,
        X_test_tensor,
        y_test_tensor,
        placket_luce_model.predict_proba_ranking,
    )
    print(f"(PL): {ece_pl}")
    ece_prefence_model = ece_func(
        possible_rankings,
        X_test_tensor,
        y_test_tensor,
        preference_model.predict_proba_ranking,
    )
    print(f"(Preference Model): {ece_prefence_model}")
    ece_mallows = ece_func(
        possible_rankings,
        X_test_tensor,
        y_test_tensor,
        mallows_model.predict_proba_ranking,
    )
    print(f"(Mallows Model): {ece_mallows}")
    ece_baseline = ece_func(
        possible_rankings,
        X_test_tensor,
        y_test_tensor,
        placket_luce_model_baseline.predict_proba_ranking,
    )
    print(f"(Baseline RPC): {ece_baseline}")

    return ece_pl, ece_mallows, ece_baseline, ece_prefence_model


def train_preference_models(
    num_epochs,
    batch_size,
    n_items,
    preference_model,
    preference_criterion,
    preference_optimizer,
    placket_luce_model,
    placket_criterion,
    placket_optimizer,
    baseline_estimator,
    X_train,
    X_test,
    y_train,
    X_train_tensor,
    y_train_tensor,
):
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
            # print("LOGITS: ", logits)
            loss = placket_criterion(y_batch, logits, placket_luce_model)
            # print("Epoch:", epoch, " Batch:", i // batch_size, " PL Loss:", loss.item())
            loss.backward()
            # for param in placket_luce_model.parameters():
            #     if param.grad is not None:
            #         print(
            #             "Gradient stats - min:",
            #             param.grad.min().item(),
            #             " max:",
            #             param.grad.max().item(),
            #             " mean:",
            #             param.grad.mean().item(),
            #         )
            #         if torch.isnan(param.grad).any():
            #             print("NaN values found in gradients.")
            #             raise ValueError("NaN values in gradients.")
            placket_optimizer.step()
            # print(f"Epoch {epoch + 1}/{num_epochs}, PL Loss: {loss.item()}")

            # placket_luce_model_brier.train()
            # placket_brier_optimizer.zero_grad()
            # logits_brier = placket_luce_model_brier(X_batch)
            # y_train_pred_brier = placket_luce_model_brier.predict(X_batch)
            # loss_brier = placket_brier_criterion(
            #         y_batch, y_train_pred_brier, logits_brier, placket_luce_model_brier
            #     )
            # loss_brier.backward()
            # placket_brier_optimizer.step()

            preference_model.train()
            preference_optimizer.zero_grad()
            logits = preference_model(X_batch).float()
            y_batch_idx = torch.tensor(
                [preference_model.idx_rankings[tuple(r.tolist())] for r in y_batch],
                device=logits.device,
            ).long()
            loss_pref = preference_criterion(logits, y_batch_idx)
            loss_pref.backward()
            preference_optimizer.step()

    baseline_estimator.fit(X_train, y_train)
    baseline_estimator_matrix = baseline_estimator.get_pairwise_matrix(X_test)
    placket_luce_weights = from_bradley_terry_to_placket_luce_map(
        np.random.default_rng(42), baseline_estimator_matrix, n_iterations=N_ITERATIONS
    )
    placket_luce_model_baseline = PlackettLuceModelWeights(
        placket_luce_weights, n_items=n_items
    )

    return placket_luce_model_baseline


def visualize_eces(res_eces, dataset_name, show=False):
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        res_eces,
        tick_labels=["PL", "Mallows", "RPC_PL", "Preference Model"],
    )
    plt.ylabel("Class-wise ECE")
    plt.title("Class-wise ECE across 5 folds")
    plt.grid(axis="y")
    plt.savefig(f"ece_boxplot_restricted_{dataset_name}.png")
    if show:
        plt.show()


def visualize_rankwise_eces(res_eces, dataset_name, show=False):
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        res_eces,
        tick_labels=["PL", "Mallows", "RPC_PL", "Preference Model"],
    )
    plt.ylabel("Rank-wise ECE")
    plt.title("Rank-wise ECE across 5 folds")
    plt.grid(axis="y")
    plt.savefig(f"rankwise_ece_boxplot_restricted_{dataset_name}.png")
    if show:
        plt.show()


def visualize_kendall(res_tau_dist, dataset_name, show=False):
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        res_tau_dist,
        tick_labels=["PL", "Mallows", "RPC_PL", "Preference Model"],
    )
    plt.ylabel("Kendall's Tau Score")
    plt.title("Kendall's Tau Score across 5 folds")
    plt.grid(axis="y")
    plt.savefig(f"kendall_tau_boxplot_restricted_{dataset_name}.png")
    if show:
        plt.show()


def visualize_rankwise_ece_multiple_T(
    dataset_name, T_values, ece_rankwise_T, show=False
):
    plt.figure(figsize=(10, 6))
    plt.plot(
        T_values,
        [ece[0] for ece in ece_rankwise_T.values()],
        marker="o",
        label="PL",
    )
    plt.plot(
        T_values,
        [ece[1] for ece in ece_rankwise_T.values()],
        marker="o",
        label="Mallows",
    )
    plt.plot(
        T_values,
        [ece[2] for ece in ece_rankwise_T.values()],
        marker="o",
        label="RPC_PL",
    )
    plt.plot(
        T_values,
        [ece[3] for ece in ece_rankwise_T.values()],
        marker="o",
        label="Preference Model",
    )
    plt.title("Rank-wise ECE over T")
    plt.xlabel("T")
    plt.ylabel("ECE")
    plt.grid()
    plt.legend()
    plt.savefig(f"rankwise_ece_T_{dataset_name}.png")
    if show:
        plt.show()


if __name__ == "__main__":
    ###### Configurations ######
    N_ITERATIONS = 1_000
    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)
    num_epochs = 50
    batch_size = 16
    dataset_name = "segment"

    ###### Load Data ######
    if dataset_name.startswith("synthetic"):
        X, y, y_true_probs = synthetic_data(
            rng, dataset_name, num_samples=1000, num_features=2, num_items=3
        )
    else:
        X, y = load_lr_data(dataset_name)

    print(
        f"Loaded dataset '{dataset_name}' with {X.shape[0]} samples and {X.shape[1]} features."
    )
    #### Define Architecture ######
    input_dim = X.shape[1]
    hidden_dims = [64, 32]
    output_dim = np.unique(y, axis=0).shape[0]  # number of unique rankings
    print("Number of unique rankings in dataset: ", output_dim)
    n_items = y.shape[1]

    #### Cross-Validation ######
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    kf.get_n_splits(X)

    ### Prepare saved variables for results ###
    possible_rankings = list(itertools.permutations(range(1, y.shape[1] + 1)))
    # possible_rankings = np.unique(y, axis=0).tolist()
    T_values = list(range(1, n_items + 1))
    res_eces = []
    res_ranking_wise_eces = []
    res_tau_dist = []
    ece_rankwise_T = {T: [] for T in T_values}
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        torch.manual_seed(42)
        np.random.seed(42)
        ### Define Models ###

        # Preference Model with Brier Loss. Has as many outputs as there are items times positions
        preference_model = PreferenceModel(
            input_dim,
            n_items,
            hidden_dims,
            output_dim,
            torch.tensor(np.unique(y, axis=0)),
        )
        # criterion = BrierPreferenceLoss(maximal_t_list_size=n_items**n_items)
        # criterion = LogLossPreferenceLoss(maximal_t_list_size=factorial(n_items))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(preference_model.parameters(), lr=0.001)

        # Plackett Luce Model
        placket_luce_model = PlackettLuceModel(input_dim, hidden_dims, y.shape[1])
        placket_criterion = PlackettLuceLoss()
        placket_optimizer = torch.optim.Adam(placket_luce_model.parameters(), lr=0.001)

        # Mallows
        unique_rankings = np.unique(y, axis=0)
        most_occurrent_ranking = unique_rankings[
            np.argmax(
                [np.sum((y == ranking).all(axis=1)) for ranking in unique_rankings]
            )
        ]
        print("Most occurrent ranking in training set: ", most_occurrent_ranking)
        mallows_model = MallowsModel(
            reference_ranking=torch.tensor(most_occurrent_ranking), dispersion=1
        )

        # RPC Baseline with Logistic Regression
        estimator = CalibratedClassifierCV(
            estimator=LogisticRegression(), cv=5, method="isotonic"
        )
        baseline_estimator = PairwiseLabelRanker(estimator=estimator, n_jobs=-1)

        print(f"Fold {fold + 1}/{n_folds}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        ####### Training Loop #######
        # It only returns one model as the other models are trained inplace
        placket_luce_model_baseline = train_preference_models(
            num_epochs,
            batch_size,
            n_items,
            preference_model,
            criterion,
            optimizer,
            placket_luce_model,
            placket_criterion,
            placket_optimizer,
            baseline_estimator,
            X_train,
            X_test,
            y_train,
            X_train_tensor,
            y_train_tensor,
        )
        ####### Evaluation #######
        placket_luce_model.eval()
        placket_luce_model_baseline.eval()

        with torch.no_grad():
            y_baseline_pred = placket_luce_model_baseline.predict(X_test_tensor)

            y_test_pred = placket_luce_model.predict(X_test_tensor)
            logits = placket_luce_model(X_test_tensor)
            test_loss = placket_criterion(y_test_tensor, logits, placket_luce_model)
            print(f"Test PL Loss: {test_loss.item()}")

            logits = preference_model(X_test_tensor)
            y_test_idx = torch.tensor(
                [
                    preference_model.idx_rankings[tuple(r.tolist())]
                    for r in y_test_tensor
                ],
                device=logits.device,
            ).long()
            y_test_pred_pref = preference_model.predict(X_test_tensor)
            test_loss_pref = criterion(
                logits,
                y_test_idx,
            )
            print(f"Test Preference Model Loss: {test_loss_pref.item()}")

            kendal_dist = tau_score(y_test_tensor, y_test_pred)
            kendal_dist_baseline = tau_score(y_test_tensor, y_baseline_pred)
            kendal_dist_pref = tau_score(y_test_tensor, y_test_pred_pref)
            print(
                f"Kendal Distance of Preference Model on Test Set: {kendal_dist_pref}"
            )
            print(f"Kendal Distance on Test Set (PL): {kendal_dist}")
            print(f"Kendal Distance of Baseline on Test Set: {kendal_dist_baseline}")
            kendal_dist_mallows = tau_score(
                y_test_tensor,
                torch.tensor(
                    np.array(
                        [most_occurrent_ranking for _ in range(y_test_tensor.shape[0])]
                    ),
                    dtype=torch.long,
                ),
            )
            print(f"Kendal Distance of Mallows on Test Set: {kendal_dist_mallows}")
            res_tau_dist.append(
                (
                    kendal_dist,
                    kendal_dist_mallows,
                    kendal_dist_baseline,
                    kendal_dist_pref,
                )
            )

        ####### Class-wise ECE Calibration #######
        ece_pl, ece_mallows, ece_baseline, ece_prefence_model = calculate_ece(
            get_classwise_ece,
            placket_luce_model,
            mallows_model,
            preference_model,
            X_test_tensor,
            y_test_tensor,
            placket_luce_model_baseline,
            possible_rankings,
        )

        res_eces.append((ece_pl, ece_mallows, ece_baseline, ece_prefence_model))

        ####### (T=1)-Rank-wise ECE Calibration #######
        T = 1
        ece_pl, ece_mallows, ece_baseline, ece_prefence_model = calculate_ece(
            partial(get_rankwise_ece, T=T),
            placket_luce_model,
            mallows_model,
            preference_model,
            X_test_tensor,
            y_test_tensor,
            placket_luce_model_baseline,
            possible_rankings,
        )
        res_ranking_wise_eces.append(
            (ece_pl, ece_mallows, ece_baseline, ece_prefence_model)
        )

        ### t-Rank-wise ECE Calibration ###
        for T in T_values:
            ece_pl, ece_mallows, ece_rpc_pl, ece_prefence_model = calculate_ece(
                partial(get_rankwise_ece, T=T),
                placket_luce_model,
                mallows_model,
                preference_model,
                X_test_tensor,
                y_test_tensor,
                placket_luce_model_baseline,
                possible_rankings,
            )
            ece_rankwise_T[T].append(
                [
                    ece_pl,
                    ece_mallows,
                    ece_rpc_pl,
                    ece_prefence_model,
                ]
            )
        # Average ECE over folds
    for T in T_values:
        ece_rankwise_T[T] = np.mean(ece_rankwise_T[T], axis=0)
    res_eces = np.array(res_eces)
    res_ranking_wise_eces = np.array(res_ranking_wise_eces)
    res_tau_dist = np.array(res_tau_dist)
    print(
        "Average Kendall's Tau Distances over all folds (PL, Mallows, Baseline, Preference Model): ",
        np.mean(res_tau_dist, axis=0),
    )
    print(
        "Average ECEs over all folds (PL, Mallows, Baseline, Preference Model): ",
        np.mean(res_eces, axis=0),
    )
    print(
        "Average Rank-wise ECEs over all folds (PL, Mallows, Baseline, Preference Model): ",
        np.mean(res_ranking_wise_eces, axis=0),
    )

    ##### Plotting of Results ######
    visualize_eces(res_eces, dataset_name, show=False)

    visualize_rankwise_eces(res_ranking_wise_eces, dataset_name, show=False)

    visualize_kendall(res_tau_dist, dataset_name, show=False)

    visualize_rankwise_ece_multiple_T(dataset_name, T_values, ece_rankwise_T)

    ###### Synthetic Data Calibration Check ######
    if dataset_name.startswith("synthetic"):
        y_probs_pl = []
        y_probs_mallows = []
        y_probs_baseline = []
        with torch.no_grad():
            for ranking in possible_rankings:
                prob_pl = placket_luce_model.predict_proba_ranking(
                    X_test_tensor, torch.tensor(ranking)
                ).numpy()
                prob_mallows = mallows_model.predict_proba_ranking(
                    None, torch.tensor(ranking)
                ).numpy()
                prob_baseline = placket_luce_model_baseline.predict_proba_ranking(
                    X_test_tensor, torch.tensor(ranking)
                )
                y_probs_mallows.append([prob_mallows.item()] * X_test_tensor.shape[0])
                y_probs_baseline.append(prob_baseline)
                y_probs_pl.append(prob_pl)
        y_probs_pl = np.array(y_probs_pl).T  # shape = (n_samples, n_rankings)
        y_probs_mallows = np.array(y_probs_mallows).T
        y_probs_baseline = np.array(y_probs_baseline).T
        plt.figure(figsize=(12, 7))
        bar_width = 0.13
        x = np.arange(len(possible_rankings))
        labels = [">".join([str(x) for x in ranking]) for ranking in possible_rankings]
        means = [
            np.mean(y_probs_pl, axis=0),
            # np.mean(y_probs_pl_brier, axis=0),
            # np.mean(y_probs_pref, axis=0),
            np.mean(y_probs_mallows, axis=0),
            np.mean(y_probs_baseline, axis=0),
            y_true_probs,
        ]
        model_labels = [
            "PL",
            "Mallows",
            "RPC_PL",
            "True Probabilities",
        ]
        colors = [
            "#1f77b4",  # blue
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
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
