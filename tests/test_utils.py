"""This module contains unit tests for utility functions in cal_pref.utils."""

import itertools
import numpy as np
import torch
from cal_pref.utils import (
    calculate_binary_ece,
    calculate_top_k_full_rank_calibration,
    check_sub_k_in_ranking,
    check_top_k_in_ranking,
    construct_sub_k_tensors,
    construct_sub_k_full_rank_tensors,
    construct_top_k_full_rank_tensors,
    construct_top_k_tensors,
    calculate_sub_k_calibration,
    calculate_top_k_calibration,
    calculate_sub_k_full_rank_calibration,
    thresholded_adaptive_calibration_error_torch,
)


def _order_to_ranks(order: torch.Tensor) -> torch.Tensor:
    """Convert orderings (best->worst item IDs) to ranks-per-item (inverse permutation)."""
    if order.ndim != 2:
        raise ValueError("order must be 2D")
    n_samples, n_items = order.shape
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1,
        order - 1,
        torch.arange(1, n_items + 1, device=order.device, dtype=order.dtype)
        .unsqueeze(0)
        .expand(n_samples, n_items),
    )
    return ranks


def test_binary_ece():

    y_true = torch.tensor([1, 0, 1, 1, 1, 0], dtype=torch.float32)
    y_prob = torch.tensor(
        [3 / 4, 1 / 2, 1 / 2, 3 / 4, 3 / 4, 3 / 4], dtype=torch.float32
    )
    ece = calculate_binary_ece(y_true, y_prob, n_bins=4)
    expected_ece = 0.0  # Perfect calibration in this example
    assert abs(ece - expected_ece) < 1e-6

    y_true = torch.tensor([1, 0, 1, 1, 0], dtype=torch.float32)
    y_prob = torch.tensor([3 / 4, 1 / 2, 3 / 4, 3 / 4, 3 / 4], dtype=torch.float32)
    ece = calculate_binary_ece(y_true, y_prob, n_bins=4)
    expected_ece = (1 / 5) * 0.5  # The first bin is miscalibrated by 0.5
    assert abs(ece - expected_ece) < 1e-6


####################################
# Unit Tests for sub-k calibration #
####################################


def test_sub_k_in_full_ranking():

    full_ranking = [3, 1, 4, 2, 5]
    sub_ranking_1 = [1, 2, 5]
    sub_ranking_2 = [4, 3, 2]

    assert check_sub_k_in_ranking(sub_ranking_1, full_ranking) == True
    assert check_sub_k_in_ranking(sub_ranking_2, full_ranking) == False


def test_construct_sub_k_tensors():

    orderings = torch.tensor([[3, 1, 4, 2, 5], [2, 1, 3, 4, 5]], dtype=torch.long)
    rankings = _order_to_ranks(orderings)
    ranking_to_probs = {(3, 1, 4, 2, 5): [0.6, 0.3], (2, 1, 3, 4, 5): [0.4, 0.7]}

    sub_k_ranking = [1, 2, 5]

    sub_k_tensors, sub_k_probs = construct_sub_k_tensors(
        sub_k_ranking, rankings, ranking_to_probs
    )

    expected = torch.tensor([1.0, 0.0], dtype=torch.float32)
    assert torch.allclose(sub_k_tensors, expected)
    expected_probs = torch.tensor([0.6, 0.3], dtype=torch.float32)
    assert torch.allclose(sub_k_probs, expected_probs)

    orderings = torch.tensor([[3, 1, 4, 2, 5], [1, 2, 3, 4, 5]], dtype=torch.long)
    rankings = _order_to_ranks(orderings)
    ranking_to_probs = {(3, 1, 4, 2, 5): [0.6, 0.3], (1, 2, 3, 4, 5): [0.4, 0.7]}

    sub_k_ranking = [1, 2, 5]

    sub_k_tensors, sub_k_probs = construct_sub_k_tensors(
        sub_k_ranking, rankings, ranking_to_probs
    )

    expected = torch.tensor([1.0, 1.0], dtype=torch.float32)
    assert torch.allclose(sub_k_tensors, expected)
    expected_probs = torch.tensor([1, 1], dtype=torch.float32)
    assert torch.allclose(sub_k_probs, expected_probs)


def test_sub_k_calibration():
    orderings = torch.tensor(
        [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]],
        dtype=torch.long,
    )
    rankings = _order_to_ranks(orderings)
    ranking_to_prob = {
        (1, 2, 3): [2 / 6] * len(rankings),
        (1, 3, 2): [1 / 12] * len(rankings),
        (2, 1, 3): [1 / 12] * len(rankings),
        (2, 3, 1): [1 / 12] * len(rankings),
        (3, 1, 2): [1 / 12] * len(rankings),
        (3, 2, 1): [2 / 6] * len(rankings),
    }
    results = calculate_sub_k_calibration(
        items=[1, 2, 3],
        y_true=rankings,
        y_pred_proba=ranking_to_prob,
        k=2,
    )
    total_ece = results["total_ece"]
    expected_ece = 0.0  # Perfect calibration in this example
    assert abs(total_ece - expected_ece) < 1e-6

    orderings = torch.tensor(
        [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]],
        dtype=torch.long,
    )
    rankings = _order_to_ranks(orderings)
    ranking_to_prob = {
        (1, 2, 3): [1 / 3] * len(rankings),
        (1, 3, 2): [0] * len(rankings),
        (2, 1, 3): [0] * len(rankings),
        (2, 3, 1): [0] * len(rankings),
        (3, 1, 2): [0] * len(rankings),
        (3, 2, 1): [2 / 3] * len(rankings),
    }

    results = calculate_sub_k_calibration(
        items=[1, 2, 3],
        y_true=rankings,
        y_pred_proba=ranking_to_prob,
        k=2,
    )
    total_ece = results["total_ece"]
    print(results["sub_rankings_ece"])
    expected_ece = 0.0  # Misscalibration in this example
    assert abs(total_ece - expected_ece) > 1e-6


####################################
# Unit Tests for top-k calibration #
####################################


def test_top_k_in_full_ranking():

    full_ranking = [3, 1, 4, 2, 5]
    top_k_1 = [3, 1, 4]
    top_k_2 = [1, 4, 2]

    assert check_top_k_in_ranking(top_k_1, full_ranking) == True
    assert check_top_k_in_ranking(top_k_2, full_ranking) == False


def test_construct_top_k_tensors():

    orderings = torch.tensor([[3, 1, 4, 2, 5], [2, 1, 3, 4, 5]], dtype=torch.long)
    rankings = _order_to_ranks(orderings)
    ranking_to_probs = {(3, 1, 4, 2, 5): [0.6, 0.3], (2, 1, 3, 4, 5): [0.4, 0.7]}

    top_k_ranking = [3, 1, 4]

    top_k_tensors, top_k_probs = construct_top_k_tensors(
        top_k_ranking, rankings, ranking_to_probs
    )

    expected = torch.tensor([1.0, 0.0], dtype=torch.float32)
    assert torch.allclose(top_k_tensors, expected)
    expected_probs = torch.tensor([0.6, 0.3], dtype=torch.float32)
    assert torch.allclose(top_k_probs, expected_probs)

    orderings = torch.tensor([[3, 1, 4, 2, 5], [1, 2, 3, 4, 5]], dtype=torch.long)
    rankings = _order_to_ranks(orderings)
    ranking_to_probs = {
        (3, 1, 4, 2, 5): [0.6, 0.3],
        (3, 1, 4, 4, 5): [0.4, 0],
        (1, 2, 3, 4, 5): [0, 0.7],
    }
    top_k_ranking = [3, 1, 4]

    top_k_tensors, top_k_probs = construct_top_k_tensors(
        top_k_ranking, rankings, ranking_to_probs
    )

    expected = torch.tensor([1.0, 0.0], dtype=torch.float32)
    assert torch.allclose(top_k_tensors, expected)
    expected_probs = torch.tensor([1, 0.3], dtype=torch.float32)
    assert torch.allclose(top_k_probs, expected_probs)


def test_top_k_calibration():
    orderings = torch.tensor(
        [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]],
        dtype=torch.long,
    )
    rankings = _order_to_ranks(orderings)
    ranking_to_prob = {
        (1, 2, 3): [1 / 3] * len(rankings),
        (1, 3, 2): [0] * len(rankings),
        (2, 1, 3): [0] * len(rankings),
        (2, 3, 1): [1 / 3] * len(rankings),
        (3, 1, 2): [0] * len(rankings),
        (3, 2, 1): [1 / 3] * len(rankings),
    }

    results = calculate_top_k_calibration(
        items=[1, 2, 3],
        y_true=rankings,
        y_pred_proba=ranking_to_prob,
        k=1,
    )
    total_ece = results["total_ece"]
    expected_ece = 0.0  # Perfect calibration in this example
    assert abs(total_ece - expected_ece) < 1e-6


#########################################
# Unit Tests for rank-wise calibration #
#######################################
def test_rankwise_calibration():
    orderings = torch.tensor(
        [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]],
        dtype=torch.long,
    )
    rankings = _order_to_ranks(orderings)
    ranking_to_prob = {
        (1, 2, 3): [1 / 3] * len(rankings),
        (1, 3, 2): [0] * len(rankings),
        (2, 1, 3): [0] * len(rankings),
        (2, 3, 1): [1 / 3] * len(rankings),
        (3, 1, 2): [0] * len(rankings),
        (3, 2, 1): [1 / 3] * len(rankings),
    }

    results = calculate_top_k_calibration(
        items=[1, 2, 3],
        y_true=rankings,
        y_pred_proba=ranking_to_prob,
        k=3,
    )
    total_ece_1 = results["total_ece"]
    expected_ece = 0.0  # Miss calibration in this example
    assert abs(total_ece_1 - expected_ece) > 1e-6

    results = calculate_top_k_calibration(
        items=[1, 2, 3],
        y_true=rankings,
        y_pred_proba=ranking_to_prob,
        k=2,
    )
    total_ece_2 = results["total_ece"]
    expected_ece = 0.0  # Miss calibration in this example
    assert abs(total_ece_2 - expected_ece) > 1e-6

    results = calculate_sub_k_calibration(
        items=[1, 2, 3],
        y_true=rankings,
        y_pred_proba=ranking_to_prob,
        k=3,
    )
    total_ece_3 = results["total_ece"]
    expected_ece = 0.0  # Miss calibration in this example
    assert abs(total_ece_3 - expected_ece) > 1e-6

    assert abs(total_ece_1 - total_ece_2) < 1e-6
    assert abs(total_ece_1 - total_ece_3) < 1e-6


#####################################
# Test Sub-k Full Rank Calibration #
#####################################
def test_construct_sub_k_full_rank_tensors():

    orderings = torch.tensor([[3, 1, 4, 2, 5], [2, 1, 3, 4, 5]], dtype=torch.long)
    rankings = _order_to_ranks(orderings)
    ranking_to_probs = [
        {(3, 1, 4, 2, 5): 0.6, (2, 1, 3, 4, 5): 0.4},
        {(2, 1, 3, 4, 5): 0.7, (3, 1, 4, 2, 5): 0.3},
    ]

    item_set = [1, 2, 5]
    possible_sub_k_rankings = list(itertools.permutations(item_set))
    rankings_to_idx = {
        ranking: idx for idx, ranking in enumerate(possible_sub_k_rankings)
    }
    sub_k_tensors, sub_k_probs = construct_sub_k_full_rank_tensors(
        possible_sub_k_rankings, rankings, ranking_to_probs, rankings_to_idx
    )

    expected = torch.tensor([0, 2])
    assert torch.equal(sub_k_tensors, expected)
    expected_probs = torch.tensor(
        [[0.6, 0, 0.4, 0, 0, 0], [0.3, 0, 0.7, 0, 0, 0]], dtype=torch.float32
    )
    assert torch.allclose(sub_k_probs, expected_probs)

    item_set = [1, 2]
    possible_sub_k_rankings = list(itertools.permutations(item_set))
    rankings_to_idx = {
        ranking: idx for idx, ranking in enumerate(possible_sub_k_rankings)
    }
    print(possible_sub_k_rankings)
    sub_k_tensors, sub_k_probs = construct_sub_k_full_rank_tensors(
        possible_sub_k_rankings, rankings, ranking_to_probs, rankings_to_idx
    )

    expected = torch.tensor([0, 1])
    assert torch.equal(sub_k_tensors, expected)
    expected_probs = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32)
    assert torch.allclose(sub_k_probs, expected_probs)


def test_sub_k_full_rank_calibration():
    orderings = torch.tensor(
        [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]],
        dtype=torch.long,
    )
    rankings = _order_to_ranks(orderings)
    ranking_to_prob = {
        (1, 2, 3): 2 / 6,
        (1, 3, 2): 1 / 12,
        (2, 1, 3): 1 / 12,
        (2, 3, 1): 1 / 12,
        (3, 1, 2): 1 / 12,
        (3, 2, 1): 2 / 6,
    }
    assert sum(ranking_to_prob.values()) == 1.0
    ranking_to_probs = [ranking_to_prob for _ in range(len(rankings))]

    results = calculate_sub_k_full_rank_calibration(
        items=[1, 2, 3],
        y_true=rankings,
        y_pred_proba=ranking_to_probs,
        k=2,
    )
    total_ece = results["total_ece"]
    expected_ece = 0.0  # This example is sub-k fully calibrated
    assert abs(total_ece - expected_ece) < 1e-6

    results_sub_k_calib = calculate_sub_k_full_rank_calibration(
        items=[1, 2, 3],
        y_true=rankings,
        y_pred_proba=ranking_to_probs,
        k=2,
        mode="kernel",
        h=1 / 6,
        p_norm=1.0,
    )
    total_ece_sub_k_calib = results_sub_k_calib["total_ece"]

    orderings = torch.tensor(
        [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]],
        dtype=torch.long,
    )
    rankings = _order_to_ranks(orderings)
    ranking_to_prob = {
        (1, 2, 3): 1 / 3,
        (1, 3, 2): 0,
        (2, 1, 3): 0,
        (2, 3, 1): 0,
        (3, 1, 2): 0,
        (3, 2, 1): 2 / 3,
    }
    assert sum(ranking_to_prob.values()) == 1.0
    ranking_to_probs = [ranking_to_prob for _ in range(len(rankings))]

    results_not_sub_k_calib = calculate_sub_k_full_rank_calibration(
        items=[1, 2, 3],
        y_true=rankings,
        y_pred_proba=ranking_to_probs,
        k=2,
        mode="kernel",
        h=1 / 6,
        p_norm=1.0,
    )
    total_ece_not_sub_k_calib = results_not_sub_k_calib["total_ece"]

    expected_ece = 0.0  # This example is sub-k fully calibrated
    assert abs(total_ece_sub_k_calib - expected_ece) < (
        total_ece_not_sub_k_calib - expected_ece
    )


def test_tace_zero_for_perfect_one_hot():
    # Perfectly calibrated deterministic predictions should have TACE ~= 0.
    labels = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
    probs = torch.nn.functional.one_hot(labels, num_classes=3).float()
    out = thresholded_adaptive_calibration_error_torch(
        probs, labels, n_bins=3, threshold=0.01
    )
    assert abs(float(out["ece"].item())) < 1e-8


def test_tace_integration_via_rank_weighting():
    # TACE is used for binary sub-k/top-k calibration (not the full-rank kernel methods).
    orderings = torch.tensor(
        [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]],
        dtype=torch.long,
    )
    rankings = _order_to_ranks(orderings)
    ranking_to_prob = {
        (1, 2, 3): [1 / 6] * len(rankings),
        (1, 3, 2): [1 / 6] * len(rankings),
        (2, 1, 3): [1 / 6] * len(rankings),
        (2, 3, 1): [1 / 6] * len(rankings),
        (3, 1, 2): [1 / 6] * len(rankings),
        (3, 2, 1): [1 / 6] * len(rankings),
    }

    results = calculate_sub_k_calibration(
        items=[1, 2, 3],
        y_true=rankings,
        y_pred_proba=ranking_to_prob,
        k=2,
        rank_weighting="tace@0.0@10",
    )
    assert np.isfinite(results["total_ece"])
    assert 0.0 <= float(results["total_ece"]) <= 1.0


#####################################
# Test Top-k Full Rank Calibration #
###################################
def test_construct_top_k_full_rank_tensors():

    orderings = torch.tensor([[3, 1, 4, 2, 5], [2, 1, 3, 4, 5]], dtype=torch.long)
    rankings = _order_to_ranks(orderings)
    ranking_to_probs = [
        {(3, 1, 4, 2, 5): 0.6, (2, 1, 3, 4, 5): 0.4},
        {(2, 1, 3, 4, 5): 0.7, (3, 1, 4, 2, 5): 0.3},
    ]

    item_set = [3, 1, 4]
    possible_sub_k_rankings = list(itertools.permutations(item_set))
    rankings_to_idx = {
        ranking: idx for idx, ranking in enumerate(possible_sub_k_rankings)
    }
    top_k_tensors, top_k_probs = construct_top_k_full_rank_tensors(
        possible_sub_k_rankings, rankings, ranking_to_probs, rankings_to_idx
    )

    expected = torch.tensor([0, -2])
    assert torch.equal(top_k_tensors, expected)
    expected_probs = torch.tensor(
        [[0.6, 0, 0, 0, 0, 0], [0.3, 0, 0, 0, 0, 0]], dtype=torch.float32
    )
    assert torch.allclose(top_k_probs, expected_probs)
    item_set = [1, 2]
    possible_sub_k_rankings = list(itertools.permutations(item_set))
    rankings_to_idx = {
        ranking: idx for idx, ranking in enumerate(possible_sub_k_rankings)
    }
    print(possible_sub_k_rankings)
    sub_k_tensors, sub_k_probs = construct_sub_k_full_rank_tensors(
        possible_sub_k_rankings, rankings, ranking_to_probs, rankings_to_idx
    )

    expected = torch.tensor([0, 1])
    assert torch.equal(sub_k_tensors, expected)
    expected_probs = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32)
    assert torch.allclose(sub_k_probs, expected_probs)


def test_top_k_full_rank_calibration():
    orderings = torch.tensor(
        [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]],
        dtype=torch.long,
    )
    rankings = _order_to_ranks(orderings)
    ranking_to_prob = {
        (1, 2, 3): 1 / 3,
        (1, 3, 2): 0,
        (2, 1, 3): 0,
        (2, 3, 1): 1 / 3,
        (3, 1, 2): 0,
        (3, 2, 1): 1 / 3,
    }
    assert sum(ranking_to_prob.values()) == 1.0
    ranking_to_probs = [ranking_to_prob for _ in range(len(rankings))]

    results = calculate_top_k_full_rank_calibration(
        items=[1, 2, 3],
        y_true=rankings,
        y_pred_proba=ranking_to_probs,
        k=1,
        mode="binning",
    )
    total_ece = results["total_ece"]
    expected_ece = 0.0  # This example is top-k fully calibrated
    assert abs(total_ece - expected_ece) < 1e-6


def test_top_k_calibration_tva_runs():
    orderings = torch.tensor(
        [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]],
        dtype=torch.long,
    )
    rankings = _order_to_ranks(orderings)
    ranking_to_prob = {
        (1, 2, 3): [1 / 6] * len(rankings),
        (1, 3, 2): [1 / 6] * len(rankings),
        (2, 1, 3): [1 / 6] * len(rankings),
        (2, 3, 1): [1 / 6] * len(rankings),
        (3, 1, 2): [1 / 6] * len(rankings),
        (3, 2, 1): [1 / 6] * len(rankings),
    }
    out = calculate_top_k_calibration(
        items=[1, 2, 3],
        y_true=rankings,
        y_pred_proba=ranking_to_prob,
        k=2,
        rank_weighting="tva@0.0@10",
    )
    assert np.isfinite(float(out["total_ece"]))
    assert 0.0 <= float(out["total_ece"]) <= 1.0


def test_top_k_calibration_tva_split_args_runs():
    orderings = torch.tensor(
        [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]],
        dtype=torch.long,
    )
    rankings = _order_to_ranks(orderings)
    ranking_to_prob = {
        (1, 2, 3): [1 / 6] * len(rankings),
        (1, 3, 2): [1 / 6] * len(rankings),
        (2, 1, 3): [1 / 6] * len(rankings),
        (2, 3, 1): [1 / 6] * len(rankings),
        (3, 1, 2): [1 / 6] * len(rankings),
        (3, 2, 1): [1 / 6] * len(rankings),
    }
    out = calculate_top_k_calibration(
        items=[1, 2, 3],
        y_true=rankings,
        y_pred_proba=ranking_to_prob,
        k=2,
        ece_method="tva@0.0@10",
        filter_mode=None,
        agg_weighting="uniform",
    )
    assert np.isfinite(float(out["total_ece"]))
    assert 0.0 <= float(out["total_ece"]) <= 1.0


def test_top_k_calibration_topl_tace_runs():
    orderings = torch.tensor(
        [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]],
        dtype=torch.long,
    )
    rankings = _order_to_ranks(orderings)
    ranking_to_prob = {
        (1, 2, 3): [1 / 6] * len(rankings),
        (1, 3, 2): [1 / 6] * len(rankings),
        (2, 1, 3): [1 / 6] * len(rankings),
        (2, 3, 1): [1 / 6] * len(rankings),
        (3, 1, 2): [1 / 6] * len(rankings),
        (3, 2, 1): [1 / 6] * len(rankings),
    }
    out = calculate_top_k_calibration(
        items=[1, 2, 3],
        y_true=rankings,
        y_pred_proba=ranking_to_prob,
        k=2,
        rank_weighting="topl_tace@2@0.0@5@1",
    )
    assert np.isfinite(float(out["total_ece"]))
    assert 0.0 <= float(out["total_ece"]) <= 1.0


def test_sub_k_calibration_tva_runs():
    orderings = torch.tensor(
        [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]],
        dtype=torch.long,
    )
    rankings = _order_to_ranks(orderings)
    ranking_to_prob = {
        (1, 2, 3): [1 / 6] * len(rankings),
        (1, 3, 2): [1 / 6] * len(rankings),
        (2, 1, 3): [1 / 6] * len(rankings),
        (2, 3, 1): [1 / 6] * len(rankings),
        (3, 1, 2): [1 / 6] * len(rankings),
        (3, 2, 1): [1 / 6] * len(rankings),
    }
    out = calculate_sub_k_calibration(
        items=[1, 2, 3],
        y_true=rankings,
        y_pred_proba=ranking_to_prob,
        k=2,
        rank_weighting="tva@0.0@10",
    )
    assert np.isfinite(float(out["total_ece"]))
    assert 0.0 <= float(out["total_ece"]) <= 1.0


def test_sub_k_calibration_filter_topl_runs():
    orderings = torch.tensor(
        [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]],
        dtype=torch.long,
    )
    rankings = _order_to_ranks(orderings)
    ranking_to_prob = {
        (1, 2, 3): [1 / 6] * len(rankings),
        (1, 3, 2): [1 / 6] * len(rankings),
        (2, 1, 3): [1 / 6] * len(rankings),
        (2, 3, 1): [1 / 6] * len(rankings),
        (3, 1, 2): [1 / 6] * len(rankings),
        (3, 2, 1): [1 / 6] * len(rankings),
    }
    out = calculate_sub_k_calibration(
        items=[1, 2, 3],
        y_true=rankings,
        y_pred_proba=ranking_to_prob,
        k=2,
        ece_method="tace@0.0@10",
        filter_mode="filter_topl@2@0",
        agg_weighting="uniform",
    )
    assert np.isfinite(float(out["total_ece"]))
    assert 0.0 <= float(out["total_ece"]) <= 1.0


def test_top_k_calibration_tace_runs():
    orderings = torch.tensor(
        [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]],
        dtype=torch.long,
    )
    rankings = _order_to_ranks(orderings)
    ranking_to_prob = {
        (1, 2, 3): [1 / 6] * len(rankings),
        (1, 3, 2): [1 / 6] * len(rankings),
        (2, 1, 3): [1 / 6] * len(rankings),
        (2, 3, 1): [1 / 6] * len(rankings),
        (3, 1, 2): [1 / 6] * len(rankings),
        (3, 2, 1): [1 / 6] * len(rankings),
    }
    results = calculate_top_k_calibration(
        items=[1, 2, 3],
        y_true=rankings,
        y_pred_proba=ranking_to_prob,
        k=1,
        rank_weighting="tace@0.0@10",
    )
    assert np.isfinite(results["total_ece"])
    assert 0.0 <= float(results["total_ece"]) <= 1.0


def test_from_bradley_terry_to_plackett_luce():
    from cal_pref.utils import (
        from_bradley_terry_to_placet_luce_old,
        from_bradley_terry_to_placket_luce_simple,
        from_bradley_terry_to_placket_luce_vectorized,
        from_bradley_terry_to_placket_luce_map,
    )

    rng = np.random.default_rng(42)
    pl_scores = np.array([0.2, 0.5, 0.3])
    bt_matrix = np.zeros((1, 3, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                bt_matrix[0, i, j] = pl_scores[i] / (pl_scores[i] + pl_scores[j])
    recovered_pl_scores = from_bradley_terry_to_placket_luce_simple(
        rng=rng, pair_order_matrices=bt_matrix, n_iterations=1000
    )[0]
    # Normalize the scores
    recovered_pl_scores /= np.sum(recovered_pl_scores)
    assert np.allclose(pl_scores, recovered_pl_scores, atol=1e-6)

    # recovered_pl_scores = from_bradley_terry_to_placket_luce_map(
    #     rng=rng, pair_order_matrices=bt_matrix, n_iterations=100_000
    # )[0]
    # # Normalize the scores
    # recovered_pl_scores /= np.sum(recovered_pl_scores)
    # assert np.allclose(pl_scores, recovered_pl_scores, atol=1e-6)

    recovered_pl_scores = from_bradley_terry_to_placket_luce_vectorized(
        rng=rng, pair_order_matrices=bt_matrix, n_iterations=100
    )[0]
    # Normalize the scores
    recovered_pl_scores /= np.sum(recovered_pl_scores)
    assert np.allclose(pl_scores, recovered_pl_scores, atol=1e-6)

    recovered_pl_scores = from_bradley_terry_to_placet_luce_old(
        rng=rng, pair_order_matrices=bt_matrix, n_iterations=1000
    )[0]
    # Normalize the scores
    recovered_pl_scores /= np.sum(recovered_pl_scores)
    assert np.allclose(pl_scores, recovered_pl_scores, atol=1e-6)
