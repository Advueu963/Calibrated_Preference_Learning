from itertools import permutations
import math
import torch
import torch.nn as nn
import numpy as np


def build_preference_mlp(input_dim: int, hidden_dims: list[int], output_dim: int):
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.ReLU())
        in_dim = h_dim
    layers.append(nn.Linear(in_dim, output_dim))

    return nn.Sequential(*layers)


def build_plackett_luce_mlp(input_dim: int, hidden_dims: list[int], n_items: int):
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.ReLU())
        in_dim = h_dim
    layers.append(nn.Linear(in_dim, n_items))  # Single output for Plackett-Luce
    return nn.Sequential(*layers)


class PreferenceModel(nn.Module):
    """Probabilistic Preference Model using training each possible ranking as a class.
    This model uses a multi-layer perceptron (MLP) to predict the logits for each
    possible ranking of items. Each unique ranking is treated as a separate class.
    The model can handle unseen rankings by assigning them a uniform low probability.
    During inference, the model outputs the probabilities for each ranking using
    the softmax function.
    """

    def __init__(
        self,
        input_dim: int,
        n_items: int,
        hidden_dims: list[int],
        output_dim: int,
        unique_rankings: torch.Tensor,
        constant_value: float = 0.0,
    ):
        super(PreferenceModel, self).__init__()
        self.mlp = build_preference_mlp(input_dim, hidden_dims, output_dim)
        self.n_unseen = math.factorial(n_items) - output_dim
        if self.n_unseen > 0:
            # Single shared logit for every unseen ranking. Each unseen ranking then gets
            # probability exp(unseen_logit) / (sum(exp(seen_logits)) + n_unseen*exp(unseen_logit)).
            # Register as buffer so it moves with .to(device) / .cuda().
            self.register_buffer("unseen_logit", torch.tensor(float(constant_value)))
        else:
            self.unseen_logit = None
        self.n_items = n_items
        self.rankings = unique_rankings
        self.idx_rankings = {
            tuple(r.tolist()): i for i, r in enumerate(unique_rankings)
        }
        self.temperature = 1.0

        # print("SELF N ITEMS: ", self.n_items)

    def forward(self, x):
        x = self.mlp(x)
        # if self.unseen_weights is not None:
        #     # print("UNSEEN WEIGHTS: ", self.unseen_weights)
        #     x = torch.hstack(
        #         (
        #             x,
        #             torch.repeat_interleave(
        #                 self.unseen_weights.unsqueeze(0), x.shape[0], dim=0
        #             ),
        #         )
        #     )
        # print(x)
        return x / self.temperature

    def _probs_seen_and_unseen_each(self, logits: torch.Tensor):
        """Return (probs_seen, p_unseen_each).

        probs_seen has shape (batch, output_dim).
        p_unseen_each has shape (batch, 1) and is the probability of any *one*
        particular unseen ranking (uniform across all unseen rankings).
        """
        # Stable normalization.
        log_z_seen = torch.logsumexp(
            logits, dim=-1, keepdim=True
        )  # log(sum(exp(seen_logits)))

        if self.n_unseen > 0:
            assert self.unseen_logit is not None
            # Total unseen mass contributes n_unseen * exp(unseen_logit)
            log_unseen_mass = self.unseen_logit + math.log(self.n_unseen)
            log_denom = torch.logaddexp(log_z_seen, log_unseen_mass)
            p_unseen_each = torch.exp(self.unseen_logit - log_denom)
        else:
            log_denom = log_z_seen
            p_unseen_each = torch.zeros(
                (logits.shape[0], 1), device=logits.device, dtype=logits.dtype
            )

        probs_seen = torch.exp(logits - log_denom)
        return probs_seen, p_unseen_each

    def calculate_probs_with_unseen(self, logits):
        # Compute probabilities over seen rankings adjusted for unseen rankings.
        probs_seen, _p_unseen_each = self._probs_seen_and_unseen_each(logits)
        return probs_seen

    def predict_proba(self, x):
        logits = self.forward(x)
        # Compute Softmax probabilities adjusted for unseen rankings
        probs = self.calculate_probs_with_unseen(logits)
        # probs = nn.functional.softmax(logits, dim=-1)
        return probs

    def predict_proba_ranking(self, x, rank):
        logits = self.forward(x)
        return self.predict_proba_ranking_logits(logits, rank)

    def predict_proba_ranking_logits(self, logits, rank):
        probs, p_unseen_each = self._probs_seen_and_unseen_each(logits)
        if len(rank.shape) == 1:
            rank = rank.expand(logits.shape[0], -1)
        idx_list = []
        seen_mask_list = []
        for r in rank:
            idx = self.idx_rankings.get(tuple(r.tolist()), -1)
            seen_mask_list.append(idx != -1)
            idx_list.append(idx if idx != -1 else 0)

        idx_ranks = torch.tensor(idx_list, device=logits.device, dtype=torch.long)
        seen_mask = torch.tensor(seen_mask_list, device=logits.device, dtype=torch.bool)

        gathered = torch.gather(probs, 1, idx_ranks.unsqueeze(1)).squeeze(1)
        unseen_vals = p_unseen_each.squeeze(1)
        return torch.where(seen_mask, gathered, unseen_vals)

    def predict_ranking_distribution(self, x, restricted_rankings=None):
        logits = self.forward(x)
        if restricted_rankings is None:
            rankings = permutations(range(1, self.n_items + 1))
        else:
            rankings = restricted_rankings
        distribution = {}
        for rank in rankings:
            rank_tensor = (
                torch.tensor(rank, device=logits.device)
                .unsqueeze(0)
                .expand(x.shape[0], -1)
            )
            probs = self.predict_proba_ranking_logits(logits, rank_tensor)
            distribution[rank] = probs.detach()
        return distribution

    def predict(self, x):
        logits = self.forward(x)
        probs = nn.functional.softmax(logits, dim=-1)
        # print("PROBS: ", probs)
        idx_ranking = torch.argmax(probs, dim=-1)

        preds = [
            (
                self.rankings[idx]
                if idx < len(self.rankings)
                else torch.zeros(self.n_items, device=logits.device)
            )
            for idx in idx_ranking
        ]
        return torch.stack(preds, dim=0).long()


class PlackettLuceModelWeights(nn.Module):
    def __init__(self, weights, n_items: int):
        super(PlackettLuceModelWeights, self).__init__()
        self.n_items = n_items
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, x):
        return self.weights

    def predict(self, x):
        ranking = torch.argsort(self.weights, dim=-1, descending=True) + 1
        return ranking

    def predict_proba(self, x):
        cumulative_sums = torch.cumsum(
            torch.sort(self.weights, dim=-1, descending=False), dim=-1
        )
        weights, _ = torch.sort(self.weights, dim=-1, descending=True)
        weights = weights / cumulative_sums
        probs = torch.prod(weights, dim=-1, keepdim=True)
        return probs

    def predict_proba_ranking_logits(self, logits, rank):
        if len(rank.shape) == 1:
            rank = rank.expand(logits.shape[0], -1)
        idx_rank_sort = torch.argsort(rank, dim=-1, descending=True)
        weights = torch.gather(
            logits, 1, idx_rank_sort
        )  # the exps that the highest rank (1) comes last
        cum_sums = torch.cumsum(weights, dim=-1)
        probs = (weights / cum_sums).prod(dim=-1)
        return probs

    def predict_ranking_distribution(self, x, restricted_rankings=None):
        logits = self.forward(x).clamp(min=1e-6)
        if restricted_rankings is None:
            rankings = permutations(range(1, self.n_items + 1))
        else:
            rankings = restricted_rankings
        distribution = {}
        for rank in rankings:
            rank_tensor = (
                torch.tensor(rank, device=logits.device)
                .unsqueeze(0)
                .expand(x.shape[0], -1)
            )
            probs = self.predict_proba_ranking_logits(logits, rank_tensor)
            distribution[rank] = probs.detach()
        return distribution

    def predict_proba_ranking(self, x, rank):
        logits = self.forward(x)
        return self.predict_proba_ranking_logits(logits, rank)


class PlackettLuceModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super(PlackettLuceModel, self).__init__()
        self.mlp = build_plackett_luce_mlp(input_dim, hidden_dims, output_dim)
        self.n_items = output_dim
        # print("SELF N ITEMS: ", self.n_items)

    def forward(self, x):
        x = self.mlp(x)
        x = x - x.max(dim=-1, keepdim=True).values  # Improve numerical stability
        x = nn.functional.softmax(x, dim=-1)
        return x.clamp_min(1e-9)

    def predict(self, x):
        weights = self.forward(x)
        # exp_logits = torch.exp(logits)
        ranking = torch.argsort(weights, dim=-1, descending=True) + 1
        return ranking

    def predict_proba(self, x):
        weights = self.forward(x)
        # exp_logits = torch.exp(logits)
        cumulative_sums = torch.cumsum(
            torch.sort(weights, dim=-1, descending=False), dim=-1
        )
        weights, _ = torch.sort(weights, dim=-1, descending=True)
        weights = weights / cumulative_sums
        probs = torch.prod(weights, dim=-1, keepdim=True)
        return probs

    def predict_proba_ranking_logits(self, weights, rank):

        if len(rank.shape) == 1:
            rank = rank.expand(weights.shape[0], -1)
        idx_rank_sort = torch.argsort(rank, dim=-1, descending=True)
        weights = torch.gather(
            weights, 1, idx_rank_sort
        )  # the exps that the highest rank (1) comes last
        cum_sums = torch.cumsum(weights, dim=-1)
        probs = (weights / cum_sums).prod(dim=-1)

        # print("PROBS: ", probs)
        if any(probs.isnan()):
            print("LOGITS: ", weights)
            print("RANK: ", rank)
            print("CUM SUMS: ", cum_sums)
            print("PROBS: ", probs)
            raise ValueError("NaN values in probabilities.")
        return probs

    def predict_ranking_distribution(self, x, restricted_rankings=None):
        weights = self.forward(x)
        if restricted_rankings is None:
            rankings = permutations(range(1, self.n_items + 1))
        else:
            rankings = restricted_rankings
        distribution = {}
        for rank in rankings:
            rank_tensor = (
                torch.tensor(rank, device=weights.device)
                .unsqueeze(0)
                .expand(x.shape[0], -1)
            )
            probs = self.predict_proba_ranking_logits(weights, rank_tensor)
            distribution[rank] = probs.detach()
        return distribution

    def predict_proba_ranking(self, x, rank):
        weights = self.forward(x)
        return self.predict_proba_ranking_logits(weights, rank)


class MallowsModel(nn.Module):
    def __init__(
        self,
        reference_ranking: torch.Tensor,
        dispersion: float,
        distance_metric: str = "kendall",
    ):
        self.reference_ranking = reference_ranking
        self.n_items = reference_ranking.shape[0]
        self.theta = dispersion
        self.dispersion = np.exp(-dispersion)
        self.distance_metric = distance_metric
        self.normalization_constant = self.compute_normalization_constant()

    def compute_distance(self, rank1: torch.Tensor):
        if self.distance_metric == "kendall":
            distance = 0
            for i in range(self.n_items):
                for j in range(i + 1, self.n_items):
                    if (rank1[i] - rank1[j]) * (
                        self.reference_ranking[i] - self.reference_ranking[j]
                    ) < 0:
                        distance += 1
            return distance
        else:
            raise NotImplementedError(
                f"Distance metric {self.distance_metric} not implemented."
            )

    def compute_normalization_constant(self):
        constant = 1.0
        for j in range(1, self.n_items + 1):

            constant *= 1 + sum([self.dispersion**k for k in range(1, j)])
        return 1 / constant

    def forward(self, x):
        return None

    def predict_proba_ranking(self, x, ranking: torch.Tensor):
        if len(ranking.shape) == 1:
            ranking = ranking.expand(x.shape[0], -1)
        distance_to_reference = torch.tensor(
            [self.compute_distance(r) for r in ranking]
        )
        probs = self.normalization_constant * (self.dispersion**distance_to_reference)
        return probs

    def predict(self, x):
        return self.reference_ranking.expand(x.shape[0], -1).long()

    def predict_ranking_distribution(self, x, restricted_rankings=None):
        if restricted_rankings is None:
            rankings = permutations(range(1, self.n_items + 1))
        else:
            rankings = restricted_rankings
        distribution = {}
        for rank in rankings:
            rank_tensor = (
                torch.tensor(rank, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
            )
            probs = self.predict_proba_ranking(x, rank_tensor)
            distribution[rank] = probs.detach()
        return distribution

    @classmethod
    def fit_from_data(
        cls,
        rankings: torch.Tensor,
        distance_metric: str = "kendall",
        max_iters: int = 50,
        tol: float = 1e-4,
    ):
        if distance_metric not in {"kendall", "cayley"}:
            raise ValueError(
                f"Unsupported distance metric '{distance_metric}'. Choose 'kendall' or 'cayley'."
            )

        rankings_np = rankings.detach().cpu().numpy()
        modal_ranking = _mean_rank_initialization(rankings_np)
        theta = 1.0
        prev_log_likelihood = None

        for _ in range(max_iters):
            distances = _distances_to_modal(modal_ranking, rankings_np, distance_metric)
            sum_distances = distances.sum()
            theta = _estimate_theta(
                sum_distances, rankings_np.shape[0], modal_ranking.size, tol
            )
            log_likelihood = _log_likelihood(
                theta, sum_distances, rankings_np.shape[0], modal_ranking.size
            )

            improved = False
            current_sum = sum_distances
            best_ranking = modal_ranking
            for neighbor in _cayley_neighbors(modal_ranking):
                neighbor_sum = _total_distance(neighbor, rankings_np, distance_metric)
                if neighbor_sum + 1e-9 < current_sum:
                    current_sum = neighbor_sum
                    best_ranking = neighbor
                    improved = True

            modal_ranking = best_ranking

            if (
                prev_log_likelihood is not None
                and abs(log_likelihood - prev_log_likelihood) < tol
                and not improved
            ):
                break
            prev_log_likelihood = log_likelihood

        reference = torch.tensor(modal_ranking, dtype=torch.long)
        model = cls(
            reference_ranking=reference,
            dispersion=theta,
            distance_metric=distance_metric,
        )
        return model


def _mean_rank_initialization(rankings: np.ndarray) -> np.ndarray:
    n_items = rankings.shape[1]
    mean_positions = np.zeros(n_items, dtype=np.float64)
    for ranking in rankings:
        for position, item in enumerate(ranking):
            mean_positions[int(item) - 1] += position
    mean_positions /= rankings.shape[0]
    ordering = np.argsort(mean_positions)
    return (ordering + 1).astype(np.int64)


def _distance_fn(distance_metric: str):
    if distance_metric == "kendall":
        return _kendall_distance
    return _cayley_distance


def _distances_to_modal(
    modal_ranking: np.ndarray, rankings: np.ndarray, distance_metric: str
) -> np.ndarray:
    distance = _distance_fn(distance_metric)
    return np.array(
        [distance(modal_ranking, ranking) for ranking in rankings], dtype=np.float64
    )


def _total_distance(
    modal_ranking: np.ndarray, rankings: np.ndarray, distance_metric: str
) -> float:
    return _distances_to_modal(modal_ranking, rankings, distance_metric).sum()


def _cayley_neighbors(ranking: np.ndarray):
    n_items = ranking.size
    neighbors = [ranking.copy()]
    for i in range(n_items):
        for j in range(i + 1, n_items):
            neighbor = ranking.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors


def _estimate_theta(
    sum_distances: float, n_samples: int, n_items: int, tol: float
) -> float:
    if sum_distances <= 1e-12:
        return 20.0

    lower, upper = 1e-6, 1.0
    while (
        _theta_derivative(upper, sum_distances, n_samples, n_items) > 0 and upper < 100
    ):
        upper *= 2.0

    for _ in range(100):
        mid = 0.5 * (lower + upper)
        derivative = _theta_derivative(mid, sum_distances, n_samples, n_items)
        if abs(derivative) < tol:
            return mid
        if derivative > 0:
            lower = mid
        else:
            upper = mid
    return mid


def _theta_derivative(
    theta: float, sum_distances: float, n_samples: int, n_items: int
) -> float:
    exp_neg_theta = math.exp(-theta)
    exp_neg_theta = min(exp_neg_theta, 1 - 1e-12)
    base_denominator = 1 - exp_neg_theta
    base_term = exp_neg_theta / max(base_denominator, 1e-12)

    derivative_log_z = 0.0
    for j in range(1, n_items):
        multiplier = j + 1
        exp_component = math.exp(-multiplier * theta)
        exp_component = min(exp_component, 1 - 1e-12)
        denominator = max(1 - exp_component, 1e-12)
        derivative_log_z += (multiplier * exp_component / denominator) - base_term

    return -n_samples * derivative_log_z - sum_distances


def _log_likelihood(
    theta: float, sum_distances: float, n_samples: int, n_items: int
) -> float:
    return -n_samples * _log_partition(theta, n_items) - theta * sum_distances


def _log_partition(theta: float, n_items: int) -> float:
    if theta < 1e-8:
        return math.log(math.factorial(n_items))

    log_z = 0.0
    exp_neg_theta = math.exp(-theta)
    log_one_minus_exp = math.log1p(-exp_neg_theta)
    for j in range(1, n_items):
        log_z += math.log1p(-math.exp(-(j + 1) * theta)) - log_one_minus_exp
    return log_z


def _kendall_distance(p: np.ndarray, q: np.ndarray) -> int:
    positions = {int(item): idx for idx, item in enumerate(q)}
    distance = 0
    n_items = len(p)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            if positions[int(p[i])] > positions[int(p[j])]:
                distance += 1
    return distance


def _cayley_distance(p: np.ndarray, q: np.ndarray) -> int:
    source = list(map(int, p))
    target = list(map(int, q))
    position = {value: idx for idx, value in enumerate(source)}
    swaps = 0
    for idx, correct_value in enumerate(target):
        current_value = source[idx]
        if current_value != correct_value:
            swaps += 1
            swap_idx = position[correct_value]
            position[current_value] = swap_idx
            source[idx], source[swap_idx] = source[swap_idx], source[idx]
            position[correct_value] = idx
    return swaps
