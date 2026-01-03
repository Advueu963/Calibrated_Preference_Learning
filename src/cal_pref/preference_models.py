from itertools import permutations
import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def _order_to_ranks_tensor(order: torch.Tensor, n_items: int) -> torch.Tensor:
    """Convert a best->worst item ordering into ranks-per-item.

    order: (..., n_items) with entries in {1..n_items}.
    returns ranks: (..., n_items) where ranks[..., j] is rank of item (j+1).
    """
    if order.dim() == 1:
        order = order.unsqueeze(0)
    order = order.to(dtype=torch.long)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1,
        order - 1,
        torch.arange(1, n_items + 1, device=order.device).unsqueeze(0).expand_as(order),
    )
    return ranks


def _ranks_to_order_tuple(ranks: tuple[int, ...]) -> tuple[int, ...]:
    """Convert ranks-per-item (tuple) to a best->worst order tuple."""
    # ranks[j] is rank of item (j+1). Smaller rank = better.
    # argsort gives item indices best->worst.
    import numpy as _np

    ranks_arr = _np.asarray(ranks)
    order = _np.argsort(ranks_arr) + 1
    return tuple(int(x) for x in order.tolist())


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
    """Probabilistic Preference Model using each ranking as a class.

    Conventions used in this repo:
    - Training labels (`y_true`) are ranks-per-item (inverse permutation):
        `y[s, j]` is the rank (1=best) of item (j+1).
    - `.predict(...)` returns ranks-per-item.
    - Predicted ranking distributions are dicts keyed by orderings (best->worst
        tuples of item IDs), via `.predict_ranking_distribution(...)`.

    This model uses a multi-layer perceptron (MLP) to predict the logits for each
    possible ranking of items. Each unique ranking is treated as a separate class.
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

    def calculate_probs_with_unseen(self, logits, with_unseen=False):
        # Compute probabilities over seen rankings adjusted for unseen rankings.
        probs_seen, _p_unseen_each = self._probs_seen_and_unseen_each(logits)
        if with_unseen:
            return probs_seen, _p_unseen_each
        return probs_seen

    def predict_proba(self, x, with_unseen=False):
        logits = self.forward(x)
        # Compute Softmax probabilities adjusted for unseen rankings
        probs = self.calculate_probs_with_unseen(logits, with_unseen=with_unseen)
        # probs = nn.functional.softmax(logits, dim=-1)
        return probs

    def predict_proba_ranking(self, x, ranking):
        """Return P(ordering | x) for each sample.

        `ranking` is a best->worst order tuple/tensor of item IDs in {1..n_items}.
        """
        logits = self.forward(x)
        ranking = torch.as_tensor(ranking, device=logits.device)
        if ranking.dim() == 1:
            ranking = ranking.unsqueeze(0)
        if ranking.shape[0] == 1 and logits.shape[0] > 1:
            ranking = ranking.expand(logits.shape[0], -1)
        ranks = _order_to_ranks_tensor(ranking, self.n_items)
        return self.predict_proba_ranking_logits(logits, ranks)

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
            rankings = list(permutations(range(1, self.n_items + 1)))
        else:
            rankings = restricted_rankings
        distribution = {}

        # We expose distributions keyed by *orderings* (best->worst item IDs).
        # Internally, this model's classes are stored as ranks-per-item.
        probs_seen, probs_unseen = self.predict_proba(x, with_unseen=True)
        for i in tqdm(range(len(rankings)), desc="Predicting ranking distribution PM"):
            order = tuple(rankings[i])
            rank_tensor = (
                torch.tensor(order, device=logits.device)
                .unsqueeze(0)
                .expand(x.shape[0], -1)
            )
            ranks_tensor = _order_to_ranks_tensor(rank_tensor, self.n_items)
            idx = self.idx_rankings.get(tuple(ranks_tensor[0].tolist()), -1)
            if idx != -1:
                prob = probs_seen[:, idx]
            else:
                prob = probs_unseen.squeeze(1)
            distribution[order] = prob.detach()
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
        # Return ranks-per-item (inverse permutation): y[i] is the rank of item (i+1).
        order = torch.argsort(self.weights, dim=-1, descending=True) + 1
        n_items = order.shape[-1]
        ranks = torch.empty_like(order)
        ranks.scatter_(
            1,
            order - 1,
            torch.arange(1, n_items + 1, device=order.device)
            .unsqueeze(0)
            .expand_as(order),
        )
        return ranks

    def predict_proba(self, x):
        cumulative_sums = torch.cumsum(
            torch.sort(self.weights, dim=-1, descending=False), dim=-1
        )
        weights, _ = torch.sort(self.weights, dim=-1, descending=True)
        weights = weights / cumulative_sums
        probs = torch.prod(weights, dim=-1, keepdim=True)
        return probs

    def predict_proba_ranking_logits(self, logits, rank):
        # `rank` is ranks-per-item (inverse permutation). Convert to an ordering.
        log_p = self.log_proba_ranking_logits(logits, rank)
        return torch.exp(log_p)

    def log_proba_ranking_logits(self, weights: torch.Tensor, rank: torch.Tensor):
        """Return log P(rank | weights) where `rank` is ranks-per-item."""
        if rank.dim() == 1:
            rank = rank.unsqueeze(0).expand(weights.shape[0], -1)
        rank = rank.to(dtype=torch.long)

        # ranks-per-item -> order best->worst (item IDs in {1..n}).
        order = torch.argsort(rank, dim=-1) + 1
        weights_ranked = torch.gather(weights, 1, order - 1)

        remaining_sums = torch.flip(
            torch.cumsum(torch.flip(weights_ranked, dims=[-1]), dim=-1), dims=[-1]
        )

        eps = torch.finfo(weights.dtype).tiny
        log_term = torch.log(weights_ranked.clamp_min(eps)) - torch.log(
            remaining_sums.clamp_min(eps)
        )
        return log_term.sum(dim=-1)

    def predict_ranking_distribution(self, x, restricted_rankings=None):
        logits = self.forward(x).clamp(min=1e-6)
        if restricted_rankings is None:
            rankings = list(permutations(range(1, self.n_items + 1)))
        else:
            rankings = restricted_rankings
        distribution = {}
        for i in tqdm(
            range(len(rankings)), desc="Predicting ranking distribution PLRPC"
        ):
            order = tuple(rankings[i])
            order_tensor = (
                torch.tensor(order, device=logits.device)
                .unsqueeze(0)
                .expand(x.shape[0], -1)
            )
            ranks_tensor = _order_to_ranks_tensor(order_tensor, self.n_items)
            probs = self.predict_proba_ranking_logits(logits, ranks_tensor)
            distribution[order] = probs.detach()
        return distribution

    def predict_proba_ranking(self, x, ranking):
        """Return P(ordering | weights) for each sample.

        `ranking` is a best->worst order tuple/tensor of item IDs.
        """
        logits = self.forward(x)
        ranking = torch.as_tensor(ranking, device=logits.device)
        if ranking.dim() == 1:
            ranking = ranking.unsqueeze(0)
        if ranking.shape[0] == 1 and logits.shape[0] > 1:
            ranking = ranking.expand(logits.shape[0], -1)
        ranks = _order_to_ranks_tensor(ranking, self.n_items)
        return self.predict_proba_ranking_logits(logits, ranks)


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
        return x

    def predict(self, x):
        # Return ranks-per-item (inverse permutation): y[i] is the rank of item (i+1).
        weights = self.forward(x)
        order = torch.argsort(weights, dim=-1, descending=True)
        n_items = order.shape[-1]
        ranks = torch.empty_like(order)
        ranks.scatter_(
            1,
            order,
            torch.arange(1, n_items + 1, device=order.device)
            .unsqueeze(0)
            .expand_as(order),
        )
        return ranks

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
        """Return P(rank | weights) for each sample.

        Notes:
                - `rank` is expected to be ranks-per-item (inverse permutation):
                    rank[i] is the rank of item (i+1).
        - `weights` are positive per-item scores (we use softmax outputs).
        """

        log_probs = self.log_proba_ranking_logits(weights, rank)
        return torch.exp(log_probs)

    def log_proba_ranking_logits(
        self, weights: torch.Tensor, rank: torch.Tensor
    ) -> torch.Tensor:
        """Return log P(rank | weights) for each sample (numerically stable).

        `rank` is ranks-per-item (inverse permutation).
        """

        if rank.dim() == 1:
            rank = rank.unsqueeze(0).expand(weights.shape[0], -1)
        rank = rank.to(dtype=torch.long)

        # Convert ranks-per-item -> order best->worst (item IDs in {1..n}).
        order = torch.argsort(rank, dim=-1) + 1
        weights_ranked = torch.gather(weights, 1, order - 1)

        # Denominator at each position is the sum of remaining weights.
        # remaining_sums[k] = sum_{j=k..n-1} weights_ranked[j]
        remaining_sums = torch.flip(
            torch.cumsum(torch.flip(weights_ranked, dims=[-1]), dim=-1), dims=[-1]
        )

        eps = torch.finfo(weights.dtype).tiny
        log_term = torch.log(weights_ranked.clamp_min(eps)) - torch.log(
            remaining_sums.clamp_min(eps)
        )
        log_prob = log_term.sum(dim=-1)

        if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
            raise ValueError("NaN/Inf in Plackett-Luce log probability.")

        return log_prob

    def predict_ranking_distribution(self, x, restricted_rankings=None):
        weights = self.forward(x)
        if restricted_rankings is None:
            rankings = list(permutations(range(1, self.n_items + 1)))
        else:
            rankings = restricted_rankings
        distribution = {}
        for i in tqdm(range(len(rankings)), desc="Predicting ranking distribution PL"):
            order = tuple(rankings[i])
            order_tensor = (
                torch.tensor(order, device=weights.device)
                .unsqueeze(0)
                .expand(x.shape[0], -1)
            )
            ranks_tensor = _order_to_ranks_tensor(order_tensor, self.n_items)
            probs = self.predict_proba_ranking_logits(weights, ranks_tensor)
            distribution[order] = probs.detach()
        return distribution

    def predict_proba_ranking(self, x, ranking):
        """Return P(ordering | weights) for each sample.

        `ranking` is a best->worst order tuple/tensor of item IDs.
        """
        weights = self.forward(x)
        ranking = torch.as_tensor(ranking, device=weights.device)
        if ranking.dim() == 1:
            ranking = ranking.unsqueeze(0)
        if ranking.shape[0] == 1 and weights.shape[0] > 1:
            ranking = ranking.expand(weights.shape[0], -1)
        ranks = _order_to_ranks_tensor(ranking, self.n_items)
        return self.predict_proba_ranking_logits(weights, ranks)


class MallowsModel(nn.Module):
    """Mallows model.

    Conventions used in this repo:
    - `.predict(...)` returns **ranks-per-item**.
    - `.predict_ranking_distribution(...)` returns a dict keyed by **orderings**
        (best->worst tuples of item IDs).
    """

    def __init__(
        self,
        reference_ranking: torch.Tensor,
        dispersion: float,
        distance_metric: str = "kendall",
    ):
        super().__init__()
        # Public convention for distributions in this repo is to use *orderings*
        # (best->worst item IDs). Internally, for pairwise comparisons we store
        # the reference ranking as ranks-per-item.
        self.reference_order = reference_ranking.to(dtype=torch.long)
        self.n_items = int(reference_ranking.shape[0])
        self.reference_ranks = _order_to_ranks_tensor(
            self.reference_order, self.n_items
        ).squeeze(0)
        self.theta = dispersion
        self.dispersion = np.exp(-dispersion)
        self.distance_metric = distance_metric
        self.normalization_constant = self.compute_normalization_constant()

    def compute_distance(self, rank1: torch.Tensor):
        """Compute distance where rank1 is ranks-per-item."""
        if self.distance_metric == "kendall":
            distance = 0
            for i in range(self.n_items):
                for j in range(i + 1, self.n_items):
                    if (rank1[i] - rank1[j]) * (
                        self.reference_ranks[i] - self.reference_ranks[j]
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
        """Return P(ordering | model) for each sample.

        `ranking` is expected to be a best->worst ordering of item IDs.
        """
        ranking = torch.as_tensor(ranking)
        if ranking.dim() == 1:
            ranking = ranking.unsqueeze(0)
        n_samples = int(getattr(x, "shape", [1])[0]) if x is not None else 1
        if ranking.shape[0] == 1 and n_samples > 1:
            ranking = ranking.expand(n_samples, -1)
        # Convert ordering -> ranks-per-item for distance computation.
        ranks = _order_to_ranks_tensor(
            ranking.to(self.reference_order.device), self.n_items
        )
        distance_to_reference = torch.tensor(
            [self.compute_distance(r) for r in ranks], device=ranks.device
        )
        probs = self.normalization_constant * (self.dispersion**distance_to_reference)
        return probs

    def predict(self, x):
        # Return ranks-per-item (inverse permutation).
        return self.reference_ranks.expand(x.shape[0], -1).long()

    def predict_ranking_distribution(self, x, restricted_rankings=None):
        if restricted_rankings is None:
            rankings = list(permutations(range(1, self.n_items + 1)))
        else:
            rankings = restricted_rankings
        distribution = {}
        for i in tqdm(
            range(len(rankings)), desc="Predicting ranking distribution Mallows"
        ):
            rank = rankings[i]
            order = tuple(rank)
            rank_tensor = (
                torch.tensor(order, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
            )
            probs = self.predict_proba_ranking(x, rank_tensor)
            distribution[order] = probs.detach()
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

        # `rankings` comes from datasets in this repo as ranks-per-item.
        rankings_np_ranks = rankings.detach().cpu().numpy()
        # Convert to best->worst orderings for the neighbor search utilities below.
        rankings_np = np.argsort(rankings_np_ranks, axis=1) + 1
        modal_ranking = _mean_rank_initialization(rankings_np_ranks)

        theta = 1.0
        prev_log_likelihood = None

        for _ in tqdm(range(max_iters), desc="Fitting Mallows Model"):
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
    """Initialize a modal *ordering* from ranks-per-item data.

    rankings is ranks-per-item (inverse permutations), shape (N, n_items).
    Returns a best->worst ordering (item IDs).
    """
    mean_ranks = rankings.astype(np.float64).mean(axis=0)
    ordering = np.argsort(mean_ranks) + 1
    return ordering.astype(np.int64)


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
