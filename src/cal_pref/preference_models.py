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
    head = nn.Linear(in_dim, output_dim)

    return nn.Sequential(*layers), head


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
    """Probabilistic Preference Model using a multi-head MLP architecture.
    Each head corresponds to an item and outputs logits for that item being ranked at each position.
    The final output is a combination of all heads, representing the joint distribution over rankings.

    """

    def __init__(
        self,
        input_dim: int,
        n_items: int,
        hidden_dims: list[int],
        output_dim: int,
        unique_rankings: torch.Tensor,
    ):
        super(PreferenceModel, self).__init__()
        self.mlp, self.head = build_preference_mlp(input_dim, hidden_dims, output_dim)
        if math.factorial(n_items) != output_dim:
            self.unseen_weights = nn.Parameter(torch.zeros(1)).float()
        else:
            self.unseen_weights = None
        self.n_items = n_items
        self.rankings = unique_rankings
        self.idx_rankings = {
            tuple(r.tolist()): i for i, r in enumerate(unique_rankings)
        }

        # print("SELF N ITEMS: ", self.n_items)

    def forward(self, x):
        x = self.mlp(x)
        x = self.head(x)
        if self.unseen_weights is not None:
            #print("UNSEEN WEIGHTS: ", self.unseen_weights)
            x = torch.hstack(
                (
                    x,
                    torch.repeat_interleave(
                        self.unseen_weights.unsqueeze(0), x.shape[0], dim=0
                    ),
                )
            )
        # print(x)
        return x

    def predict_proba(self, x):
        logits = self.forward(x)
        probs = nn.functional.softmax(logits, dim=-1)
        return probs

    def predict_proba_ranking(self, x, rank):
        logits = self.forward(x)
        return self.predict_proba_ranking_logits(logits, rank)

    def predict_proba_ranking_logits(self, logits, rank):
        probs = nn.functional.softmax(logits, dim=-1)
        if len(rank.shape) == 1:
            rank = rank.expand(logits.shape[0], -1)
        idx_ranks = torch.tensor(
            [
                self.idx_rankings.get(tuple(r.tolist()), probs.shape[1] - 1)
                for r in rank
            ],
            device=logits.device,
        )
        rank_probs = torch.gather(probs, 1, idx_ranks.unsqueeze(1)).squeeze(1)
        return rank_probs

    def predict_ranking_distribution(self, x):
        logits = self.forward(x).clamp(min=1e-6)
        rankings = permutations(range(1, self.n_items + 1))
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

    def predict_ranking_distribution(self, x):
        logits = self.forward(x).clamp(min=1e-6)
        rankings = permutations(range(1, self.n_items + 1))
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
        logits = self.forward(x)
        # exp_logits = torch.exp(logits)
        ranking = torch.argsort(logits, dim=-1, descending=True) + 1
        return ranking

    def predict_proba(self, x):
        logits = self.forward(x)
        # exp_logits = torch.exp(logits)
        cumulative_sums = torch.cumsum(
            torch.sort(logits, dim=-1, descending=False), dim=-1
        )
        logits, _ = torch.sort(logits, dim=-1, descending=True)
        logits = logits / cumulative_sums
        probs = torch.prod(logits, dim=-1, keepdim=True)
        return probs

    def predict_proba_ranking_logits(self, logits, rank):
        # Restrict logits to cause no numerical issues
        # un_exp_logits = torch.exp(logits)  # Use sqrt to reduce range of logits
        # exp_logits = un_exp_logits
        # print("EXP LOGITS: ", exp_logits)
        if len(rank.shape) == 1:
            rank = rank.expand(logits.shape[0], -1)
        idx_rank_sort = torch.argsort(rank, dim=-1, descending=True)
        logits = torch.gather(
            logits, 1, idx_rank_sort
        )  # the exps that the highest rank (1) comes last
        cum_sums = torch.cumsum(logits, dim=-1)
        probs = (logits / cum_sums).prod(dim=-1)

        # print("PROBS: ", probs)
        if any(probs.isnan()):
            print("LOGITS: ", logits)
            print("RANK: ", rank)
            print("CUM SUMS: ", cum_sums)
            print("PROBS: ", probs)
            raise ValueError("NaN values in probabilities.")
        return probs

    def predict_ranking_distribution(self, x):
        logits = self.forward(x).clamp(min=1e-6)
        rankings = permutations(range(1, self.n_items + 1))
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
        logits = self.forward(x).clamp(min=1e-6)
        return self.predict_proba_ranking_logits(logits, rank)


class MallowsModel(nn.Module):
    def __init__(
        self,
        reference_ranking: torch.Tensor,
        dispersion: float,
        distance_metric: str = "kendall",
    ):
        self.reference_ranking = reference_ranking
        self.n_items = reference_ranking.shape[0]
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

    def predict_ranking_distribution(self, x):
        rankings = permutations(range(1, self.n_items + 1))
        distribution = {}
        for rank in rankings:
            rank_tensor = (
                torch.tensor(rank, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
            )
            probs = self.predict_proba_ranking(x, rank_tensor)
            distribution[rank] = probs.detach()
        return distribution
