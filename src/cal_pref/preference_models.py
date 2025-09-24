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
    multi_heads = [
        nn.Linear(in_dim, np.sqrt(output_dim).astype(int))
        for _ in range(np.sqrt(output_dim).astype(int))
    ]

    return nn.Sequential(*layers), multi_heads


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

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super(PreferenceModel, self).__init__()
        self.mlp, self.multi_heads = build_preference_mlp(
            input_dim, hidden_dims, output_dim
        )
        self.n_items = np.sqrt(output_dim).astype(int)
        # print("SELF N ITEMS: ", self.n_items)

    def forward(self, x):
        x = self.mlp(x)
        x = torch.stack([head(x) for head in self.multi_heads], dim=1)
        x = x.view(x.shape[0], -1)
        return x

    def predict_proba(self, x):
        """Predict the probability of each possible position of each item. The output is of shape (batch_size, n_items * n_items).
        The probability of a specific ranking can be obtained by gathering the appropriate indices from the output.
        The indices for gathering can be computed as: position * n_items + (rank - 1) for each item.
        """
        logits = self(x)
        logits = logits.view(logits.shape[0], self.n_items, self.n_items)
        probs = torch.exp(logits) / torch.sum(torch.exp(logits), dim=-1, keepdim=True)
        probs = probs.view(probs.shape[0], -1)
        # print("PROBS: ", probs.shape)

        return probs

    def predict_proba_label_ranking(self, x, rank):
        """Label Ranking adapted probability gathering method.
        """
        if len(rank.shape) == 1:
            rank = rank.expand(x.shape[0], -1)
        x = self.mlp(x)
        probs = []
        lambda_vector = torch.ones((x.shape[0], self.n_items), dtype=torch.float32)
        for i, head in enumerate(self.multi_heads):
            current_rank_of_item = rank[:, i] - 1  # ranks are 1-indexed
            item_head_winner = current_rank_of_item.unsqueeze(-1)
            logits_head = head(x)
            logits_head = logits_head  # Mask out already chosen items
            exp_logits = torch.exp(logits_head) * lambda_vector
            p = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
            lambda_vector[torch.arange(x.shape[0]), item_head_winner.squeeze()] = 0
            probs.append(p[:, item_head_winner])
        probs = torch.stack(probs, dim=1)  # (batch_size, n_items, n_items)
        probs = probs.view(probs.shape[0], -1)
        return probs.prod(dim=-1)

    def predict(self, x, method: str = "lr"):
        """Predicts the ranking using either label ranking (lr) or pairwise label ranking (plr) method.
        """
        if method == "lr":
            return self.predict_lr(x)
        elif method == "plr":
            return self.predict_plr(x)
        else:
            raise ValueError("Method must be 'lr' or 'plr'")

    @torch.no_grad()
    def predict_lr(self, x):
        prediction = torch.zeros((x.shape[0], self.n_items), dtype=torch.long)
        probs = self.predict_proba(x)
        position_probs_to_mask = torch.ones(
            (x.shape[0], self.n_items), dtype=torch.long
        )

        for position in range(self.n_items):
            position_probs = probs[
                :, position * self.n_items : (position + 1) * self.n_items
            ]  # get the probability of the current position for all items
            remain_probs_after_mask = 1 - (position_probs * position_probs_to_mask).sum(
                dim=-1, keepdim=True
            )  # remaining probability mass after masking already chosen items
            position_probs = (
                position_probs + remain_probs_after_mask / (self.n_items - position)
            ) * position_probs_to_mask  # redistribute remaining probability mass to unchosen items. Masking already chosen items
            item = torch.argmax(position_probs, dim=-1)
            prediction[:, item] = position + 1
            position_probs_to_mask[:, item] = 0

        # Make sure it is label ranking
        for i in range(prediction.shape[0]):
            # print("PREDICTION: ", prediction[i])
            unique, counts = torch.unique(prediction[i], return_counts=True)
            duplicates = unique[counts > 1]
            # print("DUPLICATES: ", duplicates)
            for dup in duplicates:
                dup_indices = (prediction[i] == dup).nonzero(as_tuple=True)[0]
                probs_of_dup = torch.tensor(
                    [
                        probs[i, idx * self.n_items : (idx + 1) * self.n_items][dup - 1]
                        for idx in dup_indices
                    ]
                )
                sorted_indices = torch.argsort(probs_of_dup, descending=True)
                for offset, idx in enumerate(dup_indices[sorted_indices]):
                    prediction[i, idx] = dup + offset

        # Ensure ranks are increasing +1 each step
        for i in range(prediction.shape[0]):
            sorted_indices = torch.argsort(prediction[i])
            prediction[i, sorted_indices] = torch.arange(
                1, self.n_items + 1, dtype=prediction.dtype
            )

        return prediction

    @torch.no_grad()
    def predict_plr(self, x):
        raise NotImplementedError("Pairwise label ranking not implemented yet.")
        prediction = torch.zeros((x.shape[0], self.n_items), dtype=torch.long)
        probs = self.predict_proba(x)

        for item in range(self.n_items):
            item_probs = probs[:, item * self.n_items : (item + 1) * self.n_items]
            max_idx = torch.argmax(item_probs, dim=-1)
            prediction[:, item] = max_idx + 1

        # Ensure ranks are increasing +1 each step
        for i in range(prediction.shape[0]):
            sorted_indices = torch.argsort(prediction[i])
            prediction[i, sorted_indices] = torch.arange(
                1, self.n_items + 1, dtype=prediction.dtype
            )

        return prediction

    def get_rank_prob(self, x, rank):
        """Generic method to get the probability of a specific ranking from the underlying model."""
        probs = self.predict_proba(x)
        if len(rank.shape) == 1:
            rank = rank.expand(x.shape[0], -1)

        gather_indices = torch.arange(self.n_items) * self.n_items + (rank - 1)
        probs_of_rank = torch.gather(probs, 1, gather_indices)
        return torch.prod(probs_of_rank, dim=-1)


class PlackettLuceModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super(PlackettLuceModel, self).__init__()
        self.mlp = build_plackett_luce_mlp(input_dim, hidden_dims, output_dim)
        self.n_items = output_dim
        # print("SELF N ITEMS: ", self.n_items)

    def forward(self, x):
        x = self.mlp(x)
        return x

    def predict(self, x):
        logits = self.forward(x)
        exp_logits = torch.exp(logits)
        ranking = torch.argsort(exp_logits, dim=-1, descending=True) + 1
        return ranking

    def predict_proba(self, x):
        logits = self.forward(x)
        exp_logits = torch.exp(logits)
        cumulative_sums = torch.cumsum(
            torch.sort(exp_logits, dim=-1, descending=False), dim=-1
        )
        exp_logits, _ = torch.sort(exp_logits, dim=-1, descending=True)
        exp_logits = exp_logits / cumulative_sums
        probs = torch.prod(exp_logits, dim=-1, keepdim=True)
        return probs

    def predict_proba_ranking_logits(self, logits, rank):
        exp_logits = torch.exp(logits)
        if len(rank.shape) == 1:
            rank = rank.expand(logits.shape[0], -1)
        idx_rank_sort = torch.argsort(rank, dim=-1, descending=True)
        exp_logits = torch.gather(
            exp_logits, 1, idx_rank_sort
        )  # the exps that the highest rank (1) comes last
        cum_sums = torch.cumsum(exp_logits, dim=-1)
        probs = (exp_logits / cum_sums).prod(dim=-1)
        return probs

    def predict_proba_ranking(self, x, rank):
        logits = self.forward(x)
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

    def predict_proba_ranking(self, _, ranking: torch.Tensor):
        if len(ranking.shape) == 1:
            ranking = ranking.expand(1, -1)
        distance_to_reference = torch.tensor(
            [self.compute_distance(r) for r in ranking]
        )
        probs = self.normalization_constant * (self.dispersion**distance_to_reference)
        return probs
