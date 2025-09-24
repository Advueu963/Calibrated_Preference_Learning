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
        logits = self.forward(x)
        probs = torch.exp(logits) / torch.sum(
            torch.exp(logits + torch.arange(1, self.n_items).log().sum()),
            dim=-1,
            keepdim=True,
        )
        # print("PROBS: ", probs.shape)
        return probs

    def predict(self, x):
        prediction = torch.zeros((x.shape[0], self.n_items), dtype=torch.long)
        probs = self.predict_proba(x)

        for item in range(self.n_items):
            item_probs = probs[:, item * self.n_items : (item + 1) * self.n_items]
            max_idx = torch.argmax(item_probs, dim=-1)
            prediction[:, item] = max_idx + 1

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

    def get_rank_prob(self, x, rank):
        probs = self.predict_proba(x)
        if len(rank.shape) == 1:
            rank = rank.expand(x.shape[0], -1)

        gather_indices = torch.arange(self.n_items) * self.n_items + (rank - 1)
        probs_of_rank = torch.gather(probs, 1, gather_indices)
        return torch.sum(probs_of_rank, dim=-1)


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
        exp_logits = torch.exp(logits)
        if len(rank.shape) == 1:
            rank = rank.expand(x.shape[0], -1)
        prob = torch.ones(x.shape[0])
        for i in range(self.n_items):
            denom = torch.sum(exp_logits * (rank == i + 1), dim=-1)
            prob *= exp_logits[:, i] / denom
        return prob


class PlackettLuceLoss(nn.Module):
    def __init__(self):
        super(PlackettLuceLoss, self).__init__()

    def forward(self, y_true, logits, model):
        y_pred_probs_true_ranks = model.predict_proba_ranking_logits(logits, y_true)
        loss = -torch.log(
            y_pred_probs_true_ranks + 1e-10
        )  # Add a small constant to avoid log(0)
        return torch.mean(loss)
