import torch
import torch.nn as nn
from math import factorial


class BrierLoss(nn.Module):
    def __init__(self):
        super(BrierLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_true_one_hot = torch.nn.functional.one_hot(
            y_true, num_classes=y_pred.size(1)
        ).float()
        loss = torch.mean(torch.sum((y_pred - y_true_one_hot) ** 2, dim=1))
        return loss


class PlackettLuceLoss(nn.Module):
    """Negative log-likelihood loss for the Plackett-Luce model"""

    def __init__(self):
        super(PlackettLuceLoss, self).__init__()

    def forward(self, y_true, logits, model):
        y_pred_probs_true_ranks = model.predict_proba_ranking_logits(logits, y_true)
        loss = -torch.log(
            y_pred_probs_true_ranks + 1e-10
        )  # Add a small constant to avoid log(0)
        return torch.mean(loss)


class PlackettLuceBrierPreferenceLoss(nn.Module):
    """Adapted BrierPreferenceLoss to Plackett-Luce model probability function"""

    def __init__(self) -> None:
        super(PlackettLuceBrierPreferenceLoss, self).__init__()

    def forward(self, y_true, y_pred, logits, model):
        mask = (y_true == y_pred).all(dim=-1)

        rank_probs = model.predict_proba_ranking_logits(logits, y_true)

        valid_predictions = rank_probs >= (1 - rank_probs) / (
            factorial(y_true.shape[1]) - 1
        )  # valid if the predicted rank is more probable than random guessing for the remaining ranks
        # print("SUMMED PROBS OF PRED RANKS: ", probs_of_pred_ranks)
        loss = (
            1
            + rank_probs**2
            + (1 - rank_probs) ** 2 / (factorial(y_true.shape[1]) - 1)
            - 2
            * torch.where(
                mask,
                rank_probs,
                (1 - rank_probs) / (factorial(y_true.shape[1]) - 1),
            )
        )
        loss = torch.where(
            valid_predictions,
            loss,
            (1 - 1 / (factorial(y_true.shape[1]) - 1))
            + loss,  # Penalize invalid predictions
        )

        # print("LOSS: ", loss.shape)
        return torch.mean(loss)


class BrierPreferenceLoss(nn.Module):
    """Based on the Brier score from "From Classification Accuracy to Proper Scoring Rules: Elicitability of Probabilistic Top List Predictions" """

    def __init__(self, maximal_t_list_size: int):
        """Brier Preference Loss for ranking tasks

        Args:
            maximal_t_list_size (int): The maximal number of ranks which can be modelled. This does not have to be m!. More generally, this is the number of different possible outcomes, given the underlying ranking model.
        """
        super(BrierPreferenceLoss, self).__init__()
        self.maximal_t_list_size = maximal_t_list_size

    def forward(self, y_true, y_pred, y_pred_probs, probs_of_ranks):

        # print("Y_TRUE: ", y_true)
        # print("Y_PRED: ", y_pred)
        # print("Y_PRED_PROBS: ", y_pred_probs)
        mask = (y_true == y_pred).all(dim=-1)
        # print("MASK: ", mask)
        # print("Y_PRED SHAPE: ", y_pred)
        rank_probs = probs_of_ranks(y_pred_probs, y_pred)

        valid_predictions = rank_probs >= (1 - rank_probs) / (
            self.maximal_t_list_size - 1
        )  # valid if the predicted rank is more probable than random guessing for the remaining ranks

        # print("SUMMED PROBS OF PRED RANKS: ", probs_of_pred_ranks)
        loss = (
            1
            + rank_probs**2
            + (1 - rank_probs) ** 2 / (self.maximal_t_list_size - 1)
            - 2
            * torch.where(
                mask,
                rank_probs,
                (1 - rank_probs) / (self.maximal_t_list_size - 1),
            )
        )
        loss = torch.where(
            valid_predictions,
            loss,
            (1 - 1 / (self.maximal_t_list_size - 1))
            + loss,  # Penalize invalid predictions
        )

        # print("LOSS: ", loss.shape)
        return torch.mean(loss)


class LogLossPreferenceLoss(nn.Module):
    """Log Loss for preference learning"""

    def __init__(self, maximal_t_list_size: int) -> None:
        super(LogLossPreferenceLoss, self).__init__()
        self.maximal_t_list_size = maximal_t_list_size

    def forward(self, y_true, y_pred, y_pred_probs, probs_of_ranks):
        mask = (y_true == y_pred).all(dim=-1)

        rank_probs = probs_of_ranks(y_pred_probs, y_pred)

        loss = -torch.log(
            torch.where(
                mask,
                -torch.log(rank_probs + 1e-10),
                torch.log(torch.tensor(self.maximal_t_list_size - 1))
                - torch.log(1 - rank_probs + 1e-10),
            )
        )  # Add a small constant to avoid log(0)

        return torch.mean(loss)
