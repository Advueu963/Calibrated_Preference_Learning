class PlackettLuceBrierPreferenceLoss(nn.Module):
    def __init__(self) -> None:
        super(PlackettLuceBrierPreferenceLoss, self).__init__()

    def forward(self, y_true, y_pred, logits, model):
        mask = (y_true == y_pred).all(dim=-1)

        rank_probs = model.predict_proba_ranking_logits(logits, y_true)

        valid_predictions = rank_probs >= (1 - rank_probs) / (
            factorial(y_true.shape[1]) - 1
        )  # valid if the predicted rank is more probable than random guessing for the remaining ranks

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
    def __init__(self):
        super(BrierPreferenceLoss, self).__init__()

    def forward(self, y_true, y_pred, y_pred_probs):

        # print("Y_TRUE: ", y_true)
        # print("Y_PRED: ", y_pred)
        # print("Y_PRED_PROBS: ", y_pred_probs)
        mask = (y_true == y_pred).all(dim=-1)
        # print("MASK: ", mask)
        # print("Y_PRED SHAPE: ", y_pred)
        gather_indices = torch.arange(y_pred.shape[1]) * y_pred.shape[1] + (
            y_pred - 1
        )  # The first term is the offset for each item as the first 0,...,(n_items-1) entries correspond to item 1, the next n_items entries to item 2, etc.
        # print("GATHER INDICES: ", gather_indices)
        probs_of_pred_ranks = torch.gather(y_pred_probs, 1, gather_indices)
        # print("PROBS OF PRED RANKS: ", probs_of_pred_ranks)
        rank_probs = torch.sum(probs_of_pred_ranks, dim=-1)

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
            (10 - 1 / (factorial(y_true.shape[1]) - 1))
            * loss,  # Penalize invalid predictions
        )

        # print("LOSS: ", loss.shape)
        return torch.mean(loss)
