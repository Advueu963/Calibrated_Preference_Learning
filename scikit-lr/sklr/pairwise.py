"""The :mod:`sklr.pairwise` implements pairwise algorithms."""

# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import ABCMeta, abstractmethod

# Third party
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.multioutput import _fit_estimator
from sklearn.utils.parallel import delayed, Parallel
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils import check_X_y, check_array
import numpy as np
import torch

# Local application
from .base import PartialLabelRankerMixin, LabelRankerMixin


# =============================================================================
# Module public objects
# =============================================================================

__all__ = ["PairwisePartialLabelRanker", "PairwiseLabelRanker"]


# =============================================================================
# Functions
# =============================================================================


def _generate_y(X, Y, sample_weight):
    """Generate training target classes."""
    n_samples, n_classes = Y.shape

    for f_class in range(n_classes - 1):
        for s_class in range(f_class + 1, n_classes):
            y = np.where(
                (Y[:, f_class] == -1) | (Y[:, s_class] == -1),
                "missing",  # noqa
                np.where(
                    Y[:, f_class] < Y[:, s_class],
                    "precedes",  # noqa
                    np.where(Y[:, f_class] > Y[:, s_class], "succeeds", "tied"),
                ),
            )  # noqa

            # Drop the missing values from the training target classes
            mask = y != "missing"
            X_new = X[mask]
            y_new = y[mask]
            sample_weight_new = sample_weight[mask]

            # Duplicate the tied instances with precedes and succeeds but half of weight
            """mask = y_new != "tied"

            X_new_2 = np.concatenate((
                X_new[mask],
                X_new[~mask],
                X_new[~mask]
            ), axis=0)

            y_new_2 = np.concatenate((
                y_new[mask],
                np.full(X_new[~mask].shape[0], "precedes"),
                np.full(X_new[~mask].shape[0], "succeeds")
            ), axis=None)

            sample_weight = np.concatenate((
                np.ones(X_new[mask].shape[0]),
                np.full(X_new[~mask].shape[0], 0.5),
                np.full(X_new[~mask].shape[0], 0.5)
            ), axis=None)"""

            yield X_new, y_new, sample_weight_new


# =============================================================================
# Classes
# =============================================================================


class BasePairwise(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, estimator, *, n_jobs=None):
        """Constructor."""
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X, Y, sample_weight=None, **fit_params):
        """Fit the model on the training data and rankings."""
        X, Y = check_X_y(X, Y, multi_output=True)
        sample_weight = _check_sample_weight(sample_weight, X)

        self.n_classes_in_ = Y.shape[1]

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(self.estimator, _X, y, _sample_weight, **fit_params)
            for _X, y, _sample_weight in _generate_y(X, Y, sample_weight)
        )

        return self


class PairwiseLabelRanker(LabelRankerMixin, BasePairwise):

    def __init__(self, estimator, *, n_jobs=None):
        """Constructor."""
        super(PairwiseLabelRanker, self).__init__(estimator, n_jobs=n_jobs)

    def get_pairwise_matrix(self, X):
        """Get the pairwise preference matrix for the test data."""
        check_is_fitted(self)

        X = check_array(X)
        n_samples, _ = X.shape
        pair_order_matrices = np.zeros(
            (n_samples, self.n_classes_in_, self.n_classes_in_)
        )

        index = 0

        for f_class in range(self.n_classes_in_ - 1):
            for s_class in range(f_class + 1, self.n_classes_in_):
                proba = self.estimators_[index].predict_proba(X)

                classes = self.estimators_[index].classes_
                classes = {key: value for value, key in enumerate(classes)}

                if "precedes" in classes:
                    pair_order_matrices[:, f_class, s_class] = proba[
                        :, classes["precedes"]
                    ]
                if "succeeds" in classes:
                    pair_order_matrices[:, s_class, f_class] = proba[
                        :, classes["succeeds"]
                    ]

                index += 1

        return pair_order_matrices

    def predict(self, X):
        """Predict the target rankings for the test data."""
        check_is_fitted(self)

        X = check_array(X)
        n_samples, _ = X.shape
        # Y = np.zeros((n_samples, self.n_classes_in_ - 1), dtype=np.int64)
        Y = torch.zeros((n_samples, self.n_classes_in_), dtype=torch.int64)
        # pair_order_matrices = np.zeros((n_samples, self.n_classes_in_, self.n_classes_in_))
        precedences_matrices = self.get_pairwise_matrix(X)

        # Aggregate pairwise preferences through Borda count rule
        # s_i = sum_{j < i} p(i > j) + sum_{j > i}  1 - p(i > j)
        borda_scores = np.sum(precedences_matrices, axis=2) # (n_samples, n_classes)
        rankings = np.argsort(-borda_scores, axis=1) # (n_samples, n_classes)
        Y.scatter_(1, torch.tensor(rankings), torch.arange(1, self.n_classes_in_ + 1).unsqueeze(0).repeat(n_samples, 1))
        # print("Y after scatter
        # for sample in range(n_samples):
        #     borda_scores = np.sum(precedences_matrices[sample, :, :], axis=1)
        #     ranking = np.argsort(-borda_scores)
        #     for rank, class_index in enumerate(ranking):
        #         Y[sample, class_index] = rank + 1
        return Y

class PairwisePartialLabelRanker(PartialLabelRankerMixin, BasePairwise):

    def __init__(self, estimator, *, n_jobs=None):
        """Constructor."""
        super(PairwisePartialLabelRanker, self).__init__(estimator, n_jobs=n_jobs)

    def predict(self, X):
        """Predict the target rankings for the test data."""
        check_is_fitted(self)

        X = check_array(X)
        n_samples, _ = X.shape
        # Y = np.zeros((n_samples, self.n_classes_in_ - 1), dtype=np.int64)
        Y = np.zeros((n_samples, self.n_classes_in_), dtype=np.int64)
        # pair_order_matrices = np.zeros((n_samples, self.n_classes_in_, self.n_classes_in_))
        precedences_matrices = np.zeros(
            (n_samples, self.n_classes_in_, self.n_classes_in_, 2)
        )

        index = 0

        for f_class in range(self.n_classes_in_ - 1):
            for s_class in range(f_class + 1, self.n_classes_in_):
                proba = self.estimators_[index].predict_proba(X)

                classes = self.estimators_[index].classes_
                classes = {key: value for value, key in enumerate(classes)}

                if "precedes" in classes:
                    # pair_order_matrices[:, f_class, s_class] += proba[:, classes["precedes"]]
                    precedences_matrices[:, f_class, s_class, 0] = proba[
                        :, classes["precedes"]
                    ]
                if "tied" in classes:
                    # pair_order_matrices[:, f_class, s_class] += 0.5 * proba[:, classes["tied"]]
                    # pair_order_matrices[:, s_class, f_class] += 0.5 * proba[:, classes["tied"]]
                    precedences_matrices[:, f_class, s_class, 1] = (
                        0.5 * proba[:, classes["tied"]]
                    )
                    precedences_matrices[:, s_class, f_class, 1] = (
                        0.5 * proba[:, classes["tied"]]
                    )
                if "succeeds" in classes:
                    # pair_order_matrices[:, s_class, f_class] += proba[:, classes["succeeds"]]
                    precedences_matrices[:, s_class, f_class, 0] = proba[
                        :, classes["succeeds"]
                    ]

                index += 1

        self._rank_algorithm.init(self.n_classes_in_)

        for sample in range(n_samples):
            self._rank_algorithm._aggregate_params(
                Y[sample], None, precedences_matrices[sample]
            )

        return Y
