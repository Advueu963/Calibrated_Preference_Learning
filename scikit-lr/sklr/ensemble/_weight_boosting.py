"""This module contains weight boosting estimators."""


# =============================================================================
# Imports
# =============================================================================

# Third party
from sklearn.ensemble._weight_boosting import BaseWeightBoosting
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples
import numpy as np

# Local application
from ..base import LabelRankerMixin, PartialLabelRankerMixin
from ._base import _predict_ensemble
from ..metrics import kendall_distance, penalized_kendall_distance
from ..tree import DecisionTreeLabelRanker, DecisionTreePartialLabelRanker


# =============================================================================
# Classes
# =============================================================================

class AdaBoostLabelRanker(LabelRankerMixin, BaseWeightBoosting):
    """An AdaBoost :term:`label ranker`."""

    def __init__(self,
                 base_estimator=None,
                 *,
                 n_estimators=50,
                 learning_rate=1.0,
                 random_state=None):
        """Constructor."""
        super(AdaBoostLabelRanker, self).__init__(base_estimator,
                                                  n_estimators=n_estimators,
                                                  learning_rate=learning_rate,
                                                  random_state=random_state)

    def fit(self, X, Y, sample_weight=None):
        """Build a boosted :term:`label ranker` from the training dataset."""
        X, self._Y = self._validate_data(X, Y, multi_output=True)

        # Fake the target values to apply inheritance
        y = Y[:, 0]

        return super(AdaBoostLabelRanker, self).fit(X, y, sample_weight)

    def _validate_estimator(self):
        """Check the estimator and set the corresponding attribute."""
        default = DecisionTreeLabelRanker(max_depth=7)

        return super(AdaBoostLabelRanker, self)._validate_estimator(default)

    def _boost(self, iboost, X, Y, sample_weight, random_state):
        """Implement a single boost for label ranking."""
        estimator = self._make_estimator(random_state=random_state)

        # Weighted sampling of the training set with replacement
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        bootstrap_idx = random_state.choice(indices,
                                            size=n_samples,
                                            replace=True,
                                            p=sample_weight)

        # Fit on the bootstrapped sample and obtain a
        # prediction for all samples in the training set
        _X = _safe_indexing(X, bootstrap_idx)
        _Y = _safe_indexing(self._Y, bootstrap_idx)
        estimator.fit(_X, _Y)
        Y_predict = estimator.predict(X)

        error_vect = kendall_distance(self._Y, Y_predict, normalize=True, return_dists=True)  # noqa
        sample_mask = sample_weight > 0
        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vector = error_vect[sample_mask]
        error_max = np.max(masked_error_vector)

        if error_max != 0:
            masked_error_vector /= error_max

        # Calculate the average loss taking into account the sample weight
        estimator_error = np.sum(masked_sample_weight * masked_error_vector)

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1.0, 0.0

        elif estimator_error >= 0.5:
            # Discard current estimator only if it is not the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)

            return None, None, None

        beta = estimator_error / (1.0 - estimator_error)

        # Boost weight using the corresponding boosting algorithm
        estimator_weight = self.learning_rate * np.log(1.0 / beta)

        if not iboost == self.n_estimators - 1:
            x2 = (1.0 - masked_error_vector) * self.learning_rate
            sample_weight[sample_mask] *= np.power(beta, x2)

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        """Predict the target rankings for the test data."""
        # Discard the non-fitted estimator weights
        limit = len(self.estimators_)
        sample_weight = self.estimator_weights_[:limit]

        return _predict_ensemble(self, X, sample_weight)


class AdaBoostPartialLabelRanker(PartialLabelRankerMixin, BaseWeightBoosting):  # noqa
    """An AdaBoost :term:`partial label ranker`."""

    def __init__(self,
                 base_estimator=None,
                 *,
                 n_estimators=50,
                 learning_rate=1.0,
                 random_state=None):
        """Constructor."""
        super(AdaBoostPartialLabelRanker, self).__init__(base_estimator,
                                                         n_estimators=n_estimators,  # noqa
                                                         learning_rate=learning_rate,  # noqa
                                                         random_state=random_state)  # noqa

    def fit(self, X, Y, sample_weight=None):
        """Build a boosted :term:`label ranker` from the training dataset."""
        X, self._Y = self._validate_data(X, Y, multi_output=True)

        # Fake the target values to apply inheritance
        y = self._Y[:, 0]

        return super(AdaBoostPartialLabelRanker, self).fit(X, y, sample_weight)

    def _validate_estimator(self):
        """Check the estimator and set the corresponding attribute."""
        default = DecisionTreePartialLabelRanker(max_depth=3)

        return super(AdaBoostPartialLabelRanker, self)._validate_estimator(default)  # noqa

    def _boost(self, iboost, X, Y, sample_weight, random_state):
        """Implement a single boost for label ranking."""
        estimator = self._make_estimator(random_state=random_state)

        # Weighted sampling of the training set with replacement
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        bootstrap_idx = random_state.choice(indices,
                                            size=n_samples,
                                            replace=True,
                                            p=sample_weight)

        # Fit on the bootstrapped sample and obtain a
        # prediction for all samples in the training set
        _X = _safe_indexing(X, bootstrap_idx)
        _Y = _safe_indexing(self._Y, bootstrap_idx)
        estimator.fit(_X, _Y)
        Y_predict = estimator.predict(X)

        error_vect = penalized_kendall_distance(self._Y, Y_predict, normalize=True, return_dists=True)  # noqa
        sample_mask = sample_weight > 0
        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vector = error_vect[sample_mask]
        error_max = np.max(masked_error_vector)

        if error_max != 0:
            masked_error_vector /= error_max

        # Calculate the average loss taking into account the sample weight
        estimator_error = np.sum(masked_sample_weight * masked_error_vector)

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1.0, 0.0

        elif estimator_error >= 0.5:
            # Discard current estimator only if it is not the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)

            return None, None, None

        beta = estimator_error / (1.0 - estimator_error)

        # Boost weight using the corresponding boosting algorithm
        estimator_weight = self.learning_rate * np.log(1.0 / beta)

        if not iboost == self.n_estimators - 1:
            x2 = (1.0 - masked_error_vector) * self.learning_rate
            sample_weight[sample_mask] *= np.power(beta, x2)

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        """Predict the target rankings for the test data."""
        # Discard the non-fitted estimator weights
        limit = len(self.estimators_)
        sample_weight = self.estimator_weights_[:limit]

        return _predict_ensemble(self, X, sample_weight)
