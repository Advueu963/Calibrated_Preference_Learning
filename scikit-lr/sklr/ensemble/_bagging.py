"""This module includes bagging ensemble methods."""


# =============================================================================
# Imports
# =============================================================================

# Third party
from sklearn.ensemble._bagging import BaseBagging

# Local application
from ._base import _predict_ensemble
from ..base import LabelRankerMixin, PartialLabelRankerMixin
from ..tree import DecisionTreeLabelRanker, DecisionTreePartialLabelRanker


# =============================================================================
# Classes
# =============================================================================

class BaggingLabelRanker(LabelRankerMixin, BaseBagging):
    """A Bagging :term:`label ranker`."""

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 *,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):
        """Constructor."""
        super(BaggingLabelRanker, self).__init__(base_estimator,
                                                 n_estimators,
                                                 max_samples=max_samples,
                                                 max_features=max_features,
                                                 bootstrap=bootstrap,
                                                 bootstrap_features=bootstrap_features,  # noqa
                                                 oob_score=oob_score,
                                                 warm_start=warm_start,
                                                 n_jobs=n_jobs,
                                                 random_state=random_state,
                                                 verbose=verbose)

    def _validate_estimator(self):
        """Check the estimator and set the corresponding attribute."""
        estimator = DecisionTreeLabelRanker(max_depth=None)

        super(BaggingLabelRanker, self)._validate_estimator(estimator)

    def _set_oob_score(self, X, Y):
        """Set the score of the out-of-bag training dataset."""
        raise NotImplementedError("Computing the score of the training "
                                  "dataset with an out-of-bag estimate "
                                  "is not available yet.")

    def _validate_y(self, Y):
        """Validate the target rankings."""
        return Y

    def predict(self, X):
        """Predict the target rankings for the test data."""
        return _predict_ensemble(self, X, sample_weight=None)


class BaggingPartialLabelRanker(PartialLabelRankerMixin, BaseBagging):
    """A Bagging :term:`partial label ranker`."""

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 *,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):
        """Constructor."""
        super(BaggingPartialLabelRanker, self).__init__(base_estimator,
                                                        n_estimators,
                                                        max_samples=max_samples,  # noqa
                                                        max_features=max_features,  # noqa
                                                        bootstrap=bootstrap,
                                                        bootstrap_features=bootstrap_features,  # noqa
                                                        oob_score=oob_score,
                                                        warm_start=warm_start,
                                                        n_jobs=n_jobs,
                                                        random_state=random_state,  # noqa
                                                        verbose=verbose)

    def _validate_estimator(self):
        """Check the estimator and set the corresponding attribute."""
        estimator = DecisionTreePartialLabelRanker(max_depth=None)

        super(BaggingPartialLabelRanker, self)._validate_estimator(estimator)

    def _set_oob_score(self, X, Y):
        """Set the score of the out-of-bag training dataset."""
        raise NotImplementedError("Computing the score of the training "
                                  "dataset with an out-of-bag estimate "
                                  "is not available yet.")

    def _validate_y(self, Y):
        """Validate the target rankings."""
        return Y

    def predict(self, X):
        """Predict the target rankings for the test data."""
        return _predict_ensemble(self, X, sample_weight=None)
