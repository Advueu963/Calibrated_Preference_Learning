"""This module includes forest of trees-based ensemble methods."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import ABCMeta

# Third party
from sklearn.ensemble._forest import BaseForest

# Local application
from ._base import _predict_ensemble
from ..base import LabelRankerMixin, PartialLabelRankerMixin
from ..tree import DecisionTreeLabelRanker, DecisionTreePartialLabelRanker


# =============================================================================
# Classes
# =============================================================================

class ForestLabelRanker(LabelRankerMixin, BaseForest, metaclass=ABCMeta):
    """Base class for forest of trees-based label rankers.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self,
                 base_estimator,
                 n_estimators=100,
                 *,
                 estimator_params=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 max_samples=None):
        """Constructor."""
        super(ForestLabelRanker, self).__init__(base_estimator,
                                                n_estimators,
                                                estimator_params=estimator_params,  # noqa
                                                bootstrap=bootstrap,
                                                oob_score=oob_score,
                                                n_jobs=n_jobs,
                                                random_state=random_state,
                                                verbose=verbose,
                                                warm_start=warm_start,
                                                max_samples=max_samples)

    def _set_oob_score(self, X, Y):
        """Set the score of the out-of-bag training dataset."""
        raise NotImplementedError("Computing the score of the training "
                                  "dataset with an out-of-bag estimate "
                                  "is not available yet.")

    def predict(self, X):
        """Predict the target rankings for the test data."""
        return _predict_ensemble(self, X, sample_weight=None)


class ForestPartialLabelRanker(PartialLabelRankerMixin, BaseForest, metaclass=ABCMeta):  # noqa
    """Base class for forest of trees-based partial label rankers.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self,
                 base_estimator,
                 n_estimators=100,
                 *,
                 estimator_params=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 max_samples=None):
        """Constructor."""
        super(ForestPartialLabelRanker, self).__init__(base_estimator,
                                                       n_estimators,
                                                       estimator_params=estimator_params,  # noqa
                                                       bootstrap=bootstrap,
                                                       oob_score=oob_score,
                                                       n_jobs=n_jobs,
                                                       random_state=random_state,  # noqa
                                                       verbose=verbose,
                                                       warm_start=warm_start,
                                                       max_samples=max_samples)

    def _set_oob_score(self, X, Y):
        """Set the score of the out-of-bag training dataset."""
        raise NotImplementedError("Computing the score of the training "
                                  "dataset with an out-of-bag estimate "
                                  "is not available yet.")

    def predict(self, X):
        """Predict the target rankings for the test data."""
        return _predict_ensemble(self, X, sample_weight=None)


class RandomForestLabelRanker(ForestLabelRanker):
    """A Random Forest :term:`label ranker`."""

    def __init__(self,
                 n_estimators=100,
                 *,
                 criterion="mallows",
                 distance="kendall",
                 splitter="binary",
                 max_depth=None,
                 min_samples_split=2,
                 # min_samples_leaf=1,
                 # min_weight_fraction_leaf=0.0,
                 max_features="auto",
                 max_splits=2,
                 # max_leaf_nodes=None,
                 # min_impurity_decrease=0.0,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 # ccp_alpha=0.0,
                 max_samples=None):
        """Constructor."""
        # Get the estimator parameters from the corresponding object
        base_estimator = DecisionTreeLabelRanker(max_depth=None)
        estimator_params = base_estimator.get_params(deep=False)

        super(RandomForestLabelRanker, self).__init__(base_estimator,
                                                      n_estimators,
                                                      estimator_params=estimator_params,  # noqa
                                                      bootstrap=bootstrap,
                                                      oob_score=oob_score,
                                                      n_jobs=n_jobs,
                                                      random_state=random_state,  # noqa
                                                      verbose=verbose,
                                                      warm_start=warm_start,
                                                      max_samples=max_samples)

        self.criterion = criterion
        self.distance = distance
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # self.min_samples_leaf = min_samples_leaf
        # self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_splits = max_splits
        # self.max_leaf_nodes = max_leaf_nodes
        # self.min_impurity_decrease = min_impurity_decrease
        # self.ccp_alpha = ccp_alpha


class RandomForestPartialLabelRanker(ForestPartialLabelRanker):
    """A Random Forest :term:`partial label ranker`."""

    def __init__(self,
                 n_estimators=100,
                 *,
                 criterion="entropy",
                 splitter="binary",
                 max_depth=None,
                 min_samples_split=2,
                 # min_samples_leaf=1,
                 # min_weight_fraction_leaf=0.0,
                 max_features="auto",
                 max_splits=2,
                 # max_leaf_nodes=None,
                 # min_impurity_decrease=0.0,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 # ccp_alpha=0.0,
                 max_samples=None):
        """Constructor."""
        # Get the estimator parameters from the corresponding object
        base_estimator = DecisionTreePartialLabelRanker(max_depth=None)
        estimator_params = base_estimator.get_params(deep=False)

        super(RandomForestPartialLabelRanker, self).__init__(base_estimator,
                                                             n_estimators,
                                                             estimator_params=estimator_params,  # noqa
                                                             bootstrap=bootstrap,  # noqa
                                                             oob_score=oob_score,  # noqa
                                                             n_jobs=n_jobs,
                                                             random_state=random_state,  # noqa
                                                             verbose=verbose,
                                                             warm_start=warm_start,  # noqa
                                                             max_samples=max_samples)  # noqa

        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # self.min_samples_leaf = min_samples_leaf
        # self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_splits = max_splits
        # self.max_leaf_nodes = max_leaf_nodes
        # self.min_impurity_decrease = min_impurity_decrease
        # self.ccp_alpha = ccp_alpha
