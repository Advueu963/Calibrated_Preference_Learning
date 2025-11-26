"""The mod:`sklr.ensemble` module includes ensemble-based methods."""


# =============================================================================
# Imports
# =============================================================================

# Local application
from ._bagging import BaggingLabelRanker, BaggingPartialLabelRanker
from ._forest import RandomForestLabelRanker, RandomForestPartialLabelRanker
from ._weight_boosting import AdaBoostLabelRanker, AdaBoostPartialLabelRanker


# =============================================================================
# Module public objects
# =============================================================================

__all__ = ["AdaBoostLabelRanker", "AdaBoostPartialLabelRanker",
           "BaggingLabelRanker", "BaggingPartialLabelRanker",
           "RandomForestLabelRanker", "RandomForestPartialLabelRanker"]
