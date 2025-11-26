"""The :mod:`sklr.multioutput` module implements partial label rankers."""


# =============================================================================
# Imports
# =============================================================================

# Local application
from ._classes import PartialLabelRankerChain, BivariatePartialLabelRanker


# =============================================================================
# Module public objects
# =============================================================================

__all__ = ["PartialLabelRankerChain", "BivariatePartialLabelRanker"]
