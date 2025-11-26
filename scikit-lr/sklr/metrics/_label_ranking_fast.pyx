# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Local application
from .._types cimport BOOL_t, DTYPE_t_1D, INT64_t, INT64_t_2D, SIZE_t
from .._types cimport RANK_TYPE
from libc.stdio cimport printf


# =============================================================================
# Methods
# =============================================================================

cpdef void kendall_distance_fast(INT64_t_2D Y_true,
                                 INT64_t_2D Y_pred,
                                 BOOL_t normalize,
                                 DTYPE_t_1D dists) nogil:
    """Compute the Kendall distance."""
    cdef INT64_t n_samples = Y_true.shape[0]
    cdef INT64_t n_classes = Y_true.shape[1]

    cdef SIZE_t sample
    cdef SIZE_t f_class
    cdef SIZE_t s_class
    
    cdef INT64_t n_ranked_classes

    for sample in range(n_samples):
        n_ranked_classes = 0
        for f_class in range(n_classes):
            # Skip non-ranked classes
            if (Y_true[sample, f_class] == RANK_TYPE.RANDOM or
                    Y_pred[sample, f_class] == RANK_TYPE.RANDOM):
                continue
            else:
                n_ranked_classes += 1
            for s_class in range(f_class + 1, n_classes):
                # There exist a disagreement among the rankings
                # if the compared classes are in opposite order
                if (Y_true[sample, f_class] < Y_true[sample, s_class] and
                        Y_pred[sample, f_class] > Y_pred[sample, s_class] or
                        Y_true[sample, f_class] > Y_true[sample, s_class] and
                        Y_pred[sample, f_class] < Y_pred[sample, s_class]):
                    dists[sample] += 1

        if normalize:
            dists[sample] /= n_ranked_classes * (n_classes-1) / 2


cpdef void tau_score_fast(INT64_t_2D Y_true,
                          INT64_t_2D Y_pred,
                          DTYPE_t_1D scores) nogil:
    """Compute the Kendall tau."""
    cdef INT64_t n_samples = Y_true.shape[0]
    cdef INT64_t n_classes = Y_true.shape[1]

    cdef SIZE_t sample
    cdef SIZE_t f_class
    cdef SIZE_t s_class

    cdef INT64_t n_ranked_classes

    for sample in range(n_samples):
        n_ranked_classes = 0
        for f_class in range(n_classes):
            # Skip non-ranked classes
            if (Y_true[sample, f_class] == RANK_TYPE.RANDOM or
                    Y_pred[sample, f_class] == RANK_TYPE.RANDOM):
                continue
            else:
                n_ranked_classes += 1
            for s_class in range(f_class + 1, n_classes):
                # There exist an agreement among the rankings
                # if the compared classes are in the same order
                if (Y_true[sample, f_class] < Y_true[sample, s_class] and
                        Y_pred[sample, f_class] < Y_pred[sample, s_class] or
                        Y_true[sample, f_class] > Y_true[sample, s_class] and
                        Y_pred[sample, f_class] > Y_pred[sample, s_class]):
                    scores[sample] += 1.0
                else:
                    scores[sample] -= 1.0
        scores[sample] /= n_ranked_classes * (n_classes-1) / 2
