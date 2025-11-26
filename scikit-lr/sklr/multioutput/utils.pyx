"""This module includes various utilities."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np


# =============================================================================
# Functions
# =============================================================================

def borda(Y, total, binary, count):
    """Apply the Borda count voting system for partial rankings."""
    cdef int n_samples = Y.shape[0]
    cdef int n_classes = Y.shape[1]

    cdef float points

    cdef int labels
    cdef int counter
    cdef int position
    cdef int start
    cdef int end

    cdef int sample

    for sample in range(n_samples):
        counter = 0
        position = 0

        while counter < n_classes:
            # Get the number of labels at the current position
            labels = binary[sample, position]

            # Share the points among the labels at the current position
            start = counter
            end = counter + labels
            points = total[start:end].sum(0) / labels

            # Assign the same points to the labels at the current position
            mask = [Y[sample] == position + 1]
            mask = mask[0]
            indexes = np.where(mask)
            count[indexes] += points

            counter += labels
            position += 1


def aggregate(Y, precedences_matrices, rank_algorithm):
    """Compute the consensus ranking for each sample."""
    cdef int n_samples = Y.shape[0]
    cdef int n_classes = Y.shape[1]

    cdef long[:] consensus
    cdef double[:] count = None
    cdef double[:, :, :] precedences_matrix

    cdef int sample

    for sample in range(n_samples):
        consensus = Y[sample]
        precedences_matrix = precedences_matrices[sample]
        rank_algorithm._aggregate_params(consensus, count, precedences_matrix)
