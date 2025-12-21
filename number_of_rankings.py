"""This scripts calculate the number of rankings contained in the files, with respect to the total number of possible rankings."""

from math import factorial
import numpy as np
from cal_pref.utils import (
    load_lr_data,

)
if __name__ == "__main__":
    possible_datasets = [
        "authorship",
        "glass",
        "iris",
        "letter",
        "libras",
        "movies",
        "pendigits",
        "segment",
        "vehicle",
        "vowel",
        "wine",
        "yeast",
        "political",
    ]
    for dataset_name in possible_datasets:

        X, y = load_lr_data(dataset_name)
        n_items = y.shape[1]
        total_possible_rankings = factorial(n_items)
        observed_rankings = np.unique(y, axis=0).shape[0]
        print(
            f"Dataset: {dataset_name:12s} | N_items: {n_items:2d} | Observed rankings: {observed_rankings:5d} / {total_possible_rankings:5d} "
            f"({observed_rankings / total_possible_rankings:.4f})"
        )
