
# Calibrated Preference Learning

This repository contains the code for evaluating the calibration of Label Ranking models
    -   `label_ranking_calibration.py`: Main experiment file for label ranking calibration
    -   `rlhf_calibration.py`: Investigates the calibration of reward models
    -   `src/cal_pref`: Main source code implementing Label Ranking models, training procedures, losses and evaluations
    -   `scikit-lr`: Adapted code from [here](https://github.com/alfaro96/scikit-lr) to implement RPC model

## How to install
1. `uv sync`
2. done :D

## Ranking Representations (Important)

This repo uses two equivalent but different encodings of a ranking. We keep them
consistent by using each one in a specific place:

### 1) Ranks-per-item (datasets, training labels, `.predict()`)

- Shape: `(n_samples, n_items)`
- Meaning: `y[s, j]` is the rank (1 = best) of item `(j+1)` in sample `s`.
- Example (3 items): `y = [2, 3, 1]` means item 3 is best, then item 1, then item 2.

All datasets loaded by `cal_pref.utils.load_lr_data(...)` and
`cal_pref.utils.synthetic_data(...)` follow this convention.

All models’ `.predict(...)` return **ranks-per-item**.

### 2) Orderings / permutations (predicted distributions)

- Shape: tuples like `(2, 1, 3)`
- Meaning: best→worst ordering of **item IDs**.
- Example: `(2, 1, 3)` means `2 > 1 > 3`.

All predicted ranking distributions are represented as:

```python
{ ordering_tuple: probability_vector_per_sample }
```

where the dict keys are **orderings (best→worst)**. Keys may be full rankings
(length `n_items`) or partial orderings (e.g. `(2, 1)` meaning `2 > 1`).

In particular, all models’ `predict_ranking_distribution(...)` return a dict
keyed by **orderings**.

## Practical rule of thumb

- If it comes from the dataset or from `.predict()`: it is **ranks-per-item**.
- If it is a key in `y_pred_proba` / `distribution`: it is an **ordering**.

