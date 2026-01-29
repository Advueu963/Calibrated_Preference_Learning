
# Calibrated Preference Learning

This repository contains the code for evaluating the calibration of Label Ranking models. The project focuses on understanding and improving how well ranking models express uncertainty in their predictions, which is crucial for reliable preference learning and reward modeling.

## Overview

The codebase is organized into several key components:
- **Experiment Scripts**: Entry points for running calibration experiments
- **Core Library** (`src/cal_pref`): Reusable components for preference learning
- **External Dependencies** (`scikit-lr`): Adapted label ranking implementations
- **Data & Notebooks**: Experimental data and analysis notebooks

## Installation

### Prerequisites
- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager

## Repository Structure

### Root-Level Experiment Scripts

- **`label_ranking_calibration.py`**: Main experiment file for evaluating calibration metrics on label ranking models. Runs experiments on various datasets and models to assess how well predicted ranking distributions match true ranking frequencies.

- **`rlhf_calibration.py`**: Investigates the calibration properties of reward models trained with Reinforcement Learning from Human Feedback (RLHF). Analyzes how well these models express uncertainty in preference predictions.

- **`boxplot_visualise_ece.py`**: Visualization script for generating boxplots of Expected Calibration Error (ECE) and related metrics across different models and datasets.

- **`current-rbv2-data.csv`**: Cached experimental results or benchmark data.

- **`pyproject.toml`**: Project configuration and dependency specifications for the main package.

### `src/cal_pref/` - Core Library

The main source code implementing preference learning models, training procedures, and evaluation metrics:

- **`__init__.py`**: Package initialization and exports
- **`preference_models.py`**: Implementation of various label ranking models (e.g., neural ranking models, probabilistic models)
- **`preference_losses.py`**: Loss functions for training preference models (e.g., ranking losses, calibration-aware losses)
- **`train.py`**: Training loops and optimization procedures for preference models
- **`evaluate.py`**: Evaluation metrics for ranking quality and calibration (ECE, ranking accuracy, etc.)
- **`utils.py`**: Utility functions for data loading, preprocessing, and ranking conversions
- **`data/`**: Dataset loaders and preprocessing utilities

### `scikit-lr/` - Label Ranking Library

Adapted implementation from [scikit-lr](https://github.com/alfaro96/scikit-lr) providing baseline label ranking algorithms:

- **Core modules** (`sklr/`):
  - `pairwise.py`: Pairwise comparison-based ranking methods
  - `baseline.py`: Simple baseline models
  - `ensemble/`: Ensemble methods for label ranking
  - `metrics/`: Ranking evaluation metrics
  - `neighbors/`: k-NN based ranking methods
  - `tree/`: Decision tree-based ranking models

- **Build & Documentation**:
  - `setup.py`, `pyproject.toml`: Package configuration
  - `docs/`: Documentation source files
  - `docker/`: Containerization setup

This library is used particularly for the RPC (Ranking by Pairwise Comparison) model baseline.

### `Calibratoin_Label_Ranking/` - Analysis Notebooks

- **`conformel_classification.ipynb`**: Jupyter notebook exploring conformal prediction methods for ranking calibration and uncertainty quantification.

### `tests/` - Test Suite

- **`test_utils.py`**: Unit tests for utility functions
- Additional test files for models and evaluation metrics

### `cache_reward_bench/` - Cached Datasets

- Cached versions of the RewardBench dataset for faster experiment iterations
- Includes dataset snapshots and metadata

## Ranking Representations

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

