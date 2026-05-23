"""
sensitivity.py
==============
Sensitivity analysis for AHP-MAUT models.

Implements:
    - One-at-a-Time (OAT) sensitivity with proportional redistribution.
    - Monte Carlo simulation with uniform perturbations on criterion weights.
    - Stability metrics (ranking preservation, confidence intervals).

References
----------
Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., Gatelli, D.,
    Saisana, M., & Tarantola, S. (2008). Global Sensitivity Analysis: The Primer.
    Wiley.
Wallenius, J., Dyer, J. S., Fishburn, P. C., Steuer, R. E., Zionts, S., &
    Deb, K. (2008). Multiple criteria decision making, multiattribute utility
    theory: Recent accomplishments and what lies ahead. Management Science,
    54(7), 1336-1349.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .maut import global_utility


def redistribute_weights(weights: np.ndarray, idx: int,
                         new_value: float) -> np.ndarray:
    """
    Replace weights[idx] with new_value, then proportionally redistribute
    the remaining mass across the other indices so that the total sums to 1.

    Parameters
    ----------
    weights : np.ndarray
        Original weights summing to 1.
    idx : int
        Index to modify.
    new_value : float
        New value for weights[idx], in [0, 1].

    Returns
    -------
    new_weights : np.ndarray
        Perturbed weights summing to 1.
    """
    weights = np.asarray(weights, dtype=float)
    new_weights = weights.copy()
    new_weights[idx] = new_value
    remaining_mass = 1.0 - new_value
    others = np.arange(len(weights)) != idx
    original_sum_others = weights[others].sum()
    if original_sum_others > 0:
        new_weights[others] = weights[others] * (remaining_mass / original_sum_others)
    return new_weights


def oat_sensitivity(main_weights: np.ndarray,
                    local_weights_per_criterion: list[np.ndarray],
                    utility_matrix: np.ndarray,
                    perturbations: list[float] | None = None) -> pd.DataFrame:
    """
    One-at-a-time sensitivity analysis on main criterion weights.

    Each main criterion weight is perturbed by the given percentages, and
    the remaining main weights are proportionally rescaled to maintain
    the unit sum. Final sub-criterion weights are recomputed and U is
    re-evaluated for each scenario.

    Parameters
    ----------
    main_weights : np.ndarray
        Main criterion weights (shape: n_main).
    local_weights_per_criterion : list[np.ndarray]
        Local sub-criterion weights for each main criterion.
    utility_matrix : np.ndarray
        Utility values (shape: n_subcriteria x n_scenarios).
    perturbations : list[float], optional
        Percentage perturbations as fractions. Default: [-0.20, -0.10, 0,
        +0.10, +0.20].

    Returns
    -------
    results : pd.DataFrame
        Long-format DataFrame with columns: criterion_idx, perturbation,
        new_weights, U_A, U_B, U_C, ...
    """
    if perturbations is None:
        perturbations = [-0.20, -0.10, 0.0, 0.10, 0.20]

    n_main = len(main_weights)
    n_scenarios = utility_matrix.shape[1]
    scenario_labels = [f"U_{chr(65 + i)}" for i in range(n_scenarios)]

    rows = []
    for crit_idx in range(n_main):
        for pct in perturbations:
            base = main_weights[crit_idx]
            new_value = base * (1 + pct)
            new_value = np.clip(new_value, 0.001, 0.999)
            new_main = redistribute_weights(main_weights, crit_idx, new_value)
            final_weights = np.concatenate([
                new_main[i] * local_weights_per_criterion[i]
                for i in range(n_main)
            ])
            utilities = [global_utility(final_weights, utility_matrix[:, s])
                         for s in range(n_scenarios)]
            row = {
                "criterion_idx": crit_idx,
                "perturbation": pct,
                "new_main_weights": new_main.tolist(),
            }
            for label, u in zip(scenario_labels, utilities):
                row[label] = u
            rows.append(row)
    return pd.DataFrame(rows)


def monte_carlo_simulation(main_weights: np.ndarray,
                           local_weights_per_criterion: list[np.ndarray],
                           utility_matrix: np.ndarray,
                           n_iterations: int = 10000,
                           perturbation_range: float = 0.20,
                           random_seed: int = 42) -> pd.DataFrame:
    """
    Monte Carlo sensitivity analysis: sample main criterion weights from
    a uniform distribution around the base values, renormalize, and
    recompute global utilities for each scenario.

    Parameters
    ----------
    main_weights : np.ndarray
        Base main criterion weights (shape: n_main).
    local_weights_per_criterion : list[np.ndarray]
        Local sub-criterion weights for each main criterion.
    utility_matrix : np.ndarray
        Utility values (shape: n_subcriteria x n_scenarios).
    n_iterations : int
        Number of Monte Carlo samples.
    perturbation_range : float
        Maximum perturbation as fraction of base weight (e.g., 0.20 = +/-20%).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    samples : pd.DataFrame
        Columns: iteration, w_E, w_A, w_En, w_L, U_A, U_B, U_C, ranking_preserved.
    """
    rng = np.random.default_rng(random_seed)
    n_main = len(main_weights)
    n_scenarios = utility_matrix.shape[1]
    scenario_labels = [f"U_{chr(65 + i)}" for i in range(n_scenarios)]

    records = []
    base_ranking = tuple(np.argsort(-np.array([
        global_utility(
            np.concatenate([
                main_weights[i] * local_weights_per_criterion[i]
                for i in range(n_main)
            ]),
            utility_matrix[:, s]
        )
        for s in range(n_scenarios)
    ])))

    for iteration in range(n_iterations):
        multipliers = rng.uniform(
            low=1 - perturbation_range,
            high=1 + perturbation_range,
            size=n_main,
        )
        perturbed = main_weights * multipliers
        perturbed = perturbed / perturbed.sum()
        final_weights = np.concatenate([
            perturbed[i] * local_weights_per_criterion[i]
            for i in range(n_main)
        ])
        utilities = np.array([
            global_utility(final_weights, utility_matrix[:, s])
            for s in range(n_scenarios)
        ])
        ranking = tuple(np.argsort(-utilities))
        record = {"iteration": iteration}
        for j, label in enumerate(["w_E", "w_A", "w_En", "w_L"][:n_main]):
            record[label] = perturbed[j]
        for label, u in zip(scenario_labels, utilities):
            record[label] = u
        record["ranking_preserved"] = (ranking == base_ranking)
        records.append(record)

    return pd.DataFrame(records)


def summarize_monte_carlo(samples: pd.DataFrame,
                          scenario_columns: list[str] | None = None
                          ) -> pd.DataFrame:
    """
    Summary statistics for Monte Carlo results.

    Parameters
    ----------
    samples : pd.DataFrame
        Output from monte_carlo_simulation().
    scenario_columns : list[str], optional
        Columns to summarize. Default: detect U_* columns.

    Returns
    -------
    summary : pd.DataFrame
        Columns: scenario, mean, std, q025, q500, q975, min, max.
    """
    if scenario_columns is None:
        scenario_columns = [c for c in samples.columns if c.startswith("U_")]
    rows = []
    for col in scenario_columns:
        values = samples[col].values
        rows.append({
            "scenario": col.replace("U_", ""),
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "q025": np.percentile(values, 2.5),
            "q500": np.percentile(values, 50.0),
            "q975": np.percentile(values, 97.5),
            "min": values.min(),
            "max": values.max(),
        })
    return pd.DataFrame(rows)
