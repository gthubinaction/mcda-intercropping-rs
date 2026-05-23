"""
maut.py
=======
Multi-Attribute Utility Theory (MAUT) implementation.

Provides aggregation of individual utility values into a global utility
score using the additive model, with support for linear and exponential
utility functions.

References
----------
Keeney, R. L., & Raiffa, H. (1976). Decisions with Multiple Objectives:
    Preferences and Value Tradeoffs. Wiley.
Clemen, R. T., & Reilly, T. (2001). Making Hard Decisions with DecisionTools.
    Duxbury/Thomson Learning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def linear_utility(x: float, x_min: float, x_max: float,
                   direction: str = "more_is_better") -> float:
    """
    Linear utility function.

    Parameters
    ----------
    x : float
        Raw attribute value.
    x_min, x_max : float
        Range bounds.
    direction : str
        'more_is_better' or 'less_is_better'.

    Returns
    -------
    u : float
        Utility in [0, 1].
    """
    if x_max == x_min:
        return 0.5
    if direction == "more_is_better":
        return (x - x_min) / (x_max - x_min)
    elif direction == "less_is_better":
        return (x_max - x) / (x_max - x_min)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def exponential_utility(x: float, x_min: float, x_max: float,
                        rho: float = 1.0,
                        direction: str = "more_is_better") -> float:
    """
    Exponential utility function with risk aversion parameter rho.

    u(x) = (1 - exp(-rho * x_norm)) / (1 - exp(-rho))

    where x_norm is the linearly normalized value in [0, 1].

    Parameters
    ----------
    x : float
        Raw attribute value.
    x_min, x_max : float
        Range bounds.
    rho : float
        Risk aversion parameter. rho > 0 implies risk-averse (concave for
        gains, convex for losses). Typical values: 0.5 (mild), 1.0 (moderate),
        2.0 (strong risk aversion).
    direction : str
        'more_is_better' or 'less_is_better'.

    Returns
    -------
    u : float
        Utility in [0, 1].
    """
    x_norm = linear_utility(x, x_min, x_max, direction)
    if rho == 0:
        return x_norm
    return (1 - np.exp(-rho * x_norm)) / (1 - np.exp(-rho))


def global_utility(weights: np.ndarray, utilities: np.ndarray) -> float:
    """
    Compute global utility as weighted sum of individual utilities.

    U(x) = sum_i (w_i * u_i(x_i))

    Parameters
    ----------
    weights : np.ndarray
        Sub-criterion final weights (must sum to ~1).
    utilities : np.ndarray
        Individual utility values for each sub-criterion (in [0, 1]).

    Returns
    -------
    U : float
        Global utility in [0, 1].
    """
    if len(weights) != len(utilities):
        raise ValueError(
            f"Length mismatch: weights={len(weights)}, utilities={len(utilities)}"
        )
    return float(np.dot(weights, utilities))


def evaluate_scenarios(weights: np.ndarray, utility_matrix: np.ndarray,
                       scenario_labels: list[str] | None = None) -> pd.DataFrame:
    """
    Compute global utility for each scenario.

    Parameters
    ----------
    weights : np.ndarray
        Sub-criterion final weights (shape: n_subcriteria).
    utility_matrix : np.ndarray
        Utility values (shape: n_subcriteria x n_scenarios).
    scenario_labels : list[str], optional
        Labels for the scenarios. Defaults to ['A', 'B', 'C', ...].

    Returns
    -------
    result : pd.DataFrame
        Columns: scenario, global_utility.
    """
    n_scenarios = utility_matrix.shape[1]
    if scenario_labels is None:
        scenario_labels = [chr(65 + i) for i in range(n_scenarios)]
    utilities = [global_utility(weights, utility_matrix[:, i])
                 for i in range(n_scenarios)]
    return pd.DataFrame({"scenario": scenario_labels,
                         "global_utility": utilities})


def apply_exponential_to_matrix(utility_matrix: np.ndarray,
                                rho: float) -> np.ndarray:
    """
    Apply exponential transformation to a pre-normalized utility matrix.

    Treats existing values in [0, 1] as x_norm and applies the exponential
    transformation directly: u_exp = (1 - exp(-rho * u_lin)) / (1 - exp(-rho)).

    Parameters
    ----------
    utility_matrix : np.ndarray
        Linear utilities in [0, 1].
    rho : float
        Risk aversion parameter.

    Returns
    -------
    transformed : np.ndarray
        Exponential utilities in [0, 1].
    """
    if rho == 0:
        return utility_matrix.copy()
    return (1 - np.exp(-rho * utility_matrix)) / (1 - np.exp(-rho))
