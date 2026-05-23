"""
ahp.py
======
Analytic Hierarchy Process (AHP) implementation.

Implements pairwise comparison matrix eigenvector decomposition,
consistency index (CI), consistency ratio (CR), and weight derivation.

References
----------
Saaty, T. L. (1980). The Analytic Hierarchy Process. McGraw-Hill.
Saaty, T. L. (1990). How to make a decision: The Analytic Hierarchy Process.
    European Journal of Operational Research, 48(1), 9-26.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Saaty's Random Index for matrices of size n = 1 ... 15
# Source: Saaty (1980), updated values from Alonso & Lamata (2006)
RANDOM_INDEX = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
    11: 1.51, 12: 1.54, 13: 1.56, 14: 1.57, 15: 1.59,
}


def eigenvector_weights(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute priority weights from a pairwise comparison matrix using
    the principal eigenvector method.

    Parameters
    ----------
    matrix : np.ndarray
        Square pairwise comparison matrix (n x n) with reciprocal property.

    Returns
    -------
    weights : np.ndarray
        Normalized priority weights (sum to 1).
    lambda_max : float
        Principal eigenvalue of the matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_idx = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[max_idx].real
    principal_vector = eigenvectors[:, max_idx].real
    weights = principal_vector / principal_vector.sum()
    return weights, lambda_max


def consistency_index(lambda_max: float, n: int) -> float:
    """
    Compute the Consistency Index (CI).

    CI = (lambda_max - n) / (n - 1)

    Parameters
    ----------
    lambda_max : float
        Principal eigenvalue from the comparison matrix.
    n : int
        Matrix dimension.

    Returns
    -------
    CI : float
    """
    if n < 2:
        return 0.0
    return (lambda_max - n) / (n - 1)


def consistency_ratio(CI: float, n: int) -> float:
    """
    Compute the Consistency Ratio (CR).

    CR = CI / RI(n)

    A value below 0.10 is considered acceptable per Saaty (1980).

    Parameters
    ----------
    CI : float
        Consistency Index.
    n : int
        Matrix dimension.

    Returns
    -------
    CR : float
    """
    RI = RANDOM_INDEX.get(n, 1.59)
    if RI == 0:
        return 0.0
    return CI / RI


def evaluate_matrix(matrix: np.ndarray) -> dict:
    """
    Full AHP evaluation of a pairwise comparison matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Square pairwise comparison matrix.

    Returns
    -------
    result : dict
        Keys: weights, lambda_max, CI, CR, n, consistent (bool)
    """
    n = matrix.shape[0]
    weights, lambda_max = eigenvector_weights(matrix)
    CI = consistency_index(lambda_max, n)
    CR = consistency_ratio(CI, n)
    return {
        "weights": weights,
        "lambda_max": lambda_max,
        "CI": CI,
        "CR": CR,
        "n": n,
        "consistent": CR < 0.10,
    }


def load_pairwise_matrix(csv_path: str, sep: str = ",") -> np.ndarray:
    """
    Load a pairwise comparison matrix from a CSV file where the first
    column contains row labels.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    sep : str
        Field separator.

    Returns
    -------
    matrix : np.ndarray
    """
    df = pd.read_csv(csv_path, sep=sep, index_col=0)
    return df.values.astype(float)
