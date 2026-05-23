"""
run_all.py
==========
End-to-end reproduction script.

Validates AHP weights from pairwise matrices against published values,
computes MAUT baseline and exponential utility variants, runs OAT and
Monte Carlo sensitivity analyses, and saves all results to results/.

Run from project root:
    python run_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.ahp import evaluate_matrix, load_pairwise_matrix
from src.maut import (apply_exponential_to_matrix, evaluate_scenarios,
                      global_utility)
from src.sensitivity import (monte_carlo_simulation, oat_sensitivity,
                             summarize_monte_carlo)
from src.visualization import generate_all_figures

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in (TABLES_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load all input data files."""
    weights_df = pd.read_csv(DATA_DIR / "ahp_weights.csv")
    utilities_df = pd.read_csv(DATA_DIR / "utility_values.csv")
    main_matrix = load_pairwise_matrix(DATA_DIR / "pairwise_main_criteria.csv")
    return weights_df, utilities_df, main_matrix


def validate_ahp_main(main_matrix, published_main_weights):
    """Validate that the pairwise matrix produces the published main weights."""
    result = evaluate_matrix(main_matrix)
    print(f"\n{'='*60}")
    print("AHP Main Criteria Validation")
    print(f"{'='*60}")
    print(f"  lambda_max  = {result['lambda_max']:.4f}")
    print(f"  CI          = {result['CI']:.4f}")
    print(f"  CR          = {result['CR']:.4f} "
          f"({'consistent' if result['consistent'] else 'INCONSISTENT'})")
    print(f"  weights     = {result['weights'].round(4)}")
    print(f"  published   = {published_main_weights.round(4)}")
    diff = result["weights"] - published_main_weights
    print(f"  difference  = {diff.round(4)}")
    return result


def main():
    weights_df, utilities_df, main_matrix = load_data()

    # Published main weights (E, A, En, L)
    main_labels = ["Economic", "Agronomic", "Environmental", "Logistical"]
    main_weights = np.array([0.40, 0.25, 0.20, 0.15])
    local_weights = [
        weights_df[weights_df.criterion == "Economic"]["local_weight"].values,
        weights_df[weights_df.criterion == "Agronomic"]["local_weight"].values,
        weights_df[weights_df.criterion == "Environmental"]["local_weight"].values,
        weights_df[weights_df.criterion == "Logistical"]["local_weight"].values,
    ]

    # Validate AHP main criteria matrix
    validate_ahp_main(main_matrix, main_weights)

    # Utility matrix: rows = sub-criteria (in paper order), cols = scenarios
    utility_matrix = utilities_df[["scenario_A", "scenario_B", "scenario_C"]].values
    final_weights = weights_df["final_weight"].values

    # Baseline MAUT
    print(f"\n{'='*60}")
    print("MAUT Baseline (linear utility)")
    print(f"{'='*60}")
    baseline = evaluate_scenarios(final_weights, utility_matrix,
                                  scenario_labels=["A", "B", "C"])
    baseline["published"] = [0.8921, 0.5914, 0.2485]
    baseline["difference"] = baseline["global_utility"] - baseline["published"]
    print(baseline.to_string(index=False))
    baseline.to_csv(TABLES_DIR / "maut_baseline.csv", index=False)

    # Exponential utility variants
    print(f"\n{'='*60}")
    print("MAUT with Exponential Utility (risk aversion)")
    print(f"{'='*60}")
    rho_values = [0.5, 1.0, 2.0]
    exp_results = []
    for rho in rho_values:
        u_exp = apply_exponential_to_matrix(utility_matrix, rho=rho)
        scenarios_exp = evaluate_scenarios(final_weights, u_exp,
                                            scenario_labels=["A", "B", "C"])
        scenarios_exp["rho"] = rho
        exp_results.append(scenarios_exp)
    exp_df = pd.concat(exp_results, ignore_index=True)
    exp_pivot = exp_df.pivot(index="rho", columns="scenario", values="global_utility")
    print(exp_pivot.round(4).to_string())
    exp_df.to_csv(TABLES_DIR / "maut_exponential.csv", index=False)

    # OAT sensitivity
    print(f"\n{'='*60}")
    print("OAT Sensitivity (+/- 10%, +/- 20%)")
    print(f"{'='*60}")
    oat = oat_sensitivity(main_weights, local_weights, utility_matrix,
                          perturbations=[-0.20, -0.10, 0.0, 0.10, 0.20])
    oat["criterion"] = oat["criterion_idx"].map(dict(enumerate(main_labels)))
    cols = ["criterion", "perturbation", "U_A", "U_B", "U_C"]
    print(oat[cols].round(4).to_string(index=False))
    oat.to_csv(TABLES_DIR / "oat_sensitivity.csv", index=False)

    # Monte Carlo
    print(f"\n{'='*60}")
    print("Monte Carlo Simulation (10,000 iterations, +/- 20%)")
    print(f"{'='*60}")
    mc_samples = monte_carlo_simulation(main_weights, local_weights,
                                         utility_matrix,
                                         n_iterations=10000,
                                         perturbation_range=0.20,
                                         random_seed=42)
    mc_summary = summarize_monte_carlo(mc_samples)
    print(mc_summary.round(4).to_string(index=False))
    ranking_preserved_pct = 100 * mc_samples["ranking_preserved"].mean()
    print(f"\n  Ranking A > B > C preserved in {ranking_preserved_pct:.2f}% "
          f"of {len(mc_samples):,} iterations")
    mc_samples.to_csv(TABLES_DIR / "monte_carlo_samples.csv", index=False)
    mc_summary.to_csv(TABLES_DIR / "monte_carlo_summary.csv", index=False)

    # Figures
    print(f"\n{'='*60}")
    print("Generating figures")
    print(f"{'='*60}")
    generate_all_figures(mc_samples, oat, exp_df, baseline, FIGURES_DIR)
    print(f"  Saved to {FIGURES_DIR}/")

    print(f"\n{'='*60}")
    print("All tables and figures saved to results/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
