"""
run_all.py
==========
End-to-end reproduction script.

Derives sub-criterion local weights from the expert pairwise comparison
matrices via the principal eigenvector method, combines them with the
directly assigned main-criteria weights (elicited by panel consensus;
direct rating / point allocation), computes the MAUT baseline and
exponential variants, and runs OAT, weight Monte Carlo and utility
Monte Carlo sensitivity analyses. All results are saved to results/.

Main-criteria weights (Economic 0.40, Agronomic 0.25, Environmental 0.20,
Logistical 0.15) were obtained by direct assignment validated in a
consensus meeting with the six-member expert panel, not by pairwise
comparison; hence no main-criteria comparison matrix is used.

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

from src.ahp import evaluate_matrix
from src.maut import (apply_exponential_to_matrix, evaluate_scenarios,
                      global_utility)
from src.sensitivity import (monte_carlo_simulation, oat_sensitivity,
                             summarize_monte_carlo, utility_monte_carlo)
from src.visualization import generate_all_figures

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in (TABLES_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Main-criteria weights: directly assigned by expert-panel consensus.
MAIN_WEIGHTS = {
    "Economic": 0.40,
    "Agronomic": 0.25,
    "Environmental": 0.20,
    "Logistical": 0.15,
}
CRIT_ORDER = ["Economic", "Agronomic", "Environmental", "Logistical"]


def derive_subcriteria_weights():
    """
    Derive local sub-criterion weights from the pairwise comparison matrices
    using the principal eigenvector method, then compute final weights as
    local_weight x directly-assigned main_weight.
    """
    sub = pd.read_csv(DATA_DIR / "pairwise_subcriteria.csv")
    value_cols = list(sub.columns[2:])

    records = []
    local_per_crit = []
    print(f"\n{'='*60}")
    print("AHP — sub-criteria local weights (principal eigenvector)")
    print(f"{'='*60}")
    print("Main-criteria weights: direct assignment by panel consensus")
    print(f"  {MAIN_WEIGHTS}\n")
    for crit in CRIT_ORDER:
        block = sub[sub.parent_criterion == crit]
        cols = [c for c in value_cols if block[c].notna().all()]
        matrix = block[cols].values.astype(float)
        res = evaluate_matrix(matrix)
        local_per_crit.append(res["weights"])
        print(f"  [{crit}] lambda_max={res['lambda_max']:.4f}  "
              f"CR={res['CR']:.4f} "
              f"({'consistent' if res['consistent'] else 'INCONSISTENT'})")
        for name, lw in zip(block.subcriterion.tolist(), res["weights"]):
            fw = MAIN_WEIGHTS[crit] * lw
            records.append({
                "criterion": crit,
                "subcriterion": name,
                "main_weight": MAIN_WEIGHTS[crit],
                "local_weight": lw,
                "final_weight": fw,
                "CR": res["CR"],
            })
            print(f"     {name:<28s} local={lw:.4f}  final={fw:.4f}")
    weights_df = pd.DataFrame(records)
    weights_df.to_csv(TABLES_DIR / "ahp_weights.csv", index=False)
    print(f"\n  Sum of final weights = {weights_df.final_weight.sum():.6f}")
    return weights_df, local_per_crit


def main():
    weights_df, local_weights = derive_subcriteria_weights()
    final_weights = weights_df["final_weight"].values
    main_weights = np.array([MAIN_WEIGHTS[c] for c in CRIT_ORDER])

    utilities_df = pd.read_csv(DATA_DIR / "utility_values.csv")
    utility_matrix = utilities_df[["scenario_A", "scenario_B", "scenario_C"]].values

    # Baseline MAUT
    print(f"\n{'='*60}")
    print("MAUT Baseline (linear utility)")
    print(f"{'='*60}")
    baseline = evaluate_scenarios(final_weights, utility_matrix,
                                  scenario_labels=["A", "B", "C"])
    print(baseline.round(4).to_string(index=False))
    baseline.to_csv(TABLES_DIR / "maut_baseline.csv", index=False)

    # Exponential utility variants
    print(f"\n{'='*60}")
    print("MAUT with Exponential Utility (risk aversion)")
    print(f"{'='*60}")
    exp_results = []
    for rho in [0.5, 1.0, 2.0]:
        u_exp = apply_exponential_to_matrix(utility_matrix, rho=rho)
        scen = evaluate_scenarios(final_weights, u_exp, scenario_labels=["A", "B", "C"])
        scen["rho"] = rho
        exp_results.append(scen)
    exp_df = pd.concat(exp_results, ignore_index=True)
    print(exp_df.pivot(index="rho", columns="scenario",
                       values="global_utility").round(4).to_string())
    exp_df.to_csv(TABLES_DIR / "maut_exponential.csv", index=False)

    # OAT sensitivity
    print(f"\n{'='*60}")
    print("OAT Sensitivity (+/- 10%, +/- 20%)")
    print(f"{'='*60}")
    oat = oat_sensitivity(main_weights, local_weights, utility_matrix,
                          perturbations=[-0.20, -0.10, 0.0, 0.10, 0.20])
    oat["criterion"] = oat["criterion_idx"].map(dict(enumerate(CRIT_ORDER)))
    print(oat[["criterion", "perturbation", "U_A", "U_B", "U_C"]].round(4).to_string(index=False))
    oat.to_csv(TABLES_DIR / "oat_sensitivity.csv", index=False)

    # Monte Carlo on weights
    print(f"\n{'='*60}")
    print("Monte Carlo Simulation (10,000 iterations, +/- 20% on weights)")
    print(f"{'='*60}")
    mc_samples = monte_carlo_simulation(main_weights, local_weights, utility_matrix,
                                        n_iterations=10000, perturbation_range=0.20,
                                        random_seed=42)
    mc_summary = summarize_monte_carlo(mc_samples)
    print(mc_summary.round(4).to_string(index=False))
    print(f"\n  Ranking A > B > C preserved in "
          f"{100 * mc_samples['ranking_preserved'].mean():.2f}% "
          f"of {len(mc_samples):,} iterations")
    mc_samples.to_csv(TABLES_DIR / "monte_carlo_samples.csv", index=False)
    mc_summary.to_csv(TABLES_DIR / "monte_carlo_summary.csv", index=False)

    # Monte Carlo on utilities
    for pct, tag in [(0.10, "10pct"), (0.20, "20pct")]:
        print(f"\n{'='*60}")
        print(f"Utility Monte Carlo (10,000 iterations, +/- {int(pct*100)}% on utilities)")
        print(f"{'='*60}")
        umc = utility_monte_carlo(final_weights, utility_matrix,
                                  n_iterations=10000, perturbation_range=pct,
                                  random_seed=42)
        umc_summary = summarize_monte_carlo(umc)
        print(umc_summary.round(4).to_string(index=False))
        print(f"  Ranking preserved: {100 * umc['ranking_preserved'].mean():.2f}%")
        umc.to_csv(TABLES_DIR / f"utility_monte_carlo_{tag}_samples.csv", index=False)
        umc_summary.to_csv(TABLES_DIR / f"utility_monte_carlo_{tag}_summary.csv", index=False)

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
