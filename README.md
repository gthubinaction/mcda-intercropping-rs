# mcda-intercropping-rs

[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20788738.svg)](https://doi.org/10.5281/zenodo.20788738)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Reproducibility package for the manuscript:

> *Integrated AHP–MAUT Framework for Exploratory Multicriteria Performance
> Evaluation of Soybean–Wheat Double Cropping in Southern Brazil* —
> Silva, C., Rediske, G., Siluk, J. C. M., Marchesan, T. B. (2026).

This repository contains the full pipeline (AHP weight derivation, MAUT
aggregation, OAT sensitivity, weight-based Monte Carlo, utility-value Monte
Carlo, and exponential utility robustness checks) used in the manuscript,
along with the input data needed to reproduce every reported number, table,
and figure.

> **Note on naming:** the repository slug retains the legacy `intercropping`
> term for URL, citation, and DOI stability. The system studied is soybean–wheat
> *double cropping* (sequential cultivation within the same crop year), not
> intercropping (simultaneous cultivation of two crops in the same field).

---

## Quick start

```bash
git clone https://github.com/gthubinaction/mcda-intercropping-rs.git
cd mcda-intercropping-rs
pip install -r requirements.txt
python run_all.py
```

Outputs land in `results/tables/` and `results/figures/`.

---

## Repository structure

```
mcda-intercropping-rs/
│
├── README.md                       This file
├── CITATION.cff                    Citation metadata (software + paper)
├── LICENSE                         MIT license (code)
├── LICENSE-DATA                    CC BY 4.0 license (data and results)
├── requirements.txt                Python dependencies
├── run_all.py                      End-to-end reproduction script
│
├── data/
│   ├── ahp_weights.csv             Final AHP weights (criterion, sub-criterion, CR)
│   ├── utility_values.csv          MAUT utility values per scenario
│   ├── climatic_efficiency.csv     Battisti et al. (2013) CE data
│   ├── scenario_definitions.csv    Scenario A/B/C structural parameters
│   └── pairwise_subcriteria.csv    Aggregated pairwise comparison (sub-criteria)
│
├── src/
│   ├── ahp.py                      AHP eigenvector, CI, CR
│   ├── maut.py                     Linear and exponential utility aggregation
│   ├── sensitivity.py              OAT, weight Monte Carlo, utility Monte Carlo
│   └── visualization.py            Publication-quality figures (matplotlib)
│
├── notebooks/
│   └── 01_walkthrough.ipynb        Interactive exploration mirroring run_all.py
│
└── results/
    ├── tables/                     CSVs reproducing the manuscript's tables,
    │                               weight Monte Carlo, and utility Monte Carlo
    └── figures/                    PNG + PDF (300 dpi) for the manuscript
```

The four main-criteria weights (Economic 0.40, Agronomic 0.25, Environmental
0.20, Logistical/Commercial 0.15) are obtained by **direct assignment**
validated through panel consensus, not by a pairwise comparison matrix; there
is therefore no `pairwise_main_criteria.csv`. Only the twelve sub-criteria,
nested within each main criterion, are derived from aggregated pairwise
comparisons (`pairwise_subcriteria.csv`).

---

## What this code does

### 1. AHP weight derivation (`src/ahp.py`)

Computes priority weights from pairwise comparison matrices using the
principal eigenvector method. Returns lambda_max, Consistency Index (CI),
and Consistency Ratio (CR).

```
CI = (lambda_max - n) / (n - 1)
CR = CI / RI(n)
```

A CR below 0.10 indicates acceptable judgment coherence (Saaty, 1980).

### 2. MAUT aggregation (`src/maut.py`)

Implements the additive model:

```
U(x) = sum_i [ w_i * u_i(x_i) ]
```

Two utility function families are supported:

- **Linear**: `u(x) = (x - x_min) / (x_max - x_min)` (manuscript baseline)
- **Exponential**: `u(x) = (1 - exp(-rho * x_norm)) / (1 - exp(-rho))`,
  parameterized by risk aversion coefficient rho

### 3. Sensitivity and robustness analysis (`src/sensitivity.py`)

Four complementary procedures:

- **OAT**: each main criterion weight is perturbed by +/-10% and +/-20%, with
  proportional redistribution of the remaining mass. Ranking stability is
  checked across all perturbations.
- **Weight-based Monte Carlo**: 10,000 iterations sampling uniformly from
  [-20%, +20%] multipliers on each main weight, then renormalizing. Generates
  95% confidence intervals for U(A), U(B), U(C) and the empirical proportion
  of iterations preserving the A > B > C ranking.
- **Utility-value Monte Carlo**: stress-tests the author-assigned utility
  matrix directly. Each of the twelve utility values per scenario is perturbed
  by an independent uniform factor (+/-10% and +/-20%, 10,000 iterations each)
  and clipped to [0, 1], with the AHP weights held fixed.
- **Exponential utility comparison**: re-evaluates the global utilities under
  exponential utility with risk-aversion coefficients rho in {0.5, 1.0, 2.0}
  to test sensitivity to the utility functional form.

---

## Validation against published values

Running `run_all.py` reproduces the values reported in the manuscript
exactly: U(A) = 0.8872, U(B) = 0.5953, U(C) = 0.2421.

**Note on the pairwise matrices** (`data/pairwise_subcriteria.csv`): these are
Saaty-scale aggregated matrices consistent with the published weights.
Individual expert matrices (n = 6) are not redistributed here for
respondent confidentiality, following standard practice in AHP studies.

**Note on the utility values** (`data/utility_values.csv`): the AHP
weights are derived from elicitation with a panel of six experts
(semi-structured interviews on Saaty's 1-9 scale). The MAUT utility
values, in contrast, are author-assigned scenario-based scores derived
from the qualitative descriptions of Scenarios A, B, and C. The scenarios
themselves are anchored to climatic efficiency (CE) ranges from Battisti
et al. (2013), and within each scenario each utility reflects the
interpretation of the corresponding scenario profile, applying the linear
scaling logic ("more is better" or "less is better") per Clemen & Reilly
(2001). This is an *exploratory MAUT modeling* approach (sensu Bankes,
1993), suitable for problems characterized by primary data scarcity.
The `scenario_anchor` column in `utility_values.csv` documents which
aspect of each scenario profile each utility reflects. Future work
should refine these utilities through primary measurement.

---

## Key robustness findings

**Weight-based Monte Carlo** — 10,000 iterations, +/-20% uniform perturbation
on each main criterion weight:

| Scenario | Mean  | SD     | 95% CI            |
|----------|-------|--------|-------------------|
| A        | 0.887 | 0.001  | [0.885, 0.889]    |
| B        | 0.595 | 0.003  | [0.590, 0.601]    |
| C        | 0.243 | 0.005  | [0.234, 0.252]    |

**Utility-value Monte Carlo** — direct perturbation of the twelve author-assigned
utilities, AHP weights fixed: the A > B > C ranking is preserved in 100% of
iterations under both +/-10% and +/-20% amplitudes, with non-overlapping 95%
confidence intervals in every case.

Across all procedures the A > B > C ranking is preserved in **100%** of
iterations and the 95% confidence intervals do not overlap, providing strong
evidence of ranking stability under both weight and utility uncertainty.

---

## Dependencies

- Python 3.10+
- numpy >= 1.24
- pandas >= 2.0
- matplotlib >= 3.7
- jupyter >= 1.0 (for the walkthrough notebook)

See `requirements.txt` for pinned versions.

---

## License

- **Code** (`src/`, `run_all.py`, `notebooks/`): MIT License
  (see [LICENSE](LICENSE))
- **Data and results** (`data/`, `results/`): Creative Commons
  Attribution 4.0 International (CC BY 4.0)
  (see [LICENSE-DATA](LICENSE-DATA))

---

## Citation

If you use this software or data, please cite the archived release:

> Silva, C., Rediske, G., Siluk, J. C. M., Marchesan, T. B. (2026).
> *mcda-intercropping-rs: Reproducibility package for the AHP–MAUT double
> cropping framework* (v1.0.2). Zenodo. https://doi.org/10.5281/zenodo.20788738

Machine-readable metadata is provided in [`CITATION.cff`](CITATION.cff).

---

## Contact

Corresponding author: Cristian Silva — Programa de Pós-Graduação em
Engenharia de Produção (PPGEP), Universidade Federal de Santa Maria (UFSM),
Santa Maria, RS, Brazil.

For questions about the code or data, please open an issue in this repository.
