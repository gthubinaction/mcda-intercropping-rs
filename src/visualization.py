"""
visualization.py
================
Figure generation for the AHP-MAUT framework reproduction.

Produces three figures intended for the paper:
    Figure 1: Monte Carlo distribution of global utility per scenario.
    Figure 2: OAT sensitivity tornado chart.
    Figure 3: Exponential utility comparison across risk aversion levels.

All figures use a clean publication style (no chartjunk, large fonts, 300 dpi).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLOR_A = "#2C7BB6"
COLOR_B = "#FDAE61"
COLOR_C = "#D7191C"


def figure_monte_carlo(samples: pd.DataFrame, output_dir: Path) -> None:
    """
    Figure 1: Monte Carlo distributions of U(A), U(B), U(C).

    Two-panel layout: violin/density on left, scenario means with CIs on right.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    gridspec_kw={"width_ratios": [2, 1]})

    data = [samples["U_A"].values, samples["U_B"].values, samples["U_C"].values]
    colors = [COLOR_A, COLOR_B, COLOR_C]
    labels = ["Scenario A\n(Optimistic)", "Scenario B\n(Moderate)",
              "Scenario C\n(Pessimistic)"]

    parts = ax1.violinplot(data, positions=[1, 2, 3], widths=0.7,
                           showmeans=False, showmedians=True, showextrema=False)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Global utility $U(x)$")
    ax1.set_title("(a) Monte Carlo distribution\n(10,000 iterations, $\\pm$20% weight perturbation)")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.set_axisbelow(True)

    means = [d.mean() for d in data]
    q025 = [np.percentile(d, 2.5) for d in data]
    q975 = [np.percentile(d, 97.5) for d in data]
    yerr_lower = [m - q for m, q in zip(means, q025)]
    yerr_upper = [q - m for m, q in zip(means, q975)]

    ax2.errorbar([1, 2, 3], means, yerr=[yerr_lower, yerr_upper],
                 fmt="o", capsize=8, capthick=2, linewidth=2,
                 color="black", ecolor="gray", markersize=10,
                 markerfacecolor="white", markeredgewidth=2)
    for i, (m, color) in enumerate(zip(means, colors), 1):
        ax2.plot(i, m, "o", color=color, markersize=8, zorder=5)

    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["A", "B", "C"])
    ax2.set_xlabel("Scenario")
    ax2.set_ylabel("Global utility $U(x)$")
    ax2.set_title("(b) Mean and 95% CI")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.set_xlim(0.5, 3.5)

    for i, (m, qlo, qhi) in enumerate(zip(means, q025, q975), 1):
        ax2.annotate(f"{m:.3f}", xy=(i + 0.15, m), fontsize=9,
                     verticalalignment="center")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = output_dir / f"figure_monte_carlo.{ext}"
        plt.savefig(path)
    plt.close()


def figure_oat_tornado(oat_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Figure 2: OAT sensitivity tornado chart.

    Shows how U(A), U(B), U(C) change when each criterion weight is
    perturbed by +/-20%.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    criteria = ["Economic", "Agronomic", "Environmental", "Logistical"]
    scenarios = ["U_A", "U_B", "U_C"]
    titles = ["Scenario A (Optimistic)", "Scenario B (Moderate)",
              "Scenario C (Pessimistic)"]
    colors = [COLOR_A, COLOR_B, COLOR_C]

    for ax, scenario, title, color in zip(axes, scenarios, titles, colors):
        baseline = oat_df[oat_df["perturbation"] == 0.0][scenario].mean()

        for i, crit in enumerate(criteria):
            crit_idx = criteria.index(crit)
            sub = oat_df[oat_df["criterion_idx"] == crit_idx]
            low = sub[sub["perturbation"] == -0.20][scenario].values[0]
            high = sub[sub["perturbation"] == 0.20][scenario].values[0]
            left = min(low, high)
            width = abs(high - low)
            ax.barh(i, width, left=left, color=color, alpha=0.7,
                    edgecolor="black", linewidth=0.7)

        ax.axvline(baseline, color="black", linestyle="--", linewidth=1.5,
                   label=f"Baseline = {baseline:.3f}")
        ax.set_yticks(range(len(criteria)))
        ax.set_yticklabels(criteria)
        ax.set_xlabel("Global utility $U(x)$")
        ax.set_title(title)
        ax.legend(loc="lower right", framealpha=0.9)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

    fig.suptitle("OAT sensitivity: range of $U(x)$ under $\\pm$20% perturbation of each criterion weight",
                 y=1.02, fontsize=12)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = output_dir / f"figure_oat_tornado.{ext}"
        plt.savefig(path)
    plt.close()


def figure_exponential_utility(exp_df: pd.DataFrame, baseline_df: pd.DataFrame,
                                output_dir: Path) -> None:
    """
    Figure 3: Exponential utility comparison.

    Bar chart showing U(A), U(B), U(C) under linear and exponential
    (ρ = 0.5, 1.0, 2.0) utility functions.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    rhos = sorted(exp_df["rho"].unique())
    scenarios = ["A", "B", "C"]
    n_groups = len(scenarios)
    n_bars = 1 + len(rhos)
    bar_width = 0.18
    x = np.arange(n_groups)

    linear_vals = baseline_df["global_utility"].values
    ax.bar(x - 1.5 * bar_width, linear_vals, bar_width,
           label="Linear (baseline)", color="#555555", edgecolor="black")

    palette = ["#74A9CF", "#3690C0", "#0570B0"]
    offsets = [-0.5, 0.5, 1.5]
    for rho, color, offset in zip(rhos, palette, offsets):
        sub = exp_df[exp_df["rho"] == rho].sort_values("scenario")
        vals = sub["global_utility"].values
        ax.bar(x + offset * bar_width, vals, bar_width,
               label=f"Exp. ($\\rho$ = {rho})", color=color, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Scenario {s}" for s in scenarios])
    ax.set_ylabel("Global utility $U(x)$")
    ax.set_title("Comparison of utility function forms\n(linear vs. exponential with risk aversion $\\rho$)")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = output_dir / f"figure_exponential_utility.{ext}"
        plt.savefig(path)
    plt.close()


def generate_all_figures(samples: pd.DataFrame, oat_df: pd.DataFrame,
                         exp_df: pd.DataFrame, baseline_df: pd.DataFrame,
                         output_dir: Path) -> None:
    """Generate all paper figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_monte_carlo(samples, output_dir)
    figure_oat_tornado(oat_df, output_dir)
    figure_exponential_utility(exp_df, baseline_df, output_dir)
