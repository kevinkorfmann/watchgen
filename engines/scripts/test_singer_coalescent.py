#!/usr/bin/env python3
"""
Test SINGER-RS coalescent approximations against msprime simulations.

1. Simulate many tree sequences with msprime for different sample sizes
2. Extract empirical lineage count and TMRCA distributions
3. Run singer-rs for theoretical smooth approximations
4. Plot empirical vs analytic quantities
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import msprime
import numpy as np

SCRIPT_DIR = Path(__file__).parent
ENGINES_DIR = SCRIPT_DIR.parent
BINARY = ENGINES_DIR / "target" / "release" / "singer-rs"

# Simulation parameters
NE = 10_000          # effective population size
N_REPLICATES = 5000  # number of independent trees per sample size
SAMPLE_SIZES = [5, 10, 20, 50]  # haploid sample sizes to compare
T_MAX_COAL = 10.0    # max time in coalescent units


def simulate_coalescent_times(n, ne, n_replicates, seed=42):
    """Simulate n_replicates independent coalescent trees for n samples.

    Returns arrays of:
    - TMRCA values (in coalescent units, scaled by 2*Ne)
    - Lineage count trajectories (time, n_lineages)
    """
    rng = np.random.default_rng(seed)
    tmrcas = []
    lineage_histories = []

    for _ in range(n_replicates):
        k = n
        t = 0.0
        history = [(0.0, k)]
        while k > 1:
            rate = k * (k - 1) / 2.0
            wait = rng.exponential(1.0 / rate)
            t += wait
            k -= 1
            history.append((t, k))
        tmrcas.append(t)
        lineage_histories.append(history)

    return np.array(tmrcas), lineage_histories


def empirical_lambda(lineage_histories, t_grid):
    """Compute empirical expected lineage count at each time in t_grid."""
    lambda_emp = np.zeros(len(t_grid))
    n_rep = len(lineage_histories)

    for history in lineage_histories:
        times, counts = zip(*history)
        times = np.array(times)
        counts = np.array(counts)

        for i, t in enumerate(t_grid):
            # Find which interval t falls in
            idx = np.searchsorted(times, t, side="right") - 1
            idx = max(0, min(idx, len(counts) - 1))
            lambda_emp[i] += counts[idx]

    lambda_emp /= n_rep
    return lambda_emp


def empirical_survival(tmrcas, t_grid):
    """Compute empirical survival function P(TMRCA > t) at each t in t_grid."""
    f_bar = np.zeros(len(t_grid))
    n = len(tmrcas)
    for i, t in enumerate(t_grid):
        f_bar[i] = np.mean(tmrcas > t)
    return f_bar


def main():
    # ── Build ─────────────────────────────────────────────────
    print("Building singer-rs (release)...")
    result = subprocess.run(
        ["cargo", "build", "--release", "-p", "singer-rs"],
        cwd=ENGINES_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Build failed:")
        print(result.stderr)
        sys.exit(1)
    print("Build successful.")

    # ── Simulate empirical distributions ──────────────────────
    print(f"Simulating coalescent for n = {SAMPLE_SIZES} ({N_REPLICATES} replicates each)...")
    t_grid = np.linspace(0, T_MAX_COAL, 200)

    empirical_data = {}
    for n in SAMPLE_SIZES:
        print(f"  n = {n}...")
        tmrcas, histories = simulate_coalescent_times(
            n, NE, N_REPLICATES, seed=42 + n
        )
        lambda_emp = empirical_lambda(histories, t_grid)
        fbar_emp = empirical_survival(tmrcas, t_grid)
        empirical_data[n] = {
            "tmrcas": tmrcas,
            "lambda": lambda_emp,
            "fbar": fbar_emp,
            "mean_tmrca": np.mean(tmrcas),
        }
        expected_tmrca = 2.0 * (1.0 - 1.0 / n)
        print(
            f"    Mean TMRCA: {np.mean(tmrcas):.4f} "
            f"(expected: {expected_tmrca:.4f})"
        )

    # ── Run singer-rs for each sample size ────────────────────
    print("Running singer-rs for theoretical curves...")
    theory_data = {}
    for n in SAMPLE_SIZES:
        output_path = str(SCRIPT_DIR / f"singer_n{n}_result.json")
        cmd = [
            str(BINARY),
            "--n-samples", str(n),
            "--n-times", "200",
            "--t-max", str(T_MAX_COAL),
            "--output", output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"singer-rs failed for n={n}!")
            print(result.stderr)
            sys.exit(1)

        with open(output_path) as f:
            output = json.load(f)

        theory_data[n] = output
        Path(output_path).unlink(missing_ok=True)

    # ── Plot ──────────────────────────────────────────────────
    print("Plotting results...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "SINGER-RS: Coalescent Approximations vs Simulation",
        fontsize=13, fontweight="bold", y=0.98,
    )

    colors = {5: "#2166AC", 10: "#B2182B", 20: "#1B7837", 50: "#E08214"}

    # Panel A: Lambda(t) — Expected lineage count over time
    ax = axes[0, 0]
    for n in SAMPLE_SIZES:
        # Empirical
        ax.plot(
            t_grid, empirical_data[n]["lambda"],
            color=colors[n], lw=1.5, alpha=0.5,
        )
        # Theory
        t_theory = theory_data[n]["branch_time_grid"]
        lam_theory = theory_data[n]["lambda_values"]
        ax.plot(
            t_theory, lam_theory,
            color=colors[n], lw=2, ls="--", label=f"n={n}",
        )

    ax.set_xlabel("Time (coalescent units)")
    ax.set_ylabel("Expected lineage count $\\lambda(t)$")
    ax.set_title("A. Lineage count: simulation (solid) vs theory (dashed)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T_MAX_COAL)

    # Panel B: F_bar(t) — Survival function (TMRCA)
    ax = axes[0, 1]
    for n in SAMPLE_SIZES:
        # Empirical
        ax.plot(
            t_grid, empirical_data[n]["fbar"],
            color=colors[n], lw=1.5, alpha=0.5,
        )
        # Theory
        t_theory = theory_data[n]["branch_time_grid"]
        fbar_theory = theory_data[n]["f_bar_values"]
        ax.plot(
            t_theory, fbar_theory,
            color=colors[n], lw=2, ls="--", label=f"n={n}",
        )

    ax.set_xlabel("Time (coalescent units)")
    ax.set_ylabel("$\\bar{F}(t) = P(T_{\\mathrm{MRCA}} > t)$")
    ax.set_title("B. TMRCA survival: simulation (solid) vs theory (dashed)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T_MAX_COAL)
    ax.set_ylim(-0.05, 1.05)

    # Panel C: Joining probabilities
    ax = axes[1, 0]
    for n in SAMPLE_SIZES:
        t_theory = theory_data[n]["branch_time_grid"]
        joining = theory_data[n]["joining_probs"]
        dt = t_theory[1] - t_theory[0] if len(t_theory) > 1 else 1.0
        # Plot as density (joining_prob / dt)
        t_mids = [(t_theory[i] + t_theory[i + 1]) / 2 for i in range(len(joining))]
        density = [j / dt for j in joining]
        ax.plot(t_mids, density, color=colors[n], lw=2, label=f"n={n}")

    # Overlay empirical TMRCA density for n=10
    n_hist = 10
    if n_hist in empirical_data:
        tmrcas = empirical_data[n_hist]["tmrcas"]
        ax.hist(
            tmrcas, bins=50, density=True,
            alpha=0.3, color=colors[n_hist], edgecolor="none",
            label=f"Empirical (n={n_hist})",
        )

    ax.set_xlabel("Time (coalescent units)")
    ax.set_ylabel("Joining probability density")
    ax.set_title("C. Joining probability density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T_MAX_COAL)

    # Panel D: TMRCA distribution comparison for n=10
    ax = axes[1, 1]
    n_qq = 10
    tmrcas = np.sort(empirical_data[n_qq]["tmrcas"])
    n_pts = len(tmrcas)

    # Empirical CDF
    ecdf_y = np.arange(1, n_pts + 1) / n_pts

    # Theoretical CDF: 1 - F_bar
    t_theory = np.array(theory_data[n_qq]["branch_time_grid"])
    fbar_theory = np.array(theory_data[n_qq]["f_bar_values"])
    cdf_theory = 1.0 - fbar_theory

    ax.plot(tmrcas, ecdf_y, color="#2166AC", lw=2, alpha=0.8, label="Empirical CDF")
    ax.plot(t_theory, cdf_theory, color="#B2182B", lw=2, ls="--", label="Theory CDF")
    ax.set_xlabel(f"TMRCA (coalescent units, n={n_qq})")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"D. TMRCA CDF comparison (n={n_qq})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(T_MAX_COAL, np.percentile(tmrcas, 99.5)))

    # Add summary stats
    mean_emp = empirical_data[n_qq]["mean_tmrca"]
    expected = 2.0 * (1 - 1.0 / n_qq)
    ax.text(
        0.98, 0.05,
        f"Mean TMRCA:\n  Empirical: {mean_emp:.4f}\n  Expected:  {expected:.4f}",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = SCRIPT_DIR / "singer_coalescent_result.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    print("Done!")


if __name__ == "__main__":
    main()
