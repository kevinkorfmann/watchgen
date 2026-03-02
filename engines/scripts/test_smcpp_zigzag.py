#!/usr/bin/env python3
"""
Test SMC++-RS demographic inference against stdpopsim's Zigzag_1S14 model.

1. Simulate multiple diploid individuals using msprime via stdpopsim
2. Convert tree sequences -> binary observation sequences (het/hom bins)
3. Run smcpp-rs via subprocess
4. Parse JSON output and scale to years / Ne
5. Plot inferred vs true zigzag demography
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import msprime
import numpy as np

try:
    import stdpopsim
except ImportError:
    print("stdpopsim not installed. Install with: pip install stdpopsim")
    sys.exit(1)

# ── Parameters ──────────────────────────────────────────────
BIN_SIZE = 100      # bp per bin
MU = 1.25e-8        # per-base per-generation mutation rate
N_INDIVIDUALS = 2   # diploid individuals (n_undist = 3)
N_INTERVALS = 15    # time intervals for SMC++ inference
T_MAX = 6.0         # coalescent units
MAX_ITER = 50       # coordinate descent iterations

SCRIPT_DIR = Path(__file__).parent
ENGINES_DIR = SCRIPT_DIR.parent
BINARY = ENGINES_DIR / "target" / "release" / "smcpp-rs"


def ts_to_binary_sequences(ts, bin_size=100):
    """Convert tree sequence to binary observation sequences.

    For each diploid individual, create a binary sequence where:
    - 1 = at least one heterozygous site in the bin
    - 0 = no heterozygous sites (homozygous bin)
    """
    seq_len = int(ts.sequence_length)
    n_bins = seq_len // bin_size

    # Collect all individuals
    sample_nodes = list(ts.samples())
    n_samples = len(sample_nodes)
    n_diploids = n_samples // 2

    sequences = []
    for ind in range(n_diploids):
        hap0 = sample_nodes[2 * ind]
        hap1 = sample_nodes[2 * ind + 1]
        bins = [0] * n_bins

        for var in ts.variants():
            geno = var.genotypes
            if geno[hap0] != geno[hap1]:
                bin_idx = int(var.site.position) // bin_size
                if bin_idx < n_bins:
                    bins[bin_idx] = 1

        sequences.append(bins)

    return sequences, n_bins


def get_zigzag_true_demography(generation_time):
    """Get the true Zigzag_1S14 N(t) using msprime's DemographyDebugger."""
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("Zigzag_1S14")
    dd = model.model.debug()

    times_gen = np.concatenate([
        np.array([0]),
        np.logspace(0, 5.5, 1000),
    ])
    sizes = dd.population_size_trajectory(steps=times_gen)[:, 0]
    times_years = times_gen * generation_time
    return times_years, sizes


def main():
    # ── Build ─────────────────────────────────────────────────
    print("Building smcpp-rs (release)...")
    result = subprocess.run(
        ["cargo", "build", "--release", "-p", "smcpp-rs"],
        cwd=ENGINES_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Build failed:")
        print(result.stderr)
        sys.exit(1)
    print("Build successful.")

    # ── Simulate ──────────────────────────────────────────────
    print(f"Simulating {N_INDIVIDUALS} diploid individuals with stdpopsim (Zigzag_1S14)...")
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("Zigzag_1S14")
    generation_time = model.generation_time
    contig = species.get_contig("chr22", length_multiplier=0.1)  # ~5 Mb
    engine = stdpopsim.get_engine("msprime")

    pop_name = [p.name for p in model.model.populations][0]
    ts = engine.simulate(model, contig, {pop_name: N_INDIVIDUALS}, seed=42)

    if ts.num_mutations == 0:
        ts = msprime.sim_mutations(ts, rate=contig.mutation_rate, random_seed=42)

    print(
        f"  {ts.num_trees} trees, {ts.num_mutations} mutations, "
        f"{int(ts.sequence_length):,} bp"
    )

    # ── Convert to binary sequences ───────────────────────────
    print("Converting to binary observation sequences...")
    sequences, n_bins = ts_to_binary_sequences(ts, BIN_SIZE)
    for i, seq in enumerate(sequences):
        n_het = sum(seq)
        frac = n_het / len(seq)
        print(f"  Individual {i}: {n_het}/{len(seq)} het bins ({frac:.4f})")

    # Estimate theta from mean heterozygosity
    all_het = np.array([sum(s) / len(s) for s in sequences])
    mean_het = np.mean(all_het)
    # theta_hat = -ln(1 - het_frac) ≈ het_frac for small het_frac
    theta_hat = -np.log(1 - mean_het) if mean_het < 1 else 0.5
    rho_hat = theta_hat / 5.0  # rough ratio
    print(f"  Mean het fraction: {mean_het:.4f}")
    print(f"  Estimated theta: {theta_hat:.6f}, rho: {rho_hat:.6f}")

    # Write input file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for seq in sequences:
            f.write(" ".join(map(str, seq)) + "\n")
        input_path = f.name

    # ── Run smcpp-rs ──────────────────────────────────────────
    output_path = str(SCRIPT_DIR / "smcpp_zigzag_result.json")
    cmd = [
        str(BINARY),
        "--input", input_path,
        "--output", output_path,
        "--n-intervals", str(N_INTERVALS),
        "--t-max", str(T_MAX),
        "--theta", f"{theta_hat:.6f}",
        "--rho", f"{rho_hat:.6f}",
        "--max-iter", str(MAX_ITER),
    ]
    print(f"Running smcpp-rs ({MAX_ITER} iterations)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stderr)
    if result.returncode != 0:
        print("smcpp-rs failed!")
        print(result.stdout)
        sys.exit(1)

    # ── Parse output ──────────────────────────────────────────
    with open(output_path) as f:
        output = json.load(f)

    lambdas = np.array(output["lambdas"])
    time_breaks = np.array(output["time_breaks"])
    initial_ll = output["initial_ll"]
    final_ll = output["final_ll"]
    iterations = output["iterations"]

    print(f"Initial LL: {initial_ll:.4f}")
    print(f"Final LL:   {final_ll:.4f} ({iterations} iterations)")
    print(f"Lambda range: [{lambdas.min():.3f}, {lambdas.max():.3f}]")

    # ── Scale to years and Ne ─────────────────────────────────
    # N0 = theta / (4 * mu * bin_size)
    n0 = theta_hat / (4 * MU * BIN_SIZE)
    print(f"Inferred N0: {n0:.0f}")

    # Midpoint times in coalescent units
    t_mids = [(time_breaks[i] + time_breaks[i + 1]) / 2.0 for i in range(len(lambdas))]
    # Convert to years: t_years = 2 * N0 * t_coalescent * generation_time
    t_years = [2 * n0 * t * generation_time for t in t_mids]
    # Ne at each interval: Ne = N0 * lambda
    ne_inferred = [n0 * lam for lam in lambdas]

    # Build step-function for plotting
    x_years = []
    y_ne = []
    for i in range(len(lambdas)):
        t_lo = 2 * n0 * time_breaks[i] * generation_time
        t_hi = 2 * n0 * time_breaks[i + 1] * generation_time
        x_years.extend([max(t_lo, 1), t_hi])
        y_ne.extend([ne_inferred[i], ne_inferred[i]])

    # ── Plot ──────────────────────────────────────────────────
    print("Plotting results...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Inferred vs true demography
    ax = axes[0]

    true_years, true_sizes = get_zigzag_true_demography(generation_time)
    ax.plot(true_years, true_sizes, color="#B2182B", lw=2.5, label="True N(t)")

    ax.plot(x_years, y_ne, color="#2166AC", lw=2, label="SMC++ inferred")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Years before present")
    ax.set_ylabel("Effective population size $N_e$")
    ax.set_title("SMC++-RS: Zigzag Demography Recovery")
    ax.legend(fontsize=10)
    ax.set_xlim(1e3, 1e7)
    ax.set_ylim(1e3, 2e5)
    ax.grid(True, alpha=0.3)

    # Add inset stats
    stats_text = (
        f"$N_0$ = {n0:,.0f}\n"
        f"$\\theta$ = {theta_hat:.4f}\n"
        f"Individuals: {N_INDIVIDUALS}\n"
        f"Intervals: {N_INTERVALS}"
    )
    ax.text(
        0.02, 0.03, stats_text, transform=ax.transAxes,
        fontsize=8, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # Panel B: Lambda values as bar chart
    ax = axes[1]
    bar_x = np.arange(len(lambdas))
    colors = ["#2166AC" if lam >= 1 else "#B2182B" for lam in lambdas]
    ax.bar(bar_x, lambdas, color=colors, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.5, label="$\\lambda=1$ (reference)")
    ax.set_xlabel("Time interval index")
    ax.set_ylabel("Relative population size $\\lambda_k$")
    ax.set_title(
        f"Inferred $\\lambda_k$ (LL: {initial_ll:.1f} → {final_ll:.1f})"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = SCRIPT_DIR / "smcpp_zigzag_result.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    # Clean up
    Path(input_path).unlink(missing_ok=True)
    print("Done!")


if __name__ == "__main__":
    main()
