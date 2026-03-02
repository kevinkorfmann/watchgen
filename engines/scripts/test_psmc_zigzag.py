#!/usr/bin/env python3
"""
Test PSMC-RS against stdpopsim's HomSap Zigzag_1S14 demographic model.

1. Simulate a diploid genome using msprime via stdpopsim
2. Convert tree sequence -> PSMCFA (bin variants into 100bp windows)
3. Run psmc-rs via subprocess
4. Parse JSON output
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
BIN_SIZE = 100  # bp per bin
MU = 1.25e-8  # per-base per-generation mutation rate
N_ITERS = 25
SCRIPT_DIR = Path(__file__).parent
ENGINES_DIR = SCRIPT_DIR.parent
BINARY = ENGINES_DIR / "target" / "release" / "psmc-rs"


def ts_to_psmcfa(ts, bin_size=100):
    """Convert tree sequence to PSMCFA format (one diploid individual).

    Bins the genome into windows of `bin_size` bp. A bin is heterozygous (T)
    if it contains at least one heterozygous site, otherwise homozygous (K).
    """
    seq_len = int(ts.sequence_length)
    n_bins = seq_len // bin_size

    # Find heterozygous sites (where sample 0 and sample 1 differ)
    het_positions = []
    for var in ts.variants():
        geno = var.genotypes
        if geno[0] != geno[1]:
            het_positions.append(int(var.site.position))

    # Bin into windows
    bins = ["K"] * n_bins
    for pos in het_positions:
        bin_idx = pos // bin_size
        if bin_idx < n_bins:
            bins[bin_idx] = "T"

    return "".join(bins)


def get_zigzag_true_demography(generation_time):
    """Get the true Zigzag_1S14 N(t) using msprime's DemographyDebugger.

    Returns smooth (x_years, y_sizes) arrays capturing the exponential
    growth/decline epochs.
    """
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("Zigzag_1S14")
    dd = model.model.debug()

    # Sample at many time points on a log scale for a smooth curve
    times_gen = np.concatenate([
        np.array([0]),
        np.logspace(0, 5.5, 1000),
    ])
    sizes = dd.population_size_trajectory(steps=times_gen)[:, 0]

    times_years = times_gen * generation_time
    return times_years, sizes


def main():
    # ── Step 1: Build the Rust binary ───────────────────────
    print("Building psmc-rs (release)...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=ENGINES_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Build failed:")
        print(result.stderr)
        sys.exit(1)
    print("Build successful.")

    # ── Step 2: Simulate with stdpopsim ─────────────────────
    print("Simulating with stdpopsim (HomSap Zigzag_1S14, chr22 subset)...")
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("Zigzag_1S14")
    generation_time = model.generation_time
    contig = species.get_contig("chr22", length_multiplier=0.1)
    engine = stdpopsim.get_engine("msprime")

    pop_names = [p.name for p in model.model.populations]
    pop_name = pop_names[0]
    ts = engine.simulate(model, contig, {pop_name: 2}, seed=42)
    print(f"  Simulated tree sequence: {ts.num_trees} trees, "
          f"{ts.num_mutations} mutations, "
          f"{int(ts.sequence_length):,} bp")

    # Add mutations if needed
    if ts.num_mutations == 0:
        ts = msprime.sim_mutations(ts, rate=contig.mutation_rate, random_seed=42)
        print(f"  Added mutations: {ts.num_mutations}")

    # ── Step 3: Convert to PSMCFA ───────────────────────────
    print("Converting to PSMCFA format...")
    psmcfa_seq = ts_to_psmcfa(ts, bin_size=BIN_SIZE)
    n_het = psmcfa_seq.count("T")
    n_hom = psmcfa_seq.count("K")
    print(f"  {len(psmcfa_seq)} bins, {n_het} het ({n_het/len(psmcfa_seq):.4f}), "
          f"{n_hom} hom")

    # Write to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".psmcfa", delete=False
    ) as tmp:
        tmp.write(">chr22_sim\n")
        for i in range(0, len(psmcfa_seq), 80):
            tmp.write(psmcfa_seq[i : i + 80] + "\n")
        psmcfa_path = tmp.name

    # ── Step 4: Run psmc-rs ─────────────────────────────────
    print(f"Running psmc-rs ({N_ITERS} iterations)...")
    output_path = str(SCRIPT_DIR / "psmc_zigzag_result.json")
    cmd = [
        str(BINARY),
        "--input", psmcfa_path,
        "--output", output_path,
        "--n-iters", str(N_ITERS),
        "--mu", str(MU),
        "--bin-size", str(BIN_SIZE),
        "--generation-time", str(generation_time),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stderr)
    if result.returncode != 0:
        print("psmc-rs failed!")
        print(result.stderr)
        sys.exit(1)

    # ── Step 5: Parse JSON output ───────────────────────────
    with open(output_path) as f:
        output = json.load(f)

    print(f"Final theta: {output['final_params']['theta']:.6f}")
    print(f"Final rho: {output['final_params']['rho']:.6f}")
    print(f"N0: {output['scaled_output']['n0']:.0f}")

    # ── Step 6: Plot ────────────────────────────────────────
    print("Plotting results...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Inferred vs true demography
    ax = axes[0]

    # True zigzag demography (smooth exponential curve)
    true_years, true_sizes = get_zigzag_true_demography(generation_time)
    ax.plot(true_years, true_sizes, color="#B2182B", lw=2.5, label="True N(t)")

    # Inferred demography (step function)
    x_years = output["plot_data"]["x_years"]
    y_pops = output["plot_data"]["y_pop_sizes"]
    ax.plot(x_years, y_pops, color="#2166AC", lw=2, label="Inferred N(t)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Years before present")
    ax.set_ylabel("Effective population size $N_e$")
    ax.set_title("PSMC-RS: Zigzag Demography Recovery")
    ax.legend()
    ax.set_xlim(1e3, 1e7)
    ax.set_ylim(1e3, 2e5)
    ax.grid(True, alpha=0.3)

    # Panel B: Log-likelihood convergence
    ax = axes[1]
    iterations = [r["iteration"] for r in output["iterations"]]
    lls = [r["log_likelihood"] for r in output["iterations"]]
    ax.plot(iterations, lls, "o-", color="#2166AC", markersize=4)
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Log-likelihood")
    ax.set_title("EM Convergence")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = SCRIPT_DIR / "psmc_zigzag_result.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    # Clean up
    Path(psmcfa_path).unlink(missing_ok=True)
    print("Done!")


if __name__ == "__main__":
    main()
