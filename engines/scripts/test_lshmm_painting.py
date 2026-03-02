#!/usr/bin/env python3
"""
Test LSHMM-RS haplotype painting with stdpopsim simulation.

1. Simulate haplotypes under HomSap Zigzag_1S14 using stdpopsim
2. Build reference panel (n-1 haplotypes) and query (1 haplotype)
3. Determine true copying path from tree sequence local ancestry
4. Run lshmm-rs to infer Viterbi and posterior copying paths
5. Plot painting comparison (true vs inferred)
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

try:
    import stdpopsim
    import msprime
except ImportError:
    print("Required: pip install stdpopsim msprime")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
ENGINES_DIR = SCRIPT_DIR.parent
BINARY = ENGINES_DIR / "target" / "release" / "lshmm-rs"

# Simulation parameters
N_INDIVIDUALS = 10  # diploid individuals = 20 haploid samples
RHO_LSHMM = 0.04   # scaled recombination rate for Li-Stephens


def get_true_copying_path(ts, query_id, ref_ids):
    """Determine true copying path from tree sequence local ancestry.

    At each variant site, find which reference haplotype shares the most
    recent common ancestor with the query.
    """
    true_path = []
    variant_positions = []

    # Cache: map tree index to (best_ref_idx, min_time) to avoid
    # recomputing for sites within the same local tree
    tree_cache = {}

    for var in ts.variants():
        pos = var.site.position
        variant_positions.append(pos)
        tree = ts.at(pos)
        tree_idx = tree.index

        if tree_idx not in tree_cache:
            min_time = float("inf")
            best_ref_idx = 0
            for i, ref_id in enumerate(ref_ids):
                mrca_node = tree.mrca(query_id, ref_id)
                if mrca_node != -1:
                    t = tree.time(mrca_node)
                    if t < min_time:
                        min_time = t
                        best_ref_idx = i
            tree_cache[tree_idx] = best_ref_idx

        true_path.append(tree_cache[tree_idx])

    return np.array(true_path), np.array(variant_positions)


def count_segments(path):
    """Count the number of contiguous segments in a path."""
    if len(path) == 0:
        return 0
    count = 1
    for i in range(1, len(path)):
        if path[i] != path[i - 1]:
            count += 1
    return count


def main():
    # ── Build ─────────────────────────────────────────────────
    print("Building lshmm-rs (release)...")
    result = subprocess.run(
        ["cargo", "build", "--release", "-p", "lshmm-rs"],
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
    print(f"Simulating {N_INDIVIDUALS} diploid individuals with stdpopsim...")
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("Zigzag_1S14")
    contig = species.get_contig("chr22", length_multiplier=0.02)  # ~1 Mb
    engine = stdpopsim.get_engine("msprime")

    pop_name = [p.name for p in model.model.populations][0]
    ts = engine.simulate(model, contig, {pop_name: N_INDIVIDUALS}, seed=2024)

    if ts.num_mutations == 0:
        ts = msprime.sim_mutations(ts, rate=contig.mutation_rate, random_seed=2024)

    print(
        f"  {ts.num_trees} trees, {ts.num_mutations} mutations, "
        f"{int(ts.sequence_length):,} bp"
    )

    # ── Extract haplotypes ────────────────────────────────────
    all_samples = list(ts.samples())
    n_total = len(all_samples)
    query_id = all_samples[-1]
    ref_ids = all_samples[:-1]
    n_ref = len(ref_ids)

    print(f"  {n_ref} reference haplotypes, 1 query haplotype")

    # ── True copying path ─────────────────────────────────────
    print("Computing true copying path from tree sequence...")
    true_path, variant_positions = get_true_copying_path(ts, query_id, ref_ids)
    n_sites = len(true_path)
    true_segments = count_segments(true_path)
    print(f"  {n_sites} variant sites, {true_segments} true segments")

    # Subsample if too many sites (for visualization and speed)
    max_sites = 2000
    if n_sites > max_sites:
        step = n_sites // max_sites
        site_indices = np.arange(0, n_sites, step)[:max_sites]
    else:
        site_indices = np.arange(n_sites)

    # ── Build panel and query ─────────────────────────────────
    print("Building reference panel and query files...")
    panel_rows = []
    query_alleles = []
    for var in ts.variants():
        geno = var.genotypes
        row = [int(geno[ref_ids[i]]) for i in range(n_ref)]
        panel_rows.append(row)
        query_alleles.append(int(geno[query_id]))

    # Subsample
    panel_rows = [panel_rows[i] for i in site_indices]
    query_alleles = [query_alleles[i] for i in site_indices]
    true_path_sub = true_path[site_indices]
    positions_sub = variant_positions[site_indices]
    m = len(panel_rows)
    n = n_ref

    print(f"  Using {m} sites (subsampled) × {n} reference haplotypes")

    # Write panel TSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        for row in panel_rows:
            f.write("\t".join(map(str, row)) + "\n")
        panel_path = f.name

    # Write query file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for a in query_alleles:
            f.write(f"{a}\n")
        query_path = f.name

    # ── Run lshmm-rs ──────────────────────────────────────────
    output_path = str(SCRIPT_DIR / "lshmm_painting_result.json")
    cmd = [
        str(BINARY),
        "--panel", panel_path,
        "--query", query_path,
        "--output", output_path,
        "--rho", str(RHO_LSHMM),
    ]
    print("Running lshmm-rs...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stderr)
    if result.returncode != 0:
        print("lshmm-rs failed!")
        print(result.stdout)
        sys.exit(1)

    # ── Parse output ──────────────────────────────────────────
    with open(output_path) as f:
        output = json.load(f)

    viterbi_path = np.array(output["viterbi_path"])
    posterior_path = np.array(output["posterior_path"])

    # ── Compute accuracy ──────────────────────────────────────
    # "Correct" if inferred copies from the same haplotype as true
    viterbi_acc = np.mean(viterbi_path == true_path_sub)
    posterior_acc = np.mean(posterior_path == true_path_sub)
    viterbi_segments = count_segments(viterbi_path)
    posterior_segments = count_segments(posterior_path)

    print(f"Viterbi:   accuracy={viterbi_acc:.3f}, {viterbi_segments} segments")
    print(f"Posterior: accuracy={posterior_acc:.3f}, {posterior_segments} segments")

    # ── Plot ──────────────────────────────────────────────────
    print("Plotting results...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [1, 1, 2]})
    fig.suptitle(
        f"LSHMM-RS: Haplotype Painting ({n} ref haplotypes, {m} sites)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # Color map for reference haplotypes
    cmap = plt.cm.tab20
    n_colors = max(20, n)

    # Panel A & B: Painting strips (True vs Inferred)
    for ax, path, label, acc in [
        (axes[0], true_path_sub, "True path", None),
        (axes[1], viterbi_path, f"Viterbi (acc={viterbi_acc:.1%})", viterbi_acc),
    ]:
        img = path.reshape(1, -1)
        ax.imshow(
            img,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=n_colors - 1,
            interpolation="nearest",
            extent=[positions_sub[0], positions_sub[-1], -0.5, 0.5],
        )
        ax.set_yticks([0])
        ax.set_yticklabels([label], fontsize=10)
        ax.set_xlim(positions_sub[0], positions_sub[-1])
        ax.tick_params(axis="x", labelbottom=False)

        # Mark switching points
        for i in range(1, len(path)):
            if path[i] != path[i - 1]:
                ax.axvline(
                    (positions_sub[i - 1] + positions_sub[i]) / 2,
                    color="black", lw=0.3, alpha=0.5,
                )

    axes[1].set_xlabel("Genomic position (bp)")
    axes[1].tick_params(axis="x", labelbottom=True)

    # Panel C: Running accuracy in windows
    ax = axes[2]
    window_size = max(1, m // 50)
    match = (viterbi_path == true_path_sub).astype(float)
    if len(match) >= window_size:
        kernel = np.ones(window_size) / window_size
        running_acc = np.convolve(match, kernel, mode="valid")
        x_acc = positions_sub[window_size // 2 : window_size // 2 + len(running_acc)]
        ax.fill_between(x_acc, 0, running_acc, alpha=0.3, color="#2166AC")
        ax.plot(x_acc, running_acc, color="#2166AC", lw=1.5, label="Viterbi")

    match_post = (posterior_path == true_path_sub).astype(float)
    if len(match_post) >= window_size:
        running_acc_post = np.convolve(match_post, kernel, mode="valid")
        ax.plot(x_acc, running_acc_post, color="#B2182B", lw=1.5,
                alpha=0.8, label="Posterior")

    ax.axhline(1.0, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.set_xlabel("Genomic position (bp)")
    ax.set_ylabel("Local accuracy")
    ax.set_title(
        f"Site-by-site accuracy (window={window_size} sites)"
    )
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(positions_sub[0], positions_sub[-1])
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Add stats text
    stats_text = (
        f"Forward LL: {output['forward_ll']:.2f}\n"
        f"Viterbi LL: {output['viterbi_ll']:.2f}\n"
        f"True segments: {true_segments}\n"
        f"Inferred segments: {viterbi_segments}"
    )
    ax.text(
        0.01, 0.97, stats_text, transform=ax.transAxes,
        fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = SCRIPT_DIR / "lshmm_painting_result.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    # Clean up
    Path(panel_path).unlink(missing_ok=True)
    Path(query_path).unlink(missing_ok=True)
    print("Done!")


if __name__ == "__main__":
    main()
