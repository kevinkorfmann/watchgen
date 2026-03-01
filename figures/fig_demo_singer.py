"""
Demo: SINGER ARG sampler on msprime-simulated haplotype data.

Simulates a small genomic region, computes SINGER's transition
and emission probabilities, and shows the inferred ARG structure.
"""

import numpy as np
import matplotlib.pyplot as plt
import msprime

from watchgen.mini_singer import (
    joining_probability_exact,
    emission_probability,
    time_transition_matrix,
    psmc_transition_density,
)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "font.family": "sans-serif",
})

np.random.seed(2024)

# ── Simulate data with msprime ──────────────────────────────────
Ne = 10_000
mu = 1.25e-8
rho = 1e-8

ts = msprime.simulate(
    sample_size=8,
    Ne=Ne,
    length=50_000,
    recombination_rate=rho,
    mutation_rate=mu,
    random_seed=2024,
)

# ── Compute SINGER quantities ───────────────────────────────────
# Build time grid
n_times = 30
t_max = 4 * Ne
times = np.concatenate([[0], np.geomspace(100, t_max, n_times - 1)])
boundaries = times

# PSMC-like transition densities for different time values
t_vals = np.geomspace(100, t_max, 100)

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    f"Demo: SINGER on msprime-simulated Data ({ts.num_samples} haps, {ts.num_sites} sites)",
    fontsize=13, fontweight="bold", y=0.98,
)

# Panel A: True tree sequence from msprime
ax = axes[0, 0]
breakpoints = list(ts.breakpoints())
n_trees = ts.num_trees
tree_spans = [breakpoints[i+1] - breakpoints[i] for i in range(n_trees)]
colors = plt.cm.Set2(np.linspace(0, 1, min(n_trees, 20)))

for i, (span, bp) in enumerate(zip(tree_spans[:20], breakpoints[:20])):
    ax.barh(0, span, left=bp, height=0.5,
            color=colors[i % len(colors)],
            edgecolor="white", linewidth=0.3)
    if span > 2000:
        ax.text(bp + span/2, 0, f"T{i+1}", ha="center", va="center",
                fontsize=6, fontweight="bold")

ax.set_xlabel("Genomic position (bp)")
ax.set_yticks([0])
ax.set_yticklabels(["Trees"])
ax.set_title(f"A. True marginal trees ({n_trees} trees, {ts.num_sites} mutations)")
ax.set_xlim(0, ts.sequence_length)

# Panel B: Emission probability P(allele | tau, branch)
ax = axes[0, 1]
theta_site = 4 * Ne * mu
tau_vals = np.geomspace(100, t_max, 200)
branch_lower = 0.0

for allele_new, allele_at, color, label in [
    (0, 0, "#2166AC", "Match (0,0)"),
    (1, 0, "#B2182B", "Mismatch (1,0)"),
    (0, 1, "#1B7837", "Mismatch (0,1)"),
    (1, 1, "#E08214", "Match (1,1)"),
]:
    probs = []
    for tau in tau_vals:
        p = emission_probability(allele_new, allele_at, tau, 0.0, tau, theta_site)
        probs.append(p)
    ax.plot(tau_vals, probs, lw=1.5, color=color, label=label)

ax.set_xscale("log")
ax.set_xlabel("Branch length $\\tau$ (generations)")
ax.set_ylabel("Emission probability")
ax.set_title("B. SINGER emission probabilities")
ax.legend(fontsize=7)

# Panel C: PSMC transition density
ax = axes[1, 0]
s_vals = [500, 2000, 5000, 15000]
colors_c = ["#2166AC", "#B2182B", "#1B7837", "#E08214"]

for s_val, color in zip(s_vals, colors_c):
    densities = [psmc_transition_density(s_val, t, Ne) for t in t_vals]
    ax.plot(t_vals, densities, lw=1.5, color=color,
            label=f"$s$ = {s_val:,} gen")

ax.set_xscale("log")
ax.set_xlabel("Time $t$ (generations)")
ax.set_ylabel("Transition density")
ax.set_title("C. PSMC-like transition density $P(t' | s)$")
ax.legend(fontsize=7)

# Panel D: Tree statistics from the true tree sequence
ax = axes[1, 1]
tree_heights = []
tree_total_branches = []
for tree in ts.trees():
    roots = tree.roots
    if len(roots) == 1:
        tree_heights.append(tree.time(roots[0]))
        tree_total_branches.append(tree.total_branch_length)

ax.scatter(tree_heights, tree_total_branches, s=20, alpha=0.6,
           color="#2166AC", edgecolors="white", linewidths=0.3)
ax.set_xlabel("Tree height (generations)")
ax.set_ylabel("Total branch length")
ax.set_title("D. Tree height vs branch length (true trees)")

if tree_heights:
    ax.text(0.02, 0.95,
            f"Mean height: {np.mean(tree_heights):,.0f} gen\n"
            f"Mean branch len: {np.mean(tree_total_branches):,.0f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_singer.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_singer.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_singer.png")
