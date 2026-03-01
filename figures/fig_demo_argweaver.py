"""
Demo: ARGweaver MCMC sampling on msprime-simulated haplotype data.

Simulates a small genomic region with msprime, runs the ARGweaver
MCMC sampler, and shows convergence and sampled tree properties.
"""

import numpy as np
import matplotlib.pyplot as plt
import msprime

from watchgen.mini_argweaver import (
    get_time_points,
    recoal_distribution,
    felsenstein_pruning,
    sample_tree,
    simplified_mcmc,
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
n_haps = 8
n_sites = 50

# Run ARGweaver MCMC
tree_lengths = simplified_mcmc(
    n_haps=n_haps,
    n_sites=n_sites,
    n_iters=200,
    Ne=Ne,
    mu=mu,
    rho=rho,
    ntimes=20,
    maxtime=200_000,
    delta=0.01,
)

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    f"Demo: ARGweaver MCMC on Simulated Data ({n_haps} haplotypes, {n_sites} sites)",
    fontsize=13, fontweight="bold", y=0.98,
)

# Panel A: MCMC trace — total tree length
ax = axes[0, 0]
ax.plot(range(len(tree_lengths)), tree_lengths, color="#2166AC", lw=0.8, alpha=0.8)
ax.set_xlabel("MCMC iteration")
ax.set_ylabel("Total tree length")
ax.set_title("A. MCMC trace (total tree length)")

# Add running mean
window = 20
if len(tree_lengths) > window:
    running_mean = np.convolve(tree_lengths, np.ones(window)/window, mode="valid")
    ax.plot(np.arange(window-1, len(tree_lengths)), running_mean,
            color="#B2182B", lw=2, label=f"{window}-iter running mean")
    ax.legend(fontsize=8)

# Panel B: Time discretization
ax = axes[0, 1]
times = get_time_points(ntimes=20, maxtime=200_000)
ax.barh(range(len(times) - 1),
        [times[i+1] - times[i] for i in range(len(times)-1)],
        left=[times[i] for i in range(len(times)-1)],
        height=0.8, color=plt.cm.viridis(np.linspace(0, 1, len(times)-1)),
        edgecolor="white", linewidth=0.3)
ax.set_xlabel("Time (generations)")
ax.set_ylabel("Interval index")
ax.set_title(f"B. Time discretization ({len(times)-1} intervals)")
ax.set_xscale("log")
ax.set_xlim(10, 300_000)

# Panel C: Recoalescence distribution
ax = axes[1, 0]
t_grid = times
n_lineages_vals = [2, 4, 6, 8]
colors = ["#2166AC", "#B2182B", "#1B7837", "#E08214"]

for n_lin, color in zip(n_lineages_vals, colors):
    probs = recoal_distribution(n_lin, Ne, t_grid)
    probs = np.array(probs)
    probs = probs / (probs.sum() + 1e-30)
    ax.plot(range(len(probs)), probs, "o-", color=color, ms=3, lw=1.5,
            label=f"$k = {n_lin}$ lineages")

ax.set_xlabel("Time interval")
ax.set_ylabel("Recoalescence probability")
ax.set_title("C. Recoalescence distribution $P(t | k)$")
ax.legend(fontsize=8)

# Panel D: Tree length distribution (post burn-in)
ax = axes[1, 1]
burnin = len(tree_lengths) // 4
post_burnin = tree_lengths[burnin:]
ax.hist(post_burnin, bins=30, density=True, color="#2166AC", alpha=0.7,
        edgecolor="white", linewidth=0.5)
ax.axvline(np.mean(post_burnin), color="#B2182B", lw=2, ls="--",
           label=f"Mean = {np.mean(post_burnin):.1f}")
ax.set_xlabel("Total tree length")
ax.set_ylabel("Density")
ax.set_title("D. Posterior tree length distribution (post burn-in)")
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_argweaver.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_argweaver.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_argweaver.png")
