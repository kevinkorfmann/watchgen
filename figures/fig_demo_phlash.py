"""Demo: source-aligned PHLASH ingredients on msprime-simulated data.

This is an explanatory figure, not a PHLASH fit. It shows the observed AFS,
the normalized AFS shape score, endpoint-randomised geometric grids, and the
coalescent-time masses induced by those grids.
"""

import matplotlib.pyplot as plt
import msprime
import numpy as np

from watchgen.mini_phlash import (
    afs_log_score,
    coalescence_probabilities,
    logarithmic_grid,
)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "font.family": "sans-serif",
    }
)

rng = np.random.default_rng(2024)

# Simulate an unfolded AFS from a constant-size population.
population_size = 10_000
mutation_rate = 1.25e-8
n_haplotypes = 30
sequence_length = 500_000

ts = msprime.simulate(
    sample_size=n_haplotypes,
    Ne=population_size,
    length=sequence_length,
    mutation_rate=mutation_rate,
    random_seed=2024,
)
genotypes = ts.genotype_matrix()
derived_counts = genotypes.sum(axis=1)
observed_afs = np.bincount(
    derived_counts, minlength=n_haplotypes + 1
)[1:n_haplotypes]

# PHLASH normalizes the expected AFS before scoring it. A one-parameter shape
# family makes that distinction visible without pretending the AFS term alone
# identifies an overall population-size scale.
frequency = np.arange(1, n_haplotypes)
shape_exponents = np.linspace(0.55, 1.45, 100)
afs_scores = np.array(
    [afs_log_score(observed_afs, frequency ** (-power)) for power in shape_exponents]
)

# Draw endpoint parameters, then geometrically space the interior points exactly
# as the released model does. PHLASH uses 16 intervals in its optimized kernel.
n_grids = 12
grids = []
for _ in range(n_grids):
    log_t1 = rng.normal(np.log(1e-4), 0.45)
    log_tM = rng.normal(np.log(15.0), 0.25)
    grids.append(logarithmic_grid(log_t1, log_tM, intervals=16))

# A shared constant coalescent rate isolates the effect of the different grids on
# interval probability masses. The last mass is the open-ended tail.
grid_masses = [coalescence_probabilities(grid, np.ones(16)) for grid in grids]

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    "PHLASH ingredients on simulated sequence data "
    f"({n_haplotypes} haplotypes, {sequence_length / 1e3:.0f} kb)",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)

# Panel A: observed AFS.
ax = axes[0, 0]
show = min(20, len(observed_afs))
ax.bar(
    frequency[:show],
    observed_afs[:show],
    color="#2166AC",
    alpha=0.8,
    edgecolor="white",
    linewidth=0.5,
)
ax.set_xlabel("Derived allele count $k$")
ax.set_ylabel("Number of sites")
ax.set_title(f"A. Observed AFS ({ts.num_sites} segregating sites)")

# Panel B: normalized AFS shape score.
ax = axes[0, 1]
relative_scores = afs_scores - afs_scores.max()
best = int(np.argmax(afs_scores))
ax.plot(shape_exponents, relative_scores, color="#2166AC", lw=2)
ax.axvline(
    shape_exponents[best],
    color="#B2182B",
    ls="--",
    lw=1.5,
    label=rf"Best shape $p={shape_exponents[best]:.2f}$",
)
ax.axvline(1.0, color="#1B7837", ls=":", lw=1.5, label=r"Neutral $1/k$")
ax.set_xlabel(r"Shape exponent $p$ in $e_k \propto k^{-p}$")
ax.set_ylabel("AFS log score relative to maximum")
ax.set_title("B. Normalized AFS term compares shape")
ax.legend(fontsize=8)

# Panel C: endpoint-randomised geometric grids.
ax = axes[1, 0]
for index, grid in enumerate(grids):
    color = plt.cm.viridis(index / (n_grids - 1))
    ax.hlines(index, grid[1], grid[-1], color=color, lw=0.6, alpha=0.55)
    ax.scatter(grid[1:], np.full(15, index), marker="|", s=45, color=color)
ax.set_xlabel("Time (coalescent units)")
ax.set_ylabel("Particle grid")
ax.set_xscale("log")
ax.set_title("C. Random endpoints; geometric interior ($M=16$)")

# Panel D: exact interval masses on the sampled grids.
ax = axes[1, 1]
for index, (grid, masses) in enumerate(zip(grids[:6], grid_masses[:6])):
    color = plt.cm.Set2(index / 6)
    ax.step(
        grid[1:],
        np.cumsum(masses[:-1]),
        where="post",
        color=color,
        lw=1.6,
        label=f"Grid {index + 1}",
    )
ax.set_xlabel("Time (coalescent units)")
ax.set_ylabel("Cumulative finite-interval mass")
ax.set_xscale("log")
ax.set_ylim(0, 1.02)
ax.set_title("D. Integrated-hazard mass on each grid")
ax.legend(fontsize=7, ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_phlash.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_phlash.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_phlash.png and figures/fig_demo_phlash.pdf")
