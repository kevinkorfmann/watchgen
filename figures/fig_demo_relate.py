"""
Demo: Relate genealogy inference on msprime-simulated haplotype data.

Simulates haplotypes with msprime, runs Relate's asymmetric painting
for a focal SNP, computes pairwise distances, and compares to truth.
"""

import numpy as np
import matplotlib.pyplot as plt
import msprime

from watchgen.mini_relate import (
    forward_backward_relate,
    compute_distance_matrix,
    build_tree,
)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "font.family": "sans-serif",
})

np.random.seed(2024)

# ── Simulate with msprime ───────────────────────────────────────
Ne = 10_000
mu = 1.25e-8
rho = 1e-8
n_haps = 12

ts = msprime.simulate(
    sample_size=n_haps, Ne=Ne, length=200_000,
    recombination_rate=rho, mutation_rate=mu,
    random_seed=2024,
)

# Extract haplotype matrix
G = ts.genotype_matrix()  # sites x samples
positions = np.array([v.position for v in ts.variants()])
n_sites, n = G.shape

# ── Run Relate painting for a focal SNP ─────────────────────────
focal_snp = n_sites // 2  # middle of the region
D_mat = compute_distance_matrix(G.T, positions, recomb_rate=rho, mu=mu,
                                focal_snp=focal_snp)

# Build tree from distance matrix
tree_result = build_tree(D_mat, n_haps)

# ── True pairwise divergence ────────────────────────────────────
true_div = np.zeros((n, n))
for tree_obj in ts.trees():
    span = tree_obj.interval[1] - tree_obj.interval[0]
    for i in range(n):
        for j in range(i + 1, n):
            mrca = tree_obj.mrca(i, j)
            if mrca != -1:
                t = tree_obj.time(mrca)
                true_div[i, j] += t * span
                true_div[j, i] += t * span
true_div /= ts.sequence_length

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    f"Demo: Relate on msprime Data ({n_haps} haplotypes, {n_sites} sites, 200 kb)",
    fontsize=13, fontweight="bold", y=0.98,
)

# Panel A: Input haplotype matrix
ax = axes[0, 0]
show_sites = min(80, n_sites)
im = ax.imshow(G[:show_sites].T, aspect="auto", cmap="YlOrRd",
               interpolation="nearest")
ax.axvline(min(focal_snp, show_sites - 1), color="#1B7837", lw=2, ls="--",
           label=f"Focal SNP ({focal_snp})")
ax.set_xlabel(f"Variant site (first {show_sites} of {n_sites})")
ax.set_ylabel("Haplotype")
ax.set_title("A. Input haplotypes from VCF")
ax.legend(fontsize=7)
plt.colorbar(im, ax=ax, label="Allele", shrink=0.8, ticks=[0, 1])

# Panel B: Inferred distance matrix
ax = axes[0, 1]
im = ax.imshow(D_mat, aspect="auto", cmap="viridis", interpolation="nearest")
ax.set_xlabel("Haplotype")
ax.set_ylabel("Haplotype")
ax.set_title("B. Relate-inferred pairwise distances")
plt.colorbar(im, ax=ax, label="Distance", shrink=0.8)

# Panel C: True divergence matrix
ax = axes[1, 0]
im = ax.imshow(true_div, aspect="auto", cmap="viridis", interpolation="nearest")
ax.set_xlabel("Haplotype")
ax.set_ylabel("Haplotype")
ax.set_title("C. True pairwise divergence (msprime)")
plt.colorbar(im, ax=ax, label="Mean TMRCA (gen)", shrink=0.8)

# Panel D: True vs inferred distance scatter
ax = axes[1, 1]
true_flat = true_div[np.triu_indices(n, k=1)]
inf_flat = D_mat[np.triu_indices(n, k=1)]

true_norm = (true_flat - true_flat.min()) / (true_flat.max() - true_flat.min() + 1e-10)
inf_norm = (inf_flat - inf_flat.min()) / (inf_flat.max() - inf_flat.min() + 1e-10)

ax.scatter(true_norm, inf_norm, s=40, alpha=0.6, color="#2166AC",
           edgecolors="white", linewidths=0.3)
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="$y = x$")
ax.set_xlabel("True divergence (normalized)")
ax.set_ylabel("Inferred distance (normalized)")
ax.set_title("D. True vs inferred distances")
corr = np.corrcoef(true_norm, inf_norm)[0, 1]
ax.text(0.02, 0.95, f"$r$ = {corr:.3f}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_relate.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_relate.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_relate.png")
