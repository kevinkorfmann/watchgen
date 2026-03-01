"""
Demo: Threads segment dating on msprime-simulated IBD segments.

Simulates a tree sequence with msprime, identifies IBD segments,
and uses Threads dating estimators to infer segment ages.
"""

import numpy as np
import matplotlib.pyplot as plt
import msprime

from watchgen.mini_threads import (
    mle_recombination_only,
    mle_recombination_and_mutations,
    bayesian_recombination_only,
    bayesian_full,
    piecewise_constant_bayesian_full,
)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "font.family": "sans-serif",
})

np.random.seed(2024)

# ── Simulate data with msprime ──────────────────────────────────
Ne = 10_000
mu = 1.25e-8
rho_rate = 1e-8
L = 1_000_000  # 1 Mb

ts = msprime.simulate(
    sample_size=20,
    Ne=Ne,
    length=L,
    recombination_rate=rho_rate,
    mutation_rate=mu,
    random_seed=2024,
)

# Extract IBD-like segments from consecutive trees
# For each tree, find pairs that coalesce within 500 generations
young_threshold = 100_000  # generations — keep segments up to 100k gen
segments_data = []  # (length_cM, n_mutations, true_age)

for tree in ts.trees():
    span_bp = tree.interval[1] - tree.interval[0]
    span_cM = span_bp * rho_rate * 100  # approximate cM
    for pair in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        mrca = tree.mrca(pair[0], pair[1])
        if mrca != -1:
            age = tree.time(mrca)
            if age > young_threshold:
                continue
            # Count mutations on branches leading to this MRCA
            n_muts = 0
            for node in pair:
                u = node
                while u != mrca:
                    parent = tree.parent(u)
                    branch_len = tree.time(parent) - tree.time(u)
                    n_muts += np.random.poisson(mu * span_bp * branch_len / age if age > 0 else 0)
                    u = parent
            segments_data.append((span_cM, n_muts, age))

# ── Apply Threads dating estimators ─────────────────────────────
gamma = 1.0 / Ne
mu_measure = 2 * mu * L / 100  # scale

results = []
for span_cM, n_muts, true_age in segments_data[:200]:
    rho_seg = 2 * 0.01 * span_cM
    mu_seg = 2 * mu * span_cM / (rho_rate * 100) if span_cM > 0 else 0.001
    if rho_seg <= 0:
        continue
    t_mle = mle_recombination_only(rho_seg)
    t_bayes = bayesian_recombination_only(rho_seg, gamma)
    t_full = bayesian_full(n_muts, rho_seg, mu_seg, gamma)
    results.append((true_age, t_mle, t_bayes, t_full, span_cM, n_muts))

results = np.array(results)

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    f"Demo: Threads Dating on msprime-simulated IBD Segments ({ts.num_samples} samples, {L/1e6:.0f} Mb)",
    fontsize=13, fontweight="bold", y=0.98,
)

# Panel A: True age vs MLE estimate (log scale)
ax = axes[0, 0]
mask = (results[:, 0] > 0) & (results[:, 1] > 0)
ax.scatter(results[mask, 0], results[mask, 1], s=15, alpha=0.5, color="#2166AC",
           edgecolors="white", linewidths=0.3)
all_vals = np.concatenate([results[mask, 0], results[mask, 1]])
lo, hi = all_vals.min() * 0.5, all_vals.max() * 2
ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4, label="$y=x$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("True segment age (generations)")
ax.set_ylabel("MLE estimated age")
ax.set_title("A. MLE (recombination only)")
ax.legend(fontsize=8)

# Panel B: True age vs Bayesian estimate (log scale)
ax = axes[0, 1]
mask2 = (results[:, 0] > 0) & (results[:, 3] > 0)
ax.scatter(results[mask2, 0], results[mask2, 3], s=15, alpha=0.5, color="#B2182B",
           edgecolors="white", linewidths=0.3)
ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4, label="$y=x$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("True segment age (generations)")
ax.set_ylabel("Bayesian estimated age")
ax.set_title(f"B. Bayesian (recomb + mutations, $N_e$={Ne:,})")
ax.legend(fontsize=8)

# Panel C: Residuals by segment length
ax = axes[1, 0]
residuals_mle = results[:, 1] - results[:, 0]
residuals_bayes = results[:, 3] - results[:, 0]
ax.scatter(results[:, 4], residuals_mle, s=12, alpha=0.4, color="#2166AC",
           label="MLE residual")
ax.scatter(results[:, 4], residuals_bayes, s=12, alpha=0.4, color="#B2182B",
           label="Bayesian residual")
ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
ax.set_xlabel("Segment length (cM)")
ax.set_ylabel("Estimated $-$ True age (gen)")
ax.set_title("C. Dating residuals by segment length")
ax.legend(fontsize=8)

# Panel D: Age estimation under different demographic models
ax = axes[1, 1]
rho_fixed = 2 * 0.01 * 1.0
mu_fixed = 2 * mu * 1e6
m_vals = np.arange(0, 20)

t_const = [bayesian_full(m, rho_fixed, mu_fixed, 1.0/Ne) for m in m_vals]
t_bottle = [piecewise_constant_bayesian_full(rho_fixed, mu_fixed, m,
            [0.0, 500.0], [1.0/20_000, 1.0/2_000]) for m in m_vals]
t_expand = [piecewise_constant_bayesian_full(rho_fixed, mu_fixed, m,
            [0.0, 200.0], [1.0/2_000, 1.0/50_000]) for m in m_vals]

ax.plot(m_vals, t_const, "o-", color="#1B7837", ms=4, lw=1.5,
        label=f"Constant ($N_e$={Ne:,})")
ax.plot(m_vals, t_bottle, "s-", color="#7B3294", ms=4, lw=1.5,
        label="Expansion (20k$\\to$2k)")
ax.plot(m_vals, t_expand, "^-", color="#E08214", ms=4, lw=1.5,
        label="Bottleneck (2k$\\to$50k)")

ax.set_xlabel("Number of mutations (m)")
ax.set_ylabel("Estimated age (generations)")
ax.set_title("D. Demographic model effect (1 cM segment)")
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_threads.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_threads.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_threads.png")
