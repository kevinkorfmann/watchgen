"""Source-grounded mini-tsdate kernel checks on one simulated genealogy.

This figure deliberately does not claim to run the complete tsdate pipeline.
It compares conditional-coalescent prior means with simulated node ages and
checks the Poisson edge clock on a single tree.
"""

import matplotlib.pyplot as plt
import msprime
import numpy as np

from watchgen.mini_tsdate import (
    GammaDistribution,
    conditional_coalescent_moments,
)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "font.family": "sans-serif",
})

np.random.seed(2024)

# ── Simulate with msprime ───────────────────────────────────────
Ne = 10_000
mu = 1.25e-8

ts = msprime.simulate(
    sample_size=20,
    Ne=Ne,
    length=200_000,  # 200 kb
    recombination_rate=0,
    mutation_rate=mu,
    random_seed=2024,
)

# ── Extract true node times and mutation counts per edge ────────
true_times = {}
for node in ts.nodes():
    true_times[node.id] = node.time

# Count mutations per edge
edge_mutations = {}
for tree in ts.trees():
    for site in tree.sites():
        for mut in site.mutations:
            parent_node = tree.parent(mut.node)
            edge_key = (mut.node, parent_node)
            edge_mutations[edge_key] = edge_mutations.get(edge_key, 0) + 1

# Exact conditional-coalescent moments for this sample size. Standard
# coalescent units convert to generations by multiplying by 2*Ne.
moments = conditional_coalescent_moments(ts.num_samples, Ne=2 * Ne)

# ── Conditional-coalescent prior means ──────────────────────────
internal_nodes = [n for n in ts.nodes() if not n.is_sample()]
prior_means = []
true_node_times = []
tree = ts.first()
for node in internal_nodes:
    k = len(list(tree.samples(node.id)))
    prior_means.append(moments[k][0])
    true_node_times.append(node.time)

prior_means = np.array(prior_means)
true_node_times = np.array(true_node_times)

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    f"Mini-tsdate kernels on an msprime genealogy ({ts.num_samples} samples)",
    fontsize=13, fontweight="bold", y=0.98,
)

# Panel A: simulated node ages versus conditional-coalescent prior means
ax = axes[0, 0]
mask = (true_node_times > 0) & (prior_means > 0)
ax.scatter(true_node_times[mask], prior_means[mask], s=20, alpha=0.5,
           color="#2166AC", edgecolors="white", linewidths=0.3)
lo = min(true_node_times[mask].min(), prior_means[mask].min()) * 0.5
hi = max(true_node_times[mask].max(), prior_means[mask].max()) * 2
ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4, label="$y = x$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("True node time (generations)")
ax.set_ylabel("Conditional-coalescent prior mean")
ax.set_title("A. Simulated age vs prior mean")
ax.legend(fontsize=8)
corr = np.corrcoef(true_node_times[mask], prior_means[mask])[0, 1]
ax.text(0.02, 0.95, f"$r$ = {corr:.3f}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "gray", "alpha": 0.8})

# Panel B: Prior distributions by descendant count
ax = axes[0, 1]
t_plot = np.geomspace(10, 4 * Ne, 200)
from scipy.stats import gamma as gamma_dist

for n_desc, color in [(2, "#2166AC"), (5, "#B2182B"), (10, "#1B7837"), (20, "#E08214")]:
    prior_mean, prior_var = moments[n_desc]
    prior = GammaDistribution.from_moments(prior_mean, max(prior_var, 1.0))
    pdf = gamma_dist.pdf(t_plot, a=prior.alpha, scale=1.0/prior.beta)
    ax.plot(t_plot, pdf, lw=2, color=color, label=f"$n_d$ = {n_desc}")

ax.set_xscale("log")
ax.set_xlabel("Time (generations)")
ax.set_ylabel("Prior density")
ax.set_title("B. Coalescent prior by descendant count")
ax.legend(fontsize=8)

# Panel C: Edge likelihood for different mutation counts
ax = axes[1, 0]
t_edge = np.geomspace(10, 4 * Ne, 200)
rate = mu * 200_000
for m_count, color in [(0, "#636363"), (1, "#2166AC"), (3, "#B2182B"),
                        (5, "#1B7837"), (10, "#E08214")]:
    # Poisson likelihood: P(m | t) = (rate*t)^m * exp(-rate*t) / m!
    from scipy.stats import poisson
    lik = [poisson.pmf(m_count, rate * t) for t in t_edge]
    ax.plot(t_edge, lik, lw=1.5, color=color, label=f"$m$ = {m_count}")

ax.set_xscale("log")
ax.set_xlabel("Edge span (generations)")
ax.set_ylabel("Likelihood $P(m | t)$")
ax.set_title("C. Mutation likelihood per edge")
ax.legend(fontsize=7)

# Panel D: expected versus observed mutations on each edge
ax = axes[1, 1]
expected_muts = []
observed_muts = []
for edge in ts.edges():
    span = edge.right - edge.left
    duration = true_times[edge.parent] - true_times[edge.child]
    expected_muts.append(mu * span * duration)
    observed_muts.append(edge_mutations.get((edge.child, edge.parent), 0))

ax.scatter(expected_muts, observed_muts, s=18, alpha=0.6, color="#2166AC",
           edgecolors="white", linewidths=0.3)
hi = max(max(expected_muts), max(observed_muts), 1)
ax.plot([0, hi], [0, hi], "k--", lw=1, alpha=0.4)
ax.set_xlabel(r"Expected mutations $\mu\ell\Delta t$")
ax.set_ylabel("Observed mutations")
ax.set_title("D. Poisson clock by edge")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_tsdate.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_tsdate.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_tsdate.png")
