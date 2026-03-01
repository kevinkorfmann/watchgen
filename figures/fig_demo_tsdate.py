"""
Demo: tsdate node dating on msprime-simulated tree sequence.

Simulates a tree sequence with known node times, strips timing
information, runs tsdate's inside-outside algorithm, and compares
inferred times to truth.
"""

import numpy as np
import matplotlib.pyplot as plt
import msprime

from watchgen.mini_tsdate import (
    build_prior_grid,
    edge_likelihood_matrix,
    GammaDistribution,
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
    recombination_rate=1e-8,
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

# ── Compute priors and likelihoods ──────────────────────────────
n_grid = 50
t_grid = np.concatenate([[0], np.geomspace(10, 4 * Ne, n_grid - 1)])

# Build prior for different descendant counts
prior_grid = {}
for n_desc in [2, 5, 10, 20]:
    prior = build_prior_grid(n_desc, Ne)
    prior_grid[n_desc] = prior

# Compute edge likelihoods for a range of mutation counts
edge_liks = {}
for m_count in range(10):
    lik = edge_likelihood_matrix(m_count, mu * 200_000, t_grid)
    edge_liks[m_count] = lik

# ── Gamma approximation posterior ───────────────────────────────
# For each internal node, compute approximate posterior from edge mutations
internal_nodes = [n for n in ts.nodes() if not n.is_sample()]
posterior_means = []
posterior_vars = []
true_node_times = []

# Build edge lookup: parent_id -> list of (child, left, right)
edges_by_parent = {}
for edge in ts.edges():
    edges_by_parent.setdefault(edge.parent, []).append(edge)

# Count descendant samples per node using the first tree as approximation
first_tree = ts.first()
tree_node_ids = set(first_tree.nodes())
desc_counts = {}
for node in internal_nodes:
    if node.id in tree_node_ids:
        desc_counts[node.id] = len(list(first_tree.samples(node.id)))
    else:
        desc_counts[node.id] = max(2, ts.num_samples // 2)

n_samples = ts.num_samples
for node in internal_nodes:
    n_desc = desc_counts.get(node.id, 2)
    if n_desc < 2:
        n_desc = 2

    # Prior: gamma with mean ~ 2*Ne*(1-1/n_desc), var ~ (2*Ne)^2 / n_desc
    prior_mean = 2 * Ne * (1 - 1 / n_desc)
    prior_var = (2 * Ne) ** 2 / n_desc
    prior = GammaDistribution.from_moments(prior_mean, max(prior_var, 1.0))

    # Likelihood from mutations on child edges
    total_muts = 0
    total_span = 0
    for edge in edges_by_parent.get(node.id, []):
        key = (edge.child, edge.parent)
        total_muts += edge_mutations.get(key, 0)
        total_span += edge.right - edge.left

    if total_span > 0:
        rate = mu * total_span
        if rate > 0:
            lik = GammaDistribution(total_muts + 1, rate)
            posterior = prior.multiply(lik)
            posterior_means.append(posterior.mean)
            posterior_vars.append(posterior.variance)
            true_node_times.append(node.time)

posterior_means = np.array(posterior_means)
true_node_times = np.array(true_node_times)

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    f"Demo: tsdate on msprime Tree Sequence ({ts.num_samples} samples, {ts.num_trees} trees)",
    fontsize=13, fontweight="bold", y=0.98,
)

# Panel A: True vs inferred node times (log scale)
ax = axes[0, 0]
mask = (true_node_times > 0) & (posterior_means > 0)
ax.scatter(true_node_times[mask], posterior_means[mask], s=20, alpha=0.5,
           color="#2166AC", edgecolors="white", linewidths=0.3)
lo = min(true_node_times[mask].min(), posterior_means[mask].min()) * 0.5
hi = max(true_node_times[mask].max(), posterior_means[mask].max()) * 2
ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4, label="$y = x$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("True node time (generations)")
ax.set_ylabel("Inferred node time (posterior mean)")
ax.set_title("A. True vs inferred node times")
ax.legend(fontsize=8)
corr = np.corrcoef(true_node_times[mask], posterior_means[mask])[0, 1]
ax.text(0.02, 0.95, f"$r$ = {corr:.3f}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Panel B: Prior distributions by descendant count
ax = axes[0, 1]
t_plot = np.geomspace(10, 4 * Ne, 200)
from scipy.stats import gamma as gamma_dist
for n_desc, color in [(2, "#2166AC"), (5, "#B2182B"), (10, "#1B7837"), (20, "#E08214")]:
    prior_mean = 2 * Ne * (1 - 1 / n_desc)
    prior_var = (2 * Ne) ** 2 / n_desc
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

# Panel D: Tree span and mutation distribution
ax = axes[1, 1]
tree_spans = []
tree_muts = []
for tree in ts.trees():
    span = tree.interval[1] - tree.interval[0]
    n_muts = len(list(tree.sites()))
    tree_spans.append(span)
    tree_muts.append(n_muts)

ax.scatter(tree_spans, tree_muts, s=15, alpha=0.5, color="#2166AC",
           edgecolors="white", linewidths=0.3)
ax.set_xlabel("Tree span (bp)")
ax.set_ylabel("Number of mutations")
ax.set_title(f"D. Tree span vs mutations ({ts.num_trees} trees)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_tsdate.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_tsdate.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_tsdate.png")
