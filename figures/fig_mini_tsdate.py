"""
Figure: tsdate algorithm -- four-panel overview.

Panel A: Coalescent prior -- conditional coalescent moments for different
         sample sizes, showing gamma priors on node age by descendant count.
Panel B: Edge likelihood -- Poisson mutation likelihood as a function of
         parent time for edges with different mutation counts.
Panel C: Inside-outside posterior -- posterior distributions over a time grid
         for nodes with different roles in a small tree.
Panel D: Variational gamma -- GammaDistribution multiply/divide operations
         showing prior -> posterior evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist

from watchgen.mini_tsdate import (
    conditional_coalescent_moments,
    gamma_params_from_moments,
    build_prior_grid,
    edge_likelihood,
    make_time_grid,
    edge_likelihood_matrix,
    compute_posteriors,
    posterior_mean,
    inside_pass_logspace,
    GammaDistribution,
)

# -- Style ---------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "font.family": "sans-serif",
})

C_BLUE = "#2166AC"
C_RED = "#B2182B"
C_GREEN = "#1B7837"
C_ORANGE = "#E08214"
C_PURPLE = "#7B3294"
C_GREY = "#636363"
C_TEAL = "#01665E"

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle(
    "tsdate: Bayesian Node Dating for Tree Sequences",
    fontsize=14, fontweight="bold", y=0.98,
)

# =================================================================
# Panel A -- Coalescent Prior: gamma priors by descendant count
# =================================================================
ax = axes[0, 0]

sample_sizes = [10, 20, 50]
colors_n = [C_BLUE, C_RED, C_GREEN]
t_vals = np.linspace(0.001, 3.5, 500)

for n, col in zip(sample_sizes, colors_n):
    moments = conditional_coalescent_moments(n)
    # Plot gamma prior densities for a few representative k values
    k_values = sorted(moments.keys())
    # Pick k = 2 (smallest subtree), a middle k, and k = n (root)
    k_picks = [2, k_values[len(k_values) // 2], n]
    linestyles = ["-", "--", ":"]

    for k, ls in zip(k_picks, linestyles):
        mean, var = moments[k]
        alpha, beta = gamma_params_from_moments(mean, var)
        if alpha > 0 and beta > 0:
            pdf_vals = gamma_dist.pdf(t_vals, a=alpha, scale=1.0 / beta)
            label = f"$n={n},\\; k={k}$" if n == sample_sizes[0] or k == 2 else None
            if k == 2:
                label = f"$n={n},\\; k=2$"
            elif k == n:
                label = f"$n={n},\\; k={n}$ (root)"
            else:
                label = f"$n={n},\\; k={k}$"
            ax.plot(t_vals, pdf_vals, color=col, ls=ls, lw=1.8, label=label)

ax.set_xlabel("Node age $t$ (coalescent units)")
ax.set_ylabel("Prior density")
ax.set_title("A.  Coalescent prior by descendant count $k$")
ax.legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.9)
ax.set_xlim(0, 3.5)
ax.set_ylim(bottom=0)

# =================================================================
# Panel B -- Edge Likelihood: Poisson likelihood vs parent time
# =================================================================
ax = axes[0, 1]

lambda_e = 0.001  # mu * span
t_child = 0.0     # child is a leaf at time 0
t_parent_range = np.linspace(0.01, 5.0, 500)

mutation_counts = [0, 1, 2, 5, 10]
colors_m = [C_GREY, C_BLUE, C_GREEN, C_ORANGE, C_RED]

for m_e, col in zip(mutation_counts, colors_m):
    likelihoods = np.array([
        edge_likelihood(m_e, lambda_e, tp, t_child)
        for tp in t_parent_range
    ])
    ax.plot(t_parent_range, likelihoods, color=col, lw=2,
            label=f"$m_e = {m_e}$")

ax.set_xlabel("Parent time $t_p$ (generations)")
ax.set_ylabel("$P(m_e \\mid t_p, t_c=0)$")
ax.set_title("B.  Poisson mutation likelihood per edge")
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
ax.set_xlim(0, 5.0)
ax.set_ylim(bottom=0)

# Annotate the rate
ax.text(
    0.05, 0.92,
    f"$\\lambda_e = \\mu \\times \\ell = {lambda_e}$",
    transform=ax.transAxes, fontsize=8,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, alpha=0.8),
)

# =================================================================
# Panel C -- Inside-Outside Posterior
# =================================================================
ax = axes[1, 0]

# Build a small 5-node tree (3 leaves + 2 internal)
# Topology:
#          root (node 4)
#         /           \
#    node 3            leaf 2
#   /      \
# leaf 0   leaf 1
#
# Edges: (3->0, m=1), (3->1, m=2), (4->3, m=1), (4->2, m=0)

n_leaves = 3
K = 30
grid = make_time_grid(n=50, num_points=K, grid_type="logarithmic")

mu_rate = 1e-3
span = 1.0
lam = mu_rate * span

# Build prior from coalescent moments for n=3
moments_3 = conditional_coalescent_moments(n_leaves)

# Initialize inside messages: leaves get uniform (all 1s)
inside = np.ones((5, K))
# Leaves are at time 0: delta at grid[0]
for leaf in [0, 1, 2]:
    inside[leaf, :] = 0.0
    inside[leaf, 0] = 1.0

# Compute edge likelihood matrices
L_30 = edge_likelihood_matrix(m_e=1, lambda_e=lam, grid=grid)  # edge 3->0
L_31 = edge_likelihood_matrix(m_e=2, lambda_e=lam, grid=grid)  # edge 3->1
L_43 = edge_likelihood_matrix(m_e=1, lambda_e=lam, grid=grid)  # edge 4->3
L_42 = edge_likelihood_matrix(m_e=0, lambda_e=lam, grid=grid)  # edge 4->2

# Inside pass for node 3: combine messages from children 0 and 1
msg_from_0 = np.zeros(K)
msg_from_1 = np.zeros(K)
for i in range(K):
    msg_from_0[i] = np.sum(L_30[i, :i + 1] * inside[0, :i + 1])
    msg_from_1[i] = np.sum(L_31[i, :i + 1] * inside[1, :i + 1])

inside[3, :] = msg_from_0 * msg_from_1
# Normalize to prevent underflow
s3 = inside[3, :].sum()
if s3 > 0:
    inside[3, :] /= s3

# Inside pass for node 4 (root): combine messages from children 3 and 2
msg_from_3 = np.zeros(K)
msg_from_2 = np.zeros(K)
for i in range(K):
    msg_from_3[i] = np.sum(L_43[i, :i + 1] * inside[3, :i + 1])
    msg_from_2[i] = np.sum(L_42[i, :i + 1] * inside[2, :i + 1])

inside[4, :] = msg_from_3 * msg_from_2
s4 = inside[4, :].sum()
if s4 > 0:
    inside[4, :] /= s4

# Build outside messages
outside = np.ones((5, K))

# Apply coalescent prior to root (node 4, k=3 descendants -> root)
mean_root, var_root = moments_3[n_leaves]
alpha_root, beta_root = gamma_params_from_moments(mean_root, var_root)
prior_root = gamma_dist.pdf(grid, a=alpha_root, scale=1.0 / beta_root)
prior_root[0] = 0  # root cannot be at time 0
prior_root /= prior_root.sum() + 1e-300
outside[4, :] = prior_root

# Outside for node 3: message from parent (node 4) through edge 4->3
outside[3, :] = np.zeros(K)
for j in range(K):
    # sum over parent times i >= j
    outside[3, j] = np.sum(L_43[j:, j] * outside[4, j:] * inside[4, j:] /
                           (msg_from_3[j:] + 1e-300))

# Apply prior for node 3 (k=2 descendants)
mean_3, var_3 = moments_3[2]
alpha_3, beta_3 = gamma_params_from_moments(mean_3, var_3)
prior_3 = gamma_dist.pdf(grid, a=alpha_3, scale=1.0 / beta_3)
prior_3[0] = 0
prior_3 /= prior_3.sum() + 1e-300
outside[3, :] *= prior_3
s_out3 = outside[3, :].sum()
if s_out3 > 0:
    outside[3, :] /= s_out3

# Compute posteriors for internal nodes
post = compute_posteriors(inside, outside)

# Plot posteriors for internal nodes
node_configs = [
    (3, "Node 3 ($k{=}2$, internal)", C_BLUE, "-"),
    (4, "Node 4 ($k{=}3$, root)", C_RED, "--"),
]

for node_id, label, col, ls in node_configs:
    p = post[node_id, :]
    if p.sum() > 0:
        p = p / p.sum()
    ax.fill_between(grid, p, alpha=0.15, color=col)
    ax.plot(grid, p, color=col, lw=2, ls=ls, label=label)

    # Mark posterior mean
    pmean = np.sum(p * grid)
    ax.axvline(pmean, color=col, lw=1, ls=":", alpha=0.6)

ax.set_xlabel("Node age $t$ (coalescent units)")
ax.set_ylabel("Posterior probability")
ax.set_title("C.  Inside-outside posterior on a 3-leaf tree")
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
ax.set_xlim(0, grid[-1] * 0.6)
ax.set_ylim(bottom=0)

# Add tree schematic as text annotation
ax.text(
    0.02, 0.92,
    "Tree:  ((leaf0, leaf1):1, leaf2):0\n"
    "Mutations: $m_{3{\\to}0}{=}1,\\; m_{3{\\to}1}{=}2,\\; m_{4{\\to}2}{=}0$",
    transform=ax.transAxes, fontsize=6.5, va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, alpha=0.85),
)

# =================================================================
# Panel D -- Variational Gamma: multiply/divide operations
# =================================================================
ax = axes[1, 1]

t_vals_d = np.linspace(0.001, 6.0, 600)

# Start with a coalescent prior
prior = GammaDistribution(alpha=2.0, beta=1.5)

# A likelihood factor (from observing mutations on an edge)
likelihood_factor = GammaDistribution(alpha=3.0, beta=1.0)

# Multiply: prior x likelihood -> posterior
posterior = prior.multiply(likelihood_factor)

# A second likelihood factor from another edge
likelihood_factor_2 = GammaDistribution(alpha=2.0, beta=0.8)

# Multiply again: posterior x second factor -> updated posterior
posterior_2 = posterior.multiply(likelihood_factor_2)

# Divide out the first factor (cavity distribution)
cavity = posterior_2.divide(likelihood_factor)

# Plot each distribution
distributions = [
    (prior, "Prior: $\\Gamma(2.0, 1.5)$", C_BLUE, "-", 2.0),
    (likelihood_factor, "Likelihood factor 1: $\\Gamma(3.0, 1.0)$", C_ORANGE, "--", 1.5),
    (posterior, "After multiply: prior $\\times$ factor 1", C_RED, "-", 2.2),
    (posterior_2, "After 2nd multiply", C_PURPLE, "-", 2.2),
    (cavity, "After divide (cavity)", C_TEAL, "-.", 1.8),
]

for dist, label, col, ls, lw in distributions:
    if dist.alpha > 0 and dist.beta > 0:
        pdf_vals = gamma_dist.pdf(t_vals_d, a=dist.alpha, scale=1.0 / dist.beta)
        ax.plot(t_vals_d, pdf_vals, color=col, ls=ls, lw=lw, label=label)

# Add arrows to show the flow of operations
ax.annotate(
    "", xy=(prior.mean, 0.55), xytext=(posterior.mean, 0.55),
    arrowprops=dict(arrowstyle="<-", color=C_GREY, lw=1.5),
)
ax.text(
    (prior.mean + posterior.mean) / 2, 0.60, "multiply",
    ha="center", fontsize=7, color=C_GREY,
)

ax.annotate(
    "", xy=(posterior.mean, 0.42), xytext=(posterior_2.mean, 0.42),
    arrowprops=dict(arrowstyle="<-", color=C_GREY, lw=1.5),
)
ax.text(
    (posterior.mean + posterior_2.mean) / 2, 0.47, "multiply",
    ha="center", fontsize=7, color=C_GREY,
)

ax.set_xlabel("Node age $t$")
ax.set_ylabel("Density")
ax.set_title("D.  Variational gamma: multiply / divide updates")
ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
ax.set_xlim(0, 6.0)
ax.set_ylim(bottom=0)

# Add a note about natural parameters
ax.text(
    0.02, 0.78,
    "Natural params:\n"
    "$\\eta_1 = \\alpha - 1$\n"
    "$\\eta_2 = -\\beta$\n"
    "Multiply $\\Rightarrow$ add $\\eta$\n"
    "Divide $\\Rightarrow$ subtract $\\eta$",
    transform=ax.transAxes, fontsize=6.5, va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, alpha=0.85),
)

# -- Save ---------------------------------------------------------------
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_mini_tsdate.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_tsdate.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_tsdate.png and figures/fig_mini_tsdate.pdf")
