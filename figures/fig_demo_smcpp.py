"""
Demo: SMC++ on msprime-simulated multi-sample data.

Simulates multiple diploid genomes under a bottleneck demographic model and
illustrates the core SMC++ machinery: emission probabilities, the ODE system
tracking undistinguished lineage counts, and the effective coalescence rate
h(t) under different demographic histories.
"""

import numpy as np
import matplotlib.pyplot as plt
from watchgen.mini_psmc import (
    simulate_psmc_input,
    compute_time_intervals,
)
from watchgen.mini_smcpp import (
    emission_probability,
    solve_ode_piecewise,
    compute_h_values,
)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "font.family": "sans-serif",
})

np.random.seed(2024)

# ── Simulate multi-sample data ──────────────────────────────────
n_intervals = 15
t_max = 10.0
alpha = 0.1
theta = 0.001
rho = theta / 4.0

# True demography: bottleneck
true_lambdas = np.ones(n_intervals + 1)
for k in range(n_intervals + 1):
    frac = k / n_intervals
    if frac < 0.2:
        true_lambdas[k] = 1.0
    elif frac < 0.4:
        true_lambdas[k] = 0.3
    elif frac < 0.7:
        true_lambdas[k] = 0.8
    else:
        true_lambdas[k] = 1.5

t_bounds = compute_time_intervals(n_intervals, t_max, alpha)

def lambda_true(t):
    for k in range(n_intervals + 1):
        if t <= t_bounds[k + 1]:
            return true_lambdas[k]
    return true_lambdas[-1]

# Simulate multiple diploid sequences
n_diploids = 3
L = 100_000
sequences = []
for _ in range(n_diploids):
    seq, _ = simulate_psmc_input(L, theta, rho, lambda_true)
    sequences.append(seq)

time_breaks = compute_time_intervals(n_intervals, t_max, alpha)[:n_intervals + 2]

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    "Demo: SMC++ on Multi-sample Diploid Data (Bottleneck History)",
    fontsize=13, fontweight="bold", y=0.98,
)

# Panel A: Input data — heterozygosity across samples
ax = axes[0, 0]
window = 200
for i, seq in enumerate(sequences):
    het_rate = np.convolve(seq.astype(float),
                           np.ones(window) / window, mode="valid")
    ax.plot(np.arange(len(het_rate)), het_rate, lw=0.8, alpha=0.7,
            label=f"Sample {i+1}")
ax.set_xlabel("Genomic position (bp)")
ax.set_ylabel(f"Heterozygosity ({window}-bp window)")
ax.set_title(f"A. Input: {n_diploids} diploid genomes ({L:,} bp)")
ax.legend(fontsize=7, ncol=2)

# Panel B: Emission probabilities as a function of coalescence time
ax = axes[1, 0]
n_states = n_intervals + 1
t_mids = np.array([(t_bounds[k] + t_bounds[k + 1]) / 2.0 for k in range(n_states)])
t_mids[-1] = t_bounds[-2] * 1.1

theta_plot = 0.05  # use larger theta for visible emission variation (illustration)
em_het = [emission_probability(1, t_mids[k], theta_plot, 0, 1) for k in range(n_states)]
em_hom = [emission_probability(0, t_mids[k], theta_plot, 0, 1) for k in range(n_states)]

ax.plot(range(n_states), em_het, "o-", color="#B2182B", lw=2, ms=4,
        label="$P(\\mathrm{het} \\mid k)$")
ax.plot(range(n_states), em_hom, "s-", color="#2166AC", lw=2, ms=4,
        label="$P(\\mathrm{hom} \\mid k)$")
ax.set_xlabel("Time interval index $k$")
ax.set_ylabel("Emission probability")
ax.set_title("B. SMC++ emission probabilities")
ax.legend(fontsize=8)

# Panel C: ODE solution — p_j(t) under bottleneck demography
ax = axes[0, 1]
n_undist = 2 * n_diploids - 1  # haploid undistinguished lineages
n_intervals_ode = len(time_breaks) - 1
lambdas_for_ode = true_lambdas[:n_intervals_ode]
p_history = solve_ode_piecewise(n_undist, time_breaks, lambdas_for_ode)

t_plot = time_breaks[:-1]
colors_ode = ["#2166AC", "#B2182B", "#1B7837", "#E08214", "#762A83"]
j_show = list(range(1, min(n_undist + 1, 6)))
for j, color in zip(j_show, colors_ode):
    if j - 1 < p_history.shape[1]:
        ax.plot(range(len(p_history)), p_history[:, j - 1], "o-",
                color=color, lw=1.5, ms=3, label=f"$j={j}$")

ax.set_xlabel("Time interval index $k$")
ax.set_ylabel("$P(J(t_k) = j)$")
ax.set_title(f"C. ODE: undistinguished lineage count ($n_u={n_undist}$)")
ax.legend(fontsize=7, ncol=2)
ax.set_xlim(-0.5, len(p_history) - 0.5)

# Panel D: Effective coalescence rate h(t) under different demographies
ax = axes[1, 1]
demo_scenarios = {
    "Bottleneck": true_lambdas[:n_intervals_ode],
    "Constant":   np.ones(n_intervals_ode),
    "Expansion":  np.array([1.0 if k / n_intervals_ode < 0.4 else 3.0
                            for k in range(n_intervals_ode)]),
}
colors_d = {"Bottleneck": "#B2182B", "Constant": "#636363", "Expansion": "#1B7837"}
styles_d = {"Bottleneck": "-", "Constant": "--", "Expansion": "-."}

for label, lams in demo_scenarios.items():
    p_hist = solve_ode_piecewise(n_undist, time_breaks, lams)
    h_vals = compute_h_values(time_breaks, p_hist, lams)
    ax.plot(range(len(h_vals)), h_vals,
            color=colors_d[label], ls=styles_d[label], lw=2,
            label=label)

ax.set_xlabel("Time interval index $k$")
ax.set_ylabel("Effective coalescence rate $h(t_k)$")
ax.set_title("D. Coalescence rate $h(t)$ under different demographies")
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_smcpp.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_smcpp.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_smcpp.png")
