"""Source-matched diagnostics for the Threads segment-dating kernel.

This deliberately exercises the bounded dating model, not the complete
PBWT/Viterbi/ARG inference pipeline.
"""

import matplotlib.pyplot as plt
import numpy as np

from watchgen.mini_threads import (
    bayesian_full,
    mle_recombination_only,
    threads_date_segment,
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "font.family": "sans-serif",
})

rng = np.random.default_rng(2024)
n_segments = 4000
ne = 10_000.0
gamma = 1.0 / ne
mutation_rate = 1.25e-8
recombination_rate = 1.0e-8

# Draw exactly from the dating model: t ~ Exp(gamma), genetic length conditional
# on t ~ Exp(2t), and mutation count ~ Poisson(2*c*physical_length*t).
true_age = rng.exponential(1.0 / gamma, n_segments)
length_morgans = rng.exponential(1.0 / (2.0 * true_age))
length_bp = length_morgans / recombination_rate
rho = 2.0 * length_morgans
mu = 2.0 * mutation_rate * length_bp
mutations = rng.poisson(true_age * mu)

mle = np.array([mle_recombination_only(value) for value in rho])
bayes_full_estimates = np.array([
    bayesian_full(int(m), r, u, gamma)
    for m, r, u in zip(mutations, rho, mu)
])

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    "Threads: Source-Matched Segment-Dating Checks",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)

sample = rng.choice(n_segments, 700, replace=False)
limits = [20, 120_000]

ax = axes[0, 0]
ax.scatter(true_age[sample], mle[sample], s=11, alpha=0.35, color="#2166AC")
ax.plot(limits, limits, "k--", lw=1, alpha=0.6)
ax.set(xscale="log", yscale="log", xlim=limits, ylim=limits,
       xlabel="True age (generations)", ylabel="MLE point estimate",
       title="A. Recombination-only MLE")

ax = axes[0, 1]
ax.scatter(true_age[sample], bayes_full_estimates[sample], s=11, alpha=0.35,
           color="#B2182B")
ax.plot(limits, limits, "k--", lw=1, alpha=0.6)
ax.set(xscale="log", yscale="log", xlim=limits, ylim=limits,
       xlabel="True age (generations)", ylabel="Posterior mean",
       title="B. Bayesian estimate with mutations")

ax = axes[1, 0]
bins = np.unique(np.quantile(bayes_full_estimates, np.linspace(0, 1, 13)))
which = np.digitize(bayes_full_estimates, bins[1:-1])
estimated_means = []
true_means = []
for index in range(len(bins) - 1):
    mask = which == index
    estimated_means.append(np.mean(bayes_full_estimates[mask]))
    true_means.append(np.mean(true_age[mask]))
ax.plot(estimated_means, true_means, "o-", color="#1B7837", label="Binned means")
ax.plot(limits, limits, "k--", lw=1, alpha=0.6, label="calibrated")
ax.set(xscale="log", yscale="log", xlim=limits, ylim=limits,
       xlabel="Mean posterior estimate", ylabel="Mean true age",
       title="C. Model-based calibration")
ax.legend(fontsize=8)

ax = axes[1, 1]
m_values = np.arange(0, 26)
constant = [threads_date_segment(
    int(m), 1.0, 1_000_000, mutation_rate, [0.0], [10_000.0])
    for m in m_values]
expansion = [threads_date_segment(
    int(m), 1.0, 1_000_000, mutation_rate,
    [0.0, 500.0], [20_000.0, 2_000.0])
    for m in m_values]
ax.plot(m_values, constant, "o-", ms=4, label="constant $N_e=10{,}000$")
ax.plot(m_values, expansion, "s-", ms=4, label="piecewise demography")
ax.axvline(15.5, color="black", ls="--", lw=1, label="production shortcut")
ax.set(xlabel="Heterozygous sites ($m$)", ylabel="Estimated age (generations)",
       title="D. Production dating dispatch")
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_threads.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_threads.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_threads.png")
