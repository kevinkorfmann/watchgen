"""Source-guided figure for the discoal teaching kernel."""

import matplotlib.pyplot as plt
import numpy as np

from watchgen.mini_discoal import (
    deterministic_trajectory,
    escape_probability,
    pairwise_diversity_profile,
    stochastic_trajectory,
    structured_event_probabilities,
)


plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
N = 100
alpha = 50.0
trajectory = deterministic_trajectory(alpha, sweep_N=N)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("discoal sweep kernel: source-guided checks", fontsize=14)

ax = axes[0, 0]
t = np.arange(len(trajectory.frequencies)) * trajectory.dt_generations
ax.plot(t, trajectory.frequencies, color="#B2182B", lw=2)
ax.set(title="A. Deterministic C trajectory", xlabel="Generations backward",
       ylabel="Beneficial-allele frequency")

ax = axes[0, 1]
for seed in range(5):
    stochastic = stochastic_trajectory(
        alpha, sweep_N=N, rng=np.random.default_rng(seed)
    )
    st = np.arange(len(stochastic.frequencies)) * stochastic.dt_generations
    ax.plot(st, stochastic.frequencies, lw=0.9, alpha=0.75)
ax.set(title="B. Conditional stochastic trajectories",
       xlabel="Generations backward", ylabel="Beneficial-allele frequency")

ax = axes[1, 0]
x_grid = np.linspace(0.01, 0.99, 200)
probabilities = np.array([
    structured_event_probabilities(6, 3, x, 1e-3, N, trajectory.dt_generations)
    for x in x_grid
])
labels = ["coal B", "coal b", "B to b", "b to B"]
for column, label in enumerate(labels):
    ax.plot(x_grid, probabilities[:, column], label=label)
ax.set(title="C. One-step event probabilities", xlabel="Frequency x",
       ylabel="Probability per grid step")
ax.legend(fontsize=8)

ax = axes[1, 1]
positions = np.linspace(0, 100_000, 11)
profile = pairwise_diversity_profile(
    N, trajectory, 1e-6, positions, 50_000, replicates=20, seed=9
)
ax.plot((positions - 50_000) / 1_000, profile, "o-", color="#2166AC", ms=3)
ax.axhline(1, color="0.4", ls="--", lw=1, label="neutral expectation")
ax.set(title="D. Independent-locus pairwise-diversity proxy",
       xlabel="Distance from selected site (kb)", ylabel="Mean TMRCA / 2N")
ax.legend(fontsize=8)

fig.text(
    0.5, 0.005,
    f"Path-specific escape probability at r=0.001: "
    f"{escape_probability(0.001, trajectory):.3f}",
    ha="center", fontsize=9,
)
plt.tight_layout(rect=[0, 0.025, 1, 0.96])
plt.savefig("figures/fig_mini_discoal.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_discoal.pdf", bbox_inches="tight")
