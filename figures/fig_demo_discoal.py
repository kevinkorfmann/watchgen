"""Validation figure for the revised discoal chapter."""

import matplotlib.pyplot as plt
import numpy as np

from watchgen.mini_discoal import (
    deterministic_trajectory,
    discoal_deterministic_frequency,
    neutral_coalescent,
    simulate_linked_locus_genealogy,
)


plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
N = 200
alpha = 50.0
trajectory = deterministic_trajectory(alpha, sweep_N=N)
rng = np.random.default_rng(23)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
fig.suptitle("Revised discoal chapter: independent validation targets", fontsize=13)

ax = axes[0]
time_2N = np.linspace(0, trajectory.duration_2N, 300)
ax.plot(time_2N, discoal_deterministic_frequency(time_2N, alpha), color="#B2182B")
ax.set(title="A. Exact detSweepFreq curve", xlabel="Time backward (2N units)",
       ylabel="Beneficial-allele frequency")

ax = axes[1]
neutral = []
sweep = []
for _ in range(300):
    neutral.append(neutral_coalescent(2, N, rng)[0][-1])
    sweep.append(max(simulate_linked_locus_genealogy(
        2, N, trajectory, 0.0, rng=rng
    )))
bins = np.linspace(0, np.quantile(neutral, 0.98), 35)
ax.hist(neutral, bins=bins, density=True, alpha=0.55, label="neutral")
ax.hist(sweep, bins=bins, density=True, alpha=0.65, label="selected site")
ax.set(title="B. Pairwise TMRCA", xlabel="Generations", ylabel="Density")
ax.legend(fontsize=8)

ax = axes[2]
scalars = np.array([10, 20, 40, 80, 160])
durations = np.array([
    deterministic_trajectory(alpha, sweep_N=N, dt_scalar=float(s)).duration_generations
    for s in scalars
])
reference = deterministic_trajectory(
    alpha, sweep_N=N, dt_scalar=640.0
).duration_generations
ax.plot(scalars, np.abs(durations - reference), "o-", color="#2166AC")
ax.set(title="C. Grid-refinement convergence", xlabel="Time-step scalar",
       ylabel="Absolute duration error (generations)")

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("figures/fig_demo_discoal.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_discoal.pdf", bbox_inches="tight")
