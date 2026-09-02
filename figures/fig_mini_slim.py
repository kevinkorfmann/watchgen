"""Figure: verified mechanisms in the bounded SLiM teaching model."""

import matplotlib.pyplot as plt
import numpy as np

from watchgen.mini_slim import (
    Individual,
    Mutation,
    mutation_frequency,
    simulate,
    wright_fisher_generation,
)

plt.rcParams.update(
    {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "figure.dpi": 150}
)


def introduced_mutation_trajectory(N, selection, ticks, seed):
    """Track one identified lineage using the public WF-generation API."""

    rng = np.random.default_rng(seed)
    target = Mutation(position=25, s=selection, h=0.5)
    population = [Individual() for _ in range(N)]
    population[0].haplosome_1.append(target)
    trajectory = [(0, mutation_frequency(population, target))]
    for tick in range(1, ticks + 1):
        population = wright_fisher_generation(
            population, N, 50, 0.0, 0.0, tick, rng=rng
        )
        frequency = mutation_frequency(population, target)
        trajectory.append((tick, frequency))
        if frequency in (0.0, 1.0):
            break
    return np.asarray(trajectory), trajectory[-1][1] == 1.0


fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle(
    "SLiM: bounded Wright–Fisher teaching mechanisms",
    fontsize=14,
    fontweight="bold",
)

ax = axes[0, 0]
for attempt in range(100):
    sweep, fixed = introduced_mutation_trajectory(50, 0.15, 400, 1000 + attempt)
    if fixed:
        break
ax.plot(sweep[:, 0], sweep[:, 1], color="#D32F2F", lw=2.2)
ax.fill_between(sweep[:, 0], sweep[:, 1], alpha=0.1, color="#D32F2F")
ax.axhline(1.0, ls="--", color="#757575", lw=0.8)
ax.set(
    xlabel="Tick after introduction",
    ylabel="Beneficial-lineage frequency",
    title="A. One identified selective sweep",
    ylim=(-0.02, 1.05),
)

ax = axes[0, 1]
N_FIX = 50
TRIALS = 30
s_values = np.array([0.01, 0.05, 0.1, 0.2])
observed = []
for i, selection in enumerate(s_values):
    fixed_count = sum(
        introduced_mutation_trajectory(N_FIX, selection, 1000, 10_000 + 100 * i + j)[1]
        for j in range(TRIALS)
    )
    observed.append(fixed_count / TRIALS)
s_theory = np.linspace(0.001, 0.3, 200)
# The mini API defines ``s`` as the homozygous effect with h=1/2, making the
# corresponding per-copy genic coefficient s/2 in Kimura's formula.
p_theory = -np.expm1(-s_theory) / -np.expm1(-2 * N_FIX * s_theory)
ax.plot(s_theory, p_theory, color="#1565C0", lw=2.2, label="Kimura theory")
ax.scatter(
    s_values,
    observed,
    color="#FF6F00",
    s=70,
    edgecolors="black",
    linewidth=0.8,
    label=f"WF simulation ({TRIALS} trials)",
    zorder=3,
)
ax.set(
    xlabel="Selection coefficient $s$",
    ylabel="Fixation probability",
    title=f"B. Fixation probability ($N$={N_FIX})",
    xlim=(-0.005, 0.3),
    ylim=(0, None),
)
ax.legend(fontsize=8, loc="upper left")

ax = axes[1, 0]
N_SFS = 50
population, _ = simulate(N_SFS, 2_000, 5e-4, 1e-5, 300, seed=123)
lineage_counts = {}
for individual in population:
    for mutation in individual.haplosome_1 + individual.haplosome_2:
        lineage_counts[mutation.mutation_id] = (
            lineage_counts.get(mutation.mutation_id, 0) + 1
        )
frequencies = np.array(
    [count / (2 * N_SFS) for count in lineage_counts.values() if count < 2 * N_SFS]
)
folded = np.minimum(frequencies, 1 - frequencies)
counts, edges = np.histogram(folded, bins=np.linspace(0, 0.5, 16))
centers = (edges[:-1] + edges[1:]) / 2
ax.bar(centers, counts, width=np.diff(edges), color="#00897B", edgecolor="white")
ax.set(
    xlabel="Minor-lineage frequency",
    ylabel="Mutation lineages",
    title="C. Folded neutral frequency spectrum",
    xlim=(0, 0.5),
)

ax = axes[1, 1]
_, neutral = simulate(50, 2_000, 2e-4, 1e-5, 300, seed=99)
_, selected = simulate(
    50,
    2_000,
    2e-4,
    1e-5,
    300,
    dfe="gamma_deleterious",
    dfe_params={"shape": 0.3, "scale": 0.05},
    seed=99,
)
ax.plot(
    neutral["tick"],
    neutral["segregating_mutations"],
    color="#388E3C",
    lw=2,
    ls="--",
    label="Neutral DFE",
)
ax.plot(
    selected["tick"],
    selected["segregating_mutations"],
    color="#7B1FA2",
    lw=2,
    label="Deleterious DFE",
)
ax.set(
    xlabel="Tick",
    ylabel="Segregating mutation lineages",
    title="D. Purifying selection changes standing variation",
)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig_mini_slim.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_slim.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_slim.png and figures/fig_mini_slim.pdf")
