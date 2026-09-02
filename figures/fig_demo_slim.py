"""Demo: bounded SLiM-style forward simulation with selection."""

import matplotlib.pyplot as plt
import numpy as np

from watchgen.mini_slim import simulate

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "font.family": "sans-serif",
    }
)

N = 100
L = 50_000
MU = 5e-6
R = 1e-6
T = 300

pop_neutral, stats_neutral = simulate(N, L, MU, R, T, seed=2024)
pop_deleterious, stats_deleterious = simulate(
    N,
    L,
    MU,
    R,
    T,
    dfe="gamma_deleterious",
    dfe_params={"shape": 0.3, "scale": 0.05},
    seed=2024,
)


def mutation_burdens(population):
    """Count mutation lineages carried by each diploid individual."""

    return np.array([len(ind.haplosome_1) + len(ind.haplosome_2) for ind in population])


fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    f"Demo: bounded SLiM-style WF model ($N$={N}, $L$={L / 1e3:.0f} kb)",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)

ax = axes[0, 0]
ax.plot(
    stats_neutral["tick"],
    stats_neutral["mean_fitness"],
    color="#1B7837",
    lw=2,
    label="Neutral",
)
ax.plot(
    stats_deleterious["tick"],
    stats_deleterious["mean_fitness"],
    color="#B2182B",
    lw=2,
    label="Deleterious DFE",
)
ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.4)
ax.set(xlabel="Tick", ylabel="Mean population fitness", title="A. Fitness evolution")
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.plot(
    stats_neutral["tick"],
    stats_neutral["segregating_mutations"],
    color="#1B7837",
    lw=2,
    label="Neutral",
)
ax.plot(
    stats_deleterious["tick"],
    stats_deleterious["segregating_mutations"],
    color="#B2182B",
    lw=2,
    label="Deleterious DFE",
)
ax.set(
    xlabel="Tick",
    ylabel="Segregating mutation lineages",
    title="B. Standing variation",
)
ax.legend(fontsize=8)

ax = axes[1, 0]
neutral_burden = mutation_burdens(pop_neutral)
deleterious_burden = mutation_burdens(pop_deleterious)
upper = max(neutral_burden.max(initial=0), deleterious_burden.max(initial=0)) + 1
bins = np.arange(upper + 1) - 0.5
ax.hist(neutral_burden, bins=bins, color="#1B7837", alpha=0.6, label="Neutral")
ax.hist(
    deleterious_burden,
    bins=bins,
    color="#B2182B",
    alpha=0.6,
    label="Deleterious DFE",
)
ax.set(
    xlabel="Mutation lineages per individual",
    ylabel="Individuals",
    title=f"C. Mutation burden after {T} ticks",
)
ax.legend(fontsize=8)

ax = axes[1, 1]
s_range = np.linspace(-0.05, 0.1, 160)


def kimura_fixation(selection, population_size):
    """Diploid fixation probability for one new additive allele."""

    if abs(selection) < 1e-10:
        return 1.0 / (2 * population_size)
    # ``Mutation.s`` is the homozygous effect and h=1/2, so the per-copy
    # genic coefficient in Kimura's formula is selection / 2.
    return -np.expm1(-selection) / -np.expm1(-2 * population_size * selection)


p_fix = np.array([kimura_fixation(s, N) for s in s_range])
ax.plot(s_range, p_fix, color="#2166AC", lw=2.5)
ax.axhline(
    1 / (2 * N),
    color="#636363",
    ls="--",
    lw=1,
    label=rf"Neutral $1/(2N)={1 / (2 * N):.3f}$",
)
ax.axvline(0, color="black", ls=":", lw=0.8, alpha=0.4)
ax.set(
    xlabel="Selection coefficient $s$",
    ylabel="Fixation probability",
    title=f"D. Kimura fixation probability ($N$={N})",
    ylim=(0, 1.1 * p_fix.max()),
)
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_slim.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_slim.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_slim.png and figures/fig_demo_slim.pdf")
