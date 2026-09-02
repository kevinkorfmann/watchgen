"""Figure: verified miniature of selected PHLASH kernels.

Shows the coalescent-rate parameterisation, RBF kernel, idealized RBF-SVGD
particle direction, and endpoint-randomised geometric time grids.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from watchgen.mini_phlash import (
    effective_population_size,
    logarithmic_grid,
    rbf_kernel,
    svgd_direction,
)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
    }
)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle(
    "PHLASH: source-aligned miniature kernels",
    fontsize=14,
    fontweight="bold",
)

# Panel A: coalescent-rate parameterisation.
ax = axes[0, 0]
time = np.array([0, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
histories = {
    "Constant": np.ones_like(time),
    "Low middle rate": np.array([1.0, 1.0, 0.35, 0.35, 1.0, 1.0]),
    "High middle rate": np.array([1.0, 1.0, 2.5, 2.5, 1.0, 1.0]),
}
colors = ["#636363", "#2166AC", "#B2182B"]
for (label, rate), color in zip(histories.items(), colors):
    size = effective_population_size(rate)
    ax.step(time[1:], size[1:], where="post", color=color, lw=2, label=label)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time (coalescent units)")
ax.set_ylabel(r"$N_e(t)=1/[2\eta(t)]$")
ax.set_title("A. Coalescent rate and population size")
ax.legend(fontsize=7)

# Panel B: RBF kernel matrix.
ax = axes[0, 1]
rng = np.random.default_rng(42)
n_particles = 20
particles = rng.normal(0, 1.5, size=(n_particles, 8))
kernel, bandwidth = rbf_kernel(particles)
image = ax.imshow(kernel, cmap="magma", interpolation="nearest", origin="lower")
ax.set_xlabel("Particle index $j$")
ax.set_ylabel("Particle index $i$")
ax.set_title(rf"B. RBF kernel ($\sigma={bandwidth:.2f}$)")
plt.colorbar(image, ax=ax, label=r"$k(x_i,x_j)$", shrink=0.85)

# Panel C: idealized SVGD particle evolution for a standard-normal target.
ax = axes[1, 0]
rng_svgd = np.random.default_rng(7)
particles_2d = rng_svgd.normal(0, 3.0, size=(30, 2))
record_iterations = [0, 5, 15, 50]
snapshots = {0: particles_2d.copy()}
for iteration in range(1, max(record_iterations) + 1):
    score = -particles_2d
    particles_2d += 0.15 * svgd_direction(particles_2d, score)
    if iteration in record_iterations:
        snapshots[iteration] = particles_2d.copy()

color_map = plt.cm.cool
normalizer = Normalize(vmin=0, vmax=max(record_iterations))
for iteration in record_iterations:
    points = snapshots[iteration]
    ax.scatter(
        points[:, 0],
        points[:, 1],
        c=[color_map(normalizer(iteration))],
        s=28,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.3,
        label=f"Iteration {iteration}",
        zorder=3,
    )
angle = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(angle), np.sin(angle), "k--", lw=1, alpha=0.35, label=r"Target $1\sigma$")
ax.plot(2 * np.cos(angle), 2 * np.sin(angle), "k:", lw=0.8, alpha=0.2)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("C. Idealized RBF-SVGD direction")
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.set_aspect("equal")
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-5.5, 5.5)

# Panel D: only the two endpoints vary; interior points remain geometric.
ax = axes[1, 1]
n_grids = 15
rng_grid = np.random.default_rng(123)
for index in range(n_grids):
    log_t1 = rng_grid.normal(np.log(1e-4), 0.45)
    log_tM = rng_grid.normal(np.log(15.0), 0.25)
    grid = logarithmic_grid(log_t1, log_tM, intervals=16)
    color = plt.cm.viridis(index / (n_grids - 1))
    ax.hlines(index, grid[1], grid[-1], colors=color, lw=0.6, alpha=0.5)
    ax.scatter(
        grid[1:],
        np.full(15, index),
        marker="|",
        s=50,
        color=color,
        linewidths=0.8,
        zorder=3,
    )
ax.set_xlabel("Time (coalescent units)")
ax.set_ylabel("Particle grid")
ax.set_xscale("log")
ax.set_title("D. Random endpoints; geometric interior")
ax.set_yticks([0, 4, 9, 14])
ax.set_yticklabels(["1", "5", "10", "15"])

plt.tight_layout()
plt.savefig("figures/fig_mini_phlash.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_phlash.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_phlash.png and figures/fig_mini_phlash.pdf")
