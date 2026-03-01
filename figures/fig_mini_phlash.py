"""
Figure: PHLASH inference pipeline.

Shows the four pillars of the PHLASH algorithm: composite SFS likelihood,
RBF kernel for SVGD, particle evolution under SVGD, and random time
discretization for debiased gradients.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from watchgen.mini_phlash import (
    expected_sfs_constant,
    rbf_kernel,
    svgd_update,
    sample_random_grid,
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("PHLASH: Population History Learning by Averaging Sampled Histories",
             fontsize=14, fontweight="bold")

# ---------------------------------------------------------------------------
# Panel A: Expected SFS under constant population -- the 1/k pattern
# ---------------------------------------------------------------------------
ax = axes[0, 0]
n = 30
k_vals = np.arange(1, n)

Ne_values = [0.5, 1.0, 2.0, 5.0]
colors_a = ["#F44336", "#2196F3", "#4CAF50", "#FF9800"]
theta_base = 100.0

for Ne, color in zip(Ne_values, colors_a):
    xi = expected_sfs_constant(n, theta_base * Ne, N_e=Ne)
    ax.plot(k_vals, xi, "o-", color=color, ms=3, lw=1.5,
            label=rf"$N_e$ = {Ne} ($\theta$ = {theta_base * Ne:.0f})")

# Show the 1/k envelope for reference
ax.plot(k_vals, theta_base / k_vals, "k--", lw=1, alpha=0.4, label=r"$\theta/k$ reference")

ax.set_xlabel("Derived allele count $k$")
ax.set_ylabel(r"Expected SFS $\xi_k$")
ax.set_title("A. Expected SFS (constant population)")
ax.legend(fontsize=7, loc="upper right")
ax.set_xlim(1, n - 1)
ax.set_ylim(0, None)

# ---------------------------------------------------------------------------
# Panel B: RBF kernel matrix -- heatmap of particle similarity
# ---------------------------------------------------------------------------
ax = axes[0, 1]
rng = np.random.default_rng(42)
J_kern = 20
M_kern = 8
particles_kern = rng.normal(0, 1.5, size=(J_kern, M_kern))

K_mat, _, bw = rbf_kernel(particles_kern)

im = ax.imshow(K_mat, cmap="magma", interpolation="nearest", origin="lower")
ax.set_xlabel("Particle index $j$")
ax.set_ylabel("Particle index $i$")
ax.set_title(rf"B. RBF kernel matrix ($\sigma$ = {bw:.2f})")
plt.colorbar(im, ax=ax, label=r"$K(h_i, h_j)$", shrink=0.85)

# ---------------------------------------------------------------------------
# Panel C: SVGD particle evolution -- 2D scatter at different iterations
# ---------------------------------------------------------------------------
ax = axes[1, 0]

J_svgd = 30
rng_svgd = np.random.default_rng(7)
particles_2d = rng_svgd.normal(0, 3.0, size=(J_svgd, 2))

# Record snapshots at several iterations
snapshots = {0: particles_2d.copy()}
record_iters = [5, 15, 50]
n_total = max(record_iters)

for step in range(1, n_total + 1):
    grad_lp = -particles_2d  # target: standard normal
    particles_2d = svgd_update(particles_2d, grad_lp, epsilon=0.15)
    if step in record_iters:
        snapshots[step] = particles_2d.copy()

cmap_c = plt.cm.cool
iter_keys = [0] + record_iters
norm_c = Normalize(vmin=0, vmax=n_total)

for it in iter_keys:
    pts = snapshots[it]
    color = cmap_c(norm_c(it))
    marker = "o"
    ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=28, alpha=0.85,
               edgecolors="k", linewidths=0.3, marker=marker,
               label=f"Iter {it}", zorder=3)

# Draw the 1-sigma contour of the target N(0,I)
theta_circ = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(theta_circ), np.sin(theta_circ), "k--", lw=1, alpha=0.35,
        label=r"Target $1\sigma$")
ax.plot(2 * np.cos(theta_circ), 2 * np.sin(theta_circ), "k:", lw=0.8, alpha=0.2)

ax.set_xlabel(r"$h_1$")
ax.set_ylabel(r"$h_2$")
ax.set_title("C. SVGD particle convergence (2D)")
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.set_aspect("equal")
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-5.5, 5.5)

# ---------------------------------------------------------------------------
# Panel D: Random time discretization -- overlaid grids
# ---------------------------------------------------------------------------
ax = axes[1, 1]
M_grid = 32
n_grids_show = 15
rng_grid = np.random.default_rng(123)

cmap_d = plt.cm.viridis
colors_d = [cmap_d(i / (n_grids_show - 1)) for i in range(n_grids_show)]

for i in range(n_grids_show):
    grid = sample_random_grid(M_grid, t_max=10.0, rng=rng_grid)
    y_pos = i
    # Plot breakpoints as vertical ticks on a horizontal line
    ax.hlines(y_pos, grid[1], grid[-2], colors=colors_d[i], lw=0.6, alpha=0.5)
    ax.scatter(grid[1:-1], np.full(len(grid) - 2, y_pos),
               marker="|", s=50, color=colors_d[i], linewidths=0.8, zorder=3)

# Highlight the fixed endpoints
ax.axvline(0, color="k", ls=":", lw=0.8, alpha=0.4, label="$t = 0$")
ax.axvline(10, color="k", ls=":", lw=0.8, alpha=0.4, label="$t_{\\max}$")

ax.set_xlabel("Time (coalescent units)")
ax.set_ylabel("Random grid index")
ax.set_xscale("symlog", linthresh=0.01)
ax.set_title(f"D. Random time discretization (M = {M_grid})")
ax.set_yticks([0, 4, 9, 14])
ax.set_yticklabels(["1", "5", "10", "15"])
ax.legend(fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig("figures/fig_mini_phlash.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_phlash.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_phlash.png and figures/fig_mini_phlash.pdf")
