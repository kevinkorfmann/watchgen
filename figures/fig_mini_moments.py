"""
Figure: moments algorithm -- four-panel overview.

Panel A: Neutral SFS for different sample sizes, showing the classic theta/k pattern.
Panel B: SFS evolution under drift -- neutral SFS evolved forward under expansion
         vs contraction using integrate_sfs.
Panel C: Selection effect on SFS -- positive and negative selection compared to neutral.
Panel D: Tajima's D under expansion, constant, and contraction scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt

from watchgen.mini_moments import (
    expected_sfs_neutral,
    integrate_sfs,
    tajimas_d,
)

# -- Style ----------------------------------------------------------------
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

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle(
    "moments: Demographic Inference from the Frequency Spectrum",
    fontsize=14, fontweight="bold", y=0.98,
)

# =============================================================
# Panel A -- Neutral SFS for different sample sizes
# =============================================================
ax = axes[0, 0]

theta = 1.0
sample_sizes = [10, 20, 50, 100]
colors_a = [C_BLUE, C_RED, C_GREEN, C_ORANGE]

for n, color in zip(sample_sizes, colors_a):
    sfs = expected_sfs_neutral(n, theta=theta)
    k = np.arange(1, n)
    ax.plot(k, sfs[1:n], "o-", color=color, lw=1.5, ms=3, alpha=0.85,
            label=f"n = {n}")

# Overlay the continuous 1/k curve for reference
k_cont = np.linspace(1, 100, 500)
ax.plot(k_cont, theta / k_cont, "k--", lw=1, alpha=0.4, label=r"$\theta / k$")

ax.set_xlabel("Derived allele count $k$")
ax.set_ylabel(r"Expected SFS $\phi_k$")
ax.set_title("A.  Neutral SFS: the $\\theta / k$ law")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.8, 120)
ax.set_ylim(5e-3, 1.5)
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

# =============================================================
# Panel B -- SFS evolution under drift (expansion vs contraction)
# =============================================================
ax = axes[0, 1]

n = 30
theta_b = 1.0
phi_eq = expected_sfs_neutral(n, theta=theta_b)
T_evolve = 0.5

# Constant population
phi_const = integrate_sfs(phi_eq.copy(), n, T=T_evolve,
                          nu_func=lambda t: 1.0, theta=theta_b)
# 5x expansion
phi_expand = integrate_sfs(phi_eq.copy(), n, T=T_evolve,
                           nu_func=lambda t: 5.0, theta=theta_b)
# 5x contraction
phi_contract = integrate_sfs(phi_eq.copy(), n, T=T_evolve,
                             nu_func=lambda t: 0.2, theta=theta_b)
# Gradual 10x expansion
phi_gradual = integrate_sfs(phi_eq.copy(), n, T=T_evolve,
                            nu_func=lambda t: 1.0 + 9.0 * t / T_evolve,
                            theta=theta_b)

k_vals = np.arange(1, n)

ax.plot(k_vals, phi_eq[1:n], "s-", color=C_GREY, lw=1.5, ms=3, alpha=0.7,
        label="Equilibrium (initial)")
ax.plot(k_vals, phi_const[1:n], "o-", color=C_GREEN, lw=1.8, ms=3,
        label=r"Constant ($\nu=1$)")
ax.plot(k_vals, phi_expand[1:n], "^-", color=C_BLUE, lw=1.8, ms=3,
        label=r"Expansion ($\nu=5$)")
ax.plot(k_vals, phi_contract[1:n], "v-", color=C_RED, lw=1.8, ms=3,
        label=r"Contraction ($\nu=0.2$)")
ax.plot(k_vals, phi_gradual[1:n], "d-", color=C_PURPLE, lw=1.8, ms=3,
        label=r"Gradual exp ($\nu: 1 \to 10$)")

ax.set_xlabel("Derived allele count $k$")
ax.set_ylabel(r"SFS $\phi_k$ after $T=0.5$")
ax.set_title("B.  SFS distortion under demographic change")
ax.set_xlim(0.5, n - 0.5)
ax.set_ylim(0, None)
ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

# =============================================================
# Panel C -- Selection effect on SFS
# =============================================================
ax = axes[1, 0]

n_c = 30
theta_c = 1.0
T_sel = 1.0
phi_init_c = expected_sfs_neutral(n_c, theta=theta_c)

# Neutral reference (integrated forward to same time, constant size)
phi_neutral = integrate_sfs(phi_init_c.copy(), n_c, T=T_sel,
                            nu_func=lambda t: 1.0, theta=theta_c,
                            gamma=0)

# Weak negative selection
phi_neg_weak = integrate_sfs(phi_init_c.copy(), n_c, T=T_sel,
                             nu_func=lambda t: 1.0, theta=theta_c,
                             gamma=-5)

# Strong negative selection
phi_neg_strong = integrate_sfs(phi_init_c.copy(), n_c, T=T_sel,
                               nu_func=lambda t: 1.0, theta=theta_c,
                               gamma=-20)

# Weak positive selection
phi_pos_weak = integrate_sfs(phi_init_c.copy(), n_c, T=T_sel,
                             nu_func=lambda t: 1.0, theta=theta_c,
                             gamma=5)

# Strong positive selection
phi_pos_strong = integrate_sfs(phi_init_c.copy(), n_c, T=T_sel,
                               nu_func=lambda t: 1.0, theta=theta_c,
                               gamma=20)

k_c = np.arange(1, n_c)
w = 0.15

ax.bar(k_c - 2 * w, phi_neg_strong[1:n_c], width=w, color=C_RED, alpha=0.85,
       label=r"$\gamma = -20$ (strong purifying)")
ax.bar(k_c - w, phi_neg_weak[1:n_c], width=w, color="#E57373", alpha=0.85,
       label=r"$\gamma = -5$ (weak purifying)")
ax.bar(k_c, phi_neutral[1:n_c], width=w, color=C_GREY, alpha=0.85,
       label=r"$\gamma = 0$ (neutral)")
ax.bar(k_c + w, phi_pos_weak[1:n_c], width=w, color="#64B5F6", alpha=0.85,
       label=r"$\gamma = +5$ (weak positive)")
ax.bar(k_c + 2 * w, phi_pos_strong[1:n_c], width=w, color=C_BLUE, alpha=0.85,
       label=r"$\gamma = +20$ (strong positive)")

ax.set_xlabel("Derived allele count $k$")
ax.set_ylabel("Expected SFS count")
ax.set_title(f"C.  Selection distorts the SFS ($n={n_c}$)")
ax.set_xlim(0.3, 15)
ax.set_ylim(0, None)
ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9, ncol=1)

# =============================================================
# Panel D -- Tajima's D under different demographic scenarios
# =============================================================
ax = axes[1, 1]

np.random.seed(42)

n_d = 30
theta_d = 200.0
phi_eq_d = expected_sfs_neutral(n_d, theta=theta_d)

# Sweep over integration time to show how D evolves
T_values = np.linspace(0.02, 1.5, 25)

D_constant = []
D_expansion = []
D_contraction = []
D_gradual_exp = []

for T_val in T_values:
    # Constant size
    phi_c = integrate_sfs(phi_eq_d.copy(), n_d, T=T_val,
                          nu_func=lambda t: 1.0, theta=theta_d)
    D_constant.append(tajimas_d(phi_c))

    # Sudden expansion
    phi_e = integrate_sfs(phi_eq_d.copy(), n_d, T=T_val,
                          nu_func=lambda t: 5.0, theta=theta_d)
    D_expansion.append(tajimas_d(phi_e))

    # Sudden contraction
    phi_r = integrate_sfs(phi_eq_d.copy(), n_d, T=T_val,
                          nu_func=lambda t: 0.2, theta=theta_d)
    D_contraction.append(tajimas_d(phi_r))

    # Gradual expansion
    phi_g = integrate_sfs(phi_eq_d.copy(), n_d, T=T_val,
                          nu_func=lambda t, Tv=T_val: 1.0 + 9.0 * t / Tv,
                          theta=theta_d)
    D_gradual_exp.append(tajimas_d(phi_g))

ax.plot(T_values, D_constant, "o-", color=C_GREEN, lw=2, ms=4,
        label=r"Constant ($\nu=1$)")
ax.plot(T_values, D_expansion, "^-", color=C_BLUE, lw=2, ms=4,
        label=r"Expansion ($\nu=5$)")
ax.plot(T_values, D_contraction, "v-", color=C_RED, lw=2, ms=4,
        label=r"Contraction ($\nu=0.2$)")
ax.plot(T_values, D_gradual_exp, "d-", color=C_PURPLE, lw=2, ms=4,
        label=r"Gradual exp ($\nu: 1 \to 10$)")

ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)

# Annotate the diagnostic regions
ax.axhspan(0, max(max(D_contraction), 3), alpha=0.04, color=C_RED)
ax.axhspan(min(min(D_expansion), -3), 0, alpha=0.04, color=C_BLUE)
ax.text(T_values[-1] * 0.65, 0.3, "excess intermediate freq\n(contraction / balancing sel.)",
        fontsize=7, color=C_RED, style="italic", ha="center")
ax.text(T_values[-1] * 0.65, -0.3, "excess rare variants\n(expansion / purifying sel.)",
        fontsize=7, color=C_BLUE, style="italic", ha="center")

ax.set_xlabel("Integration time $T$ (in $2N_e$ generations)")
ax.set_ylabel("Tajima's $D$")
ax.set_title("D.  Tajima's $D$ as a demographic diagnostic")
ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)

# -- Save -----------------------------------------------------------------
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_mini_moments.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_moments.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_moments.png and figures/fig_mini_moments.pdf")
