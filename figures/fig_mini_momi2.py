"""
Figure: momi2 algorithm -- four-panel overview.

Panel A: W-matrix structure -- Polanski-Kimmel coefficients as a heatmap.
Panel B: Moran model transition -- how a delta distribution at frequency k
         spreads over time via the Moran transition matrix.
Panel C: Admixture tensor -- expected number of migrating lineages as a
         function of admixture fraction f, for different starting counts k.
Panel D: Expected time with j lineages -- branch-length contributions under
         constant-size epochs with different population sizes N.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from watchgen.mini_momi2 import (
    w_matrix,
    moran_transition,
    admixture_tensor,
    etjj_constant,
)

# -- Style ---------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "font.family": "sans-serif",
})

# Professional colour palette
C_BLUE = "#2166AC"
C_RED = "#B2182B"
C_GREEN = "#1B7837"
C_ORANGE = "#E08214"
C_PURPLE = "#7B3294"
C_GREY = "#636363"

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle(
    "momi2: Demographic Inference via Coalescent Tensor Algebra",
    fontsize=14, fontweight="bold", y=0.98,
)

# =================================================================
# Panel A -- W-matrix structure (Polanski-Kimmel coefficients)
# =================================================================
ax = axes[0, 0]

n_w = 20
W = w_matrix(n_w)

# Use a diverging colormap centred at zero to show the sign structure
vmax = np.max(np.abs(W))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(
    W, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm,
    interpolation="nearest",
)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("$W_{b,j}$", fontsize=9)

ax.set_xlabel("Lineage index $j - 2$")
ax.set_ylabel("SFS entry $b$")
ax.set_title("A.  Polanski--Kimmel $W$-matrix ($n = 20$)")

# Annotate the constant first column
ax.annotate(
    "col 0: $\\frac{6}{n+1}$",
    xy=(0, n_w // 2 - 1), xytext=(5, n_w - 4),
    fontsize=8, color=C_BLUE,
    arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.2),
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_BLUE, alpha=0.85),
)

# =================================================================
# Panel B -- Moran model transition: delta at k=5 spreading
# =================================================================
ax = axes[0, 1]

n_m = 15
k_start = 8  # initial frequency state

times = [0.0, 0.02, 0.1, 0.3, 1.0, 5.0]
colors_b = [C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE, C_GREY]
states = np.arange(n_m + 1)

for t_val, color in zip(times, colors_b):
    P = moran_transition(t_val, n_m)
    prob = P[k_start, :]  # row k_start gives P(X(t) = j | X(0) = k_start)
    ax.plot(states, prob, "o-", color=color, lw=1.5, ms=4, alpha=0.85,
            label=f"$t = {t_val}$")

ax.set_xlabel("Derived allele count $j$")
ax.set_ylabel("$P(X(t) = j \\mid X(0) = %d)$" % k_start)
ax.set_title(f"B.  Moran transition from $k = {k_start}$  ($n = {n_m}$)")
ax.legend(fontsize=7.5, ncol=2, loc="upper left", framealpha=0.9)
ax.set_xlim(-0.5, n_m + 0.5)
ax.set_ylim(-0.02, 1.05)

# Annotate absorbing states
ax.annotate(
    "absorbing\nstates",
    xy=(0, P[k_start, 0]), xytext=(2.0, 0.45),
    fontsize=7.5, color=C_GREY,
    arrowprops=dict(arrowstyle="->", color=C_GREY, lw=1),
)
ax.annotate(
    "",
    xy=(n_m, P[k_start, n_m]), xytext=(n_m - 2, 0.45),
    arrowprops=dict(arrowstyle="->", color=C_GREY, lw=1),
)

# =================================================================
# Panel C -- Admixture tensor: E[j migrating] vs f
# =================================================================
ax = axes[1, 0]

n_adm = 12
f_vals = np.linspace(0.0, 1.0, 100)

k_values = [3, 6, 9, 12]
colors_c = [C_BLUE, C_GREEN, C_ORANGE, C_RED]

for k_val, color in zip(k_values, colors_c):
    E_j = np.zeros(len(f_vals))
    Var_j = np.zeros(len(f_vals))
    for idx, f in enumerate(f_vals):
        T = admixture_tensor(n_adm, f)
        # E[j | k] = sum_j j * T[k-j, j, k]
        for j in range(k_val + 1):
            E_j[idx] += j * T[k_val - j, j, k_val]
            Var_j[idx] += j**2 * T[k_val - j, j, k_val]
        Var_j[idx] -= E_j[idx]**2

    ax.plot(f_vals, E_j, color=color, lw=2, label=f"$k = {k_val}$")
    # Show +/- 1 standard deviation band
    sd = np.sqrt(np.maximum(Var_j, 0))
    ax.fill_between(f_vals, E_j - sd, E_j + sd, color=color, alpha=0.10)

# Overlay the theoretical line E[j] = k*f as dashed reference
for k_val, color in zip(k_values, colors_c):
    ax.plot(f_vals, k_val * f_vals, color=color, ls="--", lw=1, alpha=0.5)

ax.set_xlabel("Admixture fraction $f$")
ax.set_ylabel("$E[j \\mid k]$ (migrating lineages)")
ax.set_title(f"C.  Admixture tensor: lineage splitting ($n = {n_adm}$)")
ax.legend(fontsize=8, loc="upper left", framealpha=0.9, title="Lineages $k$",
          title_fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, n_adm + 0.5)

ax.text(
    0.72, 2.5,
    "dashed: $E[j] = kf$\nband: $\\pm 1\\,\\sigma$",
    fontsize=7.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, alpha=0.8),
)

# =================================================================
# Panel D -- Expected time with j lineages (etjj_constant)
# =================================================================
ax = axes[1, 1]

n_e = 20
tau = 500  # epoch duration in generations

pop_sizes = [500, 1000, 2000, 5000]
colors_d = [C_BLUE, C_GREEN, C_ORANGE, C_RED]
j_vals = np.arange(2, n_e + 1)

for N_val, color in zip(pop_sizes, colors_d):
    etjj = etjj_constant(n_e, tau, N_val)
    ax.plot(j_vals, etjj, "s-", color=color, lw=1.8, ms=4, alpha=0.85,
            label=f"$N = {N_val}$")

# Also show the infinite-epoch limit: 1/rate = 2/(j(j-1))
etjj_inf = 2.0 / (j_vals * (j_vals - 1))
ax.plot(j_vals, etjj_inf, "k--", lw=1.5, alpha=0.5,
        label=r"$\tau \to \infty$: $2/\binom{j}{2}$")

ax.set_xlabel("Number of lineages $j$")
ax.set_ylabel("$E[T_{jj}]$ (expected sojourn time)")
ax.set_title(f"D.  Expected time with $j$ lineages ($\\tau = {tau}$ gen)")
ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
ax.set_xlim(1.5, n_e + 0.5)
ax.set_yscale("log")

ax.text(
    n_e * 0.55, etjj_inf[0] * 0.5,
    "larger $N$ $\\Rightarrow$ slower\ncoalescence",
    fontsize=8,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, alpha=0.8),
)

# -- Save ---------------------------------------------------------------
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_mini_momi2.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_momi2.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_momi2.png and figures/fig_mini_momi2.pdf")
