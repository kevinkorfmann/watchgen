"""
Figure: PSMC algorithm — four-panel overview.

Panel A: Coalescent density and survival functions under different population sizes.
Panel B: PSMC time discretization showing log-spaced intervals.
Panel C: PSMC transition matrix heatmap (the core HMM structure).
Panel D: Population history reconstruction — bottleneck demography with
         true vs inferred N(t) step functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch

from watchgen.mini_psmc import (
    coalescent_density,
    coalescent_survival,
    compute_time_intervals,
    compute_helpers,
    compute_stationary,
    compute_transition_matrix,
    build_psmc_hmm,
    scale_psmc_output,
    PSMC_HMM,
    posterior_decoding,
    simulate_psmc_input,
)

# ── Style ────────────────────────────────────────────────────
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
    "PSMC: Pairwise Sequentially Markovian Coalescent",
    fontsize=14, fontweight="bold", y=0.98,
)

# =============================================================
# Panel A — Coalescent density & survival
# =============================================================
ax = axes[0, 0]
t_vals = np.linspace(0.02, 8, 300)

pop_configs = [
    (1.0, r"$\lambda = 1$ (constant)", C_BLUE),
    (2.0, r"$\lambda = 2$ (larger $N_e$)", C_RED),
    (0.5, r"$\lambda = 0.5$ (smaller $N_e$)", C_GREEN),
]

for lam_val, label, color in pop_configs:
    dens = [coalescent_density(t, lambda u, lv=lam_val: lv) for t in t_vals]
    surv = [coalescent_survival(t, lambda u, lv=lam_val: lv) for t in t_vals]
    ax.plot(t_vals, dens, color=color, lw=2, label=f"$f(t)$, {label}")
    ax.plot(t_vals, surv, color=color, lw=1.5, ls="--", alpha=0.6,
            label=f"$S(t)$, {label}")

ax.set_xlabel("Coalescence time $t$")
ax.set_ylabel("Density / Survival")
ax.set_title("A.  Coalescent density $f(t)$ and survival $S(t)$")
ax.legend(fontsize=6.5, ncol=2, loc="upper right", framealpha=0.9)
ax.set_xlim(0, 8)
ax.set_ylim(0, 2.1)

# =============================================================
# Panel B — Time discretisation (log-spaced intervals)
# =============================================================
ax = axes[0, 1]

n_disc = 20
t_max_disc = 15.0
alpha_param = 0.1
t_boundaries = compute_time_intervals(n_disc, t_max_disc, alpha_param)

# Show the intervals as coloured bars
for k in range(n_disc + 1):
    lo = t_boundaries[k]
    hi = t_boundaries[k + 1]
    if hi > 25:
        hi = t_max_disc * 1.15  # cap the sentinel value for visualisation
    colour = plt.cm.viridis(k / (n_disc + 1))
    ax.barh(k, hi - lo, left=lo, height=0.85, color=colour, edgecolor="white",
            linewidth=0.5)

# Mark the boundaries with vertical lines
for k in range(n_disc + 2):
    tb = t_boundaries[k]
    if tb > 25:
        break
    ax.axvline(tb, color=C_GREY, lw=0.4, alpha=0.5)

ax.set_xlabel("Coalescent time $t$")
ax.set_ylabel("Interval index $k$")
ax.set_title("B.  PSMC time discretisation (log-spaced)")
ax.set_xlim(0, t_max_disc * 1.2)
ax.set_ylim(-0.5, n_disc + 1.5)
ax.text(
    t_max_disc * 0.55, n_disc - 1,
    f"$n = {n_disc}$, $t_{{\\max}} = {t_max_disc}$\n$\\alpha = {alpha_param}$",
    fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, alpha=0.8),
)

# =============================================================
# Panel C — Transition matrix heatmap
# =============================================================
ax = axes[1, 0]

n_hmm = 20
t_max_hmm = 15.0
theta_hmm = 0.001
rho_hmm = theta_hmm / 5.0

# Build with varying lambdas for a more interesting structure
np.random.seed(7)
lambdas_c = np.ones(n_hmm + 1)
# gentle wave pattern to make the matrix visually informative
for k in range(n_hmm + 1):
    lambdas_c[k] = 1.0 + 0.8 * np.sin(2 * np.pi * k / (n_hmm + 1))

transitions_c, emissions_c, initial_c = build_psmc_hmm(
    n_hmm, t_max_hmm, theta_hmm, rho_hmm, lambdas_c, alpha_param=0.1,
)

im = ax.imshow(
    transitions_c,
    origin="lower",
    aspect="auto",
    cmap="inferno",
    norm=LogNorm(vmin=max(transitions_c[transitions_c > 0].min(), 1e-6),
                 vmax=transitions_c.max()),
)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Transition probability $p_{kl}$", fontsize=9)
ax.set_xlabel("Destination interval $l$")
ax.set_ylabel("Source interval $k$")
ax.set_title("C.  PSMC transition matrix (log scale)")

# =============================================================
# Panel D — Population history reconstruction (bottleneck)
# =============================================================
ax = axes[1, 1]

# Set up a bottleneck demography via piecewise lambdas
n_inf = 20
t_max_inf = 15.0
alpha_inf = 0.1

# "True" lambdas: encode a bottleneck
# Recent past: normal (lambda=1), then drop, then recovery
true_lambdas = np.ones(n_inf + 1)
for k in range(n_inf + 1):
    frac = k / n_inf
    if frac < 0.25:
        true_lambdas[k] = 1.0
    elif frac < 0.50:
        true_lambdas[k] = 0.3       # bottleneck
    elif frac < 0.75:
        true_lambdas[k] = 0.8
    else:
        true_lambdas[k] = 1.5       # ancestral large N

# Simulation parameters
theta_inf = 0.001
rho_inf = theta_inf / 5.0

# Build the "true" lambda function for simulation:
# Use the piecewise-constant true_lambdas to form a callable
t_bounds_inf = compute_time_intervals(n_inf, t_max_inf, alpha_inf)

def lambda_true(t):
    """Piecewise constant lambda(t) from true_lambdas."""
    for k in range(n_inf + 1):
        if t <= t_bounds_inf[k + 1]:
            return true_lambdas[k]
    return true_lambdas[-1]

# Simulate data
np.random.seed(2024)
L_sim = 50000
seq_sim, _ = simulate_psmc_input(L_sim, theta_inf, rho_inf, lambda_true)

# Build HMM with flat initial lambdas (all 1s), mimicking the start of inference
init_lambdas = np.ones(n_inf + 1)
hmm_init = PSMC_HMM(n_inf, theta_inf, rho_inf, init_lambdas, t_max_inf, alpha_inf)

# Build HMM with the true lambdas to show the "oracle"
hmm_true = PSMC_HMM(n_inf, theta_inf, rho_inf, true_lambdas, t_max_inf, alpha_inf)

# Scaling parameters
mu = 1.25e-8
s = 100
gen_time = 25

# Scale the true demography
N0_true, _, t_years_true, Nt_true = scale_psmc_output(
    theta_inf, true_lambdas, t_bounds_inf, mu, s, gen_time,
)

# Scale the initial (flat) demography
N0_init, _, t_years_init, Nt_init = scale_psmc_output(
    theta_inf, init_lambdas, t_bounds_inf, mu, s, gen_time,
)

# Build step-function coordinates for plotting
def step_coords(t_years, Nt, n_intervals, t_years_max=None):
    """Return x, y arrays for a step-function plot."""
    x, y = [], []
    for k in range(n_intervals):
        x_lo = max(t_years[k], 1)  # avoid log(0)
        x_hi = t_years[k + 1]
        if t_years_max and x_hi > t_years_max:
            x_hi = t_years_max
        x.extend([x_lo, x_hi])
        y.extend([Nt[k], Nt[k]])
    return np.array(x), np.array(y)

clip_year = t_years_true[-2] * 1.1  # avoid the sentinel

x_true, y_true = step_coords(t_years_true, Nt_true, n_inf + 1, clip_year)
x_init, y_init = step_coords(t_years_init, Nt_init, n_inf + 1, clip_year)

ax.plot(x_true, y_true, color=C_RED, lw=2.5, label="True $N(t)$ (bottleneck)")
ax.plot(x_init, y_init, color=C_GREY, lw=1.5, ls=":", alpha=0.7,
        label="Initial $N(t)$ (flat, $\\lambda=1$)")

# Annotate the bottleneck region
ymin_bottle = Nt_true.min()
ax.axhspan(ymin_bottle * 0.9, ymin_bottle * 1.1, color=C_RED, alpha=0.08)
ax.annotate(
    "bottleneck", xy=(x_true[len(x_true) // 3], ymin_bottle),
    xytext=(x_true[len(x_true) // 3] * 0.6, ymin_bottle * 0.55),
    fontsize=8, color=C_RED,
    arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2),
)

ax.set_xscale("log")
ax.set_xlabel("Years before present")
ax.set_ylabel("Effective population size $N_e$")
ax.set_title("D.  Population history (simulated bottleneck)")
ax.legend(fontsize=8, loc="upper left")

# Set sensible limits
valid_years = x_true[x_true > 0]
if len(valid_years) > 0:
    ax.set_xlim(valid_years.min() * 0.8, valid_years.max() * 1.2)

# ── Save ─────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_mini_psmc.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_psmc.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_psmc.png and figures/fig_mini_psmc.pdf")
