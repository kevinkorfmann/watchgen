"""
Figure: Threads segment dating estimators.

Shows how MLE and Bayesian age estimators respond to recombination distance,
mutation count, and demographic history (bottleneck vs constant population).
"""

import numpy as np
import matplotlib.pyplot as plt
from watchgen.mini_threads import (
    mle_recombination_only,
    mle_recombination_and_mutations,
    bayesian_recombination_only,
    bayesian_full,
    piecewise_constant_bayesian_recomb_only,
    piecewise_constant_bayesian_full,
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Threads: Segment Dating Estimators", fontsize=14, fontweight="bold")

# --- Panel A: Age vs segment length (recombination distance) ---
ax = axes[0, 0]
l_cM_vals = np.linspace(0.1, 5.0, 200)
rho_vals = 2 * 0.01 * l_cM_vals
Ne = 10_000
gamma = 1.0 / Ne

t_mle_r = [mle_recombination_only(r) for r in rho_vals]
t_bayes_r = [bayesian_recombination_only(r, gamma) for r in rho_vals]

ax.plot(l_cM_vals, t_mle_r, label="MLE (recomb only)", color="#2196F3", lw=2)
ax.plot(l_cM_vals, t_bayes_r, label=f"Bayesian (N$_e$={Ne:,})", color="#FF5722", lw=2, ls="--")
ax.set_xlabel("Segment length (cM)")
ax.set_ylabel("Estimated age (generations)")
ax.set_title("A. Age vs segment length")
ax.legend(fontsize=8)
ax.set_ylim(0, 600)
ax.set_xlim(0.1, 5)

# --- Panel B: Effect of mutations on age estimate ---
ax = axes[0, 1]
mu = 2 * 1.25e-8 * 1e6  # ~1 Mb segment
rho_fixed = 2 * 0.01 * 1.0  # 1 cM
m_vals = np.arange(0, 25)

t_mle_m = [mle_recombination_and_mutations(m, rho_fixed, mu) for m in m_vals]
t_bayes_m = [bayesian_full(m, rho_fixed, mu, gamma) for m in m_vals]

ax.plot(m_vals, t_mle_m, "o-", label="MLE", color="#2196F3", ms=4, lw=1.5)
ax.plot(m_vals, t_bayes_m, "s-", label=f"Bayesian (N$_e$={Ne:,})", color="#FF5722", ms=4, lw=1.5)
ax.set_xlabel("Number of heterozygous sites (m)")
ax.set_ylabel("Estimated age (generations)")
ax.set_title("B. Effect of mutations (1 cM segment)")
ax.legend(fontsize=8)

# --- Panel C: Bayesian estimator under different demographic models ---
ax = axes[1, 0]
m_vals_c = np.arange(0, 20)

# Constant size
t_const = [bayesian_full(m, rho_fixed, mu, gamma) for m in m_vals_c]

# Bottleneck: recent large, then small
tb_bottle = [0.0, 500.0]
cr_bottle = [1.0 / 20_000, 1.0 / 2_000]  # recent expansion, ancient bottleneck
t_bottle = [piecewise_constant_bayesian_full(rho_fixed, mu, m, tb_bottle, cr_bottle)
            for m in m_vals_c]

# Growth: recent small, then large
tb_growth = [0.0, 200.0]
cr_growth = [1.0 / 2_000, 1.0 / 50_000]  # recent bottleneck, ancient large
t_growth = [piecewise_constant_bayesian_full(rho_fixed, mu, m, tb_growth, cr_growth)
            for m in m_vals_c]

ax.plot(m_vals_c, t_const, "o-", label=f"Constant (N$_e$={Ne:,})", color="#4CAF50", ms=4, lw=1.5)
ax.plot(m_vals_c, t_bottle, "s-", label="Expansion (20k→2k)", color="#9C27B0", ms=4, lw=1.5)
ax.plot(m_vals_c, t_growth, "^-", label="Bottleneck (2k→50k)", color="#FF9800", ms=4, lw=1.5)
ax.set_xlabel("Number of heterozygous sites (m)")
ax.set_ylabel("Estimated age (generations)")
ax.set_title("C. Demographic model effect")
ax.legend(fontsize=8)

# --- Panel D: Comparison of estimator families for varying N_e ---
ax = axes[1, 1]
Ne_vals = np.logspace(2, 5, 100)
gamma_vals = 1.0 / Ne_vals
rho_d = 2 * 0.01 * 1.0
m_d = 5

t_mle_ne = [mle_recombination_and_mutations(m_d, rho_d, mu)] * len(Ne_vals)
t_bayes_ne = [bayesian_full(m_d, rho_d, mu, g) for g in gamma_vals]
t_pw_ne = []
for g in gamma_vals:
    t_pw_ne.append(piecewise_constant_bayesian_full(rho_d, mu, m_d, [0.0], [g]))

ax.semilogx(Ne_vals, t_mle_ne, label="MLE (no prior)", color="#2196F3", lw=2)
ax.semilogx(Ne_vals, t_bayes_ne, label="Bayesian", color="#FF5722", lw=2, ls="--")
ax.semilogx(Ne_vals, t_pw_ne, label="Piecewise (1 epoch)", color="#4CAF50", lw=2, ls=":")
ax.set_xlabel("Effective population size (N$_e$)")
ax.set_ylabel("Estimated age (generations)")
ax.set_title(f"D. Prior sensitivity (m={m_d}, 1 cM)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig_mini_threads.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_threads.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_threads.png")
