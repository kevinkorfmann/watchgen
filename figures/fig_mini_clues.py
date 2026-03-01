"""
Figure: CLUES -- Coalescent Likelihood Under Effects of Selection.

Shows the Wright-Fisher HMM transition structure, coalescent emission
probabilities, importance sampling weights, and selection coefficient
posterior / likelihood ratio surface.
"""

import numpy as np
import matplotlib.pyplot as plt
from watchgen.mini_clues import (
    backward_mean,
    backward_std,
    build_frequency_bins,
    build_transition_matrix,
    build_normal_cdf_lookup,
    build_transition_matrix_fast,
    log_coalescent_density,
    genotype_likelihood_emission,
    logsumexp,
    likelihood_ratio_test,
    backward_algorithm,
    estimate_selection_single,
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("CLUES: Coalescent Likelihood Under Effects of Selection",
             fontsize=14, fontweight="bold")

# --- Panel A: Selection coefficient posterior (likelihood surface) ---
ax = axes[0, 0]

# Build a small model and scan the log-likelihood over s values
K = 80
freqs, logfreqs, log1minusfreqs = build_frequency_bins(K)
z_bins, z_cdf = build_normal_cdf_lookup()
N_diploid = 10_000.0
N_haploid = 2 * N_diploid
t_cutoff = 50  # generations to look back

epochs = np.arange(0.0, t_cutoff)
N_vec = N_diploid * np.ones(int(t_cutoff))
h = 0.5

# Simulate coalescence times consistent with positive selection
# (derived lineages coalesce faster = lower freq in the past)
np.random.seed(42)
n_der = 5
n_anc = 3
coal_times_der = np.sort(np.random.exponential(8, size=n_der - 1))
coal_times_anc = np.sort(np.random.exponential(15, size=n_anc - 1))
curr_freq = 0.7

# Scan log-likelihood
s_grid = np.linspace(-0.05, 0.10, 80)
log_liks = np.zeros_like(s_grid)

for i, s_val in enumerate(s_grid):
    sel = np.array([s_val])
    alpha_mat = backward_algorithm(
        sel, freqs, logfreqs, log1minusfreqs,
        z_bins, z_cdf, epochs, N_vec, h,
        coal_times_der, coal_times_anc,
        n_der, n_anc, curr_freq)
    log_liks[i] = logsumexp(alpha_mat[-2, :])

# Normalize for visualization
log_liks_norm = log_liks - log_liks.max()

# Find MLE
s_hat_idx = np.argmax(log_liks)
s_hat = s_grid[s_hat_idx]

# Approximate posterior (proportional to likelihood with flat prior)
posterior = np.exp(log_liks_norm)
posterior /= np.trapezoid(posterior, s_grid) if hasattr(np, 'trapezoid') else np.sum(posterior * np.gradient(s_grid))

ax.fill_between(s_grid, posterior, alpha=0.2, color="#1565C0")
ax.plot(s_grid, posterior, color="#1565C0", lw=2, label="Posterior density")
ax.axvline(s_hat, color="#D32F2F", ls="--", lw=1.5,
           label=f"MLE: s = {s_hat:.4f}")
ax.axvline(0.0, color="#757575", ls=":", lw=1, alpha=0.6,
           label="Neutral (s = 0)")

# Log-LR and significance
ll_neutral = log_liks[np.argmin(np.abs(s_grid))]
log_lr_val = 2 * (log_liks[s_hat_idx] - ll_neutral)
_, p_value, neg_log10_p = likelihood_ratio_test(
    log_liks[s_hat_idx], ll_neutral)

ax.text(0.97, 0.95,
        f"LR = {log_lr_val:.1f}\np = {p_value:.4f}\n"
        f"-log$_{{10}}$p = {neg_log10_p:.1f}",
        transform=ax.transAxes, fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

ax.set_xlabel("Selection coefficient s")
ax.set_ylabel("Posterior density")
ax.set_title("A. Selection coefficient inference")
ax.legend(fontsize=8, loc="upper left")

# --- Panel B: Allele frequency trajectory inference ---
ax = axes[1, 0]

# Run backward algorithm at several s values and show the frequency trajectory
# implied by the posterior mean at each epoch
s_vals_traj = [0.0, s_hat, 0.05]
colors_traj = ["#757575", "#D32F2F", "#1565C0"]
labels_traj = ["Neutral (s=0)", f"MLE (s={s_hat:.3f})", "Strong (s=0.05)"]

for s_val, color, label in zip(s_vals_traj, colors_traj, labels_traj):
    sel = np.array([s_val])
    alpha_mat = backward_algorithm(
        sel, freqs, logfreqs, log1minusfreqs,
        z_bins, z_cdf, epochs, N_vec, h,
        coal_times_der, coal_times_anc,
        n_der, n_anc, curr_freq)

    # Compute posterior mean frequency at each time step
    mean_freq = np.zeros(len(epochs) - 1)
    for t in range(len(epochs) - 1):
        row = alpha_mat[t, :]
        row_norm = row - logsumexp(row)
        probs = np.exp(row_norm)
        probs /= probs.sum()
        mean_freq[t] = np.sum(freqs * probs)

    ax.plot(np.arange(len(mean_freq)), mean_freq, color=color, lw=2,
            label=label, alpha=0.9)

# Show coalescence events
ax.scatter(coal_times_der, [curr_freq] * len(coal_times_der),
           marker="v", s=40, color="#4CAF50", zorder=5,
           label="Derived coal. events")
ax.scatter(coal_times_anc, [1 - curr_freq] * len(coal_times_anc),
           marker="^", s=40, color="#FF9800", zorder=5,
           label="Ancestral coal. events")

ax.axhline(curr_freq, color="#E0E0E0", ls="-", lw=0.8, alpha=0.5)
ax.set_xlabel("Generations into the past")
ax.set_ylabel("Derived allele frequency")
ax.set_title("B. Inferred allele frequency trajectory")
ax.legend(fontsize=7, loc="lower left", ncol=2)
ax.set_ylim(0, 1)
ax.set_xlim(0, t_cutoff - 2)

# --- Panel C: HMM transition matrix structure ---
ax = axes[0, 1]

# Build transition matrices under different selection coefficients
K_small = 60
freqs_small, _, _ = build_frequency_bins(K_small)

# Show transition row distributions for a starting frequency of ~0.3
target_freq = 0.3
i_row = np.argmin(np.abs(freqs_small - target_freq))

s_vals = [0.0, 0.02, 0.05, -0.02]
colors_c = ["#757575", "#2196F3", "#D32F2F", "#4CAF50"]
labels_c = ["Neutral", "s = 0.02", "s = 0.05", "s = -0.02"]

for s_val, color, label in zip(s_vals, colors_c, labels_c):
    logP = build_transition_matrix(freqs_small, 2 * N_diploid, s_val)
    P_row = np.exp(logP[i_row, :])
    ax.plot(freqs_small, P_row, color=color, lw=1.8, label=label, alpha=0.9)

ax.axvline(freqs_small[i_row], color="#E0E0E0", ls=":", lw=1)
ax.text(freqs_small[i_row] + 0.01, ax.get_ylim()[1] * 0.9,
        f"x = {freqs_small[i_row]:.2f}", fontsize=8, color="#757575")

# Show backward mean arrows
for s_val, color in zip([0.02, 0.05, -0.02],
                        ["#2196F3", "#D32F2F", "#4CAF50"]):
    mu_back = backward_mean(target_freq, s_val)
    ax.annotate("", xy=(mu_back, 0.001), xytext=(target_freq, 0.001),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

ax.set_xlabel("Frequency one generation into the past")
ax.set_ylabel("Transition probability")
ax.set_title(f"C. WF-HMM transitions (from x = {target_freq})")
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(0.15, 0.45)

# --- Panel D: Coalescent emission structure ---
ax = axes[1, 1]

# Show how coalescent emission probability varies with frequency and lineage count
freq_grid = np.linspace(0.05, 0.95, 100)
coal_time = np.array([0.5])
N_dip = 10_000.0

for n_lin, color, ls in [
    (2, "#1565C0", "-"),
    (5, "#2196F3", "--"),
    (10, "#E65100", "-."),
    (20, "#D32F2F", ":"),
]:
    log_probs = np.array([
        log_coalescent_density(coal_time, n_lin, 0.0, 1.0,
                               freq, N_dip, ancestral=False)
        for freq in freq_grid
    ])
    # Normalize to show relative shape
    log_probs_norm = log_probs - log_probs.max()
    probs = np.exp(log_probs_norm)
    ax.plot(freq_grid, probs, color=color, lw=1.8, ls=ls,
            label=f"n = {n_lin} lineages")

# Add ancient genotype likelihood emission overlay
ax2 = ax.twinx()
# Simulate an ancient sample that looks heterozygous (AD genotype)
gl_het = np.log(np.array([0.01, 0.70, 0.29]))  # P(R|AA), P(R|AD), P(R|DD)
gl_hom_der = np.log(np.array([0.01, 0.04, 0.95]))

for gl, color, label in [
    (gl_het, "#4CAF50", "Ancient het (AD)"),
    (gl_hom_der, "#9C27B0", "Ancient hom-der (DD)"),
]:
    anc_emissions = np.array([
        genotype_likelihood_emission(gl, np.log(max(f, 1e-12)),
                                      np.log(max(1 - f, 1e-12)))
        for f in freq_grid
    ])
    anc_probs = np.exp(anc_emissions - anc_emissions.max())
    ax2.plot(freq_grid, anc_probs, color=color, lw=1.5, ls="-",
             alpha=0.6, label=label)

ax.set_xlabel("Derived allele frequency")
ax.set_ylabel("Coalescent emission (normalized)")
ax2.set_ylabel("Ancient GL emission (normalized)", color="#4CAF50")
ax2.tick_params(axis="y", labelcolor="#4CAF50")
ax.set_title("D. Emission probabilities")

# Combine legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
          loc="upper right", ncol=2)

plt.tight_layout()
plt.savefig("figures/fig_mini_clues.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_clues.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_clues.png")
