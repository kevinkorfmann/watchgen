"""
Figure: Gamma-SMC continuous-state HMM for pairwise TMRCA inference.

Shows emission updates via Poisson-gamma conjugacy, the log-coordinate
transformation, forward-pass belief evolution along the genome, and
entropy clipping to prevent approximation drift.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist
from watchgen.mini_gamma_smc import (
    gamma_emission_update,
    to_log_coords,
    from_log_coords,
    gamma_smc_forward,
    gamma_entropy,
    entropy_clip,
    FlowField,
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Gamma-SMC: Continuous-State HMM for TMRCA Inference",
             fontsize=14, fontweight="bold")

# ---------------------------------------------------------------------------
# Panel A: Gamma emission update -- prior vs posterior after het observation
# ---------------------------------------------------------------------------
ax = axes[0, 0]

alpha_prior, beta_prior = 2.0, 2.0
theta = 0.5

# After observing a heterozygous site (y=1)
alpha_post, beta_post = gamma_emission_update(alpha_prior, beta_prior, 1, theta)

# After observing a homozygous site (y=0)
alpha_hom, beta_hom = gamma_emission_update(alpha_prior, beta_prior, 0, theta)

x = np.linspace(0.001, 5.0, 500)

pdf_prior = gamma_dist.pdf(x, a=alpha_prior, scale=1.0 / beta_prior)
pdf_post_het = gamma_dist.pdf(x, a=alpha_post, scale=1.0 / beta_post)
pdf_post_hom = gamma_dist.pdf(x, a=alpha_hom, scale=1.0 / beta_hom)

ax.plot(x, pdf_prior, color="#757575", lw=2.5, label=(
    rf"Prior: Gamma({alpha_prior:.0f}, {beta_prior:.0f})"))
ax.fill_between(x, pdf_prior, alpha=0.1, color="#757575")

ax.plot(x, pdf_post_het, color="#2196F3", lw=2, label=(
    rf"After het: Gamma({alpha_post:.0f}, {beta_post:.1f})"))
ax.fill_between(x, pdf_post_het, alpha=0.12, color="#2196F3")

ax.plot(x, pdf_post_hom, color="#F44336", lw=2, ls="--", label=(
    rf"After hom: Gamma({alpha_hom:.0f}, {beta_hom:.1f})"))
ax.fill_between(x, pdf_post_hom, alpha=0.08, color="#F44336")

ax.axvline(alpha_prior / beta_prior, color="#757575", ls=":", lw=1, alpha=0.5)
ax.axvline(alpha_post / beta_post, color="#2196F3", ls=":", lw=1, alpha=0.5)
ax.axvline(alpha_hom / beta_hom, color="#F44336", ls=":", lw=1, alpha=0.5)

ax.set_xlabel("TMRCA (t)")
ax.set_ylabel("Density")
ax.set_title(rf"A. Emission update ($\theta$={theta})")
ax.legend(fontsize=7.5, loc="upper right")
ax.set_xlim(0, 5)
ax.set_ylim(0, None)

# ---------------------------------------------------------------------------
# Panel B: Log-coordinate transformation
# ---------------------------------------------------------------------------
ax = axes[0, 1]

# Generate a grid of (alpha, beta) points
alpha_vals = np.array([1, 2, 5, 10, 20, 50, 100])
beta_vals = np.array([0.5, 1, 2, 5, 10, 20, 50, 100])

# Collect points
ab_points = []
lm_lc_points = []
colors_b = []

cmap = plt.cm.viridis
for i, a in enumerate(alpha_vals):
    for j, b in enumerate(beta_vals):
        if a / b > 0.01 and a / b < 100:  # reasonable mean range
            l_mu, l_C = to_log_coords(a, b)
            ab_points.append((a, b))
            lm_lc_points.append((l_mu, l_C))
            colors_b.append(cmap(i / (len(alpha_vals) - 1)))

ab_points = np.array(ab_points)
lm_lc_points = np.array(lm_lc_points)

ax.scatter(lm_lc_points[:, 0], lm_lc_points[:, 1], c=colors_b,
           s=40, edgecolors="k", linewidths=0.5, zorder=5)

# Annotate a few key points
highlights = [(1.0, 1.0), (5.0, 5.0), (10.0, 2.0), (50.0, 50.0)]
for a, b in highlights:
    l_mu, l_C = to_log_coords(a, b)
    ax.annotate(rf"$\alpha$={a:.0f}, $\beta$={b:.0f}",
                xy=(l_mu, l_C), fontsize=6.5,
                xytext=(5, 5), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color="#555", lw=0.5))

# Draw constant-mean lines in log-coords
for mean_val in [0.1, 1.0, 10.0]:
    alpha_line = np.logspace(0, 2.5, 100)
    beta_line = alpha_line / mean_val
    l_mu_line = np.log10(alpha_line / beta_line)
    l_C_line = np.log10(1.0 / np.sqrt(alpha_line))
    ax.plot(l_mu_line, l_C_line, color="#BDBDBD", lw=1, ls="--", zorder=1)
    ax.text(l_mu_line[-1], l_C_line[-1] - 0.06,
            rf"$\mu$={mean_val}", fontsize=6.5, color="#757575")

ax.set_xlabel(r"$\ell_\mu = \log_{10}(\alpha/\beta)$")
ax.set_ylabel(r"$\ell_C = \log_{10}(1/\sqrt{\alpha})$")
ax.set_title(r"B. Log-coordinate space ($\ell_\mu$, $\ell_C$)")
ax.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Panel C: Forward algorithm along the genome
# ---------------------------------------------------------------------------
ax = axes[1, 0]

# Build a zero-displacement flow field for the demo
l_mu_grid = np.linspace(-5, 2, 51)
l_C_grid = np.linspace(-2, 0, 50)
dl_mu_arr = np.zeros((51, 50))
# Small positive l_C displacement simulates diffusion toward the prior
dl_C_arr = 0.02 * np.ones((51, 50))
ff = FlowField(l_mu_grid, l_C_grid, dl_mu_arr, dl_C_arr)

# Create a simulated observation sequence with clustered hets
np.random.seed(123)
n_positions = 300
obs = [0] * n_positions
# Two clusters of heterozygous sites
het_positions = list(range(50, 60)) + list(range(180, 195))
# A few scattered hets
het_positions += [100, 130, 250]
for p in het_positions:
    obs[p] = 1

theta_demo = 0.01
rho_demo = 0.005

a_fwd, b_fwd = gamma_smc_forward(obs, theta_demo, rho_demo, ff)
mean_tmrca = a_fwd / b_fwd

positions = np.arange(n_positions)

# Plot mean TMRCA
color_mean = "#1565C0"
ax.plot(positions, mean_tmrca, color=color_mean, lw=1.5, label=r"Mean TMRCA ($\alpha/\beta$)")

# Shade +/- 1 std
std_tmrca = np.sqrt(a_fwd) / b_fwd
ax.fill_between(positions, mean_tmrca - std_tmrca, mean_tmrca + std_tmrca,
                color=color_mean, alpha=0.15, label=r"$\pm$ 1 std")

# Mark het positions
het_y = np.interp(het_positions, positions, mean_tmrca)
ax.scatter(het_positions, het_y, color="#F44336", s=12, zorder=5,
           label="Het sites", marker="v")

# Secondary axis for alpha (shape = precision proxy)
ax2 = ax.twinx()
ax2.plot(positions, a_fwd, color="#FF9800", lw=1, ls="--", alpha=0.7)
ax2.set_ylabel(r"Shape $\alpha$ (precision)", color="#FF9800", fontsize=9)
ax2.tick_params(axis="y", labelcolor="#FF9800")

ax.set_xlabel("Genome position")
ax.set_ylabel("Mean TMRCA")
ax.set_title("C. Forward pass belief evolution")
ax.legend(fontsize=7.5, loc="upper left")
ax.set_xlim(0, n_positions)

# ---------------------------------------------------------------------------
# Panel D: Entropy clipping
# ---------------------------------------------------------------------------
ax = axes[1, 1]

# Start with a distribution that has high entropy (diffuse)
cases = [
    (0.8, 0.5, "Diffuse"),
    (1.01, 0.7, "Slightly over"),
    (0.5, 0.25, "Very diffuse"),
]

x_ent = np.linspace(0.001, 12.0, 500)
h_max = 1.0

line_colors = ["#2196F3", "#4CAF50", "#9C27B0"]
clip_colors = ["#F44336", "#FF9800", "#FF5722"]

for idx, (a, b, label) in enumerate(cases):
    h_before = gamma_entropy(a, b)
    a_clip, b_clip = entropy_clip(a, b, h_max=h_max)
    h_after = gamma_entropy(a_clip, b_clip)

    pdf_before = gamma_dist.pdf(x_ent, a=a, scale=1.0 / b)
    pdf_after = gamma_dist.pdf(x_ent, a=a_clip, scale=1.0 / b_clip)

    # Plot before (dashed) and after (solid)
    ax.plot(x_ent, pdf_before, color=line_colors[idx], lw=1.5, ls="--", alpha=0.6,
            label=rf"{label}: H={h_before:.2f}")
    ax.plot(x_ent, pdf_after, color=clip_colors[idx], lw=2,
            label=rf"Clipped: H={h_after:.2f}")

# Add the entropy threshold reference: Gamma(1,1) = Exp(1) has H=1
pdf_ref = gamma_dist.pdf(x_ent, a=1.0, scale=1.0)
ax.plot(x_ent, pdf_ref, color="#757575", lw=1, ls=":", alpha=0.5,
        label=rf"Prior Exp(1): H={gamma_entropy(1.0, 1.0):.2f}")

ax.set_xlabel("TMRCA (t)")
ax.set_ylabel("Density")
ax.set_title(rf"D. Entropy clipping ($H_{{\max}}$={h_max:.1f})")
ax.legend(fontsize=6.5, loc="upper right", ncol=1)
ax.set_xlim(0, 12)
ax.set_ylim(0, 0.8)

plt.tight_layout()
plt.savefig("figures/fig_mini_gamma_smc.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_gamma_smc.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_gamma_smc.png")
