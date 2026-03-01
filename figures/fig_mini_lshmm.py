"""
Figure: Li & Stephens HMM -- haplotype copying model.

Shows forward/backward probabilities, Viterbi path recovery,
emission/transition structure, and posterior decoding of the copying model.
"""

import numpy as np
import matplotlib.pyplot as plt
from watchgen.mini_lshmm import (
    initial_distribution,
    transition_matrix,
    emission_matrix_haploid,
    estimate_mutation_probability,
    forwards_ls_hap,
    backwards_ls_hap,
    posterior_decoding,
    forwards_viterbi_hap,
    backwards_viterbi_hap,
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

# ---- Simulate a mosaic haplotype with known breakpoints ----
np.random.seed(123)
n = 20   # reference haplotypes
m = 200  # sites

H = np.random.binomial(1, 0.3, size=(m, n))

# True copying path: four segments copied from different reference haplotypes
true_path = np.zeros(m, dtype=int)
true_path[0:50] = 3
true_path[50:100] = 7
true_path[100:150] = 12
true_path[150:200] = 1

# Build query as imperfect copy (with ~2% mutations)
s_flat = np.array([H[l, true_path[l]] for l in range(m)])
mutation_mask = np.random.random(m) < 0.02
s_flat[mutation_mask] = 1 - s_flat[mutation_mask]
s_2d = s_flat.reshape(1, -1)

# Model parameters
mu = estimate_mutation_probability(n)
e_mat = np.zeros((m, 2))
e_mat[:, 0] = mu
e_mat[:, 1] = 1 - mu
r_arr = np.full(m, 0.04)
r_arr[0] = 0.0

# Run algorithms
F, c, ll_fwd = forwards_ls_hap(n, m, H, s_2d, e_mat, r_arr, norm=True)
B = backwards_ls_hap(n, m, H, s_2d, e_mat, c, r_arr)
gamma, posterior_path = posterior_decoding(F, B)
V, P, ll_vit = forwards_viterbi_hap(n, m, H, s_2d, e_mat, r_arr)
viterbi_path = backwards_viterbi_hap(m, V, P)

# ---- Figure ----
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Li & Stephens HMM: Haplotype Copying Model",
             fontsize=14, fontweight="bold")

# --- Panel A: Forward probabilities heatmap ---
ax = axes[0, 0]
# Show forward probabilities (sites x reference haplotypes)
im = ax.imshow(F.T, aspect="auto", cmap="YlOrRd", interpolation="nearest",
               extent=[0, m, n - 0.5, -0.5])
# Overlay true path
ax.plot(np.arange(m), true_path, color="#1565C0", lw=1.5, ls="--",
        label="True path", alpha=0.9)
ax.set_xlabel("Genomic site")
ax.set_ylabel("Reference haplotype")
ax.set_title("A. Forward probabilities")
ax.legend(fontsize=8, loc="lower right")
plt.colorbar(im, ax=ax, label="P(state | data up to site)", shrink=0.8)

# --- Panel B: Viterbi path vs true path ---
ax = axes[0, 1]
sites = np.arange(m)
ax.step(sites, true_path, color="#4CAF50", lw=2.5, alpha=0.8,
        label="True path", where="mid")
ax.step(sites, viterbi_path, color="#F44336", lw=1.5, ls="--",
        label="Viterbi path", where="mid", alpha=0.9)

# Mark breakpoints
true_breaks = [50, 100, 150]
for bp in true_breaks:
    ax.axvline(bp, color="#757575", ls=":", lw=0.8, alpha=0.6)

viterbi_accuracy = np.mean(viterbi_path == true_path)
ax.text(0.02, 0.95, f"Accuracy: {viterbi_accuracy:.1%}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

ax.set_xlabel("Genomic site")
ax.set_ylabel("Copying source (haplotype index)")
ax.set_title("B. Viterbi decoding")
ax.legend(fontsize=8, loc="center right")

# --- Panel C: Transition matrix structure ---
ax = axes[1, 0]
r_vals = [0.01, 0.05, 0.10, 0.20]
colors = ["#1565C0", "#2196F3", "#FF9800", "#F44336"]
n_show = 8

for r_val, color in zip(r_vals, colors):
    A = transition_matrix(n_show, r_val)
    # Show diagonal and off-diagonal values
    ax.plot([f"h{i}" for i in range(n_show)], A[0, :],
            "o-", color=color, ms=5, lw=1.5,
            label=f"r = {r_val}")

ax.set_xlabel("Target haplotype (from h0)")
ax.set_ylabel("Transition probability")
ax.set_title("C. Transition matrix structure")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=45)

# Add emission matrix inset
inset = ax.inset_axes([0.55, 0.45, 0.4, 0.45])
mu_vals = [0.001, 0.005, 0.01, 0.05, 0.10]
match_probs = [1 - m for m in mu_vals]
mismatch_probs = mu_vals
inset.barh(range(len(mu_vals)), match_probs, height=0.35, color="#4CAF50",
           alpha=0.8, label="Match")
inset.barh([i + 0.35 for i in range(len(mu_vals))], mismatch_probs,
           height=0.35, color="#F44336", alpha=0.8, label="Mismatch")
inset.set_yticks([i + 0.175 for i in range(len(mu_vals))])
inset.set_yticklabels([f"{m:.3f}" for m in mu_vals], fontsize=7)
inset.set_xlabel("Probability", fontsize=7)
inset.set_ylabel(r"$\mu$", fontsize=7)
inset.set_title("Emission probs", fontsize=8)
inset.tick_params(labelsize=7)
inset.legend(fontsize=6, loc="center right")

# --- Panel D: Posterior decoding (gamma) heatmap ---
ax = axes[1, 1]
# Show posterior probabilities for the 4 true source haplotypes + 2 others
show_haps = sorted(set([1, 3, 7, 12]))
gamma_show = gamma[:, show_haps]

for i, hap_idx in enumerate(show_haps):
    ax.plot(sites, gamma[:, hap_idx], lw=1.5,
            label=f"h{hap_idx}", alpha=0.85)

# Shade regions by true source
segment_colors = ["#E3F2FD", "#FFF3E0", "#E8F5E9", "#FCE4EC"]
segment_labels = [f"True: h{true_path[0]}", f"True: h{true_path[50]}",
                  f"True: h{true_path[100]}", f"True: h{true_path[150]}"]
boundaries = [0, 50, 100, 150, 200]
for i in range(4):
    ax.axvspan(boundaries[i], boundaries[i + 1], alpha=0.2,
               color=segment_colors[i])

posterior_accuracy = np.mean(posterior_path == true_path)
ax.text(0.02, 0.95, f"Posterior accuracy: {posterior_accuracy:.1%}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

ax.set_xlabel("Genomic site")
ax.set_ylabel("Posterior probability")
ax.set_title("D. Posterior decoding (forward-backward)")
ax.legend(fontsize=7, ncol=2, loc="center right")
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("figures/fig_mini_lshmm.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_lshmm.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_lshmm.png")
