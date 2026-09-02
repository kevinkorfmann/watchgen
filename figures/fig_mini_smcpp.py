"""Figure: SMC++ conditioned SFS and continuous-time transition kernel."""

import matplotlib.pyplot as plt
import numpy as np

from watchgen.mini_smcpp import (
    conditioned_sfs,
    constant_csc_transition_matrix,
    pair_interval_probabilities,
    two_locus_kernel,
)

plt.rcParams.update(
    {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "figure.dpi": 150}
)

BREAKS = np.array([0.0, 0.25, 0.75, 2.0, np.inf])
STATE_LABELS = ["0–0.25", "0.25–0.75", "0.75–2", "2–∞"]
x = np.arange(len(STATE_LABELS))

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle(
    "SMC++: distinguished-pair state and CSFS emissions",
    fontsize=14,
    fontweight="bold",
)

ax = axes[0, 0]
width = 0.25
for offset, relative_size, color in zip(
    (-width, 0, width), (0.5, 1.0, 2.0), ("#1565C0", "#4CAF50", "#E91E63")
):
    probabilities = pair_interval_probabilities(BREAKS, relative_size)
    ax.bar(
        x + offset,
        probabilities,
        width=width,
        color=color,
        label=rf"relative size $\lambda={relative_size:g}$",
    )
ax.set_xticks(x, STATE_LABELS)
ax.set(
    xlabel="Pairwise TMRCA interval",
    ylabel="Probability",
    title="A. Population size shifts pairwise TMRCA",
)
ax.legend(fontsize=7)

ax = axes[0, 1]
tau_values = np.linspace(0.0, 3.0, 80)
categories = [
    ((1, 0), "Distinguished singleton", "#1565C0"),
    ((0, 1), "Undistinguished singleton", "#4CAF50"),
    ((1, 1), "Mixed descendants", "#FF9800"),
    ((2, 0), "Distinguished pair", "#E91E63"),
]
for (a, b), label, color in categories:
    branch_lengths = [conditioned_sfs(tau, 2)[a, b] for tau in tau_values]
    ax.plot(tau_values, branch_lengths, lw=2, color=color, label=label)
ax.set(
    xlabel=r"Conditioned pair TMRCA $\tau$",
    ylabel="Expected branch length",
    title="B. Extra samples enter through the CSFS",
)
ax.legend(fontsize=7)

ax = axes[1, 0]
transition = constant_csc_transition_matrix(BREAKS, relative_size=1.0, rho=0.1)
image = ax.imshow(transition, cmap="Blues", vmin=0, vmax=1, aspect="equal")
ax.set_xticks(x, STATE_LABELS, rotation=35, ha="right")
ax.set_yticks(x, STATE_LABELS)
ax.set(
    xlabel="Next interval",
    ylabel="Current interval",
    title="C. Discretized two-locus transition",
)
plt.colorbar(image, ax=ax, label="Probability", shrink=0.82)

ax = axes[1, 1]
durations = np.linspace(0, 8, 200)
labels = (
    (0, "Linked", "#1565C0"),
    (1, "Recombined", "#FF9800"),
    (2, "Recoalesced", "#4CAF50"),
)
for state, label, color in labels:
    probabilities = [two_locus_kernel(t, 1.0, 0.2)[0, state] for t in durations]
    ax.plot(durations, probabilities, lw=2, color=color, label=label)
ax.set(
    xlabel="Elapsed coalescent time",
    ylabel="State probability",
    title="D. Exact three-state two-locus kernel",
    ylim=(0, 1.05),
)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig_mini_smcpp.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_smcpp.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_smcpp.png and figures/fig_mini_smcpp.pdf")
