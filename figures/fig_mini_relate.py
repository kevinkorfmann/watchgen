"""Generate the source-guided mini-Relate mechanism figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from watchgen import mini_relate as relate


OUT = Path(__file__).parent


def draw_tree(ax, root):
    positions = {}

    def place(node, height=0):
        if node.is_leaf:
            positions[node.id] = (node.id, 0)
        else:
            place(node.left, height + 1)
            place(node.right, height + 1)
            x = (positions[node.left.id][0] + positions[node.right.id][0]) / 2
            y = max(positions[node.left.id][1], positions[node.right.id][1]) + 1
            positions[node.id] = (x, y)

    place(root)
    for node in relate.iter_nodes(root):
        if node.is_leaf:
            continue
        x, y = positions[node.id]
        for child in (node.left, node.right):
            cx, cy = positions[child.id]
            ax.plot([cx, cx, x], [cy, y, y], color="#343a40", lw=1.8)
    for leaf in sorted(root.leaves):
        ax.text(leaf, -0.12, str(leaf), ha="center", va="top")
    ax.set_xlim(-0.5, len(root.leaves) - 0.5)
    ax.set_ylim(-0.25, positions[root.id][1] + 0.3)
    ax.axis("off")


def main():
    haplotypes = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]])
    distance = relate.directional_mutation_distance(haplotypes)
    root = relate.build_tree(distance)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    image = axes[0].imshow(distance, cmap="Blues", vmin=0)
    for i in range(distance.shape[0]):
        for j in range(distance.shape[1]):
            axes[0].text(j, i, int(distance[i, j]), ha="center", va="center")
    axes[0].set_xticks(range(distance.shape[1]))
    axes[0].set_yticks(range(distance.shape[0]))
    axes[0].set(xlabel="reference j", ylabel="target i", title="A  Derived-only distance d(i,j)")
    fig.colorbar(image, ax=axes[0], shrink=0.72)

    draw_tree(axes[1], root)
    axes[1].set_title("B  Mutual-minimum tree\n" + relate.to_newick(root) + ";")

    time = np.linspace(0, 2.5, 400)
    for k, color in zip([2, 3, 4], ["#6a4c93", "#1982c4", "#8ac926"]):
        rate = k * (k - 1) / 2
        axes[2].plot(time, rate * np.exp(-rate * time), label=f"k={k}, rate={rate:g}", color=color)
    axes[2].set(xlabel=r"interval $\tau_k$", ylabel="density", title="C  Standard-coalescent prior")
    axes[2].legend(frameon=False, fontsize=8)

    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"fig_mini_relate.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
