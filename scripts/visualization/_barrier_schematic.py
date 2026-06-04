"""
Generates a schematic 2D loss-landscape figure illustrating a loss barrier and
mode connectivity for the theory section of the thesis.

Two low-loss basins (Model A, Model B) are connected by a curved low-loss valley.
The straight linear-interpolation path between them crosses a high-loss ridge
(a large barrier), while a curved path stays inside the low-loss region
(near-zero barrier). Output: Thesis/images/loss_barrier_schematic.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "Thesis", "images")

# Endpoints (the two trained models / minima)
A = (-2.0, 0.5)
B = (2.0, 0.5)


def valley_center(x):
    """Curved centerline of the low-loss channel; dips down between the minima."""
    return 0.5 - 1.8 * np.cos(np.pi * x / 4.0)


def loss(x, y):
    bg = 2.0
    well_a = -1.6 * np.exp(-(((x + 2.0) ** 2) + ((y - 0.5) ** 2)) / 0.5)
    well_b = -1.6 * np.exp(-(((x - 2.0) ** 2) + ((y - 0.5) ** 2)) / 0.5)
    channel = -1.5 * np.exp(-((y - valley_center(x)) ** 2) / 0.22) * np.exp(-((x / 2.3) ** 6))
    return bg + well_a + well_b + channel


def main():
    os.makedirs(OUT, exist_ok=True)

    xs = np.linspace(-3.4, 3.4, 400)
    ys = np.linspace(-2.4, 1.8, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = loss(X, Y)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    filled = ax.contourf(X, Y, Z, levels=30, cmap="viridis")
    lines = ax.contour(X, Y, Z, levels=12, colors="white", linewidths=0.5, alpha=0.5)
    cbar = fig.colorbar(filled, ax=ax, pad=0.02)
    cbar.set_label(r"Loss $L(\theta)$", fontsize=11, fontweight="bold")

    # Straight linear-interpolation path (crosses the ridge -> high barrier)
    ax.plot([A[0], B[0]], [A[1], B[1]], color="#d62728", linestyle="--",
            linewidth=2.4, zorder=4, label="Linear path (high barrier)")

    # Curved low-loss path (mode-connected -> near-zero barrier)
    xc = np.linspace(A[0], B[0], 200)
    yc = valley_center(xc)
    ax.plot(xc, yc, color="#ffffff", linestyle="-", linewidth=2.6, zorder=4,
            label="Curved path (low barrier)")

    # Endpoint markers
    for (px, py), name in [(A, r"$\theta_i$"), (B, r"$\theta_j$")]:
        ax.scatter([px], [py], color="red", s=160, marker="*",
                   edgecolors="black", linewidths=1.2, zorder=6)
        ax.annotate(name, (px, py), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=13, fontweight="bold", color="white")

    # Barrier annotation at the ridge crossed by the straight path
    ax.annotate(
        "Loss barrier",
        xy=(0.0, 0.5), xytext=(0.0, 1.35),
        ha="center", fontsize=11, fontweight="bold", color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.6),
    )

    ax.set_xlabel("Parameter direction 1", fontsize=11)
    ax.set_ylabel("Parameter direction 2", fontsize=11)
    ax.set_title(
        "Loss Barrier and Mode Connectivity Between Two Minima",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.legend(loc="lower center", fontsize=10, framealpha=0.9, ncol=2)
    ax.set_aspect("equal")

    out_path = os.path.join(OUT, "loss_barrier_schematic.png")
    plt.tight_layout()
    plt.savefig(out_path, format="png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
