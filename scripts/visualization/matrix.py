import numpy as np
import matplotlib.pyplot as plt

from common import load_predictions, compute_correlation_matrix


def create_correlation_heatmap(
    corr_matrix: np.ndarray,
    output_path: str = "correlation_matrix.svg",
) -> None:
    n = corr_matrix.shape[0]
    off_diagonal = corr_matrix[~np.eye(n, dtype=bool)]
    data_min = off_diagonal.min()
    data_max = off_diagonal.max()

    vmin = np.floor(data_min * 100) / 100 - 0.01
    vmax = 1.0

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr_matrix, cmap=plt.cm.viridis, vmin=vmin, vmax=vmax, aspect='equal')

    ticks = np.arange(vmin, vmax + 0.01, 0.01)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t:.2f}' for t in ticks])
    cbar.set_label('Correlation', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    tick_positions = np.arange(0, n, 5)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_positions, fontsize=10)
    ax.set_yticklabels(tick_positions, fontsize=10)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.3, alpha=0.5)

    ax.set_xlabel('Model Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Model Index', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Correlation Matrix of {n} Model Predictions\n'
        f'Off-diagonal range: [{data_min:.4f}, {data_max:.4f}]',
        fontsize=16, fontweight='bold', pad=15,
    )

    stats_text = (
        f"Mean: {off_diagonal.mean():.4f}\n"
        f"Std: {off_diagonal.std():.4f}\n"
        f"Min: {data_min:.4f}\n"
        f"Max: {data_max:.4f}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(1.15, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to: {output_path}")
    plt.close()


def main():
    predictions = load_predictions("predictions")
    if not predictions:
        print("ERROR: No predictions found.")
        return

    print(f"Loaded {len(predictions)} model predictions")
    corr_matrix = compute_correlation_matrix(predictions)
    if corr_matrix.size == 0:
        print("ERROR: Could not compute correlation matrix.")
        return

    create_correlation_heatmap(corr_matrix, "correlation_matrix.svg")
    np.save('correlation_matrix.npy', corr_matrix)


if __name__ == '__main__':
    main()
