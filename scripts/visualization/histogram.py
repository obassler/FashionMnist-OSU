import numpy as np
import matplotlib.pyplot as plt

from common import load_predictions, compute_correlation_matrix, extract_upper_triangle


def create_correlation_histogram(
    correlations: np.ndarray,
    output_path: str = "correlation_histogram.svg",
) -> None:
    data_range = correlations.max() - correlations.min()
    if data_range > 0.1:
        bin_size = 0.01
    elif data_range > 0.05:
        bin_size = 0.005
    else:
        bin_size = 0.002

    min_corr = np.floor(correlations.min() / bin_size) * bin_size
    max_corr = np.ceil(correlations.max() / bin_size) * bin_size
    bins = np.arange(min_corr, max_corr + bin_size, bin_size)

    fig, ax = plt.subplots(figsize=(12, 7))
    counts, _, patches = ax.hist(
        correlations, bins=bins,
        color='#0064C8', edgecolor='darkblue', linewidth=1.0, alpha=0.8,
    )

    ax.set_title('Distribution of Pairwise Model Correlations',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Correlation Value', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency (Number of Model Pairs)', fontsize=13, fontweight='bold')
    ax.set_xticks(bins)
    ax.set_xticklabels([f'{b:.3f}' for b in bins], rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    stats_text = (
        f"Total pairs: {len(correlations)}\n"
        f"Mean: {correlations.mean():.4f}\n"
        f"Std: {correlations.std():.4f}\n"
        f"Min: {correlations.min():.4f}\n"
        f"Max: {correlations.max():.4f}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace', bbox=props)

    for count, patch in zip(counts, patches):
        if count > 0:
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            ax.annotate(f'{int(count)}', xy=(x, y), ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Saved histogram to: {output_path}")
    plt.close()


def main():
    predictions_dir = "predictions"
    output_path = "correlation_histogram.svg"

    predictions = load_predictions(predictions_dir)
    if not predictions:
        print("ERROR: No predictions found.")
        return

    print(f"Loaded {len(predictions)} model predictions")
    corr_matrix = compute_correlation_matrix(predictions)
    if corr_matrix.size == 0:
        print("ERROR: Could not compute correlation matrix.")
        return

    correlations = extract_upper_triangle(corr_matrix)
    print(f"Pairwise correlations: {len(correlations)} in [{correlations.min():.4f}, {correlations.max():.4f}]")

    create_correlation_histogram(correlations, output_path)
    np.save('correlation_matrix.npy', corr_matrix)


if __name__ == '__main__':
    main()
