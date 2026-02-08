import numpy as np
import matplotlib.pyplot as plt

from prediction_utils import load_predictions, compute_correlation_matrix, extract_upper_triangle


def create_correlation_histogram(
    correlations: np.ndarray,
    min_bins: int = 10,
    output_path: str = "correlation_histogram.svg"
) -> None:
    """Create histogram of correlation values using matplotlib (SVG output)."""

    # Calculate bin size to get at least min_bins
    data_range = correlations.max() - correlations.min()
    bin_size = data_range / min_bins

    # Round bin_size down to a nice number (0.005, 0.01, etc.)
    if bin_size > 0.01:
        bin_size = 0.01
    elif bin_size > 0.005:
        bin_size = 0.005
    else:
        bin_size = 0.002

    # Calculate bin edges
    min_corr = np.floor(correlations.min() / bin_size) * bin_size
    max_corr = np.ceil(correlations.max() / bin_size) * bin_size
    bins = np.arange(min_corr, max_corr + bin_size, bin_size)

    num_bins = len(bins) - 1
    print(f"Using {num_bins} bins with size {bin_size}")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create histogram
    counts, bin_edges, patches = ax.hist(
        correlations,
        bins=bins,
        color='#0064C8',
        edgecolor='darkblue',
        linewidth=1.0,
        alpha=0.8
    )

    # Labels and title
    ax.set_title('Distribution of Pairwise Model Correlations',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Correlation Value', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency (Number of Model Pairs)', fontsize=13, fontweight='bold')

    # Set x-axis ticks at bin edges
    ax.set_xticks(bins)
    ax.set_xticklabels([f'{b:.3f}' for b in bins], rotation=45, ha='right', fontsize=9)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add statistics box
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

    # Add count labels on top of bars
    for count, patch in zip(counts, patches):
        if count > 0:
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            ax.annotate(f'{int(count)}', xy=(x, y), ha='center', va='bottom',
                       fontsize=8, fontweight='bold')

    plt.tight_layout()

    # Save as SVG
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Saved histogram to: {output_path}")

    plt.close()


def main():
    # Configuration
    predictions_dir = "predictions"
    min_bins = 10
    output_path = "correlation_histogram.svg"

    print("Loading predictions...")
    predictions = load_predictions(predictions_dir)

    if not predictions:
        print("ERROR: No predictions found. Cannot generate histogram.")
        return

    print(f"Loaded {len(predictions)} model predictions")

    print("Computing correlation matrix...")
    corr_matrix = compute_correlation_matrix(predictions)

    if corr_matrix.size == 0:
        print("ERROR: Could not compute correlation matrix.")
        return

    print("Extracting upper triangle values...")
    correlations = extract_upper_triangle(corr_matrix)

    print(f"Total pairwise correlations: {len(correlations)}")
    print(f"Correlation range: [{correlations.min():.4f}, {correlations.max():.4f}]")

    print("Creating histogram...")
    create_correlation_histogram(correlations, min_bins, output_path)

    # Save the correlation matrix for reference
    np.save('correlation_matrix.npy', corr_matrix)
    print("Saved correlation matrix to: correlation_matrix.npy")


if __name__ == '__main__':
    main()
