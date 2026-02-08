"""
Loss Landscape Visualization

Implements the filter-normalized random direction method from:
"Visualizing the Loss Landscape of Neural Nets" (Li et al., 2018)
https://arxiv.org/abs/1712.09913

Creates 2D contour plots showing the loss surface around a trained model.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from typing import Tuple, List, Optional

from data.datamodule import FashionMNISTDataModule


class FashionMNISTModelOld(nn.Module):
    """
    Original model architecture for loading old checkpoints.
    Matches the structure used when checkpoints were saved.
    """
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = torch.relu(self.bn1(torch.max_pool2d(self.conv1(x), 2)))
        x = torch.relu(self.bn2(torch.max_pool2d(self.conv2(x), 2)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load model from checkpoint, handling both old and new architectures."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Check which architecture the checkpoint uses
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Remove 'model.' prefix if present (from Lightning)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('model.', '') if k.startswith('model.') else k
        cleaned_state_dict[new_key] = v

    # Detect architecture by checking key names
    if 'conv1.weight' in cleaned_state_dict:
        print("Detected OLD model architecture (conv1, bn1, etc.)")
        model = FashionMNISTModelOld()
    elif 'feature_extractor.0.weight' in cleaned_state_dict:
        print("Detected NEW model architecture (feature_extractor, classifier)")
        from models.lit_model import FashionMNISTModel
        model = FashionMNISTModel()
    else:
        raise ValueError(f"Unknown model architecture in checkpoint. Keys: {list(cleaned_state_dict.keys())[:5]}")

    model_keys = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in cleaned_state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state_dict)

    return model


def get_weights(model: nn.Module) -> List[torch.Tensor]:
    """Extract all trainable parameters from model as a list of tensors."""
    return [p.data.clone() for p in model.parameters()]


def set_weights(model: nn.Module, weights: List[torch.Tensor]) -> None:
    """Set model parameters from a list of tensors."""
    for p, w in zip(model.parameters(), weights):
        p.data.copy_(w)


def get_random_direction(weights: List[torch.Tensor]) -> List[torch.Tensor]:
    """Generate a random direction with same shape as weights."""
    return [torch.randn_like(w) for w in weights]


def normalize_direction_filter_wise(
    direction: List[torch.Tensor],
    weights: List[torch.Tensor],
    ignore_bn: bool = True
) -> List[torch.Tensor]:
    """
    Apply filter-wise normalization to direction vectors.

    For each filter in direction d, normalize it to have the same norm
    as the corresponding filter in weights θ:
        d_i ← (d_i / ||d_i||) × ||θ_i||

    This removes scale invariance issues as described in Li et al. (2018).

    Args:
        direction: Random direction vectors
        weights: Model weights (θ)
        ignore_bn: Whether to zero out BatchNorm parameters

    Returns:
        Filter-normalized direction
    """
    normalized = []

    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            # Bias or 1D parameter (e.g., BatchNorm) - either zero or normalize simply
            if ignore_bn and d.dim() == 1:
                normalized.append(torch.zeros_like(d))
            else:
                # Simple normalization for 1D
                d_norm = d.norm()
                w_norm = w.norm()
                if d_norm > 1e-10:
                    normalized.append(d * (w_norm / d_norm))
                else:
                    normalized.append(torch.zeros_like(d))
        elif d.dim() == 2:
            # Fully connected layer: treat each row (neuron) as a "filter"
            d_normalized = torch.zeros_like(d)
            for i in range(d.shape[0]):
                d_norm = d[i].norm()
                w_norm = w[i].norm()
                if d_norm > 1e-10:
                    d_normalized[i] = d[i] * (w_norm / d_norm)
            normalized.append(d_normalized)
        elif d.dim() == 4:
            # Convolutional layer: each filter is d[i, :, :, :]
            d_normalized = torch.zeros_like(d)
            for i in range(d.shape[0]):
                d_norm = d[i].norm()
                w_norm = w[i].norm()
                if d_norm > 1e-10:
                    d_normalized[i] = d[i] * (w_norm / d_norm)
            normalized.append(d_normalized)
        else:
            # Other dimensions - simple normalization
            d_norm = d.norm()
            w_norm = w.norm()
            if d_norm > 1e-10:
                normalized.append(d * (w_norm / d_norm))
            else:
                normalized.append(torch.zeros_like(d))

    return normalized


def compute_loss_at_direction(
    model: nn.Module,
    base_weights: List[torch.Tensor],
    direction1: List[torch.Tensor],
    direction2: List[torch.Tensor],
    alpha: float,
    beta: float,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute loss and accuracy at: θ' = θ + α*d1 + β*d2

    Args:
        model: Neural network model
        base_weights: Original trained weights (θ)
        direction1: First direction (d1)
        direction2: Second direction (d2)
        alpha: Coefficient for direction1
        beta: Coefficient for direction2
        dataloader: Data loader for evaluation
        device: Computation device
        max_batches: Limit number of batches for faster computation

    Returns:
        (loss, accuracy) tuple
    """
    # Compute new weights: θ' = θ + α*d1 + β*d2
    new_weights = [
        w + alpha * d1 + beta * d2
        for w, d1, d2 in zip(base_weights, direction1, direction2)
    ]

    # Set new weights
    set_weights(model, new_weights)

    # Evaluate
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def compute_loss_surface(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
    resolution: int = 21,
    max_batches: Optional[int] = None,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D loss surface around the current model weights.

    Args:
        model: Trained model
        dataloader: Data loader for evaluation
        device: Computation device
        x_range: Range for first direction (alpha)
        y_range: Range for second direction (beta)
        resolution: Number of points along each axis
        max_batches: Limit batches for faster computation
        seed: Random seed for direction generation

    Returns:
        (X, Y, loss_surface, acc_surface) - meshgrid coordinates and surfaces
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get base weights
    base_weights = get_weights(model)

    # Generate two random directions
    print("Generating random directions...")
    direction1 = get_random_direction(base_weights)
    direction2 = get_random_direction(base_weights)

    # Apply filter-wise normalization
    print("Applying filter-wise normalization...")
    direction1 = normalize_direction_filter_wise(direction1, base_weights)
    direction2 = normalize_direction_filter_wise(direction2, base_weights)

    # Create coordinate grid
    alphas = np.linspace(x_range[0], x_range[1], resolution)
    betas = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(alphas, betas)

    # Compute loss at each point
    loss_surface = np.zeros_like(X)
    acc_surface = np.zeros_like(X)

    total_points = resolution * resolution
    print(f"Computing loss surface ({total_points} points)...")

    pbar = tqdm(total=total_points, desc="Computing loss landscape")

    for i in range(resolution):
        for j in range(resolution):
            alpha = alphas[j]
            beta = betas[i]

            loss, acc = compute_loss_at_direction(
                model, base_weights, direction1, direction2,
                alpha, beta, dataloader, device, max_batches
            )

            loss_surface[i, j] = loss
            acc_surface[i, j] = acc
            pbar.update(1)

    pbar.close()

    # Restore original weights
    set_weights(model, base_weights)

    return X, Y, loss_surface, acc_surface


def plot_2d_contour(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    title: str = "Loss Landscape",
    output_path: str = "loss_landscape_2d.svg",
    levels: int = 30,
    cmap: str = "viridis",
    z_label: str = "Cross-Entropy Loss",
    acc_surface: Optional[np.ndarray] = None,
    resolution: int = 21,
    seed: int = 42,
    max_batches: Optional[int] = None,
    checkpoint_path: str = ""
) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))

    contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    cs = ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.3, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f',
              levels=cs.levels[::max(1, len(cs.levels) // 8)])

    cbar = fig.colorbar(contour, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label(z_label, fontsize=12, fontweight='bold')

    center_z = Z[Z.shape[0] // 2, Z.shape[1] // 2]
    ax.plot(0, 0, 'r*', markersize=20, markeredgecolor='black', markeredgewidth=1.5,
            zorder=10)
    ax.annotate(
        f'Trained model\n{z_label.split()[0]}: {center_z:.4f}',
        xy=(0, 0),
        xytext=(0.15, 0.15),
        fontsize=10, fontweight='bold', color='red',
        arrowprops=dict(arrowstyle='->', color='red', lw=2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9)
    )

    min_idx = np.unravel_index(Z.argmin(), Z.shape)
    min_alpha = X[min_idx]
    min_beta = Y[min_idx]
    if abs(min_alpha) > 0.05 or abs(min_beta) > 0.05:
        ax.plot(min_alpha, min_beta, 'g^', markersize=12, markeredgecolor='black',
                markeredgewidth=1.0, zorder=10)
        ax.annotate(
            f'Global min: {Z.min():.4f}\n({min_alpha:.2f}, {min_beta:.2f})',
            xy=(min_alpha, min_beta),
            xytext=(min_alpha + 0.15, min_beta - 0.15),
            fontsize=9, fontweight='bold', color='darkgreen',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.9)
        )

    ax.set_xlabel(r'Direction $\mathbf{d_1}$ coefficient ($\alpha$)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Direction $\mathbf{d_2}$ coefficient ($\beta$)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    loss_ratio = Z.max() / max(center_z, 1e-10)
    if center_z <= Z.min() + (Z.max() - Z.min()) * 0.1:
        basin_desc = "Trained model sits near the minimum (good convergence)"
    else:
        basin_desc = "Minimum is offset from trained model"

    if loss_ratio > 10:
        sharpness_desc = "Sharp landscape (loss grows rapidly)"
    elif loss_ratio > 3:
        sharpness_desc = "Moderate sharpness"
    else:
        sharpness_desc = "Flat landscape (loss changes slowly)"

    stats_text = (
        f"Surface Statistics\n"
        f"{'=' * 34}\n"
        f"Center {z_label.split()[0]:8s}: {center_z:.4f}\n"
        f"Min {z_label.split()[0]:11s}: {Z.min():.4f}\n"
        f"Max {z_label.split()[0]:11s}: {Z.max():.4f}\n"
        f"Max/Center ratio : {loss_ratio:.1f}x\n"
        f"{'=' * 34}\n"
        f"{basin_desc}\n"
        f"{sharpness_desc}"
    )

    if acc_surface is not None:
        center_acc = acc_surface[acc_surface.shape[0] // 2, acc_surface.shape[1] // 2]
        stats_text += (
            f"\n{'=' * 34}\n"
            f"Center Accuracy  : {center_acc * 100:.2f}%\n"
            f"Min Accuracy     : {acc_surface.min() * 100:.2f}%\n"
            f"Max Accuracy     : {acc_surface.max() * 100:.2f}%"
        )

    props = dict(boxstyle='round', facecolor='#f8f8f8', alpha=0.95, edgecolor='#333333', linewidth=1.2)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace', bbox=props)

    method_text = (
        f"Method: Filter-normalized random directions (Li et al., 2018)\n"
        f"Grid: {resolution}x{resolution} | "
        f"Range: [{X.min():.1f}, {X.max():.1f}] x [{Y.min():.1f}, {Y.max():.1f}] | "
        f"Seed: {seed}"
    )
    if max_batches is not None:
        method_text += f" | Batches: {max_batches}"
    if checkpoint_path:
        method_text += f"\nCheckpoint: {checkpoint_path}"
    method_text += f"\nFormula: " + r"$\theta' = \theta + \alpha \cdot d_1 + \beta \cdot d_2$"

    ax.text(0.5, -0.08, method_text, transform=ax.transAxes, fontsize=8,
            ha='center', va='top', fontfamily='monospace', color='#555555',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='#cccccc', alpha=0.9))

    ax.axhline(y=0, color='white', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='white', linewidth=0.5, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14)
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Saved 2D contour plot to: {output_path}")
    plt.close()


def plot_3d_surface(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    title: str = "Loss Landscape (3D)",
    output_path: str = "loss_landscape_3d.svg",
    cmap: str = "viridis",
    z_label: str = "Cross-Entropy Loss",
    acc_surface: Optional[np.ndarray] = None,
    resolution: int = 21,
    seed: int = 42,
    max_batches: Optional[int] = None,
    checkpoint_path: str = ""
) -> None:
    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.85,
                           rstride=1, cstride=1, antialiased=True)

    center_z = Z[Z.shape[0] // 2, Z.shape[1] // 2]
    ax.scatter([0], [0], [center_z], color='red', s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10, depthshade=False)

    ax.plot([0, 0], [0, 0], [Z.min(), center_z], color='red',
            linestyle='--', linewidth=1.0, alpha=0.7)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, aspect=20)
    cbar.set_label(z_label, fontsize=12, fontweight='bold')

    ax.set_xlabel(r'Direction $d_1$ ($\alpha$)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel(r'Direction $d_2$ ($\beta$)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel(z_label, fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=25)

    loss_ratio = Z.max() / max(center_z, 1e-10)
    stats_text = (
        f"Center: {center_z:.4f}\n"
        f"Min: {Z.min():.4f}\n"
        f"Max: {Z.max():.4f}\n"
        f"Ratio: {loss_ratio:.1f}x"
    )
    if acc_surface is not None:
        center_acc = acc_surface[acc_surface.shape[0] // 2, acc_surface.shape[1] // 2]
        stats_text += f"\nCenter Acc: {center_acc * 100:.1f}%"

    ax.text2D(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
              fontfamily='monospace', verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='#f8f8f8', alpha=0.95,
                        edgecolor='#333333', linewidth=1.2))

    method_text = (
        f"Filter-normalized random directions (Li et al., 2018)\n"
        f"Grid: {resolution}x{resolution} | Seed: {seed}"
    )
    if max_batches is not None:
        method_text += f" | Batches: {max_batches}"
    method_text += f"\n" + r"$\theta' = \theta + \alpha \cdot d_1 + \beta \cdot d_2$"

    ax.text2D(0.5, -0.02, method_text, transform=ax.transAxes, fontsize=8,
              ha='center', va='top', fontfamily='monospace', color='#555555',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0',
                        edgecolor='#cccccc', alpha=0.9))

    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Saved 3D surface plot to: {output_path}")
    plt.close()


def main():
    """Main function to generate loss landscape visualization."""

    # Configuration
    checkpoint_path = "checkpoints/best-checkpoint.ckpt"  # Adjust path as needed
    data_dir = "./data"
    batch_size = 256
    resolution = 21  # 21x21 = 441 points (higher = smoother but slower)
    x_range = (-1.0, 1.0)  # Range for direction 1
    y_range = (-1.0, 1.0)  # Range for direction 2
    max_batches = 10  # Limit batches for faster computation (None = all)
    seed = 42

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    data_module = FashionMNISTDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,  # Avoid multiprocessing issues
        seed=seed
    )
    data_module.setup('test')
    test_loader = data_module.test_dataloader()

    # Load model from checkpoint
    print(f"Loading model from: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        model = load_model_from_checkpoint(checkpoint_path, device)
    else:
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please provide a valid checkpoint path.")
        return

    model = model.to(device)
    model.eval()

    # Compute loss surface
    print(f"\nComputing loss surface (resolution={resolution}x{resolution})...")
    print(f"X range: {x_range}, Y range: {y_range}")

    X, Y, loss_surface, acc_surface = compute_loss_surface(
        model=model,
        dataloader=test_loader,
        device=device,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        max_batches=max_batches,
        seed=seed
    )

    # Print statistics
    print("\n--- Loss Surface Statistics ---")
    print(f"Min Loss:    {loss_surface.min():.4f}")
    print(f"Max Loss:    {loss_surface.max():.4f}")
    print(f"Center Loss: {loss_surface[resolution//2, resolution//2]:.4f}")
    print(f"Min Acc:     {acc_surface.min():.4f}")
    print(f"Max Acc:     {acc_surface.max():.4f}")
    print(f"Center Acc:  {acc_surface[resolution//2, resolution//2]:.4f}")

    # Save surfaces for later use
    np.savez('loss_landscape_data.npz', X=X, Y=Y, loss=loss_surface, acc=acc_surface)
    print("\nSaved loss landscape data to: loss_landscape_data.npz")

    # Create visualizations
    print("\nGenerating visualizations...")

    plot_kwargs = dict(
        resolution=resolution,
        seed=seed,
        max_batches=max_batches,
        checkpoint_path=checkpoint_path,
    )

    plot_2d_contour(
        X, Y, loss_surface,
        title="Loss Landscape (Filter-Normalized Random Directions)",
        output_path="loss_landscape_2d.svg",
        z_label="Cross-Entropy Loss",
        acc_surface=acc_surface,
        **plot_kwargs,
    )

    plot_3d_surface(
        X, Y, loss_surface,
        title="Loss Landscape (3D View)",
        output_path="loss_landscape_3d.svg",
        z_label="Cross-Entropy Loss",
        acc_surface=acc_surface,
        **plot_kwargs,
    )

    plot_2d_contour(
        X, Y, acc_surface,
        title="Accuracy Landscape (Filter-Normalized Random Directions)",
        output_path="accuracy_landscape_2d.svg",
        cmap="RdYlGn",
        z_label="Accuracy",
        **plot_kwargs,
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
