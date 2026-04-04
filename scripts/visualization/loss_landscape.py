"""
Loss Landscape Visualization

Implements methods from:
"Visualizing the Loss Landscape of Neural Nets" (Li et al., 2018)
https://arxiv.org/abs/1712.09913

Methods:
  1. 1D Linear Interpolation between two models
  2. 2D Contour Plots with filter-normalized random directions
  3. PCA-based trajectory visualization from training checkpoints
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from typing import Tuple, List, Optional
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datamodule import FashionMNISTDataModule
from plot_fixes import plot_multi_optimizer_interpolation, plot_pca_trajectory


# ============================================================
# Model loading
# ============================================================

class FashionMNISTModelOld(nn.Module):
    """Original model architecture for loading old checkpoints."""
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

    state_dict = checkpoint.get('state_dict', checkpoint)

    # Remove 'model.' prefix if present (from Lightning)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('model.', '') if k.startswith('model.') else k
        cleaned_state_dict[new_key] = v

    if 'conv1.weight' in cleaned_state_dict:
        print("Detected OLD model architecture (conv1, bn1, etc.)")
        model = FashionMNISTModelOld()
    elif 'feature_extractor.0.weight' in cleaned_state_dict:
        print("Detected NEW model architecture (feature_extractor, classifier)")
        from models.lit_model import FashionMNISTModel
        model = FashionMNISTModel()
    else:
        raise ValueError(f"Unknown architecture. Keys: {list(cleaned_state_dict.keys())[:5]}")

    model_keys = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in cleaned_state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state_dict)

    return model


# ============================================================
# Weight manipulation utilities
# ============================================================

def get_weights(model: nn.Module) -> List[torch.Tensor]:
    """Extract all trainable parameters from model as a list of tensors."""
    return [p.data.clone() for p in model.parameters()]


def set_weights(model: nn.Module, weights: List[torch.Tensor]) -> None:
    """Set model parameters from a list of tensors."""
    for p, w in zip(model.parameters(), weights):
        p.data.copy_(w)


def get_state(model: nn.Module) -> List[torch.Tensor]:
    """Extract full model state (parameters + buffers like BN running stats)."""
    return [v.data.clone() for v in model.state_dict().values()]


def set_state(model: nn.Module, state: List[torch.Tensor]) -> None:
    """Set full model state (parameters + buffers)."""
    for (name, param), val in zip(model.state_dict().items(), state):
        param.copy_(val)


def flatten_weights(weights: List[torch.Tensor]) -> np.ndarray:
    """Flatten a list of weight tensors into a single 1D numpy array."""
    return np.concatenate([w.cpu().numpy().flatten() for w in weights])


def unflatten_weights(flat: np.ndarray, reference: List[torch.Tensor]) -> List[torch.Tensor]:
    """Reconstruct weight tensors from a flat numpy array using reference shapes."""
    result = []
    offset = 0
    for w in reference:
        n = w.numel()
        result.append(torch.from_numpy(flat[offset:offset + n].reshape(w.shape)).float())
        offset += n
    return result


# ============================================================
# Direction generation and normalization
# ============================================================

def get_random_direction(weights: List[torch.Tensor]) -> List[torch.Tensor]:
    """Generate a random Gaussian direction with same shape as weights."""
    return [torch.randn_like(w) for w in weights]


def normalize_direction_filter_wise(
    direction: List[torch.Tensor],
    weights: List[torch.Tensor],
    ignore_bn: bool = True
) -> List[torch.Tensor]:
    """
    Apply filter-wise normalization: d_{i,j} <- (d_{i,j} / ||d_{i,j}||) * ||theta_{i,j}||

    This removes scale invariance artifacts (Li et al., 2018, Eq. 2).
    """
    normalized = []

    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore_bn and d.dim() == 1:
                normalized.append(torch.zeros_like(d))
            else:
                d_norm = d.norm()
                w_norm = w.norm()
                if d_norm > 1e-10:
                    normalized.append(d * (w_norm / d_norm))
                else:
                    normalized.append(torch.zeros_like(d))
        elif d.dim() in (2, 4):
            # 2D = Linear (each row = one filter), 4D = Conv2d (each d[i] = one filter)
            d_normalized = torch.zeros_like(d)
            for i in range(d.shape[0]):
                d_norm = d[i].norm()
                w_norm = w[i].norm()
                if d_norm > 1e-10:
                    d_normalized[i] = d[i] * (w_norm / d_norm)
            normalized.append(d_normalized)
        else:
            d_norm = d.norm()
            w_norm = w.norm()
            if d_norm > 1e-10:
                normalized.append(d * (w_norm / d_norm))
            else:
                normalized.append(torch.zeros_like(d))

    return normalized


# ============================================================
# Loss evaluation
# ============================================================

def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Tuple[float, float]:
    """Evaluate model, return (loss, accuracy)."""
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

    return total_loss / total_samples, total_correct / total_samples


# ============================================================
# METHOD 1: 1D Linear Interpolation
# ============================================================

def compute_1d_interpolation(
    model: nn.Module,
    state_a: List[torch.Tensor],
    state_b: List[torch.Tensor],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_points: int = 51,
    alpha_range: Tuple[float, float] = (-0.5, 1.5),
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1D linear interpolation: theta(alpha) = (1-alpha)*theta_A + alpha*theta_B

    At alpha=0 we have model A, at alpha=1 we have model B.
    The range extends beyond [0,1] to show the landscape around both solutions.
    Interpolates FULL state (parameters + BN running stats) to avoid BN mismatch.

    Returns: (alphas, losses, accuracies)
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], num_points)
    losses = np.zeros(num_points)
    accuracies = np.zeros(num_points)

    for idx, alpha in enumerate(tqdm(alphas, desc="1D interpolation")):
        interpolated = [
            (1 - alpha) * sa + alpha * sb
            for sa, sb in zip(state_a, state_b)
        ]
        set_state(model, interpolated)
        loss, acc = evaluate_model(model, dataloader, device, max_batches)
        losses[idx] = loss
        accuracies[idx] = acc

    return alphas, losses, accuracies


def plot_1d_interpolation(
    alphas: np.ndarray,
    losses: np.ndarray,
    accuracies: np.ndarray,
    output_path: str = "interpolation_1d.svg",
    label_a: str = "Model A",
    label_b: str = "Model B",
) -> None:
    """
    Plot 1D interpolation -- loss and accuracy curves side by side.
    Styled similar to Li et al. (2018), Figure 3 top row.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(alphas, losses, 'b-', linewidth=2)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label=label_a)
    ax1.axvline(x=1, color='green', linestyle='--', linewidth=1, alpha=0.7, label=label_b)
    ax1.scatter([0], [losses[np.argmin(np.abs(alphas))]], color='red', s=80, zorder=5)
    ax1.scatter([1], [losses[np.argmin(np.abs(alphas - 1))]], color='green', s=80, zorder=5)
    ax1.set_xlabel(r'Interpolation coefficient $\alpha$', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.set_title('Loss Along Interpolation Path', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Check for barrier
    idx_0 = np.argmin(np.abs(alphas))
    idx_1 = np.argmin(np.abs(alphas - 1))
    loss_a = losses[idx_0]
    loss_b = losses[idx_1]
    max_between = losses[min(idx_0, idx_1):max(idx_0, idx_1) + 1].max()
    barrier = max_between - max(loss_a, loss_b)
    if barrier > 0.01:
        ax1.annotate(
            f'Barrier: {barrier:.3f}',
            xy=(alphas[min(idx_0, idx_1) + np.argmax(losses[min(idx_0, idx_1):max(idx_0, idx_1) + 1])],
                max_between),
            xytext=(0.5, max_between + (losses.max() - losses.min()) * 0.1),
            fontsize=10, ha='center', color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='darkred')
        )

    # Accuracy curve
    ax2.plot(alphas, accuracies * 100, 'b-', linewidth=2)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label=label_a)
    ax2.axvline(x=1, color='green', linestyle='--', linewidth=1, alpha=0.7, label=label_b)
    ax2.scatter([0], [accuracies[np.argmin(np.abs(alphas))] * 100], color='red', s=80, zorder=5)
    ax2.scatter([1], [accuracies[np.argmin(np.abs(alphas - 1))] * 100], color='green', s=80, zorder=5)
    ax2.set_xlabel(r'Interpolation coefficient $\alpha$', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Along Interpolation Path', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        r'1D Linear Interpolation: $\theta(\alpha) = (1-\alpha)\,\theta_A + \alpha\,\theta_B$',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Saved 1D interpolation plot to: {output_path}")
    plt.close()


# ============================================================
# METHOD 1b: 1D Random Direction (Li et al., 2018, Figure 3 top)
# ============================================================

def compute_1d_direction(
    model: nn.Module,
    base_weights: List[torch.Tensor],
    direction: List[torch.Tensor],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_points: int = 51,
    alpha_range: Tuple[float, float] = (-1.0, 1.0),
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1D perturbation along a filter-normalized random direction:
        f(alpha) = L(theta* + alpha * d)

    This matches Li et al. (2018) Figure 3 top row.

    Returns: (alphas, losses, accuracies)
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], num_points)
    losses = np.zeros(num_points)
    accuracies = np.zeros(num_points)

    for idx, alpha in enumerate(tqdm(alphas, desc="1D direction")):
        perturbed = [w + alpha * d for w, d in zip(base_weights, direction)]
        set_weights(model, perturbed)
        loss, acc = evaluate_model(model, dataloader, device, max_batches)
        losses[idx] = loss
        accuracies[idx] = acc

    set_weights(model, base_weights)  # restore
    return alphas, losses, accuracies


def plot_1d_direction(
    alphas: np.ndarray,
    losses: np.ndarray,
    accuracies: np.ndarray,
    output_path: str = "direction_1d.svg",
    seed: int = 42,
    checkpoint_path: str = "",
) -> None:
    """
    Plot 1D loss and accuracy along a filter-normalized random direction.
    Styled to match Li et al. (2018) Figure 3 top row.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(alphas, losses, 'b-', linewidth=2)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Trained model')
    center_idx = np.argmin(np.abs(alphas))
    ax1.scatter([0], [losses[center_idx]], color='red', s=80, zorder=5)
    ax1.set_xlabel(r'Perturbation coefficient $\alpha$', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.set_title('Loss Along Random Direction', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(alphas, accuracies * 100, 'b-', linewidth=2)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Trained model')
    ax2.scatter([0], [accuracies[center_idx] * 100], color='red', s=80, zorder=5)
    ax2.set_xlabel(r'Perturbation coefficient $\alpha$', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Along Random Direction', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        r'1D Random Direction: $f(\alpha) = L(\theta^* + \alpha \cdot d)$'
        f'\nFilter-normalized (Li et al., 2018) | Seed: {seed}',
        fontsize=14, fontweight='bold', y=1.04
    )
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Saved 1D direction plot to: {output_path}")
    plt.close()


# ============================================================
# METHOD 1c: Multi-optimizer 1D interpolation comparison
# ============================================================

def plot_multi_optimizer_interpolation(
    results: dict,
    output_path: str = "interpolation_multi_optimizer.svg",
) -> None:
    """
    Plot 1D interpolation curves for multiple optimizers on a single figure.

    Args:
        results: dict mapping optimizer_name -> (alphas, losses, accuracies)
    """
    colors = {
        "adamw": "#1f77b4",
        "sgd": "#d62728",
        "adam": "#2ca02c",
        "rmsprop": "#ff7f0e",
        "vanilla": "#9467bd",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for opt_name, (alphas, losses, accuracies) in results.items():
        color = colors.get(opt_name.lower(), "gray")
        label = "Vanilla SGD (no momentum/wd)" if opt_name.lower() == "vanilla" else opt_name.upper()

        ax1.plot(alphas, losses, '-', color=color, linewidth=2, label=label)
        idx_0 = np.argmin(np.abs(alphas))
        idx_1 = np.argmin(np.abs(alphas - 1))
        ax1.scatter([0], [losses[idx_0]], color=color, s=60, zorder=5, marker='o')
        ax1.scatter([1], [losses[idx_1]], color=color, s=60, zorder=5, marker='s')

        ax2.plot(alphas, accuracies * 100, '-', color=color, linewidth=2, label=label)
        ax2.scatter([0], [accuracies[idx_0] * 100], color=color, s=60, zorder=5, marker='o')
        ax2.scatter([1], [accuracies[idx_1] * 100], color=color, s=60, zorder=5, marker='s')

    ax1.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.axvline(x=1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.set_xlabel(r'Interpolation coefficient $\alpha$', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.set_title('Loss Along Interpolation Path', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axvline(x=1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_xlabel(r'Interpolation coefficient $\alpha$', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Along Interpolation Path', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        r'Optimizer Comparison: $\theta(\alpha) = (1-\alpha)\,\theta_{seed_0} + \alpha\,\theta_{seed_1}$'
        '\nEach curve shows interpolation between two models trained with the same optimizer but different seeds',
        fontsize=13, fontweight='bold', y=1.06
    )
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Saved multi-optimizer interpolation plot to: {output_path}")
    plt.close()


def run_multi_optimizer_interpolation(
    optimizer_dirs: dict,
    data_dir: str = "./data",
    batch_size: int = 256,
    num_points: int = 51,
    alpha_range: Tuple[float, float] = (-0.5, 1.5),
    max_batches: Optional[int] = None,
    output_dir: str = ".",
):
    """
    Run 1D interpolation for multiple optimizers and plot comparison.

    Args:
        optimizer_dirs: dict mapping optimizer_name -> (checkpoint_a_path, checkpoint_b_path)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_module = FashionMNISTDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=0)
    data_module.setup('test')
    test_loader = data_module.test_dataloader()

    results = {}

    for opt_name, (ckpt_a, ckpt_b) in optimizer_dirs.items():
        print(f"\n{'='*60}")
        print(f"Interpolating {opt_name.upper()}: {ckpt_a} <-> {ckpt_b}")
        print(f"{'='*60}")

        model_a = load_model_from_checkpoint(ckpt_a, device).to(device)
        state_a = get_state(model_a)

        model_b = load_model_from_checkpoint(ckpt_b, device).to(device)
        state_b = get_state(model_b)

        model_a.eval()

        alphas, losses, accuracies = compute_1d_interpolation(
            model_a, state_a, state_b, test_loader, device,
            num_points, alpha_range, max_batches
        )

        set_state(model_a, state_a)
        results[opt_name] = (alphas, losses, accuracies)

        # Save individual data
        np.savez(
            os.path.join(output_dir, f'interpolation_1d_{opt_name}.npz'),
            alphas=alphas, losses=losses, accuracies=accuracies
        )

    plot_multi_optimizer_interpolation(
        results,
        output_path=os.path.join(output_dir, "interpolation_multi_optimizer.svg"),
    )

    print("\nMulti-optimizer interpolation done!")


# ============================================================
# METHOD 2: 2D Contour Plots (filter-normalized random directions)
# ============================================================

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
    Compute 2D loss surface: f(alpha, beta) = L(theta* + alpha*d1 + beta*d2)
    using filter-normalized random directions d1, d2.

    Returns: (X, Y, loss_surface, acc_surface)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    base_weights = get_weights(model)

    print("Generating and normalizing random directions...")
    direction1 = normalize_direction_filter_wise(get_random_direction(base_weights), base_weights)
    direction2 = normalize_direction_filter_wise(get_random_direction(base_weights), base_weights)

    alphas = np.linspace(x_range[0], x_range[1], resolution)
    betas = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(alphas, betas)

    loss_surface = np.zeros_like(X)
    acc_surface = np.zeros_like(X)

    total_points = resolution * resolution
    pbar = tqdm(total=total_points, desc="Computing 2D loss surface")

    for i in range(resolution):
        for j in range(resolution):
            new_weights = [
                w + alphas[j] * d1 + betas[i] * d2
                for w, d1, d2 in zip(base_weights, direction1, direction2)
            ]
            set_weights(model, new_weights)
            loss, acc = evaluate_model(model, dataloader, device, max_batches)
            loss_surface[i, j] = loss
            acc_surface[i, j] = acc
            pbar.update(1)

    pbar.close()
    set_weights(model, base_weights)  # restore

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
    """
    Paper-style contour plot: lines only, log-spaced levels so contours
    cluster densely near the minimum — matching Li et al. (2018) figures.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Log-spaced levels: dense near minimum, sparse at high loss
    z_min = Z.min()
    z_max = Z.max()
    # Shift to positive range for log spacing
    log_min = np.log(max(z_min, 1e-8))
    log_max = np.log(z_max)
    log_levels = np.linspace(log_min, log_max, levels)
    contour_levels = np.exp(log_levels)

    # Contour lines only — no fill
    cs = ax.contour(
        X, Y, Z,
        levels=contour_levels,
        colors='black',
        linewidths=0.8,
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f',
              levels=cs.levels[::max(1, len(cs.levels) // 8)])

    # Mark the trained model (center)
    ax.plot(0, 0, 'r*', markersize=18, markeredgecolor='black', markeredgewidth=1.0, zorder=10)

    ax.set_xlabel(r'$\delta$', fontsize=14)
    ax.set_ylabel(r'$\eta$', fontsize=14)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=11)

    plt.tight_layout()
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


# ============================================================
# METHOD 3: PCA-based trajectory visualization
# ============================================================

def load_trajectory_checkpoints(
    checkpoint_dir: str,
    device: torch.device,
    pattern: str = "epoch_*.ckpt"
) -> Tuple[List[List[torch.Tensor]], List[int]]:
    """
    Load all epoch checkpoints from a directory.
    Returns list of weight vectors and corresponding epoch numbers.
    """
    import glob

    files = sorted(glob.glob(os.path.join(checkpoint_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No checkpoints matching '{pattern}' found in {checkpoint_dir}"
        )

    weights_list = []
    epochs = []

    for f in tqdm(files, desc="Loading checkpoints"):
        # Extract epoch number from filename
        basename = os.path.basename(f)
        epoch_str = basename.replace("epoch_", "").replace(".ckpt", "")
        try:
            epoch = int(epoch_str)
        except ValueError:
            continue

        checkpoint = torch.load(f, map_location=device, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Clean keys (remove Lightning prefix)
        cleaned = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.', '') if k.startswith('model.') else k
            cleaned[new_key] = v

        # Only keep trainable parameter keys, in the same order as model.parameters()
        # This ensures compatibility with set_weights() which iterates model.parameters()
        param_keys = [k for k in cleaned.keys()
                      if 'running_' not in k and 'num_batches_tracked' not in k]
        # Use model definition order (not alphabetical) by matching against a reference model
        param_tensors = [cleaned[k].cpu() for k in param_keys if k in cleaned]

        weights_list.append(param_tensors)
        epochs.append(epoch)

    print(f"Loaded {len(weights_list)} checkpoints (epochs {epochs[0]}-{epochs[-1]})")
    return weights_list, epochs


def compute_pca_directions(
    weights_list: List[List[torch.Tensor]],
    n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run PCA on the training trajectory to find the principal directions
    of weight variation during training.

    Returns:
        pca_components: shape (n_components, n_params) - the PCA direction vectors
        projected: shape (n_checkpoints, n_components) - trajectory in PCA space
        explained_variance_ratio: how much variance each PC explains
    """
    # Flatten all checkpoints into a matrix: (n_checkpoints, n_params)
    matrix = np.array([flatten_weights(w) for w in weights_list])

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(matrix)

    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
    if n_components >= 2:
        print(f"  PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")

    return pca.components_, projected, pca.explained_variance_ratio_


def compute_pca_loss_surface(
    model: nn.Module,
    base_weights: List[torch.Tensor],
    pca_components: np.ndarray,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
    resolution: int = 21,
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute loss surface along the two PCA directions.
    theta' = theta_final + alpha * pc1 + beta * pc2

    The PCA components are already in the right scale (they represent actual
    weight differences observed during training), so no filter normalization
    is needed here.

    Returns: (X, Y, loss_surface, acc_surface)
    """
    # Convert PCA components to weight tensor lists
    pc1_tensors = unflatten_weights(pca_components[0], base_weights)
    pc2_tensors = unflatten_weights(pca_components[1], base_weights)

    # Move to device
    pc1_tensors = [t.to(device) for t in pc1_tensors]
    pc2_tensors = [t.to(device) for t in pc2_tensors]

    alphas = np.linspace(x_range[0], x_range[1], resolution)
    betas = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(alphas, betas)

    loss_surface = np.zeros_like(X)
    acc_surface = np.zeros_like(X)

    total_points = resolution * resolution
    pbar = tqdm(total=total_points, desc="Computing PCA loss surface")

    for i in range(resolution):
        for j in range(resolution):
            new_weights = [
                w + alphas[j] * p1 + betas[i] * p2
                for w, p1, p2 in zip(base_weights, pc1_tensors, pc2_tensors)
            ]
            set_weights(model, new_weights)
            loss, acc = evaluate_model(model, dataloader, device, max_batches)
            loss_surface[i, j] = loss
            acc_surface[i, j] = acc
            pbar.update(1)

    pbar.close()
    set_weights(model, base_weights)  # restore

    return X, Y, loss_surface, acc_surface


def plot_pca_trajectory(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    projected: np.ndarray,
    epochs: List[int],
    explained_variance: np.ndarray,
    output_path: str = "pca_trajectory.svg",
    title: str = "Training Trajectory on PCA Loss Surface",
    cmap: str = "viridis",
    levels: int = 30,
) -> None:
    """
    Plot the 2D PCA loss surface with the training trajectory overlaid.
    Similar to Li et al. (2018) Figure 6 concept.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.3, alpha=0.4)
    cbar = fig.colorbar(contour, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label('Cross-Entropy Loss', fontsize=12, fontweight='bold')

    # Plot trajectory
    traj_x = projected[:, 0]
    traj_y = projected[:, 1]

    # Color trajectory by epoch
    n_pts = len(traj_x)
    colors = plt.cm.hot(np.linspace(0, 0.8, n_pts))

    for i in range(n_pts - 1):
        ax.plot(traj_x[i:i+2], traj_y[i:i+2], '-', color=colors[i], linewidth=2.0)

    # Mark start, end, and some milestones
    ax.scatter(traj_x[0], traj_y[0], color='blue', s=150, marker='o',
               edgecolors='black', linewidths=1.5, zorder=10, label=f'Start (epoch {epochs[0]})')
    ax.scatter(traj_x[-1], traj_y[-1], color='red', s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10, label=f'End (epoch {epochs[-1]})')

    # Mark every ~25% of training
    n_marks = 4
    for k in range(1, n_marks):
        idx = int(k * (n_pts - 1) / n_marks)
        ax.scatter(traj_x[idx], traj_y[idx], color=colors[idx], s=60, marker='D',
                   edgecolors='black', linewidths=1, zorder=9)
        ax.annotate(f'ep {epochs[idx]}', (traj_x[idx], traj_y[idx]),
                    textcoords="offset points", xytext=(8, 8), fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax.set_xlabel(
        f'PC1 ({explained_variance[0]*100:.1f}% variance explained)',
        fontsize=13, fontweight='bold'
    )
    ax.set_ylabel(
        f'PC2 ({explained_variance[1]*100:.1f}% variance explained)',
        fontsize=13, fontweight='bold'
    )
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper right')

    method_text = (
        f"PCA directions from {len(epochs)} training checkpoints\n"
        r"$\theta' = \theta_{final} + \alpha \cdot PC_1 + \beta \cdot PC_2$"
    )
    ax.text(0.5, -0.06, method_text, transform=ax.transAxes, fontsize=9,
            ha='center', va='top', fontfamily='monospace', color='#555555',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='#cccccc', alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Saved PCA trajectory plot to: {output_path}")
    plt.close()


def plot_pca_trajectory_3d(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    projected: np.ndarray,
    epochs: List[int],
    explained_variance: np.ndarray,
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    weights_list: List[List[torch.Tensor]],
    device: torch.device,
    max_batches: Optional[int] = None,
    output_path: str = "pca_trajectory_3d.svg",
    title: str = "Training Trajectory on PCA Loss Surface (3D)",
    cmap: str = "viridis",
) -> None:
    """
    3D surface plot with training trajectory drawn on top.
    The trajectory floats at the actual loss value for each checkpoint.
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.7,
                           rstride=1, cstride=1, antialiased=True)

    # Evaluate loss at each checkpoint to get z-values for the trajectory
    traj_x = projected[:, 0]
    traj_y = projected[:, 1]
    traj_z = np.zeros(len(traj_x))

    print("Evaluating loss at trajectory points for 3D plot...")
    for idx, weights in enumerate(tqdm(weights_list, desc="Trajectory losses")):
        weights_on_device = [w.to(device) for w in weights]
        set_weights(model, weights_on_device)
        loss, _ = evaluate_model(model, dataloader, device, max_batches)
        traj_z[idx] = loss

    n_pts = len(traj_x)
    colors = plt.cm.hot(np.linspace(0, 0.8, n_pts))

    for i in range(n_pts - 1):
        ax.plot(traj_x[i:i+2], traj_y[i:i+2], traj_z[i:i+2],
                '-', color=colors[i], linewidth=2.5)

    ax.scatter(traj_x[0], traj_y[0], traj_z[0], color='blue', s=150, marker='o',
               edgecolors='black', linewidths=1.5, depthshade=False, label=f'Start (ep {epochs[0]})')
    ax.scatter(traj_x[-1], traj_y[-1], traj_z[-1], color='red', s=200, marker='*',
               edgecolors='black', linewidths=1.5, depthshade=False, label=f'End (ep {epochs[-1]})')

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, aspect=20)
    cbar.set_label('Cross-Entropy Loss', fontsize=12, fontweight='bold')

    ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel('Loss', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=25)
    ax.legend(fontsize=10)

    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Saved 3D PCA trajectory plot to: {output_path}")
    plt.close()


# ============================================================
# Main entry points
# ============================================================

def run_2d_landscape(
    checkpoint_path: str = "checkpoints/best-checkpoint.ckpt",
    data_dir: str = "./data",
    batch_size: int = 256,
    resolution: int = 21,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
    max_batches: Optional[int] = None,
    seed: int = 42,
    output_dir: str = ".",
):
    """Generate 2D and 3D loss landscape with filter-normalized random directions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_module = FashionMNISTDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=0, seed=seed)
    data_module.setup('test')
    test_loader = data_module.test_dataloader()

    model = load_model_from_checkpoint(checkpoint_path, device).to(device)
    model.eval()

    X, Y, loss_surface, acc_surface = compute_loss_surface(
        model, test_loader, device, x_range, y_range, resolution, max_batches, seed
    )

    np.savez(os.path.join(output_dir, 'loss_landscape_data.npz'),
             X=X, Y=Y, loss=loss_surface, acc=acc_surface)

    plot_kwargs = dict(resolution=resolution, seed=seed, max_batches=max_batches,
                       checkpoint_path=checkpoint_path)

    plot_2d_contour(X, Y, loss_surface,
                    title="Loss Landscape (Filter-Normalized Random Directions)",
                    output_path=os.path.join(output_dir, "loss_landscape_2d.svg"),
                    z_label="Cross-Entropy Loss", acc_surface=acc_surface, **plot_kwargs)

    plot_3d_surface(X, Y, loss_surface,
                    title="Loss Landscape (3D View)",
                    output_path=os.path.join(output_dir, "loss_landscape_3d.svg"),
                    z_label="Cross-Entropy Loss", acc_surface=acc_surface, **plot_kwargs)

    plot_2d_contour(X, Y, acc_surface,
                    title="Accuracy Landscape (Filter-Normalized Random Directions)",
                    output_path=os.path.join(output_dir, "accuracy_landscape_2d.svg"),
                    cmap="RdYlGn", z_label="Accuracy", **plot_kwargs)

    print("2D landscape done!")


def run_1d_direction(
    checkpoint_path: str = "checkpoints/best-checkpoint.ckpt",
    data_dir: str = "./data",
    batch_size: int = 256,
    num_points: int = 51,
    alpha_range: Tuple[float, float] = (-1.0, 1.0),
    max_batches: Optional[int] = None,
    seed: int = 42,
    output_dir: str = ".",
):
    """Generate 1D loss plot along a filter-normalized random direction (Li et al., 2018, Fig. 3 top)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    data_module = FashionMNISTDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=0, seed=seed)
    data_module.setup('test')
    test_loader = data_module.test_dataloader()

    model = load_model_from_checkpoint(checkpoint_path, device).to(device)
    model.eval()

    base_weights = get_weights(model)
    direction = normalize_direction_filter_wise(get_random_direction(base_weights), base_weights)

    alphas, losses, accuracies = compute_1d_direction(
        model, base_weights, direction, test_loader, device,
        num_points, alpha_range, max_batches
    )

    np.savez(os.path.join(output_dir, 'direction_1d_data.npz'),
             alphas=alphas, losses=losses, accuracies=accuracies)

    plot_1d_direction(
        alphas, losses, accuracies,
        output_path=os.path.join(output_dir, "direction_1d.svg"),
        seed=seed, checkpoint_path=checkpoint_path,
    )

    print("1D direction done!")


def run_1d_interpolation(
    checkpoint_a: str,
    checkpoint_b: str,
    data_dir: str = "./data",
    batch_size: int = 256,
    num_points: int = 51,
    alpha_range: Tuple[float, float] = (-0.5, 1.5),
    max_batches: Optional[int] = None,
    output_dir: str = ".",
    label_a: str = "Model A",
    label_b: str = "Model B",
):
    """Generate 1D interpolation plot between two trained models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_module = FashionMNISTDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=0)
    data_module.setup('test')
    test_loader = data_module.test_dataloader()

    # Load both models — use full state (params + BN buffers)
    print(f"Loading model A: {checkpoint_a}")
    model_a = load_model_from_checkpoint(checkpoint_a, device).to(device)
    state_a = get_state(model_a)

    print(f"Loading model B: {checkpoint_b}")
    model_b = load_model_from_checkpoint(checkpoint_b, device).to(device)
    state_b = get_state(model_b)

    # Use model_a as the evaluator
    model_a.eval()

    alphas, losses, accuracies = compute_1d_interpolation(
        model_a, state_a, state_b, test_loader, device,
        num_points, alpha_range, max_batches
    )

    # Restore
    set_state(model_a, state_a)

    np.savez(os.path.join(output_dir, 'interpolation_1d_data.npz'),
             alphas=alphas, losses=losses, accuracies=accuracies)

    plot_1d_interpolation(
        alphas, losses, accuracies,
        output_path=os.path.join(output_dir, "interpolation_1d.svg"),
        label_a=label_a, label_b=label_b,
    )

    print("1D interpolation done!")


def run_pca_trajectory(
    checkpoint_dir: str,
    data_dir: str = "./data",
    batch_size: int = 256,
    resolution: int = 21,
    max_batches: Optional[int] = None,
    output_dir: str = ".",
    margin: float = 0.3,
):
    """
    Generate PCA trajectory visualization.
    Requires epoch checkpoints saved as epoch_0.ckpt, epoch_1.ckpt, ...
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_module = FashionMNISTDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=0)
    data_module.setup('test')
    test_loader = data_module.test_dataloader()

    # Load all epoch checkpoints
    weights_list, epochs = load_trajectory_checkpoints(checkpoint_dir, device)

    # Run PCA
    print("Running PCA on training trajectory...")
    pca_components, projected, explained_variance = compute_pca_directions(weights_list)

    # Determine grid range from trajectory extent + margin
    x_min, x_max = projected[:, 0].min(), projected[:, 0].max()
    y_min, y_max = projected[:, 1].min(), projected[:, 1].max()
    x_pad = (x_max - x_min) * margin
    y_pad = (y_max - y_min) * margin
    x_range = (x_min - x_pad, x_max + x_pad)
    y_range = (y_min - y_pad, y_max + y_pad)
    print(f"Grid range: x={x_range}, y={y_range}")

    # Load the final model for evaluation
    # Use the last checkpoint's weights as the base (center of PCA space)
    final_weights = weights_list[-1]

    # We need a model instance to evaluate
    # Try to find the original checkpoint to load architecture
    import glob
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "epoch_*.ckpt"))
    model = load_model_from_checkpoint(ckpt_files[-1], device).to(device)

    # The PCA is centered on the mean of all checkpoints.
    # projected coordinates are relative to this mean.
    # So base_weights for the surface should be the PCA mean.
    mean_flat = np.mean([flatten_weights(w) for w in weights_list], axis=0)
    base_weights = unflatten_weights(mean_flat, weights_list[0])
    base_weights = [w.to(device) for w in base_weights]

    # Compute loss surface along PCA directions
    X, Y, loss_surface, acc_surface = compute_pca_loss_surface(
        model, base_weights, pca_components, test_loader, device,
        x_range, y_range, resolution, max_batches
    )

    np.savez(os.path.join(output_dir, 'pca_trajectory_data.npz'),
             X=X, Y=Y, loss=loss_surface, acc=acc_surface,
             projected=projected, epochs=np.array(epochs),
             explained_variance=explained_variance)

    # 2D trajectory plot
    plot_pca_trajectory(
        X, Y, loss_surface, projected, epochs, explained_variance,
        output_path=os.path.join(output_dir, "pca_trajectory_2d.svg"),
    )

    # 3D trajectory plot
    plot_pca_trajectory_3d(
        X, Y, loss_surface, projected, epochs, explained_variance,
        test_loader, model, weights_list, device, max_batches,
        output_path=os.path.join(output_dir, "pca_trajectory_3d.svg"),
    )

    print("PCA trajectory done!")


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Loss Landscape Visualization")
    subparsers = parser.add_subparsers(dest="command", help="Visualization method")

    # 2D landscape
    p2d = subparsers.add_parser("landscape2d", help="2D/3D loss landscape with random directions")
    p2d.add_argument("--checkpoint", default="checkpoints/best-checkpoint.ckpt")
    p2d.add_argument("--data-dir", default="./data")
    p2d.add_argument("--batch-size", type=int, default=256)
    p2d.add_argument("--resolution", type=int, default=21)
    p2d.add_argument("--x-range", type=float, nargs=2, default=[-1.0, 1.0])
    p2d.add_argument("--y-range", type=float, nargs=2, default=[-1.0, 1.0])
    p2d.add_argument("--max-batches", type=int, default=None)
    p2d.add_argument("--seed", type=int, default=42)
    p2d.add_argument("--output-dir", default=".")

    # 1D random direction (Li et al., 2018, Figure 3 top)
    p1dr = subparsers.add_parser("direction1d", help="1D loss along filter-normalized random direction (Fig. 3 top)")
    p1dr.add_argument("--checkpoint", default="checkpoints/best-checkpoint.ckpt")
    p1dr.add_argument("--data-dir", default="./data")
    p1dr.add_argument("--batch-size", type=int, default=256)
    p1dr.add_argument("--num-points", type=int, default=51)
    p1dr.add_argument("--alpha-range", type=float, nargs=2, default=[-1.0, 1.0])
    p1dr.add_argument("--max-batches", type=int, default=None)
    p1dr.add_argument("--seed", type=int, default=42)
    p1dr.add_argument("--output-dir", default=".")

    # 1D interpolation
    p1d = subparsers.add_parser("interp1d", help="1D interpolation between two models")
    p1d.add_argument("--checkpoint-a", required=True, help="Path to first model checkpoint")
    p1d.add_argument("--checkpoint-b", required=True, help="Path to second model checkpoint")
    p1d.add_argument("--label-a", default="Model A (seed 0)")
    p1d.add_argument("--label-b", default="Model B (seed 1)")
    p1d.add_argument("--data-dir", default="./data")
    p1d.add_argument("--batch-size", type=int, default=256)
    p1d.add_argument("--num-points", type=int, default=51)
    p1d.add_argument("--alpha-range", type=float, nargs=2, default=[-0.5, 1.5])
    p1d.add_argument("--max-batches", type=int, default=None)
    p1d.add_argument("--output-dir", default=".")

    # PCA trajectory
    ppca = subparsers.add_parser("pca", help="PCA trajectory visualization")
    ppca.add_argument("--checkpoint-dir", required=True, help="Directory with epoch_*.ckpt files")
    ppca.add_argument("--data-dir", default="./data")
    ppca.add_argument("--batch-size", type=int, default=256)
    ppca.add_argument("--resolution", type=int, default=21)
    ppca.add_argument("--max-batches", type=int, default=None)
    ppca.add_argument("--output-dir", default=".")

    # Multi-optimizer interpolation comparison
    pmulti = subparsers.add_parser("interp1d-multi", help="Compare 1D interpolation across optimizers")
    pmulti.add_argument(
        "--optimizers", nargs="+", required=True,
        help="Optimizers to compare, e.g.: adamw sgd adam"
    )
    pmulti.add_argument(
        "--base-dir", default="landscape_models",
        help="Base directory containing optimizer subdirs (e.g. landscape_models/adamw/seed_0/)"
    )
    pmulti.add_argument("--seed-a", type=int, default=0, help="First seed for interpolation")
    pmulti.add_argument("--seed-b", type=int, default=1, help="Second seed for interpolation")
    pmulti.add_argument("--data-dir", default="./data")
    pmulti.add_argument("--batch-size", type=int, default=256)
    pmulti.add_argument("--num-points", type=int, default=51)
    pmulti.add_argument("--alpha-range", type=float, nargs=2, default=[-0.5, 1.5])
    pmulti.add_argument("--max-batches", type=int, default=None)
    pmulti.add_argument("--output-dir", default=".")

    # Default: run all with 2D landscape (backward compatible)
    args = parser.parse_args()

    if args.command == "landscape2d":
        run_2d_landscape(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            resolution=args.resolution,
            x_range=tuple(args.x_range),
            y_range=tuple(args.y_range),
            max_batches=args.max_batches,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    elif args.command == "direction1d":
        run_1d_direction(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_points=args.num_points,
            alpha_range=tuple(args.alpha_range),
            max_batches=args.max_batches,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    elif args.command == "interp1d":
        run_1d_interpolation(
            checkpoint_a=args.checkpoint_a,
            checkpoint_b=args.checkpoint_b,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_points=args.num_points,
            alpha_range=tuple(args.alpha_range),
            max_batches=args.max_batches,
            output_dir=args.output_dir,
            label_a=args.label_a,
            label_b=args.label_b,
        )
    elif args.command == "pca":
        run_pca_trajectory(
            checkpoint_dir=args.checkpoint_dir,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            resolution=args.resolution,
            max_batches=args.max_batches,
            output_dir=args.output_dir,
        )
    elif args.command == "interp1d-multi":
        optimizer_dirs = {}
        for opt in args.optimizers:
            ckpt_a = os.path.join(args.base_dir, opt, f"seed_{args.seed_a}", f"best_seed_{args.seed_a}.ckpt")
            ckpt_b = os.path.join(args.base_dir, opt, f"seed_{args.seed_b}", f"best_seed_{args.seed_b}.ckpt")
            if not os.path.exists(ckpt_a):
                print(f"WARNING: {ckpt_a} not found — skipping {opt}")
                continue
            if not os.path.exists(ckpt_b):
                print(f"WARNING: {ckpt_b} not found — skipping {opt}")
                continue
            optimizer_dirs[opt] = (ckpt_a, ckpt_b)

        if not optimizer_dirs:
            print("ERROR: No valid optimizer checkpoints found.")
            print(f"Expected structure: {args.base_dir}/<optimizer>/seed_<N>/best_seed_<N>.ckpt")
            print(f"Train models first: python train_landscape.py seeds --optimizer <name> --output-dir {args.base_dir}/<name>")
        else:
            run_multi_optimizer_interpolation(
                optimizer_dirs=optimizer_dirs,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                num_points=args.num_points,
                alpha_range=tuple(args.alpha_range),
                max_batches=args.max_batches,
                output_dir=args.output_dir,
            )
    else:
        # Default: backward compatible 2D landscape
        run_2d_landscape()


if __name__ == '__main__':
    main()
