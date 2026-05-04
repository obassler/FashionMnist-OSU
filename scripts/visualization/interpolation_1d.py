import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from typing import Tuple, List, Optional

from common import (
    FashionMNISTDataModule,
    load_model_from_checkpoint,
    get_weights, set_weights, get_state, set_state,
    get_random_direction, normalize_direction_filter_wise,
    evaluate_model,
)


DISPLAY_NAMES = {
    "adamw":   "AdamW",
    "sgd":     "SGD",
    "adam":     "Adam",
    "rmsprop": "RMSProp",
    "vanilla": "Vanilla SGD",
}

COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
]


def compute_1d_interpolation(
    model: torch.nn.Module,
    state_a: List[torch.Tensor],
    state_b: List[torch.Tensor],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_points: int = 51,
    alpha_range: Tuple[float, float] = (-0.5, 1.5),
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

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


def compute_1d_direction(
    model: torch.nn.Module,
    base_weights: List[torch.Tensor],
    direction: List[torch.Tensor],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_points: int = 51,
    alpha_range: Tuple[float, float] = (-1.0, 1.0),
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    alphas = np.linspace(alpha_range[0], alpha_range[1], num_points)
    losses = np.zeros(num_points)
    accuracies = np.zeros(num_points)

    for idx, alpha in enumerate(tqdm(alphas, desc="1D direction")):
        perturbed = [w + alpha * d for w, d in zip(base_weights, direction)]
        set_weights(model, perturbed)
        loss, acc = evaluate_model(model, dataloader, device, max_batches)
        losses[idx] = loss
        accuracies[idx] = acc

    set_weights(model, base_weights)
    return alphas, losses, accuracies


def plot_1d_direction(
    alphas: np.ndarray,
    losses: np.ndarray,
    accuracies: np.ndarray,
    output_path: str = "direction_1d.svg",
    seed: int = 42,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(alphas, losses, 'b-', linewidth=2)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Trained model')
    center_idx = np.argmin(np.abs(alphas))
    ax1.scatter([0], [losses[center_idx]], color='red', s=80, zorder=5)
    ax1.set_xlabel(r'Perturbation coefficient $\alpha$', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.set_title('Loss Along Random Direction', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

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


def plot_multi_optimizer_interpolation(
    results: dict,
    output_path: str = "interpolation_multi_optimizer.svg",
) -> None:
    opt_names = list(results.keys())
    n_opts = len(opt_names)

    fig, axes = plt.subplots(
        2, n_opts,
        figsize=(4.2 * n_opts, 7.5),
        squeeze=False,
    )

    for col, opt_name in enumerate(opt_names):
        color = COLORS[col % len(COLORS)]
        label = DISPLAY_NAMES.get(opt_name.lower(), opt_name.upper())
        alphas, losses, accuracies = results[opt_name]

        idx_0 = np.argmin(np.abs(alphas))
        idx_1 = np.argmin(np.abs(alphas - 1.0))

        for row, (values, ylabel) in enumerate([
            (losses,           "Cross-Entropy Loss"),
            (accuracies * 100, "Accuracy (%)"),
        ]):
            ax = axes[row][col]

            ax.plot(alphas, values, color=color, linewidth=2.2)
            ax.scatter([0], [values[idx_0]], color=color, s=65, zorder=5, marker="o")
            ax.scatter([1], [values[idx_1]], color=color, s=65, zorder=5, marker="s")
            ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.9, alpha=0.55)
            ax.axvline(x=1, color="gray", linestyle="--", linewidth=0.9, alpha=0.55)
            ax.set_xlabel(r"Interpolation $\alpha$", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.25)

            if row == 0:
                ax.set_title(label, fontsize=12, fontweight="bold", color=color)

    axes[0][0].set_ylabel("Cross-Entropy Loss", fontsize=10, fontweight="bold")
    axes[1][0].set_ylabel("Accuracy (%)",       fontsize=10, fontweight="bold")

    legend_elements = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=8, label=r"$\alpha=0$  (seed 0 model)"),
        Line2D([0], [0], marker="s", color="gray", linestyle="None",
               markersize=8, label=r"$\alpha=1$  (seed 1 model)"),
        Line2D([0], [0], color="gray", linestyle="--",
               linewidth=1.2, label="Model endpoints"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=3,
        fontsize=10, frameon=True,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.suptitle(
        r"Optimizer Comparison — "
        r"$\theta(\alpha) = (1-\alpha)\,\theta_{\mathrm{seed\,0}} + \alpha\,\theta_{\mathrm{seed\,1}}$",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    plt.savefig(output_path, format="svg", dpi=300, bbox_inches="tight")
    print(f"Saved multi-optimizer plot -> {output_path}")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_module = FashionMNISTDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=0)
    data_module.setup('test')
    test_loader = data_module.test_dataloader()

    results = {}

    for opt_name, (ckpt_a, ckpt_b) in optimizer_dirs.items():
        print(f"Interpolating {opt_name}: {ckpt_a} <-> {ckpt_b}")
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

        np.savez(
            os.path.join(output_dir, f'interpolation_1d_{opt_name}.npz'),
            alphas=alphas, losses=losses, accuracies=accuracies
        )

    plot_multi_optimizer_interpolation(
        results,
        output_path=os.path.join(output_dir, "interpolation_multi_optimizer.svg"),
    )


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
        seed=seed,
    )


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_module = FashionMNISTDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=0)
    data_module.setup('test')
    test_loader = data_module.test_dataloader()

    model_a = load_model_from_checkpoint(checkpoint_a, device).to(device)
    state_a = get_state(model_a)
    model_b = load_model_from_checkpoint(checkpoint_b, device).to(device)
    state_b = get_state(model_b)

    model_a.eval()

    alphas, losses, accuracies = compute_1d_interpolation(
        model_a, state_a, state_b, test_loader, device,
        num_points, alpha_range, max_batches
    )

    set_state(model_a, state_a)

    np.savez(os.path.join(output_dir, 'interpolation_1d_data.npz'),
             alphas=alphas, losses=losses, accuracies=accuracies)

    plot_1d_interpolation(
        alphas, losses, accuracies,
        output_path=os.path.join(output_dir, "interpolation_1d.svg"),
        label_a=label_a, label_b=label_b,
    )

