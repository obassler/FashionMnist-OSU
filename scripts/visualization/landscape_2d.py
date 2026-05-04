import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from tqdm import tqdm
from typing import Tuple, List, Optional

from common import (
    FashionMNISTDataModule,
    load_model_from_checkpoint,
    get_weights, set_weights,
    get_random_direction, normalize_direction_filter_wise,
    evaluate_model,
)


def compute_loss_surface(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
    resolution: int = 21,
    max_batches: Optional[int] = None,
    seed: int = 42,
    filter_normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    base_weights = get_weights(model)

    direction1 = get_random_direction(base_weights)
    direction2 = get_random_direction(base_weights)

    if filter_normalize:
        direction1 = normalize_direction_filter_wise(direction1, base_weights)
        direction2 = normalize_direction_filter_wise(direction2, base_weights)

    alphas = np.linspace(x_range[0], x_range[1], resolution)
    betas = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(alphas, betas)

    loss_surface = np.zeros_like(X)
    acc_surface = np.zeros_like(X)

    total_points = resolution * resolution
    label = "normalized" if filter_normalize else "raw"
    pbar = tqdm(total=total_points, desc=f"Computing 2D loss surface ({label})")

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
    set_weights(model, base_weights)

    return X, Y, loss_surface, acc_surface


def _contour_on_ax(ax, X, Y, Z, title, levels=30):
    z_min = Z.min()
    z_max = Z.max()
    log_min = np.log(max(z_min, 1e-8))
    log_max = np.log(z_max)
    log_levels = np.linspace(log_min, log_max, levels)
    contour_levels = np.exp(log_levels)

    cs = ax.contour(X, Y, Z, levels=contour_levels, colors='black', linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f',
              levels=cs.levels[::max(1, len(cs.levels) // 8)])
    ax.plot(0, 0, 'r*', markersize=18, markeredgecolor='black', markeredgewidth=1.0, zorder=10)
    ax.set_xlabel(r'$\delta$', fontsize=14)
    ax.set_ylabel(r'$\eta$', fontsize=14)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=10)


def plot_2d_comparison(
    X_norm, Y_norm, Z_norm,
    X_raw, Y_raw, Z_raw,
    output_path: str = "loss_landscape_2d.svg",
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    _contour_on_ax(ax1, X_norm, Y_norm, Z_norm, "With Filter Normalization")
    _contour_on_ax(ax2, X_raw, Y_raw, Z_raw, "Without Filter Normalization")

    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Saved 2D comparison plot to: {output_path}")
    plt.close()


def _surface_on_ax(fig, ax, X, Y, Z, title, cmap='viridis'):
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.85,
                           rstride=1, cstride=1, antialiased=True)
    center_z = Z[Z.shape[0] // 2, Z.shape[1] // 2]
    ax.scatter([0], [0], [center_z], color='red', s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10, depthshade=False)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, aspect=20)
    cbar.set_label('Cross-Entropy Loss', fontsize=11, fontweight='bold')
    ax.set_xlabel(r'$d_1$ ($\alpha$)', fontsize=10, fontweight='bold', labelpad=8)
    ax.set_ylabel(r'$d_2$ ($\beta$)', fontsize=10, fontweight='bold', labelpad=8)
    ax.set_zlabel('Loss', fontsize=10, fontweight='bold', labelpad=8)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.view_init(elev=30, azim=45)


def plot_3d_comparison(
    X_norm, Y_norm, Z_norm,
    X_raw, Y_raw, Z_raw,
    output_path: str = "loss_landscape_3d.svg",
) -> None:
    fig = plt.figure(figsize=(24, 11))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    _surface_on_ax(fig, ax1, X_norm, Y_norm, Z_norm, "With Filter Normalization")
    _surface_on_ax(fig, ax2, X_raw, Y_raw, Z_raw, "Without Filter Normalization")

    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Saved 3D comparison plot to: {output_path}")
    plt.close()


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_module = FashionMNISTDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=0, seed=seed)
    data_module.setup('test')
    test_loader = data_module.test_dataloader()

    model = load_model_from_checkpoint(checkpoint_path, device).to(device)
    model.eval()

    X_norm, Y_norm, loss_norm, acc_norm = compute_loss_surface(
        model, test_loader, device, x_range, y_range, resolution, max_batches, seed,
        filter_normalize=True,
    )

    X_raw, Y_raw, loss_raw, acc_raw = compute_loss_surface(
        model, test_loader, device, x_range, y_range, resolution, max_batches, seed,
        filter_normalize=False,
    )

    np.savez(os.path.join(output_dir, 'loss_landscape_data.npz'),
             X_norm=X_norm, Y_norm=Y_norm, loss_norm=loss_norm, acc_norm=acc_norm,
             X_raw=X_raw, Y_raw=Y_raw, loss_raw=loss_raw, acc_raw=acc_raw)

    plot_2d_comparison(
        X_norm, Y_norm, loss_norm,
        X_raw, Y_raw, loss_raw,
        output_path=os.path.join(output_dir, "loss_landscape_2d.svg"),
    )

    plot_3d_comparison(
        X_norm, Y_norm, loss_norm,
        X_raw, Y_raw, loss_raw,
        output_path=os.path.join(output_dir, "loss_landscape_3d.svg"),
    )

