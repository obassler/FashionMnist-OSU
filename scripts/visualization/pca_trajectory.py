import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List, Optional
from sklearn.decomposition import PCA

from common import flatten_weights


def load_trajectory_checkpoints(
    checkpoint_dir: str,
    device: torch.device,
    pattern: str = "epoch_*.ckpt"
) -> Tuple[List[List[torch.Tensor]], List[int]]:
    files = sorted(glob.glob(os.path.join(checkpoint_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No checkpoints matching '{pattern}' found in {checkpoint_dir}"
        )

    weights_list = []
    epochs = []

    for f in tqdm(files, desc="Loading checkpoints"):
        basename = os.path.basename(f)
        epoch_str = basename.replace("epoch_", "").replace(".ckpt", "")
        try:
            epoch = int(epoch_str)
        except ValueError:
            continue

        checkpoint = torch.load(f, map_location=device, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)

        cleaned = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.', '') if k.startswith('model.') else k
            cleaned[new_key] = v

        param_keys = [k for k in cleaned.keys()
                      if 'running_' not in k and 'num_batches_tracked' not in k]
        param_tensors = [cleaned[k].cpu() for k in param_keys if k in cleaned]

        weights_list.append(param_tensors)
        epochs.append(epoch)

    print(f"Loaded {len(weights_list)} checkpoints (epochs {epochs[0]}-{epochs[-1]})")
    return weights_list, epochs


def compute_pca_directions(
    weights_list: List[List[torch.Tensor]],
    n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.array([flatten_weights(w) for w in weights_list])

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(matrix)
    return pca.components_, projected, pca.explained_variance_ratio_


def plot_pca_trajectory(
    projected: np.ndarray,
    epochs: List[int],
    explained_variance: np.ndarray,
    output_path: str = "pca_trajectory_2d.svg",
    title: str = "Training Trajectory in PCA Space",
) -> None:
    order = np.argsort(epochs)
    projected = projected[order]
    epochs_sorted = np.array(epochs)[order]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(projected[:, 0], projected[:, 1], color="lightgray", linewidth=1, zorder=1)

    sc = ax.scatter(
        projected[:, 0], projected[:, 1],
        c=epochs_sorted, cmap="cool", s=70,
        edgecolors="black", linewidths=0.6, zorder=3,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Epoch", fontsize=10)

    ax.set_xlabel(f"PC1 ({explained_variance[0] * 100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({explained_variance[1] * 100:.1f}% variance)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, format="svg", dpi=300, bbox_inches="tight")
    print(f"Saved PCA plot -> {output_path}")
    plt.close()


def run_pca_trajectory(checkpoint_dir: str, output_dir: str = "."):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_list, epochs = load_trajectory_checkpoints(checkpoint_dir, device)
    _, projected, explained_variance = compute_pca_directions(weights_list)

    np.savez(os.path.join(output_dir, 'pca_trajectory_data.npz'),
             projected=projected, epochs=np.array(epochs),
             explained_variance=explained_variance)

    plot_pca_trajectory(
        projected, epochs, explained_variance,
        output_path=os.path.join(output_dir, "pca_trajectory_2d.svg"),
    )
