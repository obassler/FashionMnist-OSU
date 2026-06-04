import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.prediction_utils import load_predictions


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_test_labels(data_dir: str) -> np.ndarray:
    ds = FashionMNIST(data_dir, train=False, download=True)
    return np.array(ds.targets)


def compute_ensemble_confusion(preds: np.ndarray, labels: np.ndarray, k: int = 10) -> np.ndarray:
    n_models, n_samples = preds.shape
    conf = np.zeros((k, k))
    for c in range(k):
        mask = labels == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        for n in range(n_models):
            counts = np.bincount(preds[n, mask], minlength=k)
            conf[c] += counts / n_c
        conf[c] /= n_models
    return conf


def compute_per_class_disagreement(preds: np.ndarray, labels: np.ndarray, k: int = 10) -> np.ndarray:
    n_models = preds.shape[0]
    disagreement = np.zeros(k)
    for c in range(k):
        mask = labels == c
        if mask.sum() == 0:
            continue
        sub = preds[:, mask]
        per_sample = np.zeros(sub.shape[1])
        for j in range(sub.shape[1]):
            p = np.bincount(sub[:, j], minlength=k) / n_models
            per_sample[j] = 1.0 - np.sum(p ** 2)
        disagreement[c] = float(per_sample.mean())
    return disagreement


def plot_per_class(
    conf: np.ndarray,
    disagreement: np.ndarray,
    output_path: str,
    n_models: int,
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6.2),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )

    off_diag = conf.copy()
    np.fill_diagonal(off_diag, np.nan)
    vmax = np.nanmax(off_diag) if np.isfinite(np.nanmax(off_diag)) else 0.1

    im = ax1.imshow(off_diag, cmap="Reds", vmin=0.0, vmax=vmax, aspect="equal")
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    ax1.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax1.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax1.set_xlabel("Predicted class", fontsize=11)
    ax1.set_ylabel("True class", fontsize=11)
    ax1.set_title(
        f"Ensemble-Averaged Confusion (N={n_models}, diagonal hidden)",
        fontsize=12, fontweight="bold",
    )

    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            v = conf[i, j]
            if v >= 0.01:
                ax1.text(
                    j, i, f"{v*100:.1f}",
                    ha="center", va="center",
                    fontsize=7,
                    color="white" if v > vmax * 0.55 else "black",
                )
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Mean P(predicted | true)", fontsize=10)

    order = np.argsort(disagreement)[::-1]
    sorted_vals = disagreement[order]
    sorted_names = [CLASS_NAMES[i] for i in order]

    colors = plt.cm.viridis(sorted_vals / max(sorted_vals.max(), 1e-9))
    bars = ax2.barh(range(10), sorted_vals * 100, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(sorted_names, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean pairwise disagreement (%)", fontsize=11)
    ax2.set_title(
        "Per-Class Disagreement Across Models",
        fontsize=12, fontweight="bold",
    )
    ax2.grid(True, axis="x", alpha=0.3)

    for bar, v in zip(bars, sorted_vals):
        ax2.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{v*100:.2f}%",
            va="center", fontsize=9,
        )

    fig.suptitle(
        "Functional Disagreement of 100 Independently Trained Models",
        fontsize=13.5, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, format="svg", dpi=300, bbox_inches="tight")
    print(f"Saved per-class disagreement plot -> {output_path}")
    plt.close()


def run_per_class(
    predictions_dir: str,
    data_dir: str = "./data",
    output_dir: str = ".",
):
    preds_list = load_predictions(predictions_dir)
    if not preds_list:
        raise FileNotFoundError(f"No predictions found in {predictions_dir}")

    labels = load_test_labels(data_dir)
    m = min(min(len(p) for p in preds_list), len(labels))
    preds = np.stack([p[:m] for p in preds_list])
    labels = labels[:m]

    n_models = preds.shape[0]
    print(f"Loaded {n_models} prediction sets over {m} test samples")

    conf = compute_ensemble_confusion(preds, labels)
    disagreement = compute_per_class_disagreement(preds, labels)

    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "per_class_data.npz"),
        confusion=conf, disagreement=disagreement,
        class_names=np.array(CLASS_NAMES),
    )

    plot_per_class(
        conf, disagreement,
        output_path=os.path.join(output_dir, "per_class_disagreement.svg"),
        n_models=n_models,
    )
