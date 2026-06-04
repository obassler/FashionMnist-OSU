import os
import itertools
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List, Optional

from common import (
    FashionMNISTDataModule,
    load_model_from_checkpoint,
    get_state, set_state,
    evaluate_model,
)


def reset_bn_stats(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.reset_running_stats()
            m.momentum = None


def recalibrate_bn(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> None:
    has_bn = any(
        isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        for m in model.modules()
    )
    if not has_bn:
        return

    reset_bn_stats(model)
    model.train()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            model(x.to(device))
    model.eval()


def barrier_for_pair(
    model: nn.Module,
    state_a: List[torch.Tensor],
    state_b: List[torch.Tensor],
    eval_loader: torch.utils.data.DataLoader,
    bn_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_points: int = 11,
    bn_batches: int = 20,
    eval_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    param_names = [n for n, _ in model.named_parameters()]
    param_keys = set(param_names)
    state_keys = list(model.state_dict().keys())

    alphas = np.linspace(0.0, 1.0, num_points)
    losses = np.zeros(num_points)

    for idx, alpha in enumerate(alphas):
        interp = []
        for key, sa, sb in zip(state_keys, state_a, state_b):
            if key in param_keys:
                interp.append((1 - alpha) * sa + alpha * sb)
            else:
                interp.append(sa.clone())
        set_state(model, interp)
        recalibrate_bn(model, bn_loader, device, max_batches=bn_batches)
        loss, _ = evaluate_model(model, eval_loader, device, max_batches=eval_batches)
        losses[idx] = loss

    return alphas, losses


def compute_barrier_matrix(
    checkpoints: List[Tuple[str, str]],
    data_dir: str = "./data",
    batch_size: int = 256,
    num_points: int = 11,
    bn_batches: int = 20,
    eval_batches: Optional[int] = None,
    cache_curves_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dm = FashionMNISTDataModule(
        data_dir=data_dir, batch_size=batch_size, num_workers=0,
        use_augmentation=False,
    )
    dm.setup("fit")
    dm.setup("test")
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()

    states, names = [], []
    model = None
    for name, path in checkpoints:
        m = load_model_from_checkpoint(path, device).to(device)
        m.eval()
        states.append(get_state(m))
        names.append(name)
        if model is None:
            model = m

    n = len(states)
    barriers = np.full((n, n), np.nan)
    endpoint_loss = np.zeros(n)
    curves = {}

    print("Evaluating endpoints (with BN recalibration)...")
    for i, st in enumerate(tqdm(states, desc="endpoints")):
        set_state(model, st)
        recalibrate_bn(model, train_loader, device, max_batches=bn_batches)
        loss, _ = evaluate_model(model, test_loader, device, max_batches=eval_batches)
        endpoint_loss[i] = loss
        barriers[i, i] = 0.0

    pairs = list(itertools.combinations(range(n), 2))
    print(f"Computing {len(pairs)} pair barriers ({num_points} points each)...")
    for i, j in tqdm(pairs, desc="pairs"):
        alphas, losses = barrier_for_pair(
            model, states[i], states[j],
            test_loader, train_loader, device,
            num_points=num_points,
            bn_batches=bn_batches,
            eval_batches=eval_batches,
        )
        endpoint_max = max(losses[0], losses[-1])
        barrier = float(losses.max() - endpoint_max)
        barriers[i, j] = barrier
        barriers[j, i] = barrier
        curves[f"{names[i]}__{names[j]}"] = {
            "alphas": alphas, "losses": losses,
        }

    if cache_curves_path is not None:
        np.savez(cache_curves_path, **{
            k: np.stack([v["alphas"], v["losses"]]) for k, v in curves.items()
        })

    return barriers, endpoint_loss, names, curves


def plot_barrier_heatmap(
    barriers: np.ndarray,
    endpoint_loss: np.ndarray,
    names: List[str],
    output_path: str,
) -> None:
    n = barriers.shape[0]
    fig, ax = plt.subplots(figsize=(8.5, 7))

    display = barriers.copy()
    off = display.copy()
    np.fill_diagonal(off, np.nan)
    vmax = float(np.nanmax(off)) if np.isfinite(np.nanmax(off)) else 1.0
    vmin = 0.0

    im = ax.imshow(display, cmap="magma_r", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(
        "Loss Barrier Between Trained Models\n"
        r"(linear interpolation $\theta(\alpha)=(1-\alpha)\theta_i+\alpha\theta_j$, BN recalibrated)",
        fontsize=12, fontweight="bold", pad=10,
    )

    for i in range(n):
        for j in range(n):
            v = display[i, j]
            if np.isnan(v):
                continue
            color = "white" if v > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Barrier height $\max_\alpha L(\alpha)-\max(L_i,L_j)$", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, format="svg", dpi=300, bbox_inches="tight")
    print(f"Saved barrier heatmap -> {output_path}")
    plt.close()


def run_barrier_heatmap(
    base_dir: str = "landscape_models",
    optimizers: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    data_dir: str = "./data",
    batch_size: int = 256,
    num_points: int = 11,
    bn_batches: int = 20,
    eval_batches: Optional[int] = None,
    output_dir: str = ".",
):
    if optimizers is None:
        optimizers = ["adam", "adamw", "rmsprop", "sgd", "vanilla"]
    if seeds is None:
        seeds = [0, 1]

    checkpoints = []
    for opt in optimizers:
        for s in seeds:
            path = os.path.join(base_dir, opt, f"seed_{s}", f"best_seed_{s}.ckpt")
            if not os.path.exists(path):
                print(f"WARNING: {path} not found, skipping")
                continue
            checkpoints.append((f"{opt}/s{s}", path))

    if len(checkpoints) < 2:
        raise FileNotFoundError(f"Need at least 2 checkpoints, found {len(checkpoints)}")

    os.makedirs(output_dir, exist_ok=True)
    barriers, endpoint_loss, names, curves = compute_barrier_matrix(
        checkpoints,
        data_dir=data_dir,
        batch_size=batch_size,
        num_points=num_points,
        bn_batches=bn_batches,
        eval_batches=eval_batches,
        cache_curves_path=os.path.join(output_dir, "barrier_curves.npz"),
    )

    np.savez(
        os.path.join(output_dir, "barrier_matrix_data.npz"),
        barriers=barriers,
        endpoint_loss=endpoint_loss,
        names=np.array(names),
    )

    plot_barrier_heatmap(
        barriers, endpoint_loss, names,
        output_path=os.path.join(output_dir, "barrier_heatmap.svg"),
    )
