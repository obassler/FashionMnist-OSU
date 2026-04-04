"""
Training script for loss landscape analysis.

Trains models and saves the checkpoints needed for visualization:
  1. Multiple models with different seeds → for 1D interpolation comparison
  2. Per-epoch checkpoints for one model → for PCA trajectory visualization

Usage:
  # Train 8 models with different seeds (for 1D interpolation + correlation analysis)
  python train_landscape.py seeds --num-seeds 8

  # Train one model and save every-epoch checkpoints (for PCA trajectory)
  python train_landscape.py trajectory --seed 42

  # Both at once
  python train_landscape.py all --num-seeds 8 --trajectory-seed 42
"""

import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datamodule import FashionMNISTDataModule
from models.lit_model import FashionMNISTModel

# Best hyperparameters from Optuna sweep
BEST_PARAMS = dict(
    learning_rate=0.0063359488139514445,
    weight_decay=0.008912117369121994,
    max_lr=0.01,
    num_epochs=200,
    batch_size=128,
    accumulate_grad_batches=2,
    gradient_clip_val=0.5,
    precision="16-mixed",
)


class EpochCheckpointCallback(Callback):
    """Save a checkpoint at the end of every N epochs."""

    def __init__(self, save_dir: str, every_n_epochs: int = 1):
        self.save_dir = save_dir
        self.every_n_epochs = every_n_epochs
        os.makedirs(save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs == 0:
            path = os.path.join(self.save_dir, f"epoch_{epoch}.ckpt")
            trainer.save_checkpoint(path)


def train_single_model(
    seed: int,
    output_dir: str,
    save_epoch_checkpoints: bool = False,
    epoch_checkpoint_interval: int = 1,
    num_epochs: int = BEST_PARAMS["num_epochs"],
    optimizer_name: str = "adamw",
):
    """Train a single model and save its best checkpoint."""
    pl.seed_everything(seed, workers=True)

    print(f"\n{'='*60}")
    print(f"Training model with seed={seed}, optimizer={optimizer_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    data_module = FashionMNISTDataModule(
        batch_size=BEST_PARAMS["batch_size"],
        data_dir="./data",
        num_workers=0,
        seed=seed,
    )

    model = FashionMNISTModel(
        learning_rate=BEST_PARAMS["learning_rate"],
        weight_decay=BEST_PARAMS["weight_decay"],
        max_lr=BEST_PARAMS["max_lr"],
        save_predictions_dir=output_dir,
        optimizer_name=optimizer_name,
    )

    callbacks = []

    # Always save best checkpoint
    best_ckpt = ModelCheckpoint(
        dirpath=output_dir,
        filename=f"best_seed_{seed}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        verbose=True,
    )
    callbacks.append(best_ckpt)

    # Optionally save epoch checkpoints for PCA
    if save_epoch_checkpoints:
        epoch_dir = os.path.join(output_dir, "epoch_checkpoints")
        epoch_cb = EpochCheckpointCallback(
            save_dir=epoch_dir,
            every_n_epochs=epoch_checkpoint_interval,
        )
        callbacks.append(epoch_cb)
        print(f"Saving epoch checkpoints every {epoch_checkpoint_interval} epoch(s) to {epoch_dir}")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=BEST_PARAMS["gradient_clip_val"],
        accumulate_grad_batches=BEST_PARAMS["accumulate_grad_batches"],
        precision=BEST_PARAMS["precision"],
        logger=False,  # No W&B for this utility script
    )

    trainer.fit(model, datamodule=data_module)
    test_results = trainer.test(model, datamodule=data_module, ckpt_path="best")

    test_acc = test_results[0]["test_acc"] if test_results else 0.0
    test_loss = test_results[0]["test_loss"] if test_results else 0.0
    print(f"Seed {seed} — Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    return test_acc, test_loss


def train_multiple_seeds(num_seeds: int, base_output_dir: str, num_epochs: int, optimizer_name: str = "adamw"):
    """Train multiple models with different seeds."""
    results = []
    for seed in range(num_seeds):
        output_dir = os.path.join(base_output_dir, f"seed_{seed}")
        test_acc, test_loss = train_single_model(
            seed=seed,
            output_dir=output_dir,
            save_epoch_checkpoints=False,
            num_epochs=num_epochs,
            optimizer_name=optimizer_name,
        )
        results.append((seed, test_acc, test_loss))

    print(f"\n{'='*60}")
    print("All models trained. Results:")
    print(f"{'='*60}")
    for seed, acc, loss in results:
        print(f"  Seed {seed}: Acc={acc:.4f}, Loss={loss:.4f}")

    accs = [r[1] for r in results]
    print(f"\nMean Acc: {sum(accs)/len(accs):.4f}")
    print(f"Checkpoint paths for 1D interpolation:")
    for seed, _, _ in results:
        path = os.path.join(base_output_dir, f"seed_{seed}", f"best_seed_{seed}.ckpt")
        print(f"  {path}")


def train_trajectory(seed: int, output_dir: str, epoch_interval: int, num_epochs: int, optimizer_name: str = "adamw"):
    """Train one model with per-epoch checkpoints for PCA visualization."""
    train_single_model(
        seed=seed,
        output_dir=output_dir,
        save_epoch_checkpoints=True,
        epoch_checkpoint_interval=epoch_interval,
        num_epochs=num_epochs,
        optimizer_name=optimizer_name,
    )
    epoch_dir = os.path.join(output_dir, "epoch_checkpoints")
    n_files = len([f for f in os.listdir(epoch_dir) if f.endswith(".ckpt")])
    print(f"\nSaved {n_files} epoch checkpoints to: {epoch_dir}")
    print(f"Use with: python loss_landscape.py pca --checkpoint-dir {epoch_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train models for loss landscape analysis")
    subparsers = parser.add_subparsers(dest="command")

    # Common optimizer help text
    opt_help = "Optimizer to use: adamw, adam, sgd, rmsprop (default: adamw)"

    # Train multiple seeds
    p_seeds = subparsers.add_parser("seeds", help="Train N models with different seeds")
    p_seeds.add_argument("--num-seeds", type=int, default=8)
    p_seeds.add_argument("--output-dir", default="landscape_models")
    p_seeds.add_argument("--num-epochs", type=int, default=BEST_PARAMS["num_epochs"])
    p_seeds.add_argument("--optimizer", default="adamw", help=opt_help)

    # Train trajectory
    p_traj = subparsers.add_parser("trajectory", help="Train one model with epoch checkpoints")
    p_traj.add_argument("--seed", type=int, default=42)
    p_traj.add_argument("--epoch-interval", type=int, default=5,
                        help="Save checkpoint every N epochs (default=5, use 1 for full resolution)")
    p_traj.add_argument("--output-dir", default="landscape_trajectory")
    p_traj.add_argument("--num-epochs", type=int, default=BEST_PARAMS["num_epochs"])
    p_traj.add_argument("--optimizer", default="adamw", help=opt_help)

    # Both
    p_all = subparsers.add_parser("all", help="Train seeds + trajectory")
    p_all.add_argument("--num-seeds", type=int, default=8)
    p_all.add_argument("--trajectory-seed", type=int, default=42)
    p_all.add_argument("--epoch-interval", type=int, default=5)
    p_all.add_argument("--output-dir", default="landscape_models")
    p_all.add_argument("--num-epochs", type=int, default=BEST_PARAMS["num_epochs"])
    p_all.add_argument("--optimizer", default="adamw", help=opt_help)

    args = parser.parse_args()

    if args.command == "seeds":
        train_multiple_seeds(args.num_seeds, args.output_dir, args.num_epochs, args.optimizer)
    elif args.command == "trajectory":
        train_trajectory(args.seed, args.output_dir, args.epoch_interval, args.num_epochs, args.optimizer)
    elif args.command == "all":
        train_multiple_seeds(args.num_seeds, args.output_dir, args.num_epochs, args.optimizer)
        train_trajectory(
            args.trajectory_seed,
            os.path.join(args.output_dir, "trajectory"),
            args.epoch_interval,
            args.num_epochs,
            args.optimizer,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
