import os
import argparse

import numpy as np

from interpolation_1d import (
    run_1d_direction,
    run_1d_interpolation,
    run_multi_optimizer_interpolation,
    plot_multi_optimizer_interpolation,
)
from landscape_2d import run_2d_landscape
from pca_trajectory import run_pca_trajectory, plot_pca_trajectory


def replot_from_cache(npz_dir: str, output_dir: str, optimizers):
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for opt in optimizers:
        path = os.path.join(npz_dir, f"interpolation_1d_{opt}.npz")
        if not os.path.exists(path):
            print(f"skipping {opt}: not found ({path})")
            continue
        data = np.load(path)
        results[opt] = (data["alphas"], data["losses"], data["accuracies"])

    if results:
        plot_multi_optimizer_interpolation(
            results, output_path=os.path.join(output_dir, "interpolation_multi_optimizer.svg"),
        )

    pca_path = os.path.join(npz_dir, "pca_trajectory_data.npz")
    if os.path.exists(pca_path):
        data = np.load(pca_path)
        plot_pca_trajectory(
            data["projected"], data["epochs"].tolist(), data["explained_variance"],
            output_path=os.path.join(output_dir, "pca_trajectory_2d.svg"),
        )


def main():
    parser = argparse.ArgumentParser(description="Loss Landscape Visualization")
    subparsers = parser.add_subparsers(dest="command", help="Visualization method")

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

    p1dr = subparsers.add_parser("direction1d", help="1D loss along filter-normalized random direction")
    p1dr.add_argument("--checkpoint", default="checkpoints/best-checkpoint.ckpt")
    p1dr.add_argument("--data-dir", default="./data")
    p1dr.add_argument("--batch-size", type=int, default=256)
    p1dr.add_argument("--num-points", type=int, default=51)
    p1dr.add_argument("--alpha-range", type=float, nargs=2, default=[-1.0, 1.0])
    p1dr.add_argument("--max-batches", type=int, default=None)
    p1dr.add_argument("--seed", type=int, default=42)
    p1dr.add_argument("--output-dir", default=".")

    p1d = subparsers.add_parser("interp1d", help="1D interpolation between two models")
    p1d.add_argument("--checkpoint-a", required=True)
    p1d.add_argument("--checkpoint-b", required=True)
    p1d.add_argument("--label-a", default="Model A (seed 0)")
    p1d.add_argument("--label-b", default="Model B (seed 1)")
    p1d.add_argument("--data-dir", default="./data")
    p1d.add_argument("--batch-size", type=int, default=256)
    p1d.add_argument("--num-points", type=int, default=51)
    p1d.add_argument("--alpha-range", type=float, nargs=2, default=[-0.5, 1.5])
    p1d.add_argument("--max-batches", type=int, default=None)
    p1d.add_argument("--output-dir", default=".")

    ppca = subparsers.add_parser("pca", help="PCA trajectory visualization")
    ppca.add_argument("--checkpoint-dir", required=True)
    ppca.add_argument("--output-dir", default=".")

    preplot = subparsers.add_parser("replot", help="Replot SVGs from cached .npz files (multi-optimizer interp + PCA trajectory)")
    preplot.add_argument("--npz-dir", default="../outputs")
    preplot.add_argument("--output-dir", default="../outputs")
    preplot.add_argument("--optimizers", nargs="+",
                         default=["adam", "adamw", "rmsprop", "sgd", "vanilla"])

    pmulti = subparsers.add_parser("interp1d-multi", help="Compare 1D interpolation across optimizers")
    pmulti.add_argument("--optimizers", nargs="+", required=True)
    pmulti.add_argument("--base-dir", default="landscape_models")
    pmulti.add_argument("--seed-a", type=int, default=0)
    pmulti.add_argument("--seed-b", type=int, default=1)
    pmulti.add_argument("--data-dir", default="./data")
    pmulti.add_argument("--batch-size", type=int, default=256)
    pmulti.add_argument("--num-points", type=int, default=51)
    pmulti.add_argument("--alpha-range", type=float, nargs=2, default=[-0.5, 1.5])
    pmulti.add_argument("--max-batches", type=int, default=None)
    pmulti.add_argument("--output-dir", default=".")

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
            output_dir=args.output_dir,
        )
    elif args.command == "replot":
        replot_from_cache(
            npz_dir=args.npz_dir,
            output_dir=args.output_dir,
            optimizers=args.optimizers,
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
        run_2d_landscape()


if __name__ == '__main__':
    main()
