# FashionMNIST Loss Landscape Exploration

> A bachelor's project focused on analyzing and visualizing the **loss landscape** of convolutional neural networks trained on the FashionMNIST dataset using PyTorch Lightning.

---

## Overview

This repository explores how neural network optimization behaves in the context of **loss landscapes**. Using a CNN trained on FashionMNIST, we analyze curvature, convergence patterns, and training dynamics across different optimizers and seeds. The project leverages modern PyTorch workflows, Hydra-based configuration, and a suite of visualization tools to investigate deep learning behavior beyond accuracy metrics.

---

## Model Architecture

A simple but effective CNN:

```
Conv2d(1, 32, kernel_size=5) → BatchNorm → ReLU
→ Conv2d(32, 64, kernel_size=5) → BatchNorm → ReLU
→ Flatten → Linear(1024, 128) → ReLU
→ Linear(128, 64) → ReLU
→ Linear(64, 10)
```

- **Loss**: CrossEntropyLoss
- **Optimizer**: AdamW
- **Scheduler**: OneCycleLR

---

## Quick Start

```bash
git clone https://github.com/obassler/FashionMnist-OSU.git
cd FashionMnist-OSU

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd scripts
python training/train.py          # single training run
python training/train_joblib.py   # parallel multi-seed training
```

> A GPU is recommended for training; CPU works for visualization and small experiments.

---

## Visualizations

All loss-landscape figures are produced through a single CLI in `scripts/visualization/loss_landscape.py`:

```bash
cd scripts/visualization

# 2D / 3D loss surface around a trained model (with and without filter normalization)
python loss_landscape.py landscape2d --checkpoint ../checkpoints/best-checkpoint.ckpt

# 1D loss along a filter-normalized random direction
python loss_landscape.py direction1d --checkpoint ../checkpoints/best-checkpoint.ckpt

# 1D linear interpolation between two trained models
python loss_landscape.py interp1d --checkpoint-a A.ckpt --checkpoint-b B.ckpt

# 1D interpolation comparing several optimizers side by side
python loss_landscape.py interp1d-multi --optimizers adam adamw sgd rmsprop vanilla

# PCA projection of the training trajectory
python loss_landscape.py pca --checkpoint-dir ../landscape_trajectory

# Replot SVGs from cached .npz outputs (no recomputation)
python loss_landscape.py replot
```

Standalone scripts for prediction-correlation analysis:

```bash
python histogram.py   # histogram of pairwise prediction correlations
python matrix.py      # correlation heatmap across trained models
```

Outputs land in `scripts/outputs/` (figures + cached `.npz` data).

---

## Requirements

> Python 3.9+ recommended

```txt
torch>=1.12.0
torchvision>=0.13.0
pytorch-lightning>=1.8.0
torchmetrics>=0.11.0
hydra-core>=1.3.0
hydra-optuna-sweeper>=1.2.0
hydra-joblib-launcher
omegaconf>=2.3.0
wandb>=0.13.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
pyyaml>=6.0
tqdm>=4.64.0
```

---

## Project Structure

```
FashionMnist-OSU/
│
├── requirements.txt
├── README.md
└── scripts/
    ├── configs/
    │   ├── default.yaml
    │   └── best_params.yaml
    ├── data/
    │   └── datamodule.py
    ├── models/
    │   └── lit_model.py
    ├── training/
    │   ├── train.py
    │   ├── train_joblib.py
    │   └── train_landscape.py
    ├── utils/
    │   ├── prediction_utils.py
    │   └── Utils.py
    ├── visualization/
    │   ├── loss_landscape.py        # CLI entry point
    │   ├── common.py                # shared model/weight utilities
    │   ├── interpolation_1d.py
    │   ├── landscape_2d.py
    │   ├── pca_trajectory.py
    │   ├── histogram.py
    │   └── matrix.py
    ├── checkpoints/
    ├── outputs/
    ├── predictions/
    ├── landscape_models/
    └── landscape_trajectory/
```

---

## About

This project is part of a **Bachelor's thesis** at the University of Ostrava, focused on improving understanding of deep learning dynamics through **loss landscape analysis**.

---

## TODO / Roadmap

- [x] Basic CNN model training
- [x] Loss & accuracy tracking via Weights & Biases
- [x] Loss surface visualization (2D / 3D projections)
- [x] 1D interpolation and random-direction analysis
- [x] PCA projection of training trajectory
- [x] Multi-optimizer comparison
- [x] Unified CLI entry point (`loss_landscape.py`)
- [x] Hydra sweeps + parallel training (joblib)
- [x] YAML config + W&B logging

---

## Author

**Ondřej Bassler**
[GitHub](https://github.com/obassler)
