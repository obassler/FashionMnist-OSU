# FashionMNIST Loss Landscape Exploration

> A bachelor's thesis project (University of Ostrava) analyzing and visualizing the **loss landscape** of convolutional neural networks trained on FashionMNIST.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-792EE5?logo=pytorchlightning&logoColor=white)
![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)
![Optuna](https://img.shields.io/badge/HPO-Optuna-2c5bb4)
![W&B](https://img.shields.io/badge/Tracking-Weights%20%26%20Biases-FFBE00?logo=weightsandbiases&logoColor=black)

---

## Overview

This repository explores how neural network optimization behaves in the context of **loss landscapes**. A CNN is trained on FashionMNIST many times (different seeds and optimizers) and the resulting models are compared beyond plain accuracy:

- **2D / 3D loss surfaces** around a trained model (with and without filter normalization)
- **1D interpolation** between independently trained models and across optimizers (Adam, AdamW, SGD, RMSprop, vanilla SGD)
- **Loss barriers** between pairs of models, with batch-norm recalibration
- **Prediction disagreement** across 100 trained models, overall and per class

Best validation accuracy of a single model: **~91.9 %**.

---

## Model Architecture

A simple but effective CNN:

```
Conv2d(1, 32, kernel_size=5) → BatchNorm → ReLU → MaxPool(2)
→ Conv2d(32, 64, kernel_size=5) → BatchNorm → ReLU → MaxPool(2)
→ Flatten → Linear(64·4·4 = 1024, 128) → ReLU
→ Linear(128, 64) → ReLU
→ Linear(64, 10)
```

- **Loss**: CrossEntropyLoss
- **Optimizer**: AdamW
- **Scheduler**: OneCycleLR
- **Training**: 16-bit mixed precision, gradient clipping, gradient accumulation, deterministic seeds

---

## Experiment Pipeline

1. **Hyperparameter search** – Hydra + Optuna sweeper (TPE sampler, 100 trials) over learning rate, weight decay, max LR and gradient clipping (`configs/default.yaml`).
2. **Final training** – best parameters (`configs/best_params.yaml`) trained across many seeds in parallel via the Hydra joblib launcher.
3. **Analysis** – checkpoints and predictions feed the visualization CLI below. All runs are logged to Weights & Biases.

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

# Per-class disagreement heatmap + bar across the 100 trained models
python loss_landscape.py per-class --predictions-dir ../predictions

# Pairwise loss-barrier heatmap with batch-norm recalibration
python loss_landscape.py barrier-heatmap --base-dir ../landscape_models

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

Python 3.9+ (developed on 3.10). All dependencies are listed in [`requirements.txt`](requirements.txt).

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
    │   ├── per_class.py
    │   ├── barrier_heatmap.py
    │   ├── histogram.py
    │   └── matrix.py
    ├── checkpoints/
    ├── outputs/
    ├── predictions/
    └── landscape_models/
```

---

## TODO / Roadmap

- [x] Basic CNN model training
- [x] Loss & accuracy tracking via Weights & Biases
- [x] Loss surface visualization (2D / 3D projections)
- [x] 1D interpolation and random-direction analysis
- [x] Per-class disagreement and BN-recalibrated loss-barrier matrix
- [x] Multi-optimizer comparison
- [x] Unified CLI entry point (`loss_landscape.py`)
- [x] Hydra sweeps + parallel training (joblib)
- [x] YAML config + W&B logging

---

## Author

**Ondřej Bassler**
[GitHub](https://github.com/obassler)
