#  FashionMNIST Loss Landscape Exploration

![FashionMNIST Banner](https://user-images.githubusercontent.com/26833433/239359139-ce0a434e-9056-43e0-a306-3214f193dcce.png)

> A bachelor’s project focused on analyzing and visualizing the **loss landscape** of convolutional neural networks trained on the FashionMNIST dataset using PyTorch Lightning.

---

##  Overview

This repository explores how neural network optimization behaves in the context of **loss landscapes**. Using a CNN trained on the FashionMNIST dataset, we analyze curvature, convergence patterns, and training dynamics. The project leverages **modern PyTorch workflows**, advanced scheduling, and visualization tools to investigate deep learning behavior beyond just accuracy metrics.

---

##  Model Architecture

A simple but effective CNN architecture:

```python
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

##  Quick Start

```bash
# Clone the repo
git clone https://github.com/obassler/FashionMnist-OSU.git
cd FashionMnist-OSU

# (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run training
cd scripts
python train.py
```

> Make sure you have a GPU for faster training, although CPU works for experimentation.

---

##  Experiments

This project is built around **experimenting with the loss landscape** of CNNs:

-  Training dynamics with **OneCycleLR**
-  Model accuracy tracked via `torchmetrics`
-  Visualizations (coming soon) using **Matplotlib**, **Seaborn**, and **TensorBoard**
-  Exploring curvature and optimization smoothness

Plots and visualizations will be available in the `assets/` directory or via TensorBoard logs.

---

##  Requirements

> Python 3.9+ recommended

```txt
torch>=1.12.0
torchvision>=0.13.0
pytorch-lightning>=1.8.0
torchmetrics>=0.11.0
hydra-core>=1.3.0
omegaconf>=2.3.0
wandb>=0.13.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
hydra-optuna-sweeper>=1.2.0
pyyaml>=6.0
tqdm>=4.64.0
```

---

##  Project Structure

```
FashionMnist-OSU/
│
├── requirements.txt
├── README.md
└── scripts/
    ├── train.py
    ├── train_joblib.py
    ├── loss_landscape.py
    ├── histogram.py
    ├── matrix.py
    ├── prediction_utils.py
    ├── Utils.py
    ├── configs/
    │   ├── default.yaml
    │   └── best_params.yaml
    ├── data/
    │   └── datamodule.py
    └── models/
        └── lit_model.py
```

---

##  About

This project was developed as part of a **Bachelor's thesis** at University of Ostrava, focusing on improving understanding of deep learning dynamics via **loss landscape analysis**.

---

##  TODO / Roadmap

- [x] Basic CNN model training
- [x] Loss & accuracy tracking with TensorBoard
- [x] Loss surface visualization (2D/3D projections)
- [ ] Add CLI or main.py entry point
- [x] Add experiment logging support (e.g. YAML config, wandb)
- [x] Hydra sweeps functionality 

---

##  Author

**Ondřej Bassler**  
[GitHub](https://github.com/obassler)
