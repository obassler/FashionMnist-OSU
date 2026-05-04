import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Optional
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datamodule import FashionMNISTDataModule


class FashionMNISTModelOld(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = torch.relu(self.bn1(torch.max_pool2d(self.conv1(x), 2)))
        x = torch.relu(self.bn2(torch.max_pool2d(self.conv2(x), 2)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('model.', '') if k.startswith('model.') else k
        cleaned_state_dict[new_key] = v

    if 'conv1.weight' in cleaned_state_dict:
        model = FashionMNISTModelOld()
    elif 'feature_extractor.0.weight' in cleaned_state_dict:
        from models.lit_model import FashionMNISTModel
        model = FashionMNISTModel()
    else:
        raise ValueError(f"Unknown architecture. Keys: {list(cleaned_state_dict.keys())[:5]}")

    model_keys = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in cleaned_state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state_dict)

    return model


def get_weights(model: nn.Module) -> List[torch.Tensor]:
    return [p.data.clone() for p in model.parameters()]


def set_weights(model: nn.Module, weights: List[torch.Tensor]) -> None:
    for p, w in zip(model.parameters(), weights):
        p.data.copy_(w)


def get_state(model: nn.Module) -> List[torch.Tensor]:
    return [v.data.clone() for v in model.state_dict().values()]


def set_state(model: nn.Module, state: List[torch.Tensor]) -> None:
    for param, val in zip(model.state_dict().values(), state):
        param.copy_(val)


def flatten_weights(weights: List[torch.Tensor]) -> np.ndarray:
    return np.concatenate([w.cpu().numpy().flatten() for w in weights])


def unflatten_weights(flat: np.ndarray, reference: List[torch.Tensor]) -> List[torch.Tensor]:
    result = []
    offset = 0
    for w in reference:
        n = w.numel()
        result.append(torch.from_numpy(flat[offset:offset + n].reshape(w.shape)).float())
        offset += n
    return result


def get_random_direction(weights: List[torch.Tensor]) -> List[torch.Tensor]:
    return [torch.randn_like(w) for w in weights]


def normalize_direction_filter_wise(
    direction: List[torch.Tensor],
    weights: List[torch.Tensor],
    ignore_bn: bool = True
) -> List[torch.Tensor]:
    normalized = []

    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore_bn and d.dim() == 1:
                normalized.append(torch.zeros_like(d))
            else:
                d_norm = d.norm()
                w_norm = w.norm()
                if d_norm > 1e-10:
                    normalized.append(d * (w_norm / d_norm))
                else:
                    normalized.append(torch.zeros_like(d))
        elif d.dim() in (2, 4):
            d_normalized = torch.zeros_like(d)
            for i in range(d.shape[0]):
                d_norm = d[i].norm()
                w_norm = w[i].norm()
                if d_norm > 1e-10:
                    d_normalized[i] = d[i] * (w_norm / d_norm)
            normalized.append(d_normalized)
        else:
            d_norm = d.norm()
            w_norm = w.norm()
            if d_norm > 1e-10:
                normalized.append(d * (w_norm / d_norm))
            else:
                normalized.append(torch.zeros_like(d))

    return normalized


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples


def load_predictions(predictions_dir: str) -> list:
    predictions = []
    predictions_path = Path(predictions_dir)

    if not predictions_path.exists():
        return predictions

    json_files = sorted(predictions_path.glob("*.json"), key=lambda p: int(p.stem))

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                pred_array = data.get("test_predictions.pt", [])
                if pred_array:
                    predictions.append(np.array(pred_array))
        except (json.JSONDecodeError, ValueError):
            pass

    return predictions


def compute_correlation_matrix(predictions: list) -> np.ndarray:
    if not predictions:
        return np.array([])

    min_length = min(len(p) for p in predictions)
    predictions = [p[:min_length] for p in predictions]

    n = len(predictions)
    corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                try:
                    corr, _ = pearsonr(predictions[i], predictions[j])
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                except Exception:
                    corr_matrix[i, j] = 0.0
                    corr_matrix[j, i] = 0.0

    return corr_matrix


def extract_upper_triangle(corr_matrix: np.ndarray) -> np.ndarray:
    n = corr_matrix.shape[0]
    upper_triangle_indices = np.triu_indices(n, k=1)
    return corr_matrix[upper_triangle_indices]
