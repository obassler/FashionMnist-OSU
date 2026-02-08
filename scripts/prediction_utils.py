import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr


def load_predictions(predictions_dir: str, num_files: int = 100) -> list:
    predictions = []

    for i in range(num_files):
        file_path = Path(predictions_dir) / f"{i}.json"
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                pred_array = data.get("test_predictions.pt", [])
                if pred_array:
                    predictions.append(np.array(pred_array))
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
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
