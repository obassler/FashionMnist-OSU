import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr

predictions_folder = "predictions"
num_files = 100

predictions = []
for i in range(num_files):
    file_path = Path(predictions_folder) / f"{i}.json"
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

if predictions:
    min_length = min(len(p) for p in predictions)
    predictions = [p[:min_length] for p in predictions]

n = len(predictions)
corr_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(i, n):
        if i == j:
            corr_matrix[i, j] = 1.0
        else:
            pred_i = predictions[i]
            pred_j = predictions[j]
            try:
                corr, _ = pearsonr(pred_i, pred_j)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
            except Exception:
                corr_matrix[i, j] = 0.0
                corr_matrix[j, i] = 0.0

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={'label': 'Correlation', 'shrink': 0.8},
            xticklabels=True, yticklabels=True, ax=ax, linewidths=0.5, linecolor='gray',
            annot=True, fmt='.4f', annot_kws={'size': 6})

plt.title('Correlation Matrix of 100 Model Predictions', fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Prediction File Index', fontsize=12, fontweight='bold')
plt.ylabel('Prediction File Index', fontsize=12, fontweight='bold')

ax.set_xticklabels(range(100), rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(range(100), rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig('correlation_matrix_100x100.svg', format='svg', dpi=300, bbox_inches='tight')
plt.close()

np.save('correlation_matrix.npy', corr_matrix)