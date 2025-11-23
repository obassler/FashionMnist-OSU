import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def load_all_predictions(predictions_dir):
    all_preds = []
    for i in range(100):
        file_path = os.path.join(predictions_dir, f"{i}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
                for preds in data.values():
                    if isinstance(preds, list):
                        all_preds.append(np.array(preds))
                        break
    return all_preds

def group_predictions(predictions, group_size=5):
    grouped = []
    for i in range(0, len(predictions), group_size):
        group = predictions[i:i+group_size]
        if group:
            avg_pred = np.mean(group, axis=0)
            grouped.append(avg_pred)
    return grouped

def calculate_group_to_overall_correlation(all_predictions, grouped_predictions):
    correlations = []

    if not all_predictions:
        return []

    all_preds_array = np.array(all_predictions)
    overall_avg_prediction = np.mean(all_preds_array, axis=0)

    vec_overall = overall_avg_prediction.flatten()

    for i, group_avg in enumerate(grouped_predictions):
        vec_group = group_avg.flatten()
     
        if len(vec_group) == len(vec_overall) and len(vec_group) > 1:
            corr, _ = pearsonr(vec_group, vec_overall)
            correlations.append(corr)
        else:
            print(f"Warning: Skipping correlation for group {i}. Vector lengths are {len(vec_group)} and {len(vec_overall)}.")
    
    return correlations

predictions_dir = "predictions"

all_predictions = load_all_predictions(predictions_dir)
grouped_predictions = group_predictions(all_predictions, group_size=5)
correlations = calculate_group_to_overall_correlation(all_predictions, grouped_predictions)

if not correlations:
    print("ERROR: No correlations were calculated. Cannot generate plot.")
    exit()

fig, ax = plt.subplots(figsize=(12, 7))

group_size = 5
num_groups = len(correlations)
group_labels = [f"{i*group_size}-{(i+1)*group_size - 1}" for i in range(num_groups)]
x_positions = np.arange(num_groups)

ax.bar(x_positions, correlations, color='#0064C8', alpha=0.8, edgecolor='darkblue', linewidth=1.0)

ax.set_xticks(x_positions)
ax.set_xticklabels(group_labels, rotation=45, ha="right")

ax.set_title('Histogram of models grouped by 5 and showing their corelation frequency', fontsize=14, fontweight='bold')
ax.set_xlabel('Model Groups by 5', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')

ax.set_ylim(min(0.9, min(correlations) - 0.01), max(1.0, max(correlations) + 0.01))
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('histogram.svg', format='svg', dpi=300, bbox_inches='tight')
plt.close()
