import os
import pickle
import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def save_model_checkpoint(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model checkpoint saved to {filepath}")


def load_model_checkpoint(model, filepath):
    model.load_state_dict(torch.load(filepath))
    return model


def save_pickle(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filepath}")


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def save_yaml(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data


def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_class_labels():
    return {
        0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
        5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"
    }


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def compare_multiple_runs(results_list, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    run_ids = [r['run_id'] for r in results_list]
    val_accs = [r['best_val_acc'] for r in results_list]

    ax1.bar(run_ids, val_accs, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Run ID')
    ax1.set_ylabel('Best Validation Accuracy')
    ax1.set_title('Validation Accuracy Across Runs')
    ax1.set_ylim(0, 1)

    for i, acc in enumerate(val_accs):
        ax1.text(run_ids[i], acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')

    mean_acc = np.mean(val_accs)
    std_acc = np.std(val_accs)

    ax2.axhline(y=mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.3f}')
    ax2.axhline(y=mean_acc + std_acc, color='orange', linestyle=':', label=f'Mean + Std: {mean_acc + std_acc:.3f}')
    ax2.axhline(y=mean_acc - std_acc, color='orange', linestyle=':', label=f'Mean - Std: {mean_acc - std_acc:.3f}')
    ax2.scatter(run_ids, val_accs, alpha=0.7, s=50)
    ax2.set_xlabel('Run ID')
    ax2.set_ylabel('Best Validation Accuracy')
    ax2.set_title('Statistical Summary')
    ax2.legend()
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nTraining Summary Statistics:")
    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"Standard Deviation: {std_acc:.4f}")
    print(f"Min Accuracy: {min(val_accs):.4f}")
    print(f"Max Accuracy: {max(val_accs):.4f}")


def create_results_summary(results_list, save_path=None):
    summary = {
        'num_runs': len(results_list),
        'mean_accuracy': np.mean([r['best_val_acc'] for r in results_list]),
        'std_accuracy': np.std([r['best_val_acc'] for r in results_list]),
        'min_accuracy': min([r['best_val_acc'] for r in results_list]),
        'max_accuracy': max([r['best_val_acc'] for r in results_list]),
        'results': results_list
    }

    if save_path:
        save_yaml(summary, save_path)

    return summary


def plot_dataset_distribution(train_targets, val_targets, test_targets, save_path=None):
    class_labels = get_class_labels()

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].hist(train_targets, bins=10, color='#056fff', rwidth=0.6, alpha=0.7)
    axes[0].set_title('Training Set Class Distribution')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_xticks(range(10))
    axes[0].set_xticklabels(class_labels.values(), rotation=45)

    axes[1].hist(val_targets, bins=10, color='#256812', rwidth=0.6, alpha=0.7)
    axes[1].set_title('Validation Set Class Distribution')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')
    axes[1].set_xticks(range(10))
    axes[1].set_xticklabels(class_labels.values(), rotation=45)

    axes[2].hist(test_targets, bins=10, color='#101010', rwidth=0.6, alpha=0.7)
    axes[2].set_title('Test Set Class Distribution')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('Count')
    axes[2].set_xticks(range(10))
    axes[2].set_xticklabels(class_labels.values(), rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sample_images(dataloader, num_samples=8, save_path=None):
    class_labels = get_class_labels()

    batch = next(iter(dataloader))
    images, labels = batch

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(min(num_samples, len(images))):
        img = images[i].squeeze()
        label = class_labels[labels[i].item()]

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Class: {label}')
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
