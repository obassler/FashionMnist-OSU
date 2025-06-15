import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datamodule import FashionMNISTDataModule
from models.lit_model import FashionMNISTModel


class FashionMNISTEvaluator:
    def __init__(self, model_path, config_path=None):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.class_labels = {
            0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
            5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"
        }

    def load_model(self):
        self.model = FashionMNISTModel.load_from_checkpoint(self.model_path)
        self.model.eval()
        self.model.to(self.device)

    def setup_data(self, batch_size=512):
        self.data_module = FashionMNISTDataModule(batch_size=batch_size)
        self.data_module.setup()

    def evaluate_model(self):
        test_loader = self.data_module.test_dataloader()

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                x_flat = x.reshape(x.shape[0], -1)
                logits = self.model(x_flat)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_targets), np.array(all_probs)

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(self.class_labels.values()),
                    yticklabels=list(self.class_labels.values()))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_sample_predictions(self, num_samples=8, save_path=None):
        test_loader = self.data_module.test_dataloader()

        batch = next(iter(test_loader))
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            x_flat = x.reshape(x.shape[0], -1)
            logits = self.model(x_flat)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()

        for i in range(num_samples):
            img = x[i].cpu().squeeze()
            true_label = self.class_labels[y[i].item()]
            pred_label = self.class_labels[preds[i].item()]
            confidence = probs[i][preds[i]].item()

            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}')
            axes[i].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_class_accuracy(self, y_true, y_pred, save_path=None):
        class_accuracies = []
        for class_id in range(10):
            mask = y_true == class_id
            if mask.sum() > 0:
                acc = (y_pred[mask] == y_true[mask]).mean()
                class_accuracies.append(acc)
            else:
                class_accuracies.append(0)

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(10), class_accuracies)
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.xticks(range(10), list(self.class_labels.values()), rotation=45)
        plt.ylim(0, 1)

        for bar, acc in zip(bars, class_accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self, save_dir=None):
        print("Loading model...")
        self.load_model()

        print("Setting up data...")
        self.setup_data()

        print("Evaluating model...")
        y_pred, y_true, y_probs = self.evaluate_model()

        accuracy = (y_pred == y_true).mean()
        print(f"\nOverall Test Accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=list(self.class_labels.values())))

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        print("\nGenerating confusion matrix...")
        self.plot_confusion_matrix(y_true, y_pred,
                                   save_path=f"{save_dir}/confusion_matrix.png" if save_dir else None)

        print("Generating sample predictions...")
        self.plot_sample_predictions(save_path=f"{save_dir}/sample_predictions.png" if save_dir else None)

        print("Generating class accuracy plot...")
        self.plot_class_accuracy(y_true, y_pred,
                                 save_path=f"{save_dir}/class_accuracy.png" if save_dir else None)

        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': y_true,
            'probabilities': y_probs
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate FashionMNIST model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')

    args = parser.parse_args()

    evaluator = FashionMNISTEvaluator(args.model_path)
    results = evaluator.generate_report(args.save_dir)

    print(f"\nEvaluation complete! Results saved to {args.save_dir}")


if __name__ == '__main__':
    main()