import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
import os
import logging

log = logging.getLogger(__name__)

class FashionMNISTModel(pl.LightningModule):
    def __init__(self, input_size: int = 784, num_classes: int = 10,
                 learning_rate: float = 0.0001, weight_decay: float = 1e-4,
                 max_lr: float = 0.01, save_path_dir: str = "."):
        super().__init__()
        self.save_hyperparameters()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict({
            'train_acc': Accuracy(task="multiclass", num_classes=num_classes),
            'val_acc': Accuracy(task="multiclass", num_classes=num_classes),
            'test_acc': Accuracy(task="multiclass", num_classes=num_classes),
        })
        self.test_step_outputs = []
        self.save_path_dir = save_path_dir

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

    def _common_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch)
        acc = self.metrics['train_acc'](preds, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch)
        acc = self.metrics['val_acc'](preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch)
        acc = self.metrics['test_acc'](preds, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.test_step_outputs.append(preds)
        return preds

    def on_test_epoch_end(self):
        try:
            all_preds = torch.cat(self.test_step_outputs)
            save_path = os.path.join(self.save_path_dir, "test_predictions.pt")
            torch.save(all_preds, save_path)
            log.info(f"MODEL-SIDE SAVE: Saved {len(all_preds)} predictions to {save_path}")
        except Exception as e:
            log.error(f"MODEL-SIDE SAVE FAILED: Could not save predictions. Error: {e}")
        finally:
            self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.hparams.max_lr,
                total_steps=self.trainer.estimated_stepping_batches,
            ),
            'interval': 'step',
        }
        return [optimizer], [scheduler]