import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
import os
import logging

log = logging.getLogger(__name__)


class FashionMNISTModel(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 784,
        num_classes: int = 10,
        learning_rate: float = 0.0001,
        weight_decay: float = 1e-4,
        max_lr: float = 0.01,
        save_predictions_dir: str = None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.test_step_outputs = []
        self.save_predictions_dir = save_predictions_dir

        self.class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
        ]

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
        acc = self.train_accuracy(preds, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch)
        acc = self.val_accuracy(preds, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch)
        acc = self.test_accuracy(preds, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)

        self.test_step_outputs.append(preds)
        return loss

    def on_test_epoch_end(self):
        if self.save_predictions_dir is not None:
            try:
                all_preds = torch.cat(self.test_step_outputs)
                save_path = os.path.join(self.save_predictions_dir, "test_predictions.pt")
                torch.save(all_preds, save_path)
                log.info(f"Saved {len(all_preds)} predictions to {save_path}")
            except Exception as e:
                log.error(f"Could not save predictions: {e}")
        self.test_step_outputs.clear()

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            self.log('epoch', float(self.current_epoch))

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.max_lr,
                total_steps=self.trainer.estimated_stepping_batches,
            ),
            'interval': 'step',
            'frequency': 1
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
