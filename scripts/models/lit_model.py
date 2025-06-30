import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy
import wandb


class FashionMNISTModel(pl.LightningModule):
    def __init__(self, input_size: int = 784, num_classes: int = 10,
                 learning_rate: float = 0.0001, weight_decay: float = 1e-4,
                 max_lr: float = 0.01):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
        ]

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, 28, 28)
        x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        x = F.relu(self.bn2(F.max_pool2d(self.conv2(x), 2)))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            self.log('epoch', self.current_epoch)

            val_acc = self.trainer.callback_metrics.get('val_acc', 0)
            if hasattr(val_acc, 'item'):
                val_acc = val_acc.item()

            if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
                self.logger.experiment.log({
                    'epoch': self.current_epoch,
                    'val_acc_custom': val_acc
                })

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=float(self.hparams.weight_decay)
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
