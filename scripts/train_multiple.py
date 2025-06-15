import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from data.datamodule import FashionMNISTDataModule
from models.lit_model import FashionMNISTModel


def setup_logging(base_log_dir):
    os.makedirs(base_log_dir, exist_ok=True)


def train_model(config, run_name="run"):
    data_module = FashionMNISTDataModule(
        batch_size=config['data']['batch_size'],
        data_dir=config['data']['data_dir']
    )

    model = FashionMNISTModel(
        input_size=config['model']['input_size'],
        num_classes=config['model']['num_classes'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        max_lr=config['training']['max_lr']
    )

    logger = TensorBoardLogger(
        save_dir=config['paths']['log_dir'],
        name='sweep_logs',
        version=run_name
    )

    early_stopping = EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=config['training']['patience'],
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        callbacks=[early_stopping],
        logger=logger,
        deterministic=True,
        accelerator='auto',
        devices='auto',
        log_every_n_steps=config['logging']['log_every_n_steps'],
        enable_progress_bar=True
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    val_acc = trainer.callback_metrics.get("val_acc")
    return val_acc.item() if val_acc else 0.0
