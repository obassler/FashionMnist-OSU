import argparse
import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datamodule import FashionMNISTDataModule
from models.lit_model import FashionMNISTModel


def load_config(config_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    full_path = os.path.join(root_dir, config_path)

    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Config file not found: {full_path}")

    with open(full_path, 'r') as file:
        return yaml.safe_load(file)


def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )


def train_single_model(config, run_id=None):
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

    early_stopping = EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=config['training']['patience'],
        verbose=True
    )

    logger = TensorBoardLogger(
        save_dir=config['paths']['log_dir'],
        name='fashion_mnist',
        version=f'run_{run_id}' if run_id else None
    )

    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        callbacks=[early_stopping],
        logger=logger,
        accelerator='auto',
        devices='auto',
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=config['logging']['log_every_n_steps']
    )

    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)

    return trainer, model


def main():
    parser = argparse.ArgumentParser(description='Train FashionMNIST model with PyTorch Lightning')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of training runs to perform')
    args = parser.parse_args()

    config = load_config(args.config)

    pl.seed_everything(config['training']['seed'])

    os.makedirs(config['paths']['log_dir'], exist_ok=True)

    log_file = os.path.join(config['paths']['log_dir'], 'training.log')
    setup_logging(log_file)

    results = []

    for run_id in range(1, args.num_runs + 1):
        logging.info(f"\n{'=' * 50}\nTraining Run {run_id}/{args.num_runs}\n{'=' * 50}")

        trainer, model = train_single_model(config, run_id)

        val_acc = trainer.callback_metrics.get('val_acc')
        val_acc = val_acc.item() if val_acc is not None else 0.0

        results.append({
            'run_id': run_id,
            'val_acc': val_acc,
        })

        logging.info(f"Run {run_id} completed. Best Validation Accuracy: {val_acc:.4f}")

    if args.num_runs > 1:
        logging.info(f"\n{'=' * 50}\nTRAINING SUMMARY\n{'=' * 50}")
        for result in results:
            logging.info(f"Run {result['run_id']}: Val Acc = {result['val_acc']:.4f}")
        avg_acc = sum(r['val_acc'] for r in results) / len(results)
        logging.info(f"\nAverage Validation Accuracy: {avg_acc:.4f}")


if __name__ == '__main__':
    main()
