import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from hydra.core.hydra_config import HydraConfig
import wandb
import logging
import torch
import os

from data.datamodule import FashionMNISTDataModule
from models.lit_model import FashionMNISTModel

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="best_params")
def train(cfg: DictConfig) -> float:
    if wandb.run is not None:
        wandb.finish()

    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    pl.seed_everything(cfg.training.seed, workers=True)

    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Run output will be saved in: {output_dir}")

    try:
        job_num = HydraConfig.get().job.num
        if job_num is not None and cfg.data.num_workers > 0:
            log.warning("Multi-run detected. Setting data.num_workers to 0.")
            cfg.data.num_workers = 0
    except (ValueError, AttributeError):
        job_num = None
        log.info("Running a single job.")

    run_name = f"{cfg.wandb.run_name}_seed_{cfg.training.seed}"
    if job_num is not None:
        run_name = f"{run_name}_job_{job_num}"

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=run_name,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        log_model=cfg.wandb.log_model,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    data_module = FashionMNISTDataModule(
        batch_size=cfg.data.batch_size,
        data_dir=cfg.data.data_dir,
        num_workers=cfg.data.num_workers,
        seed=cfg.training.seed
    )

    model = FashionMNISTModel(
        input_size=cfg.model.input_size,
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        max_lr=cfg.training.max_lr,
        save_predictions_dir=output_dir
    )

    callbacks = []

    if cfg.training.checkpointing.enable:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=cfg.training.checkpointing.filename,
            monitor=cfg.training.checkpointing.monitor,
            mode=cfg.training.checkpointing.mode,
            save_top_k=cfg.training.checkpointing.save_top_k,
            save_last=cfg.training.checkpointing.get('save_last', True),
            verbose=cfg.training.checkpointing.get('verbose', True)
        )
        callbacks.append(checkpoint_callback)

    if 'early_stopping' in cfg.training and cfg.training.early_stopping.get('enable', False):
        early_stopping = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            mode=cfg.training.early_stopping.mode,
            patience=cfg.training.early_stopping.patience,
            verbose=cfg.training.early_stopping.verbose
        )
        callbacks.append(early_stopping)

    trainer = pl.Trainer(
        max_epochs=cfg.training.num_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        callbacks=callbacks,
        logger=wandb_logger,
        deterministic=cfg.training.deterministic,
        enable_progress_bar=cfg.training.enable_progress_bar,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        precision=cfg.training.precision,
        val_check_interval=cfg.training.get('val_check_interval', 1.0),
        limit_train_batches=cfg.training.get('limit_train_batches', 1.0),
        limit_val_batches=cfg.training.get('limit_val_batches', 1.0)
    )

    log.info("Starting model training...")
    trainer.fit(model, datamodule=data_module)

    log.info("Starting testing with the best model...")
    test_results = trainer.test(model, datamodule=data_module, ckpt_path='best')

    final_val_acc = trainer.callback_metrics.get("val_acc", torch.tensor(0.0))
    if hasattr(final_val_acc, 'item'):
        final_val_acc = final_val_acc.item()

    final_test_acc = test_results[0]['test_acc'] if test_results else 0.0
    final_test_loss = test_results[0]['test_loss'] if test_results else 0.0

    log.info(f"Final Val Accuracy: {final_val_acc:.4f}")
    log.info(f"Final Test Accuracy: {final_test_acc:.4f}")
    log.info(f"Final Test Loss: {final_test_loss:.4f}")

    wandb.log({
        "final_val_acc": final_val_acc,
        "final_test_acc": final_test_acc,
        "final_test_loss": final_test_loss
    })

    log.info("Run finished successfully. Closing W&B.")
    wandb.finish()

    return final_val_acc


if __name__ == "__main__":
    train()
