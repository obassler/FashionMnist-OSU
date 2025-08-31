import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import logging
import os

from data.datamodule import FashionMNISTDataModule
from models.lit_model import FashionMNISTModel

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def train(cfg: DictConfig) -> float:
    # Each job will run in a unique directory created by Hydra.
    # We can use this to our advantage for logging and checkpointing.
    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    pl.seed_everything(cfg.training.seed, workers=True)

    # This combines the base name from your config with the specific seed and job number.
    run_name = f"{cfg.wandb.run_name}_seed_{cfg.training.seed}_job_{hydra.core.hydra_config.HydraConfig.get().job.num}"

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=run_name,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        # Logs will be saved in a unique directory managed by the logger within Hydra's output folder
        save_dir=".",
        log_model=cfg.wandb.log_model,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )

    data_module = FashionMNISTDataModule(
        batch_size=cfg.data.batch_size,
        data_dir=cfg.data.data_dir,
        num_workers=cfg.data.num_workers
    )

    data_module.setup('fit')

    total_steps = (data_module.train_size // cfg.data.batch_size) * cfg.training.num_epochs

    model = FashionMNISTModel(
        input_size=cfg.model.input_size,
        num_classes=cfg.model.num_classes,
        weight_decay=cfg.training.weight_decay,
        max_lr=cfg.training.max_lr,
        total_steps=total_steps
    )

    callbacks = []

    if cfg.training.early_stopping.enable:
        early_stopping = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            mode=cfg.training.early_stopping.mode,
            patience=cfg.training.early_stopping.patience,
            verbose=cfg.training.early_stopping.verbose
        )
        callbacks.append(early_stopping)

    if cfg.training.checkpointing.enable:
        # --- FIX 1: REMOVED `dirpath` ---
        # ModelCheckpoint will now automatically save to the logger's directory,
        # which is correctly placed inside the unique Hydra output folder.
        checkpoint_callback = ModelCheckpoint(
            filename=cfg.training.checkpointing.filename,
            monitor=cfg.training.checkpointing.monitor,
            mode=cfg.training.checkpointing.mode,
            save_top_k=cfg.training.checkpointing.save_top_k,
            save_last=cfg.training.checkpointing.save_last,
            verbose=cfg.training.checkpointing.verbose
        )
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        max_epochs=cfg.training.num_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        callbacks=callbacks,
        logger=wandb_logger,
        deterministic=cfg.training.deterministic,
        enable_progress_bar=cfg.training.enable_progress_bar,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        precision=cfg.training.precision
    )

    trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)

    val_acc = trainer.callback_metrics.get("val_acc", 0.0)
    if hasattr(val_acc, 'item'):
        val_acc = val_acc.item()

    # The W&B logger will log this automatically. If you need it for other purposes, it's fine.
    wandb.log({"final_val_acc": val_acc})

    # --- FIX 2: REMOVED `wandb.finish()` ---
    # The WandbLogger will handle finishing the run automatically when the script ends.
    # Calling it manually can interfere with the Joblib process cleanup.

    return val_acc


if __name__ == "__main__":
    train()

