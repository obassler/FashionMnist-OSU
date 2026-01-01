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

from data.datamodule_final import FashionMNISTDataModule
from models.model_final import FashionMNISTModel

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="best_params")
def train(cfg: DictConfig) -> float:
    if wandb.run is not None:
        wandb.finish()

    pl.seed_everything(cfg.training.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Run output will be saved in: {output_dir}")

    if HydraConfig.get().job.num is not None and cfg.data.num_workers > 0:
        log.warning("Multi-run detected. Setting data.num_workers to 0.")
        cfg.data.num_workers = 0

    wandb_logger = WandbLogger(
        project=cfg.wandb.project, name=f"run_seed_{cfg.training.seed}",
        tags=cfg.wandb.tags, notes=cfg.wandb.notes, log_model=cfg.wandb.log_model,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    data_module = FashionMNISTDataModule(
        batch_size=cfg.data.batch_size, data_dir=cfg.data.data_dir,
        num_workers=cfg.data.num_workers, seed=cfg.training.seed
    )

    model = FashionMNISTModel(
        input_size=cfg.model.input_size, num_classes=cfg.model.num_classes,
        learning_rate=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay,
        max_lr=cfg.training.max_lr,
        save_path_dir=output_dir
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename=cfg.training.checkpointing.filename,
        monitor=cfg.training.checkpointing.monitor, mode=cfg.training.checkpointing.mode,
        save_top_k=cfg.training.checkpointing.save_top_k,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.num_epochs, accelerator=cfg.training.accelerator,
        devices=cfg.training.devices, callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger, deterministic=cfg.training.deterministic,
        enable_progress_bar=cfg.training.enable_progress_bar,
        log_every_n_steps=cfg.logging.log_every_n_steps, precision=cfg.training.precision
    )

    log.info("Starting model training...")
    trainer.fit(model, datamodule=data_module)

    log.info("Starting testing with the best model...")
    test_results = trainer.test(model, datamodule=data_module, ckpt_path='best')
    log.info("Testing complete. Model has now saved its own predictions.")

    final_val_acc = trainer.callback_metrics.get("val_acc", torch.tensor(0.0)).item()
    final_test_acc = test_results[0]['test_acc']
    final_test_loss = test_results[0]['test_loss']

    log.info(f"Final Test Accuracy: {final_test_acc:.4f}, Final Test Loss: {final_test_loss:.4f}")
    wandb.log({
        "final_val_acc": final_val_acc,
        "final_test_acc": final_test_acc,
        "final_test_loss": final_test_loss
    })
    # -----------------------------

    log.info("Run finished successfully. Closing W&B.")
    wandb.finish()

    return final_val_acc

if __name__ == "__main__":
    train()
