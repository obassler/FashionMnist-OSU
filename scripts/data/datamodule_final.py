import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from typing import Optional

class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 11,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.data_train = None
        self.data_val = None
        self.data_test = None


    def prepare_data(self):
        FashionMNIST(self.hparams.data_dir, train=True, download=True)
        FashionMNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            fashion_mnist_full = FashionMNIST(self.hparams.data_dir, train=True, transform=self.transform)
            
            n_samples = len(fashion_mnist_full)
            n_val = int(self.hparams.val_split * n_samples)
            n_train = n_samples - n_val

            self.data_train, self.data_val = random_split(
                fashion_mnist_full, [n_train, n_val], 
                generator=torch.Generator().manual_seed(self.hparams.seed)
            )

        if stage == "test" or stage is None:
            self.data_test = FashionMNIST(self.hparams.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers = False
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers = False
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers = False
        )