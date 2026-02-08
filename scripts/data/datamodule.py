import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from typing import Optional


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.1,
        seed: int = 42,
        pin_memory: bool = True,
        use_augmentation: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()

        if use_augmentation:
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.train_size = None
        self.val_size = None

    def prepare_data(self):
        FashionMNIST(self.hparams.data_dir, train=True, download=True)
        FashionMNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            full_train = FashionMNIST(
                self.hparams.data_dir,
                train=True,
                transform=self.train_transform
            )

            n_samples = len(full_train)
            n_val = int(self.hparams.val_split * n_samples)
            n_train = n_samples - n_val

            self.train_size = n_train
            self.val_size = n_val

            self.data_train, self.data_val = random_split(
                full_train,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(self.hparams.seed)
            )

            val_dataset = FashionMNIST(
                self.hparams.data_dir,
                train=True,
                transform=self.test_transform
            )
            self.data_val = torch.utils.data.Subset(
                val_dataset,
                self.data_val.indices
            )

        if stage == "test" or stage is None:
            self.data_test = FashionMNIST(
                self.hparams.data_dir,
                train=False,
                transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size * 2,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0
        )
