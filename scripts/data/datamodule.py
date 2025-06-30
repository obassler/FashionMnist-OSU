import pytorch_lightning as pl
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 512, data_dir: str = "datasets/",
                 num_workers: int = 4, pin_memory: bool = True):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.train_size = 50000
        self.val_size = 10000

    def prepare_data(self):
        datasets.FashionMNIST(root=self.data_dir, train=True, download=True)
        datasets.FashionMNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            full_dataset = datasets.FashionMNIST(
                root=self.data_dir,
                train=True,
                transform=self.transform
            )

            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [self.train_size, self.val_size],
                generator=torch.Generator().manual_seed(42)
            )

            self.val_dataset.dataset.transform = self.test_transform

        if stage == "test" or stage is None:
            self.test_dataset = datasets.FashionMNIST(
                root=self.data_dir,
                train=False,
                transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
