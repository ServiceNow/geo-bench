from torch.utils.data import DataLoader
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule
import os
import random

from datasets.datasets import SatDataset, EurosatDataset, ChangeDetectionDataset, ForestNetDataset
from utils.utils import get_embeddings, RandomFlip, RandomRotation, Compose, ToTensor


class DataModule(LightningDataModule):
    def __init__(self, args, encoder=None):
        super().__init__()
        self.data_dir = args.data_dir
        self.encoder = encoder
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.patch_size = args.patch_size

        if self.dataset == "sat":
            self.train_dataset = SatDataset(self.data_dir, split="train", transform=T.ToTensor())
            self.val_dataset = SatDataset(self.data_dir, split="val", transform=T.ToTensor())

        elif self.dataset == "eurosat":
            self.train_dataset = EurosatDataset(self.data_dir, split="train", transform=T.ToTensor())
            self.val_dataset = EurosatDataset(self.data_dir, split="val", transform=T.ToTensor())
        elif self.dataset == "forestnet":
            self.train_dataset = ForestNetDataset(self.data_dir, split="train", transform=T.ToTensor())
            self.val_dataset = ForestNetDataset(self.data_dir, split="val", transform=T.ToTensor())

        elif self.dataset == "oscd":
            self.train_dataset = ChangeDetectionDataset(
                self.data_dir,
                split="train",
                transform=Compose([RandomFlip, RandomRotation, ToTensor]),
                patch_size=self.patch_size,
            )
            self.val_dataset = ChangeDetectionDataset(
                self.data_dir, split="test", transform=ToTensor, patch_size=self.patch_size
            )

    def setup(self, stage=None):

        if self.encoder:
            get_embeddings(self.encoder, self.train_dataset)
            get_embeddings(self.encoder, self.val_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def add_encoder(self, encoder):
        self.encoder = encoder

    def get_num_classes(self):
        return self.train_dataset.num_classes
