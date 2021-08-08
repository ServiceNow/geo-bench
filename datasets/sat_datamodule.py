from torch.utils.data import DataLoader
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule
import os
import random

from datasets.sat_dataset import SatDataset
from utils.utils import get_embeddings


class SatDataModule(LightningDataModule):
    def __init__(self, args, encoder=None):
        super().__init__()
        self.data_dir = args.data_dir
        self.encoder = encoder
        self.num_workers = args.num_workers
        self.bs = args.batch_size

    @property
    def num_classes(self):

        return 6

    def setup(self, stage=None):

        self.train_dataset = SatDataset(self.data_dir, split="train", transform=T.ToTensor())
        self.val_dataset = SatDataset(self.data_dir, split="val", transform=T.ToTensor())

        if self.encoder:
            get_embeddings(self.encoder, self.train_dataset)
            get_embeddings(self.encoder, self.val_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def add_encoder(self, encoder):
        self.encoder = encoder
