from torch.utils.data import DataLoader
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule
import os
import random

from datasets.eurosat_dataset import EurosatDataset
from utils.utils import get_embeddings


class EurosatDataModule(LightningDataModule):
    def __init__(self, args, encoder=None):
        super().__init__()
        self.data_dir = args.data_dir
        self.encoder = encoder
        self.bs = args.batch_size
        self.num_workers = args.num_workers

    @property
    def num_classes(self):
        return 10

    def setup(self, stage=None):

        # if not os.path.exists("datasets/eurosat/train.txt") or not os.path.exists("datasets/eurosat/val.txt"):

        im_paths = []
        for root, dirs, files in os.walk("datasets/eurosat"):
            for file in files:
                if file == "train.txt" or file == "val.txt":
                    continue

                im_paths.append(file)

        random.shuffle(im_paths)

        train_paths = im_paths[: int(len(im_paths) * 0.6)]
        val_paths = im_paths[int(len(im_paths) * 0.6) :]

        with open("datasets/eurosat/train.txt", "w") as f:
            for p in train_paths:
                f.write(p + "\n")

        with open("datasets/eurosat/val.txt", "w") as f:
            for p in val_paths:
                f.write(p + "\n")

        self.train_dataset = EurosatDataset(self.data_dir, split="train", transform=T.ToTensor())
        self.val_dataset = EurosatDataset(self.data_dir, split="val", transform=T.ToTensor())

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
