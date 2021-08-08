import random

from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from pytorch_lightning import LightningDataModule

from datasets.oscd_dataset import ChangeDetectionDataset


class RandomFlip:
    def __call__(self, *xs):
        if random.random() > 0.5:
            xs = tuple(TF.hflip(x) for x in xs)
        return xs


class RandomRotation:
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, *xs):
        angle = random.choice(self.angles)
        return tuple(TF.rotate(x, angle) for x in xs)


class RandomSwap:
    def __call__(self, x1, x2, y):
        if random.random() > 0.5:
            return x2, x1, y
        else:
            return x1, x2, y


class ToTensor:
    def __call__(self, *xs):
        return tuple(TF.to_tensor(x) for x in xs)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *xs):
        for t in self.transforms:
            xs = t(*xs)
        return xs


class ChangeDetectionDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.patch_size = args.patch_size
        self.num_workers = args.num_workers
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size

    def setup(self, stage=None):
        self.train_dataset = ChangeDetectionDataset(
            self.data_dir,
            split="train",
            transform=Compose([RandomFlip(), RandomRotation(), ToTensor()]),
            patch_size=self.patch_size,
        )
        self.val_dataset = ChangeDetectionDataset(
            self.data_dir, split="test", transform=ToTensor(), patch_size=self.patch_size
        )

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
