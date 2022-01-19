from torch.utils.data import DataLoader
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule
from PIL import Image
import os
import random

# from deepforest import main, dataset
from ccb.datasets.datasets import (
    SatDataset,
    EurosatDataset,
    ChangeDetectionDataset,
    ForestNetDataset,
    BigEarthNetDataset,
    GeoClefDataset,
    Sen12FloodDataset,
)
from ccb.utils.utils import get_embeddings, RandomFlip, RandomRotation, Compose, ToTensor, random_subset

# import torchrs.transforms
# from torchrs.datasets import ETCI2021

# transform = Compose([ToTensor()])

# dataset = ETCI2021(
#     root="path/to/dataset/",
#     split="train",  # or 'val', 'test'
#     transform=transform
# )

# x = dataset[0]
# """
# x: dict(
#     vv:         (3, 256, 256)
#     vh:         (3, 256, 256)
#     flood_mask: (1, 256, 256)
#     water_mask: (1, 256, 256)
# )
# """


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
            self.train_dataset = EurosatDataset(
                self.data_dir, split="train", transform=T.Compose([T.Resize((224, 224)), T.ToTensor()])
            )
            self.val_dataset = EurosatDataset(
                self.data_dir, split="val", transform=T.Compose([T.Resize((224, 224)), T.ToTensor()])
            )
        elif self.dataset == "forestnet":
            self.train_dataset = ForestNetDataset(self.data_dir, split="train", transform=T.ToTensor())
            self.val_dataset = ForestNetDataset(self.data_dir, split="val", transform=T.ToTensor())

        elif self.dataset == "bigearthnet":
            self.train_dataset = BigEarthNetDataset(
                self.data_dir,
                split="train",
                transform=T.Compose([T.Resize((128, 128), interpolation=Image.BICUBIC), T.ToTensor()]),
            )
            self.val_dataset = BigEarthNetDataset(
                self.data_dir,
                split="val",
                transform=T.Compose([T.Resize((128, 128), interpolation=Image.BICUBIC), T.ToTensor()]),
            )
        elif self.dataset == "sen12flood":
            self.train_dataset = Sen12FloodDataset(
                self.data_dir,
                split="train",
                transform=T.Compose([T.Resize((224, 224), interpolation=Image.BICUBIC), T.ToTensor()]),
            )
            self.val_dataset = Sen12FloodDataset(
                self.data_dir,
                split="val",
                transform=T.Compose([T.Resize((224, 224), interpolation=Image.BICUBIC), T.ToTensor()]),
            )

        # elif self.dataset == "deepforest":
        #     m = main.deepforest()
        #     m.use_release()
        #     print(m.config["train"])
        #     self.train_dataset = dataset.TreeDataset(csv_file=m.config["train"]["csv_file"],
        #                             root_dir=m.config["train"]["root_dir"],
        #                             transforms=m.transforms(augment=True),
        #                             label_dict=m.label_dict)
        #     self.val_dataset = dataset.TreeDataset(csv_file=m.config["validation"]["csv_file"],
        #                             root_dir=m.config["validation"]["root_dir"],
        #                             transforms=m.transforms(augment=False),
        #                             label_dict=m.label_dict)

        elif self.dataset == "geoclef":
            self.train_dataset = GeoClefDataset(self.data_dir, split="train", transform=ToTensor)
            self.val_dataset = GeoClefDataset(self.data_dir, split="val", transform=ToTensor)

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

        if args.train_frac < 1:
            self.train_dataset = random_subset(self.train_dataset, args.train_frac, args.seed)
            print(len(self.train_dataset))
        if args.val_frac < 1:
            self.val_dataset = random_subset(self.val_dataset, args.val_frac, args.seed)

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
