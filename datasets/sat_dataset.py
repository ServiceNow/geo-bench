from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image
from scipy.io import loadmat
import numpy as np


class SatDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.split = split
        self.transform = transform

        self.class_to_idx = {
            "building": 0,
            "barren land": 1,
            "trees": 2,
            "grassland": 3,
            "road": 4,
            "water": 5,
        }

        mat = loadmat(data_dir)

        if self.split == "train":
            self.data = mat["train_x"]
            self.targets = mat["train_y"]

        else:
            self.data = mat["test_x"]
            self.targets = mat["test_y"]

        self.data = np.transpose(self.data, (3, 0, 1, 2))[:, :, :, :3]

        self.targets = np.transpose(self.targets, (1, 0))
        self.targets = np.where(self.targets == 1)[1]

        self.embeddings = None

    def __getitem__(self, index):

        target = self.targets[index].astype(np.long)

        if self.embeddings is not None:
            return self.embeddings[index], target

        img = self.data[index].astype(np.float32)

        if self.transform is not None and len(img.shape) in (2, 3):
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.data.shape[0]

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings
