from pathlib import Path
from torch.utils.data import Dataset
from itertools import product
import random
import rasterio
import numpy as np
from PIL import Image
from scipy.io import loadmat
import os

ALL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
RGB_BANDS = ["B04", "B03", "B02"]

QUANTILES = {
    "min_q": {"B02": 885.0, "B03": 667.0, "B04": 426.0},
    "max_q": {"B02": 2620.0, "B03": 2969.0, "B04": 3698.0},
}


class SatDataset(Dataset):
    def __init__(self, data_dir, split, transform=None, shuffle=True):
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

        if shuffle:
            print(self.data.shape, self.targets.shape)
            idx = np.random.shuffle(np.arange(self.targets.shape[0]))
            self.data = self.data[idx][0]
            self.targets = self.targets[idx][0]
            print(self.data.shape, self.targets.shape)

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

    @property
    def num_classes(self):
        return 10

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings


def read_image(path, bands, normalize=True):
    patch_id = next(path.iterdir()).name[:-8]
    channels = []
    for b in bands:
        ch = rasterio.open(path / f"{patch_id}_{b}.tif").read(1)
        if normalize:
            min_v = QUANTILES["min_q"][b]
            max_v = QUANTILES["max_q"][b]
            ch = (ch - min_v) / (max_v - min_v)
            ch = np.clip(ch, 0, 1)
            ch = (ch * 255).astype(np.uint8)
        channels.append(ch)
    img = np.dstack(channels)
    img = Image.fromarray(img)
    return img


class ChangeDetectionDataset(Dataset):
    def __init__(self, root, split="all", bands=None, transform=None, patch_size=96, shuffle=True):
        self.root = Path(root)
        self.split = split
        self.bands = bands if bands is not None else RGB_BANDS
        self.transform = transform

        with open(self.root / f"{split}.txt") as f:
            names = f.read().strip().split(",")

        self.samples = []
        for name in names:
            fp = next((self.root / name / "imgs_1").glob(f"*{self.bands[0]}*"))
            img = rasterio.open(fp)
            limits = product(range(0, img.width, patch_size), range(0, img.height, patch_size))
            for lim in limits:
                self.samples.append((self.root / name, (lim[0], lim[1], lim[0] + patch_size, lim[1] + patch_size)))

        if shuffle:
            random.shuffle(self.samples)

    def __getitem__(self, index):
        path, limits = self.samples[index]

        img_1 = read_image(path / "imgs_1", self.bands)
        img_2 = read_image(path / "imgs_2", self.bands)
        cm = Image.open(path / "cm" / "cm.png").convert("L")

        img_1 = img_1.crop(limits)
        img_2 = img_2.crop(limits)
        cm = cm.crop(limits)

        if self.transform is not None:
            img_1, img_2, cm = self.transform(img_1, img_2, cm)

        return img_1, img_2, cm

    def __len__(self):
        return len(self.samples)


class EurosatDataset(Dataset):
    def __init__(self, root, split, transform=None, shuffle=True):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.embeddings = None

        im_paths = []
        for root, dirs, files in os.walk(self.root):
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

        with open(self.root / f"{split}.txt") as f:
            filenames = f.read().splitlines()

        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for fn in filenames:
            cls_name = fn.split("_")[0]
            self.samples.append(self.root / cls_name / fn)

        if shuffle:
            random.shuffle(self.samples)

    def __getitem__(self, index):

        path = self.samples[index]
        target = self.class_to_idx[path.parts[-2]]

        if self.embeddings is not None:
            return self.embeddings[index], target

        img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.samples)

    @property
    def num_classes(self):
        return 10

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings
