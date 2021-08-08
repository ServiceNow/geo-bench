from pathlib import Path
from itertools import product
import random
from torch.utils.data import Dataset
import rasterio
import numpy as np
from PIL import Image

ALL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
RGB_BANDS = ["B04", "B03", "B02"]

QUANTILES = {
    "min_q": {"B02": 885.0, "B03": 667.0, "B04": 426.0},
    "max_q": {"B02": 2620.0, "B03": 2969.0, "B04": 3698.0},
}


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
