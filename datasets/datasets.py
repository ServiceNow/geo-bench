from pathlib import Path
from torch.utils.data import Dataset
from itertools import product
import random
import rasterio
import numpy as np
from PIL import Image
from scipy.io import loadmat
import os

import pandas as pd
import torch

import json
from tqdm import tqdm

from torchvision.datasets.utils import download_and_extract_archive, download_url
import skimage.exposure

# class DeepForestDataset(Dataset):
#     def __init__(self, data_dir, split, transform=None, shuffle=True):
#         self.split = split
#         self.transform = transform
#         self.embeddings = None

#         m = deepforest.main.deepforest()
#         m.use_release()

#         if split == "val":
#             split = "validation"

#         self.dataset = dataset.TreeDataset(csv_file=m.config[split]["csv_file"],
#             root_dir=m.config[split]["root_dir"],
#             transforms=m.transforms(augment=True),
#             label_dict=m.label_dict)

#     def __getitem__(self, index):


#     def __len__(self):
#         return self.data.shape[0]

#     @property
#     def num_classes(self):
#         return 10

#     def set_embeddings(self, embeddings):
#         self.embeddings = embeddings


def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class BigEarthNetDataset(Dataset):

    ALL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    RGB_BANDS = ["B04", "B03", "B02"]

    BAND_STATS = {
        "mean": {
            "B01": 340.76769064,
            "B02": 429.9430203,
            "B03": 614.21682446,
            "B04": 590.23569706,
            "B05": 950.68368468,
            "B06": 1792.46290469,
            "B07": 2075.46795189,
            "B08": 2218.94553375,
            "B8A": 2266.46036911,
            "B09": 2246.0605464,
            "B11": 1594.42694882,
            "B12": 1009.32729131,
        },
        "std": {
            "B01": 554.81258967,
            "B02": 572.41639287,
            "B03": 582.87945694,
            "B04": 675.88746967,
            "B05": 729.89827633,
            "B06": 1096.01480586,
            "B07": 1273.45393088,
            "B08": 1365.45589904,
            "B8A": 1356.13789355,
            "B09": 1302.3292881,
            "B11": 1079.19066363,
            "B12": 818.86747235,
        },
    }

    LABELS = [
        "Agro-forestry areas",
        "Airports",
        "Annual crops associated with permanent crops",
        "Bare rock",
        "Beaches, dunes, sands",
        "Broad-leaved forest",
        "Burnt areas",
        "Coastal lagoons",
        "Complex cultivation patterns",
        "Coniferous forest",
        "Construction sites",
        "Continuous urban fabric",
        "Discontinuous urban fabric",
        "Dump sites",
        "Estuaries",
        "Fruit trees and berry plantations",
        "Green urban areas",
        "Industrial or commercial units",
        "Inland marshes",
        "Intertidal flats",
        "Land principally occupied by agriculture, with significant areas of " "natural vegetation",
        "Mineral extraction sites",
        "Mixed forest",
        "Moors and heathland",
        "Natural grassland",
        "Non-irrigated arable land",
        "Olive groves",
        "Pastures",
        "Peatbogs",
        "Permanently irrigated land",
        "Port areas",
        "Rice fields",
        "Road and rail networks and associated land",
        "Salines",
        "Salt marshes",
        "Sclerophyllous vegetation",
        "Sea and ocean",
        "Sparsely vegetated areas",
        "Sport and leisure facilities",
        "Transitional woodland/shrub",
        "Vineyards",
        "Water bodies",
        "Water courses",
    ]

    NEW_LABELS = [
        "Urban fabric",
        "Industrial or commercial units",
        "Arable land",
        "Permanent crops",
        "Pastures",
        "Complex cultivation patterns",
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
        "Agro-forestry areas",
        "Broad-leaved forest",
        "Coniferous forest",
        "Mixed forest",
        "Natural grassland and sparsely vegetated areas",
        "Moors, heathland and sclerophyllous vegetation",
        "Transitional woodland/shrub",
        "Beaches, dunes, sands",
        "Inland wetlands",
        "Coastal wetlands",
        "Inland waters",
        "Marine waters",
    ]

    GROUP_LABELS = {
        "Continuous urban fabric": "Urban fabric",
        "Discontinuous urban fabric": "Urban fabric",
        "Non-irrigated arable land": "Arable land",
        "Permanently irrigated land": "Arable land",
        "Rice fields": "Arable land",
        "Vineyards": "Permanent crops",
        "Fruit trees and berry plantations": "Permanent crops",
        "Olive groves": "Permanent crops",
        "Annual crops associated with permanent crops": "Permanent crops",
        "Natural grassland": "Natural grassland and sparsely vegetated areas",
        "Sparsely vegetated areas": "Natural grassland and sparsely vegetated areas",
        "Moors and heathland": "Moors, heathland and sclerophyllous vegetation",
        "Sclerophyllous vegetation": "Moors, heathland and sclerophyllous vegetation",
        "Inland marshes": "Inland wetlands",
        "Peatbogs": "Inland wetlands",
        "Salt marshes": "Coastal wetlands",
        "Salines": "Coastal wetlands",
        "Water bodies": "Inland waters",
        "Water courses": "Inland waters",
        "Coastal lagoons": "Marine waters",
        "Estuaries": "Marine waters",
        "Sea and ocean": "Marine waters",
    }
    # url = 'http://bigearth.net/downloads/BigEarthNet-v1.0.tar.gz'
    url = "http://bigearth.net/downloads/BigEarthNet-S1-v1.0.tar.gz"
    subdir = "BigEarthNet-v1.0"
    list_file = {
        "train": "https://storage.googleapis.com/remote_sensing_representations/bigearthnet-train.txt",
        "val": "https://storage.googleapis.com/remote_sensing_representations/bigearthnet-val.txt",
        "test": "https://storage.googleapis.com/remote_sensing_representations/bigearthnet-test.txt",
    }
    bad_patches = [
        "http://bigearth.net/static/documents/patches_with_seasonal_snow.csv",
        "http://bigearth.net/static/documents/patches_with_cloud_and_shadow.csv",
    ]

    def __init__(
        self, root, split, bands=None, transform=None, target_transform=None, download=False, use_new_labels=True
    ):
        self.root = Path(root)
        self.split = split
        self.bands = bands if bands is not None else self.RGB_BANDS
        self.transform = transform
        self.target_transform = target_transform
        self.use_new_labels = use_new_labels

        if not os.path.exists(self.root):
            os.makedirs(self.root)
            print("a")
            download_and_extract_archive(self.url, self.root)
            download_url(self.list_file[self.split], self.root, f"{self.split}.txt")
            for url in self.bad_patches:
                download_url(url, self.root)

        bad_patches = set()
        for url in self.bad_patches:
            filename = Path(url).name
            with open(self.root / filename) as f:
                bad_patches.update(f.read().splitlines())

        self.samples = []
        with open(self.root / f"{self.split}.txt") as f:
            for patch_id in f.read().splitlines():
                if patch_id not in bad_patches:
                    self.samples.append(self.root / self.subdir / patch_id)

    def __getitem__(self, index):
        path = self.samples[index]
        patch_id = path.name

        channels = []
        for b in self.bands:
            ch = rasterio.open(path / f"{patch_id}_{b}.tif").read(1)
            ch = normalize(ch, mean=self.BAND_STATS["mean"][b], std=self.BAND_STATS["std"][b])
            channels.append(ch)
        img = np.dstack(channels)
        img = Image.fromarray(img)

        with open(path / f"{patch_id}_labels_metadata.json", "r") as f:
            labels = json.load(f)["labels"]
        if self.use_new_labels:
            target = self.get_multihot_new(labels)
        else:
            target = self.get_multihot_old(labels)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.samples)

    def get_multihot_old(self, labels):
        target = np.zeros((len(self.LABELS),), dtype=np.float32)
        for label in labels:
            target[self.LABELS.index(label)] = 1
        return target

    def get_multihot_new(self, labels):
        target = np.zeros((len(self.NEW_LABELS),), dtype=np.float32)
        for label in labels:
            if label in self.GROUP_LABELS:
                target[self.NEW_LABELS.index(self.GROUP_LABELS[label])] = 1
            elif label not in set(self.NEW_LABELS):
                continue
            else:
                target[self.NEW_LABELS.index(label)] = 1
        return target


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
            idx = np.random.shuffle(np.arange(self.targets.shape[0]))
            self.data = self.data[idx][0]
            self.targets = self.targets[idx][0]
        print(np.unique(self.targets))

    def __getitem__(self, index):

        target = self.targets[index].astype(np.long)

        if self.embeddings is not None:
            return self.embeddings[index], target

        img = self.data[index].astype(np.float32)

        if self.transform is not None and len(img.shape) in (2, 3):
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.get_embeddings:
            return self.embeddings.shape[0]
        return self.data.shape[0]

    @property
    def num_classes(self):
        return 6

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings


class ChangeDetectionDataset(Dataset):

    ALL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    RGB_BANDS = ["B04", "B03", "B02"]

    QUANTILES = {
        "min_q": {"B02": 885.0, "B03": 667.0, "B04": 426.0},
        "max_q": {"B02": 2620.0, "B03": 2969.0, "B04": 3698.0},
    }

    def __init__(self, root, split="all", bands=None, transform=None, patch_size=96, shuffle=True):
        self.root = Path(root)
        self.split = split
        self.bands = bands if bands is not None else self.RGB_BANDS
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

        img_1 = self.read_image(path / "imgs_1", self.bands)
        img_2 = self.read_image(path / "imgs_2", self.bands)
        cm = Image.open(path / "cm" / "cm.png").convert("L")

        img_1 = img_1.crop(limits)
        img_2 = img_2.crop(limits)
        cm = cm.crop(limits)

        if self.transform is not None:
            img_1, img_2, cm = self.transform(img_1, img_2, cm)
        return img_1, img_2, cm

    def __len__(self):
        return len(self.samples)

    @property
    def num_classes(self):
        return 1

    def read_image(self, path, bands, normalize=True):
        patch_id = next(path.iterdir()).name[:-8]
        channels = []
        for b in bands:
            ch = rasterio.open(path / f"{patch_id}_{b}.tif").read(1)
            if normalize:
                min_v = self.QUANTILES["min_q"][b]
                max_v = self.QUANTILES["max_q"][b]
                ch = (ch - min_v) / (max_v - min_v)
                ch = np.clip(ch, 0, 1)
                ch = (ch * 255).astype(np.uint8)
            channels.append(ch)
        img = np.dstack(channels)
        img = Image.fromarray(img)
        return img


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

        with open(os.path.join(root, "train.txt"), "w") as f:
            for p in train_paths:
                f.write(p + "\n")

        with open(os.path.join(root, "val.txt"), "w") as f:
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


class ForestNetDataset(Dataset):
    def __init__(self, data_dir, split, transform=None, shuffle=True):
        self.split = split
        self.transform = transform

        # self.class_to_idx = {
        #     "Timber plantation": 0,
        #     "Other": 1,
        #     "Grassland shrubland": 2,
        #     "Small-scale agriculture": 3,
        #     "Other large-scale plantations": 4,
        #     "Small-scale mixed plantation": 5,
        #     "Oil palm plantation": 6,
        #     "Logging": 7,
        #     "Mining": 8,
        #     "Small-scale oil palm plantation": 9,
        #     "Secondary forest": 10,
        #     "Fish pond": 11,
        # }

        self.class_to_idx = {"Plantation": 0, "Other": 1, "Grassland shrubland": 2, "Smallholder agriculture": 3}

        self.metadata_df = pd.read_csv(os.path.join(data_dir, "{}.csv".format(split)))
        self.samples = (
            self.metadata_df["example_path"]
            .apply(lambda x: os.path.join(data_dir, x, "images/visible/composite.png"))
            .to_list()
        )
        # print(self.metadata_df["merged_label"].unique(), "===============")

        self.targets = self.metadata_df["merged_label"].apply(lambda x: self.class_to_idx[x]).to_numpy()
        self.embeddings = None

        if shuffle:
            tmp = list(zip(self.samples, self.targets))
            random.shuffle(tmp)
            self.samples, self.targets = zip(*tmp)

    def __getitem__(self, index):

        target = self.targets[index]

        if self.embeddings is not None:
            return self.embeddings[index], target

        path = self.samples[index]
        img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.samples)

    @property
    def num_classes(self):
        return 4

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings


class GeoClefDataset(Dataset):
    """
    TODO: add per-image classification labels
    TODO: add augmentations to seg masks

    =============================
    Class Numbers for USA dataset
    =============================
    label new_code old_code land_cover_class
    0 18 11 Open Water
    1 19 12 Perennial Ice/Snow
    2 20 21 Developed, Open Space
    3 21 22 Developed, Low Intensity
    4 22 23 Developed, Medium Intensity
    5 23 24 Developed, High Intensity
    6 24 31 Barren Land (Rock/Sand/Clay)
    7 25 41 Deciduous Forest
    8 26 42 Evergreen Forest
    9 27 43 Mixed Forest
    10 28 52 Shrub/Scrub
    11 29 71 Grassland/Herbaceous
    12 30 81 Pasture/Hay
    13 31 82 Cultivated Crops
    14 32 90 Woody Wetlands
    15 33 95 Emergent Herbaceous Wetlands
    16       Other
    """

    def __init__(self, data_dir, split, transform=None, shuffle=True):
        self.split = split
        self.transform = transform
        self.embeddings = None
        self.root = data_dir
        self.samples = []

        if not os.path.exists(os.path.join(data_dir, "train.txt")):
            with open(os.path.join(data_dir, "annotations_train.json")) as f:
                d = json.load(f)

                # add training samples which were downloaded in patches_us_01
                for sample_d in tqdm(d["images"]):
                    sample_p = os.path.join(data_dir, "patches_us_01", sample_d["file_name"])

                    if os.path.exists(sample_p):
                        self.samples.append(sample_p)

            random.shuffle(self.samples)
            split_idx = int(0.8 * len(self.samples))

            with open(os.path.join(data_dir, "train.txt"), "w") as f:
                for p in self.samples[:split_idx]:
                    f.write(p + "\n")

            with open(os.path.join(data_dir, "val.txt"), "w") as f:
                for p in self.samples[split_idx:]:
                    f.write(p + "\n")

        if split == "train":
            self.samples = open(os.path.join(data_dir, "train.txt")).read().split()
        elif split == "val":
            self.samples = open(os.path.join(data_dir, "val.txt")).read().split()

        # if split == "train":
        #     with open(os.path.join(data_dir, "annotations_train.json")) as f:
        #         d = json.load(f)

        #     for sample_d in tqdm(d["images"][:100000]):
        #         sample_p = os.path.join(data_dir, "patches_us_01", sample_d["file_name"])
        #         if os.path.exists(sample_p):
        #             self.samples.append(sample_p)

        # elif split == "val":
        #     with open(os.path.join(data_dir, "annotations_val.json")) as f:
        #         d = json.load(f)

        #     for sample_d in tqdm(d["images"]):
        #         sample_p = os.path.join(data_dir, "patches_us_01", sample_d["file_name"])
        #         if os.path.exists(sample_p):
        #             self.samples.append(sample_p)

        if shuffle:
            random.shuffle(self.samples)

    def __getitem__(self, index):

        data = np.load(self.samples[index])

        img1 = data[:, :, :3].astype(np.float32)
        img2 = np.zeros_like(img1)
        target = data[:, :, 4] - 18

        # target_oh = np.zeros(target.shape + (self.num_classes,))

        # for i, unique_value in enumerate(np.unique(target)):
        #     target_oh[:, :, i][target == unique_value] = 1

        if self.transform is not None:
            img1, img2 = self.transform(img1, img2)

        target[target > 15] = 16
        target[target < 0] = 16

        return img1, img2, target.astype(np.long)

    def __len__(self):

        return len(self.samples)

    @property
    def num_classes(self):
        return 17

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings


class Sen12FloodDataset(Dataset):
    def __init__(self, root, split, transform=None, shuffle=True):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.embeddings = None

        im_paths = []
        for fn in os.listdir(self.root):
            if fn == "train.txt" or fn == "val.txt" or fn == "collection.json":
                continue
            if "B01.tif" not in os.listdir(os.path.join(self.root, fn)):
                continue
            im_paths.append(fn)

        random.shuffle(im_paths)

        train_paths = im_paths[: int(len(im_paths) * 0.6)]
        val_paths = im_paths[int(len(im_paths) * 0.6) :]

        with open(os.path.join(root, "train.txt"), "w") as f:
            for p in train_paths:
                f.write(p + "\n")

        with open(os.path.join(root, "val.txt"), "w") as f:
            for p in val_paths:
                f.write(p + "\n")

        with open(self.root / f"{split}.txt") as f:
            filenames = f.read().splitlines()

        self.samples = []
        self.targets = []

        for fn in filenames:
            self.samples.append(os.path.join(self.root, fn))
            with open(os.path.join(self.root, fn, "labels.geojson").replace("source", "labels")) as f:
                label_d = json.load(f)
                label = label_d["properties"]["FLOODING"]
                if label:
                    self.targets.append(1)
                else:
                    self.targets.append(0)

    def __getitem__(self, index):

        path = self.samples[index]
        target = self.targets[index]
        if self.embeddings is not None:
            return self.embeddings[index], target

        r = rasterio.open(path + "/B04.tif").read()
        g = rasterio.open(path + "/B03.tif").read()
        b = rasterio.open(path + "/B02.tif").read()

        img = np.concatenate((r, g, b), 0)
        img = skimage.exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))

        print(img.shape)

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.samples)

    @property
    def num_classes(self):
        return 2

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings
