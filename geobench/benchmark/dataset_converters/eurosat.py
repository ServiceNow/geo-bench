"""Eurosat dataset."""
# EuroSat will be automatically downloaded by TorchGeo (https://github.com/microsoft/torchgeo)

import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
from torch import Tensor
from torchgeo.datasets import EuroSAT
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from geobench import io

DATASET_NAME = "eurosat"
SRC_DATASET_DIR = Path(io.src_datasets_dir, DATASET_NAME)  # type: ignore
DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)  # type: ignore


class GeoEuroSAT(EuroSAT):
    """Wrapper around EuroSAT Dataset to extract geo information."""

    all_band_names = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B08A",
        "B09",
        "B10",
        "B11",
        "B12",
    )

    rgb_bands = ("B04", "B03", "B02")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    """Wrapper to extract geo information."""

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: Sequence[str] = BAND_SETS["all"],
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new EuroSAT dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            bands: a sequence of band names to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        super().__init__(root, split, bands, transforms, download, checksum)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image, label, crs, bounds = self._load_image(index)

        image = torch.index_select(image, dim=0, index=self.band_indices).float()
        sample = {"image": image, "label": label, "crs": crs, "bound": bounds}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, index: int) -> Tuple[Tensor, Tensor]:
        """Load a single image and it's class label.

        Args:
            index: index to return

        Returns:
            the image
            the image class label
        """
        path, _ = self.imgs[0]
        img, label = ImageFolder.__getitem__(self, index)
        with rasterio.open(path, "r") as src:
            crs = src.crs
            bounds = src.bounds
        array: "np.typing.NDArray[np.int_]" = np.array(img)
        tensor = torch.from_numpy(array).float()
        # Convert from HxWxC to CxHxW
        tensor = tensor.permute((2, 0, 1))
        label = torch.tensor(label)
        return tensor, label, crs, bounds


def make_sample(images: "np.typing.NDArray[np.float_]", label, sample_name: str) -> io.Sample:
    """Create a sample from images and label.

    Args:
        images: image array to be contained in sample
        label: label to be contained in sample
        sample_name: name of samplefrom torchgeo.datasets import BigEarthNet

    Returns:
        sample
    """
    n_bands, _height, _width = images.shape

    transform = None  # TODO can't find the GPS coordinates from torch geo.
    crs = None

    bands = []
    for band_idx in range(n_bands):
        band_data = images[band_idx, :, :]

        band_info = io.sentinel2_13_bands[band_idx]

        band = io.Band(
            data=band_data.astype(np.int16),
            band_info=band_info,
            spatial_resolution=10,
            transform=transform,
            crs=crs,
            convert_to_int16=False,
        )
        bands.append(band)

    return io.Sample(bands, label=label, sample_name=sample_name)


def convert(max_count=None, dataset_dir=DATASET_DIR) -> None:
    """Convert Eurosat dataset.

    Args:
        max_count: maximum number of samples
        dataset_dir: path to dataset directory
    """
    dataset_dir.mkdir(exist_ok=True, parents=True)
    partition = io.dataset.Partition()

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(64, 64),
        n_time_steps=1,
        bands_info=io.sentinel2_13_bands,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=io.Classification(10, class_names=EuroSAT.classes),
        # eval_loss=io.Accuracy,
        spatial_resolution=10,
    )
    task_specs.save(dataset_dir, overwrite=True)

    sample_id = 0
    for split_name in ["train", "val", "test"]:
        eurosat_dataset = GeoEuroSAT(
            root=SRC_DATASET_DIR, split=split_name, transforms=None, download=True, checksum=True
        )
        for tg_sample in tqdm(eurosat_dataset):
            sample_name = f"id_{sample_id:04d}"

            images = np.array(tg_sample["image"])
            label = tg_sample["label"]
            sample = make_sample(images, int(label), sample_name)
            sample.write(dataset_dir)
            _split_name = {"val": "valid"}.get(split_name, split_name)
            partition.add(_split_name, sample_name)

            sample_id += 1

            if max_count is not None and sample_id >= max_count:
                break

        if max_count is not None and sample_id >= max_count:
            break

    partition.save(dataset_dir, "original", as_default=True)


if __name__ == "__main__":
    convert()
