"""CV4A Kenya Crop Type dataset."""
import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch import Tensor
from torchgeo.datasets import CV4AKenyaCropType
from tqdm import tqdm

from geobench import io

# Deprecated:
# we need to re-write this scripts so that it can properly splits into train / test
# and extract georefence. torchgeo is not an option.

# Notes
# * torchgeo doesn't seem to provide coordinates in general as a general interface
# * should we use the radiant mlhub api_key as a constant?


DATASET_NAME = "CV4AKenyaCropType"
SRC_DATASET_DIR = io.src_datasets_dir  # type: ignore
DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)  # type: ignore

DATES = [
    datetime.datetime.strptime(date, "%Y%m%d").date()
    for date in [
        "20190606",
        "20190701",
        "20190706",
        "20190711",
        "20190721",
        "20190805",
        "20190815",
        "20190825",
        "20190909",
        "20190919",
        "20190924",
        "20191004",
        "20191103",
    ]
]

max_band_value = {
    "06 - Vegetation Red Edge": 1.4976,
    "02 - Blue": 1.7024,
    "03 - Green": 1.6,
    "12 - SWIR": 1.2458,
    "05 - Vegetation Red Edge": 1.5987,
    "04 - Red": 1.5144,
    "01 - Coastal aerosol": 1.7096,
    "07 - Vegetation Red Edge": 1.4803,
    "11 - SWIR": 1.0489,
    "09 - Water vapour": 1.6481,
    "08A - Vegetation Red Edge": 1.4244,
    "08 - NIR": 1.4592,
}

BAND_INFO_LIST: List[Any] = io.sentinel2_13_bands[:]
dropped_band = BAND_INFO_LIST.pop(10)
assert dropped_band.name == "10 - SWIR - Cirrus"
BAND_INFO_LIST.append(io.CloudProbability(alt_names=("CPL", "CLD")))

LABEL_BAND = io.SegmentationClasses("label", spatial_resolution=10, n_classes=8)


class GeoCV4AKenyaCropType(CV4AKenyaCropType):
    """Geo wrapper around crop type dataset."""

    band_names = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
        "CLD",
    )

    rgb_bands = ["B04", "B03", "B02"]

    def __init__(
        self,
        root: str = "data",
        chip_size: int = 256,
        stride: int = 128,
        bands: Tuple[str, ...] = band_names,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize a new CV4A Kenya Crop Type Dataset instance.

        Args:
            root: root directory where dataset can be found
            chip_size: size of chips
            stride: spacing between chips, if less than chip_size, then there
                will be overlap between chips
            bands: the subset of bands to load
            download: if True, download dataset and store it in the root directory
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)
            verbose: if True, print messages when new tiles are loaded

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        super().__init__(root, chip_size, stride, bands, None, download, api_key, checksum, verbose)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data, labels, field ids, and metadata at that index
        """
        tile_index, y, x = self.chips_metadata[index]
        tile_name = self.tile_names[tile_index]

        img, transform, crs, bounds = self._load_all_imagery(self.bands)
        labels, field_ids = self._load_label_tile(tile_name)

        img = img[:, :, y : y + self.chip_size, x : x + self.chip_size]
        labels = labels[y : y + self.chip_size, x : x + self.chip_size]
        field_ids = field_ids[y : y + self.chip_size, x : x + self.chip_size]

        sample = {
            "image": img,
            "mask": labels,
            "field_ids": field_ids,
            "tile_index": torch.tensor(tile_index),
            "x": torch.tensor(x),
            "y": torch.tensor(y),
            "transform": transform,
            "crs": crs,
            "bounds": bounds,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_all_image_tiles(self, tile_name: str, bands: Tuple[str, ...] = band_names) -> Tensor:
        """Load all the imagery (across time) for a single _tile_.

        Optionally allows for subsetting of the bands that are loaded.

        Args:
            tile_name: name of tile to load
            bands: tuple of bands to load

        Returns
            imagery of shape (13, number of bands, 3035, 2016) where 13 is the number of
                points in time, 3035 is the tile height, and 2016 is the tile width

        Raises:
            AssertionError: if ``tile_name`` is invalid
        """
        assert tile_name in self.tile_names

        if self.verbose:
            print(f"Loading all imagery for {tile_name}")

        img = torch.zeros(
            len(self.dates),
            len(bands),
            self.tile_height,
            self.tile_width,
            dtype=torch.float32,
        )

        for date_index, date in enumerate(self.dates):
            single_scene, transform, crs, bounds = self._load_single_scene(date, self.bands)
            img[date_index] = single_scene

        return img, transform, crs, bounds

    def _load_single_image_tile(self, tile_name: str, date: str, bands: Tuple[str, ...]) -> Tensor:
        """Load the imagery for a single tile for a single date.

        Optionally allows for subsetting of the bands that are loaded.

        Args:
            tile_name: name of tile to load
            date: date of tile to load
            bands: bands to load

        Returns:
            array containing a single image tile

        Raises:
            AssertionError: if ``tile_name`` or ``date`` is invalid
        """
        assert tile_name in self.tile_names
        assert date in self.dates

        if self.verbose:
            print(f"Loading imagery for {tile_name} at {date}")

        img = torch.zeros(len(bands), self.tile_height, self.tile_width, dtype=torch.float32)
        for band_index, band_name in enumerate(self.bands):
            filepath = os.path.join(
                self.root,
                "ref_african_crops_kenya_02_source",
                f"{tile_name}_{date}",
                f"{band_name}.tif",
            )

            with rasterio.open(filepath) as src:
                transform = src.transform  # same transform for every bands
                crs = src.crs
                array = src.read().astype(np.float32)
                img[band_index] = torch.from_numpy(array)
                roi = src.bounds

        return img, crs, transform, roi


def make_sample(
    images: "np.typing.NDArray[np.int_]", mask: "np.typing.NDArray[np.int_]", sample_name: str
) -> io.Sample:
    """Create a sample from images and label.

    Args:
        images: image array to be contained in sample
        mask: label to be contained in sample
        sample_name: name of sample

    Returns:
        sample
    """
    n_dates, n_bands, _height, _width = images.shape

    transform = None  # TODO can't find the GPS coordinates from torch geo.
    crs = None

    bands = []
    for date_idx in range(n_dates):
        for band_idx in range(n_bands):
            band_data = images[date_idx, band_idx, :, :]

            band_info = BAND_INFO_LIST[band_idx]

            if band_info.name in max_band_value:
                band_data = band_data / max_band_value[band_info.name] * 10000  # type: ignore

            band = io.Band(
                data=band_data,
                band_info=band_info,
                date=DATES[date_idx],
                spatial_resolution=10,
                transform=transform,
                crs=crs,
                # convert_to_int16=False,
            )
            bands.append(band)

    label = io.Band(data=mask, band_info=LABEL_BAND, spatial_resolution=10, transform=transform, crs=crs)
    return io.Sample(bands, label=label, sample_name=sample_name)


def convert(max_count=None, dataset_dir=DATASET_DIR) -> None:
    """Convert CV4A Kenya crop type dataset.

    Args:
        max_count: maximum number of samples
        dataset_dir: path to dataset directory
    """
    dataset_dir.mkdir(exist_ok=True, parents=True)

    cv4a_dataset = GeoCV4AKenyaCropType(
        root=SRC_DATASET_DIR,
        download=False,
        checksum=True,
        api_key="e46c4efbca1274862accc0f1616762c9c72791e00523980eea3db3c48acd106c",
        chip_size=128,
        verbose=True,
    )

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(128, 128),
        n_time_steps=13,
        bands_info=BAND_INFO_LIST,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=LABEL_BAND,
        eval_loss=io.SegmentationAccuracy,  # TODO probably not the final loss eval loss. To be discussed.
        spatial_resolution=10,
    )
    task_specs.save(dataset_dir)

    # trying to understand train / test split.
    set_map = {}
    trn, tst = cv4a_dataset.get_splits()

    for id in trn:
        set_map[id] = 0
    for id in tst:
        set_map[id] = 1
    set_map[0] = 0

    partition = io.Partition()

    j = 0
    for i, tg_sample in enumerate(tqdm(cv4a_dataset)):

        if np.all(np.array(tg_sample["field_ids"]) == 0):
            continue

        tile_id, x_start, y_start = cv4a_dataset.chips_metadata[i]
        sample_name = f"tile={tile_id}_x={x_start:04d}_y={y_start:04d}"
        # uids = np.unique(tg_sample["field_ids"])

        images = np.array(tg_sample["image"])
        mask = np.array(tg_sample["mask"])

        # set_count = np.bincount([set_map[id] for id in uids])

        sample = make_sample(images, mask, sample_name)
        sample.write(dataset_dir)
        partition.add("train", sample_name)  # by default everything goes in train

        j += 1
        if max_count is not None and j >= max_count:
            break

    partition.save(dataset_dir, "nopartition", as_default=True)


if __name__ == "__main__":
    convert(10)
