"""Benin Smallholder Cashew dataset."""
# Smallholder Cashew GeobenchDataset will be downloaded by torchgeo
#
# 1) This requires Radiant MLHub package and API token
#   pip install radiant_mlhub
# 2) Sign up for a MLHub account here: https://mlhub.earth/
# 3) Type this in your terminal:
#   mlhub configure
# and enter your API key.
#
# More info on the dataset: https://mlhub.earth/10.34911/rdnt.hfv20i

import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from torch import Tensor
from torchgeo.datasets import BeninSmallHolderCashews
from tqdm import tqdm

from geobench import io

# Classification labels
LABELS = (
    "no data",
    "well-managed plantation",
    "poorly-managed plantation",
    "non-plantation",
    "residential",
    "background",
    "uncertain",
)
DATES: List[datetime.date] = [
    datetime.datetime.strptime(date, "%Y-%m-%d").date()
    for date in [
        "2019-11-05",
        "2019-11-10",
        "2019-11-15",
        "2019-11-20",
        "2019-11-30",
        "2019-12-05",
        "2019-12-10",
        "2019-12-15",
        "2019-12-20",
        "2019-12-25",
        "2019-12-30",
        "2020-01-04",
        "2020-01-09",
        "2020-01-14",
        "2020-01-19",
        "2020-01-24",
        "2020-01-29",
        "2020-02-08",
        "2020-02-13",
        "2020-02-18",
        "2020-02-23",
        "2020-02-28",
        "2020-03-04",
        "2020-03-09",
        "2020-03-14",
        "2020-03-19",
        "2020-03-24",
        "2020-03-29",
        "2020-04-03",
        "2020-04-08",
        "2020-04-13",
        "2020-04-18",
        "2020-04-23",
        "2020-04-28",
        "2020-05-03",
        "2020-05-08",
        "2020-05-13",
        "2020-05-18",
        "2020-05-23",
        "2020-05-28",
        "2020-06-02",
        "2020-06-07",
        "2020-06-12",
        "2020-06-17",
        "2020-06-22",
        "2020-06-27",
        "2020-07-02",
        "2020-07-07",
        "2020-07-12",
        "2020-07-17",
        "2020-07-22",
        "2020-07-27",
        "2020-08-01",
        "2020-08-06",
        "2020-08-11",
        "2020-08-16",
        "2020-08-21",
        "2020-08-26",
        "2020-08-31",
        "2020-09-05",
        "2020-09-10",
        "2020-09-15",
        "2020-09-20",
        "2020-09-25",
        "2020-09-30",
        "2020-10-10",
        "2020-10-15",
        "2020-10-20",
        "2020-10-25",
        "2020-10-30",
    ]
]

noclouds_25 = [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    13,
    15,
    16,
    17,
    19,
    20,
    22,
    23,
    27,
    28,
    30,
    33,
    37,
    38,
    69,
]  # 25 dates with the least clouds

BAND_INFO_LIST: List[Any] = io.sentinel2_13_bands[:]
dropped_band = BAND_INFO_LIST.pop(10)
assert dropped_band.name == "10 - SWIR - Cirrus"
BAND_INFO_LIST.append(io.CloudProbability(alt_names=("CPL", "CLD"), spatial_resolution=10))


SPATIAL_RESOLUTION = 0.5  # meters, to be confirmed
N_TIMESTEPS = 70
LABEL_BAND = io.SegmentationClasses("label", spatial_resolution=SPATIAL_RESOLUTION, n_classes=len(LABELS))
GROUP_BY_TIMESTEP = False
NOCLOUDS = True

# Paths
DATASET_NAME = "smallholder_cashew"
SRC_DATASET_DIR = Path(io.src_datasets_dir, DATASET_NAME)  # type: ignore
DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)  # type: ignore


class GeoBeninCashew(BeninSmallHolderCashews):
    """Geo Wrapper to extract geo information from dataste."""

    all_bands = (
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
    rgb_bands = ("B04", "B03", "B02")

    def __init__(
        self,
        root: str = "data",
        chip_size: int = 256,
        stride: int = 128,
        bands: Tuple[str, ...] = all_bands,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize a new Benin Smallholder Cashew Plantations Dataset instance.
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
            a dict containing image, mask, transform, crs, and metadata at index.
        """
        y, x = self.chips_metadata[index]

        img, transform, crs, bounds = self._load_all_imagery(self.bands)
        labels = self._load_mask(transform)

        img = img[:, :, y : y + self.chip_size, x : x + self.chip_size]
        labels = labels[y : y + self.chip_size, x : x + self.chip_size]

        sample = {
            "image": img,
            "mask": labels,
            "x": torch.tensor(x),
            "y": torch.tensor(y),
            "transform": transform,
            "crs": crs,
            "bounds": bounds,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_single_scene(self, date: str, bands: Tuple[str, ...]) -> Tuple[Tensor, rasterio.Affine, CRS]:
        """Load the imagery for a single date.

        Optionally allows for subsetting of the bands that are loaded.

        Args:
            date: date of the imagery to load
            bands: bands to load

        Returns:
            Tensor containing a single image tile, rasterio affine transform,
            mapping pixel coordinates to geo coordinates, and coordinate
            reference system of transform.

        Raises:
            AssertionError: if  ``date`` is invalid
        """
        assert date in self.dates

        if self.verbose:
            print(f"Loading imagery at {date}")

        img = torch.zeros(len(bands), self.tile_height, self.tile_width, dtype=torch.float32)
        for band_index, band_name in enumerate(self.bands):
            filepath = os.path.join(
                self.root,
                "ts_cashew_benin_source",
                f"ts_cashew_benin_source_00_{date}",
                f"{band_name}.tif",
            )
            with rasterio.open(filepath) as src:
                transform = src.transform  # same transform for every bands
                crs = src.crs
                array = src.read().astype(np.float32)
                img[band_index] = torch.from_numpy(array)
                roi = src.bounds

        return img, transform, crs, roi

    def _load_all_imagery(self, bands: Tuple[str, ...] = all_bands) -> Tuple[Tensor, rasterio.Affine, CRS]:
        """Load all the imagery (across time) for the dataset.

        Optionally allows for subsetting of the bands that are loaded.

        Args:
            bands: tuple of bands to load

        Returns:
            imagery of shape (70, number of bands, 1186, 1122) where 70 is the number
            of points in time, 1186 is the tile height, and 1122 is the tile width
            rasterio affine transform, mapping pixel coordinates to geo coordinates
            coordinate reference system of transform
        """
        if self.verbose:
            print("Loading all imagery")

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


def get_sample_name(total_samples) -> str:
    """Return the name of the samples.

    Args:
        total_sample:

    Returns:
        sample name
    """
    return f"sample_{total_samples}"


def convert(max_count=None, dataset_dir=DATASET_DIR) -> None:
    """Convert Benin Smallholder Cashews dataset.

    Args:
        max_count: maximum number of samples
        dataset_dir: path to dataset directory
    """
    dataset_dir.mkdir(exist_ok=True, parents=True)

    print("Loading dataset from torchgeo")
    cashew = GeoBeninCashew(root=SRC_DATASET_DIR, download=True, checksum=True)

    if GROUP_BY_TIMESTEP:
        n_time_steps = len(noclouds_25) if NOCLOUDS else N_TIMESTEPS
    else:
        n_time_steps = 1

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(256, 256),
        n_time_steps=n_time_steps,
        bands_info=BAND_INFO_LIST,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=LABEL_BAND,
        # eval_loss=io.SegmentationAccuracy,  # TODO probably not the final loss eval loss. To be discussed.
        # either 50cm or 40cm, Airbus Pleiades 50cm, https://radiantearth.blob.core.windows.net/mlhub/technoserve-cashew-benin/Documentation.pdf
        spatial_resolution=SPATIAL_RESOLUTION,
    )

    partition = io.Partition()

    task_specs.save(dataset_dir, overwrite=True)

    print("Saving timesteps as separate bands")
    total_samples = 0
    for tg_sample in tqdm(cashew):

        images = tg_sample["image"].numpy()
        mask = tg_sample["mask"].numpy()
        n_timesteps, n_bands, _height, _width = images.shape

        label = io.Band(
            data=mask, band_info=LABEL_BAND, spatial_resolution=SPATIAL_RESOLUTION, transform=None, crs=None
        )
        split = np.random.choice(("train", "valid", "test"), p=(0.8, 0.1, 0.1))
        grouped_bands = []
        for date_idx in range(n_timesteps):
            current_bands = []
            if NOCLOUDS and date_idx not in noclouds_25:
                continue

            for band_idx in range(n_bands):
                band_data = images[date_idx, band_idx, :, :]

                band_info = BAND_INFO_LIST[band_idx]

                band = io.Band(
                    data=band_data,
                    band_info=band_info,
                    date=DATES[date_idx],
                    spatial_resolution=SPATIAL_RESOLUTION,
                    transform=tg_sample["transform"],  # TODO can't find the GPS coordinates from torch geo.
                    crs=tg_sample["crs"],
                    convert_to_int16=False,
                )
                current_bands.append(band)
                grouped_bands.append(band)

            if not GROUP_BY_TIMESTEP:
                sample = io.Sample(current_bands, label=label, sample_name=get_sample_name(total_samples))
                sample.write(dataset_dir)
                partition.add(split, get_sample_name(total_samples))
                total_samples += 1

            if max_count is not None and total_samples >= max_count:
                break

        if GROUP_BY_TIMESTEP:
            sample = io.Sample(grouped_bands, label=label, sample_name=get_sample_name(total_samples))
            sample.write(dataset_dir)
            partition.add(split, get_sample_name(total_samples))
            total_samples += 1

        if max_count is not None and total_samples >= max_count:
            break

    # partition.resplit_iid(split_names=("train", "valid", "test"), ratios=(0.8, 0.1, 0.1))
    partition.save(dataset_dir, "default")
    print(f"Done. GROUP_BY_TIMESTEP={GROUP_BY_TIMESTEP}, total_samples={total_samples}")


if __name__ == "__main__":
    convert()
