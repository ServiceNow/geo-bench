"""CV4A Kenya Crop Type dataset."""
import datetime
from pathlib import Path
from typing import Any, List

import numpy as np
from torchgeo.datasets import cv4a_kenya_crop_type
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

    cv4a_dataset = cv4a_kenya_crop_type.CV4AKenyaCropType(
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
