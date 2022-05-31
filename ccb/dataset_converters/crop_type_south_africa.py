import datetime
import gc
import json
import os
import re
from pathlib import Path
from typing import List

import numpy as np
from ccb import io
from rasterio.crs import CRS
from rasterio.transform import Affine
from tqdm import tqdm

import crop_type_utils

DATASET_NAME = "southAfricaCropType"
SRC_DATASET_DIR = Path(io.CCB_DIR, "source", DATASET_NAME)
SRC_TRANSFORM = Affine(10.0, 0.0, 331040.0, 0.0, -10.0, -3714560.0)
SRC_CRS = CRS.from_epsg(32634)
LABEL_DIRECTORY_REGEX = r"""_(?P<id>[0-9]{4})$"""
IMG_DIRECTORY_REGEX = r"""
    _(?P<id>[0-9]{4})
    _(?P<year>[0-9]{4})
    _(?P<month>[0-9]{2})
    _(?P<day>[0-9]{2})$"""
PARTITION_CRS = SRC_CRS
PARTITION_DIR = Path(io.CCB_DIR, "converted", DATASET_NAME)

PATCH_SIZE = 256
HEIGHT = 256
WIDTH = 256

crop_labels = [
    "No Data",
    "Lucerne/Medics",
    "Planted pastures (perennial)",
    "Fallow",
    "Wine grapes",
    "Weeds",
    "Small grain grazing",
    "Wheat",
    "Canola",
    "Rooibos",
]

CLASS2IDX = {c: i for i, c in enumerate(crop_labels)}

BANDNAMES = [
    "B01.tif",
    "B02.tif",
    "B03.tif",
    "B04.tif",
    "B05.tif",
    "B06.tif",
    "B07.tif",
    "B08.tif",
    "B8A.tif",
    "B09.tif",
    "B11.tif",
    "B12.tif",
    "CLM.tif",
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

BAND_INFO_LIST = io.sentinel2_13_bands[:]
dropped_band = BAND_INFO_LIST.pop(10)
assert dropped_band.name == "10 - SWIR - Cirrus"
BAND_INFO_LIST.append(
    io.CloudProbability(alt_names=("CPL", "CLM")),
)

LABEL_BAND = io.SegmentationClasses("label", spatial_resolution=10, n_classes=len(crop_labels))


def make_sample(images: np.array, mask: np.array, sample_name: str, dates: List[datetime.date]) -> io.Sample:
    """Make a single sample from the dataset.

    Args:
        images: image tensor of shape T x C x H x W
        mask: corresponding labels
        sample_name: name of sample to save in the partition

    Returns:
        io Sample object
    """
    n_dates, n_bands, _height, _width = images.shape

    bands = []
    for date_idx in range(n_dates):
        for band_idx in range(n_bands):
            band_data = images[date_idx, band_idx, :, :]

            band_info = BAND_INFO_LIST[band_idx]

            if band_info.name in max_band_value:
                band_data = band_data / max_band_value[band_info.name] * 10000

            band = io.Band(
                data=band_data,
                band_info=band_info,
                date=dates[date_idx],
                spatial_resolution=10,
                transform=SRC_TRANSFORM,
                crs=PARTITION_CRS,
                convert_to_int16=False,
            )
            bands.append(band)

    label = io.Band(data=mask, band_info=LABEL_BAND, spatial_resolution=10, transform=SRC_TRANSFORM, crs=PARTITION_CRS)
    return io.Sample(bands, label=label, sample_name=sample_name)


def convert(max_count=None, partition_dir=PARTITION_DIR) -> None:
    """Convert dataset to desired format.

    Args:
        max_count: max number of samples to save in the dataset partition
        dataset_dir: path directory to save partition in
    """
    partition_dir.mkdir(exist_ok=True, parents=True)

    img_dir = os.path.join(SRC_DATASET_DIR, "ref_south_africa_crops_competition_v1_train_source_s2")
    label_dir = os.path.join(SRC_DATASET_DIR, "ref_south_africa_crops_competition_v1_train_labels")

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(256, 256),
        n_time_steps=76,  # this is not the same for all images but the max
        bands_info=BAND_INFO_LIST,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=LABEL_BAND,
        eval_loss=io.SegmentationAccuracy,  # TODO probably not the final loss eval loss. To be discussed.
        spatial_resolution=10,
    )
    task_specs.save(partition_dir, overwrite=True)

    partition = io.Partition()

    # iterate over dataset based on criteria and add to partition
    # go per label and find corresponding imagery

    label_dir_regex = re.compile(LABEL_DIRECTORY_REGEX, re.VERBOSE)
    label_dir_paths = sorted(os.listdir(label_dir))[2:]  # skip aux files
    img_dir_regex = re.compile(IMG_DIRECTORY_REGEX, re.VERBOSE)
    img_dir_paths = sorted(os.listdir(img_dir))[1:]  # skip aux files

    j = 0
    for dirpath in tqdm(label_dir_paths):
        path = os.path.join(label_dir, dirpath)
        # to extract date information and id to match input images
        match = re.search(label_dir_regex, path)
        id = "_source_s2_" + match.group("id")

        matched_image_dirs = [os.path.join(img_dir, dir) for dir in img_dir_paths if id in dir]

        label = crop_type_utils.load_tif_mask(
            filepath=path,
            dest_crs=PARTITION_CRS,
        )

        img_list = [
            crop_type_utils.load_image_bands(filepath, BANDNAMES, PARTITION_CRS) for filepath in matched_image_dirs
        ]

        dates = crop_type_utils.collect_dates(matched_image_dirs, img_dir_regex)
        dates = [datetime.datetime.strptime(date, "%Y%m%d").date() for date in dates]

        # stack to array of T x C x H x W
        imgs = np.stack(img_list, axis=0)
        del img_list
        mask = np.stack(label, axis=0)

        # dataset is said to have all images 256,256 but found 270,270
        if imgs.shape[-2:] != (PATCH_SIZE, PATCH_SIZE) or mask.shape[-2:] != (PATCH_SIZE, PATCH_SIZE):
            imgs = imgs[:, :, 0:PATCH_SIZE, 0:PATCH_SIZE]
            mask = mask[:, 0:PATCH_SIZE, 0:PATCH_SIZE]

        sample_name = dirpath
        sample = make_sample(imgs, mask, sample_name, dates)
        sample.write(partition_dir)

        partition.add("train", sample_name)  # by default everything goes in train

        j += 1
        if max_count is not None and j >= max_count:
            break


if __name__ == "__main__":
    convert()
