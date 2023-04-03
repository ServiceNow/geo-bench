"""Crop Type South Africa dataset."""
import os
import re
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from tqdm import tqdm

from geobench import io

DATASET_NAME = "southAfricaCropType"
SRC_DATASET_DIR = Path(io.CCB_DIR, "source", DATASET_NAME)  # type: ignore
IMG_DIR = "ref_south_africa_crops_competition_v1_train_source_s2"
LABEL_DIR = "ref_south_africa_crops_competition_v1_train_labels"
SRC_TRANSFORM = Affine(10.0, 0.0, 331040.0, 0.0, -10.0, -3714560.0)
SRC_CRS = CRS.from_epsg(32634)
LABEL_DIRECTORY_REGEX = r"""_(?P<id>[0-9]{4})$"""
IMG_DIRECTORY_REGEX = r"""
    _(?P<id>[0-9]{4})
    _(?P<year>[0-9]{4})
    _(?P<month>[0-9]{2})
    _(?P<day>[0-9]{2})$"""
PARTITION_CRS = SRC_CRS
DATASET_DIR = Path(io.CCB_DIR, "converted", DATASET_NAME)

CLOUD_P = (0.0, 0.1)

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

BAND_INFO_LIST: List[Any] = io.sentinel2_13_bands[:]
dropped_band = BAND_INFO_LIST.pop(10)
assert dropped_band.name == "10 - SWIR - Cirrus"
BAND_INFO_LIST.extend([io.CloudProbability(alt_names=("CPL", "CLM"))])


LABEL_BAND = io.SegmentationClasses("label", spatial_resolution=10, n_classes=len(crop_labels))


def compute_area_with_labels(mask: "np.typing.NDArray[np.int_]") -> float:
    """Compute percentage of mask that contain labels.

    Args:
        mask: mask to compute labels on

    Return:
        percentage
    """
    num_px_with_label = np.count_nonzero(mask)
    num_px_total = mask.shape[0] * mask.shape[1]
    return num_px_with_label / num_px_total


def load_images(
    filepaths: List[str], band_names: List[str], dest_crs: CRS, cloud_p: Sequence[float] = (0.0, 0.1), num_imgs: int = 5
) -> "np.typing.NDArray[np.int_]":
    """Load the desired input image.

    Args:
        filepaths: paths to files to load
        bandnames: band names with file extension as they can be found in data directory
        dest_crs: CRS of data in partition that is being created
        cloud_p: cloud probability interval to accept samples
        num_imgs: num images to return per label

    Returns:
        image of shape num_imgs x C x H x W
    """
    accepted_imgs = []
    accepted_pct = []
    for path in filepaths:
        img = load_image_bands(filepath=path, bandnames=band_names, dest_crs=dest_crs)
        pct_area_cloud = compute_area_with_labels(img[-1, :, :])
        if pct_area_cloud >= cloud_p[0] and pct_area_cloud <= cloud_p[1]:
            accepted_imgs.append(img)
            accepted_pct.append(pct_area_cloud)
        else:
            continue

    sorted_pct = np.argsort(np.array(accepted_pct))[:num_imgs]
    imgs = np.stack(accepted_imgs, axis=0)
    imgs = imgs[sorted_pct, :, :, :]
    return imgs


def load_image_bands(filepath: str, bandnames: List[str], dest_crs: CRS) -> "np.typing.NDArray[np.int_]":
    """Load seperate band images.

    Args:
        filepath: filepath that contains bands
        bandnames: band names with file extension as they can be found in data directory
        dest_crs: CRS of data in partition that is being created

    Returns:
        images at this filepath of shape C x H x W
    """
    # load imagery
    band_list = []
    for bandname in bandnames:
        band_filename = os.path.join(filepath, bandname)
        src = load_warp_file(band_filename, dest_crs)
        band = src.read()
        band_list.append(band)

    # stack along band channel dimension
    data = np.concatenate(band_list, axis=0, dtype=np.int16)
    return data


def load_warp_file(filepath: str, dest_crs: CRS) -> DatasetReader:
    """Load and warp a file to the correct CRS and resolution.

    Args:
        filepath: file to load and warp
        dest_crs: CRS of data in partition that is being created

    Returns:
        file handle of warped VRT
    """
    src = rasterio.open(filepath)

    # Only warp if necessary
    if src.crs != dest_crs:
        vrt = WarpedVRT(src, crs=dest_crs)
        src.close()
        return vrt
    else:
        return src


def load_tif_mask(filepath: str, dest_crs: CRS) -> "np.typing.NDArray[np.int_]":
    """Load the mask.

    Args:
        filepaths: one or more files to load and merge
        query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

    Returns:
        labels at this query index
    """
    # load label
    label_filename = os.path.join(filepath, "labels.tif")

    src = load_warp_file(label_filename, dest_crs=dest_crs)
    label = src.read().astype(np.int16).squeeze(0)

    return label


def make_sample(
    images: "np.typing.NDArray[np.int_]", mask: "np.typing.NDArray[np.int_]", sample_name: str
) -> io.Sample:
    """Make a single sample from the dataset.

    Args:
        images: image tensor of shape T x C x H x W
        mask: corresponding labels
        sample_name: name of sample to save in the partition

    Returns:
        io Sample object
    """
    n_bands, _height, _width = images.shape
    assert n_bands == 13

    bands = []

    for band_idx in range(n_bands):
        band_data = images[band_idx, :, :]

        band_info = BAND_INFO_LIST[band_idx]

        band = io.Band(
            data=band_data,
            band_info=band_info,
            spatial_resolution=10,
            transform=SRC_TRANSFORM,
            crs=PARTITION_CRS,
            convert_to_int16=False,
        )
        bands.append(band)

    label = io.Band(data=mask, band_info=LABEL_BAND, spatial_resolution=10, transform=SRC_TRANSFORM, crs=PARTITION_CRS)
    return io.Sample(bands, label=label, sample_name=sample_name)


def convert(max_count=None, dataset_dir=DATASET_DIR) -> None:
    """Convert dataset to desired format.

    Args:
        max_count: max number of samples to save in the dataset partition
        dataset_dir: path directory to save partition in
    """
    dataset_dir.mkdir(exist_ok=True, parents=True)

    img_dir = os.path.join(SRC_DATASET_DIR, IMG_DIR)
    label_dir = os.path.join(SRC_DATASET_DIR, LABEL_DIR)

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(256, 256),
        # n_time_steps=76,  # this is not the same for all images but the max
        n_time_steps=1,  # we yield different instance for each time steps.
        bands_info=BAND_INFO_LIST,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=LABEL_BAND,
        # eval_loss=io.SegmentationAccuracy,  # TODO probably not the final loss eval loss. To be discussed.
        spatial_resolution=10,
    )
    task_specs.save(dataset_dir, overwrite=True)

    partition = io.Partition()

    label_dir_regex = re.compile(LABEL_DIRECTORY_REGEX, re.VERBOSE)
    label_dir_paths = sorted(os.listdir(label_dir))[2:]  # skip aux files
    img_dir_paths = sorted(os.listdir(img_dir))[1:]  # skip aux files

    j = 0
    for dirpath in tqdm(label_dir_paths):
        path = os.path.join(label_dir, dirpath)
        # to extract date information and id to match input images
        match = re.search(label_dir_regex, path)
        if match is not None:
            id = "_source_s2_" + match.group("id")

        matched_image_dirs = [os.path.join(img_dir, dir) for dir in img_dir_paths if id in dir]

        mask = load_tif_mask(filepath=path, dest_crs=PARTITION_CRS)

        imgs = load_images(filepaths=matched_image_dirs, band_names=BANDNAMES, dest_crs=PARTITION_CRS, cloud_p=CLOUD_P)

        # dataset is said to have all images 256,256 but found 270,270
        if imgs.shape[-2:] != (PATCH_SIZE, PATCH_SIZE) or mask.shape[-2:] != (PATCH_SIZE, PATCH_SIZE):
            imgs = imgs[:, :, 0:PATCH_SIZE, 0:PATCH_SIZE]
            mask = mask[0:PATCH_SIZE, 0:PATCH_SIZE]

        split = np.random.choice(("train", "valid", "test"), p=(0.8, 0.1, 0.1))

        for i in range(imgs.shape[0]):
            img = imgs[i, :, :, :]
            sample_name = dirpath + f"_{i}"
            sample = make_sample(img, mask, sample_name)
            sample.write(dataset_dir)
            partition.add(split, sample_name)
            j += 1

            if max_count is not None and j >= max_count:
                break

        if max_count is not None and j >= max_count:
            break

    partition.save(dataset_dir, "original", as_default=True)


if __name__ == "__main__":
    convert()
