"""Xview2 dataset."""

# Need to download XView2 dataset manually:
# Register at https://xview2.org/signup
# Copy link address to download Challenge training set (~7.8GB) and test set (~2.6GB) from https://xview2.org/download-links
# Insert link address to the curl to download datasets:
# curl -o $CC_BENCHMARK_SOURCE_DATASETS/xview2/train_images_labels_targets.tar.gz --remote-name "https://download.xview2.org/train_images_labels_targets.tar.gz?Expires=<>&Signature=<>__&Key-Pair-Id=<>"
# curl -o $CC_BENCHMARK_SOURCE_DATASETS/xview2/test_images_labels_targets.tar.gz --remote-name "INSERT_TEST_DATA_LINK_HERE"
# Verify download by visually comparing SHASUMs (todo: verify files through torchgeo.)
# shasum -a 1 train_images_labels_targets.tar.gz
# shasum -a 1 test_images_labels_targets.tar.gz
# Todo: move the geotransforms from dataset_converters to other location SRC_DATASET_DIR. The geotransforms.json of the original xview2 is inaccurate. It has been corrected in https://arxiv.org/abs/2104.04785.
# python xview2.py
# Can delete raw files (only after xview2.py was run once and torchgeo extracted tar.gz)
# rm train_images_labels_targets.tar.gz
# rm test_images_labels_targets.tar.gz

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import rasterio
from torchgeo.datasets import XView2
from tqdm import tqdm

from geobench import io

DATASET_NAME = "xview2"
SRC_DATASET_DIR = Path(io.src_datasets_dir, DATASET_NAME)  # type: ignore
DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)  # type: ignore


# # Todo: move to io.dataset.py
# class Worldview3(io.SpectralBand):
#     pass


# # Worldview3
# # Source: https://www.spaceimagingme.com/downloads/sensors/
# #      datasheets/DG_WorldView3_DS_2014.pdf
# # Todo: verify that band ordering is BGR
# worldview3_rgb_bands = [
#     Worldview3("01 - Blue", ("1", "01", "blue"), spatial_resolution=1.24, wavelength=0.51),  # .45-.51
#     Worldview3("02 - Green", ("2", "02", "green"), 1.24, 0.58),  # .51-.58
#     Worldview3("03 - Red", ("3", "03", "red"), 1.24, 0.69),  # .63-.69
# ]

rgb_bands = io.make_rgb_bands(spatial_resolution=1.24)

# Todo: document class labels: background, no damage, minor damage, major damage, destroyed
LABEL_BAND = io.SegmentationClasses("label", spatial_resolution=1.24, n_classes=5)


def convert_origin_transform_gdal_to_rasterio(transform_origin):
    """
    Convert geotransform in origin format, i.e., coordinates of upper left image corner, from gdal to rasterio format.

    Args:
        geo_t list: Geotransforms of upper left image corner in GDALDataset::GetGeoTransform() format

    Returns:
        transform rasterio.transform.Affine: rasterio transform
    """
    ul_e = transform_origin[0]  # Upper Left easting coordinate (i.e., horizontal)
    ul_n = transform_origin[3]  # The Upper left northing coordinate (i.e., vertical)
    ew_px_space = transform_origin[1]  # The E-W pixel spacing
    ns_px_space = transform_origin[5]  # The N-S pixel spacing, negative as we will be counting from the UL corner
    # TODO: check if ul_e needs to be converted from East to West coordinates!
    transform = rasterio.transform.from_origin(ul_e, ul_n, ew_px_space, ns_px_space)

    return transform


def make_sample(image_A, image_B, mask, sample_name):
    """Create one xview2 sample.

    One sample contains two images of pre- and post-disaster
    satellite imagery and a semantic segmentation mask. The mask classifies the building
    damage in the post-disaster image via pixel-level classes, such as, "destroyed building",
    "building with minor damage", or "no building".

    Args:
        image_A :dict(
            'image' np.array(3,1024,1024, dtype=np.uint8): RGB images
            'transform' rasterio.transform.Affine: geotransform
            'crs' rasterio.crs.CRS: Coordinate reference system, e.g., EPSG '4326'
            'spatial_resolution' float: Ground sampling distance in meters. Note that it
                varies from ~1.25 to 2m.
            'date' str: Date in format - "2018-02-05T17:10:18.000Z"
            'meta_info' dict(): Various metadata about the image, including
                "off_nadir_angle",  "pan_resolution", "sun_azimuth",
                "sun_elevation", "target_azimuth", "disaster_type",
                "img_name"
            )
        image_B : Same as image_A, but post-disaster
        mask np.array(1024,1024, dtype=np.int64): Semantic segmentation mask, classes encoded in integer value
        sample_name str: Sample name; 'id_{:04d}'

    Returns:
        sample io.Sample: GeobenchDataset sample
    """
    # Todo: Convert images and mask into uint16.
    for j, image in enumerate([image_A, image_B]):
        n_bands, _height, _width = image["image"].shape
        # Todo: Convert date to format Union[datetime.datetime, datetime.date]
        date = None  # convert(image['date'])
        bands = []
        for band_idx in range(n_bands):
            band_data = image["image"][band_idx, :, :]
            band_info = rgb_bands[band_idx]
            band = io.Band(
                data=band_data,
                band_info=band_info,
                spatial_resolution=image["spatial_resolution"],
                date=date,
                transform=image["transform"],
                crs=image["crs"],
                meta_info=image["meta_info"],
                convert_to_int16=False,
            )
            bands.append(band)

    label = io.Band(
        data=mask,
        band_info=LABEL_BAND,
        spatial_resolution=image_B["spatial_resolution"],
        transform=image_B["transform"],
        crs=image_B["crs"],
    )
    return io.Sample(bands, label=label, sample_name=sample_name)


def convert(max_count=None, dataset_dir=DATASET_DIR) -> None:
    """Convert torchgeo.XView2 dataset into ccb dataset.

    Args:
        max_count int: Maximum number of images to be converted
        dataset_dir string: GeobenchDataset directory
    """
    dataset_dir.mkdir(exist_ok=True, parents=True)  # Creates path to converted data
    partition = io.Partition()  # Creates dictionary to store train, val, test filenames

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(1024, 1024),
        n_time_steps=1,
        bands_info=rgb_bands,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=LABEL_BAND,
        eval_loss=io.SegmentationAccuracy,  # TODO probably not the final
        # loss eval loss. To be discussed.
        spatial_resolution=1.24,  # Note varying res in dataset.
    )
    task_specs.save(dataset_dir)

    offset = 0

    # Load geotransforms
    with open(SRC_DATASET_DIR / "xview2-geotransforms.json") as json_file:
        transforms_json = json.load(json_file)

    for split_name in ["train", "test"]:
        xview2_dataset = XView2(root=SRC_DATASET_DIR, split=split_name, transforms=None, checksum=True)
        for i, tg_sample in enumerate(tqdm(xview2_dataset)):
            # tg_sample dict(
            #   'image': torch.Tensor(2,3,1024,1024)
            #   'mask': torch.Tensor(2,1024,1024)) -
            sample_name = f"id_{i+offset:04d}"
            image_A: Dict[str, Any] = {}
            image_B: Dict[str, Any] = {}
            for j, image in enumerate([image_A, image_B]):
                # TODO: why are we converting torch.Tensor to np.array here; seems like it would be more efficient to keep as Tensor?
                image["image"] = np.array(tg_sample["image"][j, ...])
                if j == 1:  # We only use the damage mask of the post-disaster image. The
                    # building/no-building of the pre-disaster image is discarded.
                    mask = np.array(tg_sample["mask"][j, ...])

                # Get geotransform and coordinate reference system; image and mask share same transform
                filename = xview2_dataset.files[i]["image" + str(j + 1)].split("/")[-1]
                transform_origin = transforms_json[filename][0]
                image["transform"] = convert_origin_transform_gdal_to_rasterio(transform_origin)
                image["crs"] = rasterio.crs.CRS.from_string(transforms_json[filename][1])

                # Extract metadata from xview2 dataset
                dir_meta_info = Path(SRC_DATASET_DIR, split_name, "labels")
                filename_meta_info = filename.split(".")[0] + ".json"
                with open(dir_meta_info / filename_meta_info) as json_file:
                    image["meta_info"] = json.load(json_file)["metadata"]
                image["spatial_resolution"] = image["meta_info"].pop("gsd")
                image["date"] = image["meta_info"].pop("capture_date")

            sample = make_sample(image_A, image_B, mask, sample_name)
            sample.write(dataset_dir)
            partition.add(split_name, sample_name)

            offset += 1

            if max_count is not None and i + 1 >= max_count:
                break

        if max_count is not None and offset >= max_count:
            break

    partition.resplit_iid(split_names=("valid", "test"), ratios=(0.5, 0.5))
    partition.save(dataset_dir, "original", as_default=True)


if __name__ == "__main__":
    convert()
