"""Nz cattle detection dataset."""
# Downloaded from "https://zenodo.org/record/5908869"

# to authors
# * coordintates are lon-lat (not lat-lon)
# * specify the coordintates are for the center.
# * can we change "Kapiti_Coast" to "Kapiti-Coast"

import datetime
from pathlib import Path
from typing import List, Union

import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

from geobench import io
from geobench.benchmark.rasterize_detection import point_to_boxes, rasterize_box

SEGMENTATION = True

if SEGMENTATION:
    DATASET_NAME = "nz_cattle_segmentation"
else:
    DATASET_NAME = "nz_cattle_detection"

SRC_DATASET_DIR = Path(io.src_datasets_dir, "nz_cattle")  # type: ignore
DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)  # type: ignore


BAND_INFO_LIST = io.make_rgb_bands(0.1)

if SEGMENTATION:
    label_type = io.SegmentationClasses(  # type: ignore
        "label", spatial_resolution=0.1, n_classes=2, class_names=["no cattle", "cattle"]
    )
else:
    label_type = io.PointAnnotation()  # type: ignore


def parse_file_name(name):
    """Parse file name and extract information.

    Args:
        name: filename

    Returns:
        location, date, transform and crs information
    """
    name = name.replace("Kapiti_Coast", "Kapiti-Coast")
    _index, location, year, lon_lat = name.split("_")[:4]
    lon_center, lat_center = [float(val) for val in lon_lat.split(",")]
    year = int(year[1:-1].split("-")[-1])
    date = datetime.date(year=year, month=1, day=1)
    transform_center = rasterio.transform.from_origin(lon_center, lat_center, 0.1, 0.1)
    lon_corner, lat_corner = transform_center * [-250, -250]
    transform = rasterio.transform.from_origin(lon_corner, lat_corner, 0.1, 0.1)

    crs = rasterio.crs.CRS.from_epsg(4326)

    return location, date, transform, crs


def load_sample(img_path: Path) -> io.Sample:
    """Create sample from a given image path.

    Args:
        img_path: path to image

    Return:
        created sample
    """
    label_path = img_path.with_suffix(".png.mask.0.txt")
    with Image.open(img_path) as im:
        data = np.array(im)[:, :, :3]

    location, date, transform, crs = parse_file_name(img_path.stem)
    coords = []
    with open(label_path, "r") as fd:
        for line in fd:
            coord = [int(val) for val in line.split(",")]
            coords.append(coord)

    bands = []
    for i in range(3):
        band_data = io.Band(
            data=data[:, :, i],
            band_info=BAND_INFO_LIST[i],
            spatial_resolution=0.1,
            transform=transform,
            crs=crs,
            date=date,
            meta_info={"location": location},
        )
        bands.append(band_data)

    if SEGMENTATION:
        label_data = rasterize_box(boxes=point_to_boxes(points=coords, radius=4), img_shape=data.shape[:2])
        label = io.Band(
            data=label_data,
            band_info=label_type,
            spatial_resolution=0.1,
            transform=transform,
            crs=crs,
            date=date,
            meta_info={"location": location},
        )
    # else:
    #     label = coords

    return io.Sample(bands, label=label, sample_name=img_path.stem)


def convert(max_count=None, dataset_dir=DATASET_DIR) -> None:
    """Convert Nz Cattle detection dataset.

    Args:
        max_count: maximum number of samples
        dataset_dir: path to dataset directory
    """
    dataset_dir.mkdir(exist_ok=True, parents=True)

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(500, 500),
        n_time_steps=1,
        bands_info=BAND_INFO_LIST,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=label_type,
        eval_loss=io.SegmentationAccuracy(),  # TODO decide on the loss
        spatial_resolution=0.1,
    )
    task_specs.save(dataset_dir, overwrite=True)
    partition = io.Partition()

    path_list = list(Path(SRC_DATASET_DIR, "cow_images").iterdir())

    sample_count = 0
    partition = io.Partition()  # default partition: everything in train
    for file in tqdm(path_list):
        if file.suffix == ".png":
            sample = load_sample(img_path=file)
            sample.write(dataset_dir)

            partition.add("train", sample.sample_name)

            sample_count += 1
            if max_count is not None and sample_count >= max_count:
                break

    partition.resplit_iid(split_names=("train", "valid", "test"), ratios=(0.8, 0.1, 0.1))
    partition.save(dataset_dir, "iid", as_default=True)


if __name__ == "__main__":
    convert()
