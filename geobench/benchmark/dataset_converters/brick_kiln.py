"""BrickKiln dataset."""
# Downloaded from "https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg13/brick_kiln.html"
# Try this command for downloading on headless server:
#   wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aOWHRY72LlHNv7nwbAcPEHWcuuYnaEtx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1aOWHRY72LlHNv7nwbAcPEHWcuuYnaEtx" -O brick_kiln_v1.0.tar.gz && rm -rf /tmp/cookies.txt


import csv
import os
from pathlib import Path

import h5py
import numpy as np
import rasterio
from tqdm import tqdm

from geobench import io

DATASET_NAME = "brick_kiln_v1.0"
SRC_DATASET_DIR = Path(io.src_datasets_dir, DATASET_NAME)  # type: ignore
DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)  # type: ignore


def load_examples_bloc(file_path):
    """Load a .h5py bloc of images with their labels.

    Args:
        file_path: path to bloc of images

    Returns:
        images, labels, bounds, and file id
    """
    file_id = file_path.stem.split("_")[1]

    with h5py.File(file_path) as data:

        images = data["images"][:]
        labels = data["labels"][:]
        bounds = data["bounds"][:]

        return images, labels, bounds, file_id


def read_list_eval_partition(csv_file):
    """Read List eval partition.

    The CSV file contains redundant information and the information for the original partition.

    Args:
        csv_file: path to csv file

    Returns:
        information
    """
    with open(csv_file) as fd:
        reader = csv.reader(fd, delimiter=",")
        data = []
        for i, row in enumerate(reader):
            if i == 0:
                continue
            data.append([float(e) for e in row])

    data = np.array(data)

    y = data[:, 0].astype(np.int32)
    partition = data[:, 1].astype(np.int32)
    hdf5_file = data[:, 2].astype(np.int32)
    hdf5_idx = data[:, 3].astype(np.int32)
    gps = data[:, 4:8]
    indices = data[:, 8:11].astype(np.int32)

    id_map = {}
    for i, (file_id, sample_idx) in enumerate(zip(hdf5_file, hdf5_idx)):
        id_map[(file_id, sample_idx)] = i

    return y, partition, hdf5_file, hdf5_idx, gps, indices, id_map


def make_sample(src_bands, label, coord_box, sample_name) -> io.Sample:
    """Create a sample.

    Convert the data in src_bands. Instantiate each Band separately and combine them into Sample

    Args:
        src_bands:
        label:
        coord_box:
        sample_name: name of sample

    Returns:
        sample
    """
    lon_top_left, lat_top_left, lon_bottom_right, lat_bottom_right = coord_box
    transform = rasterio.transform.from_bounds(
        west=lon_top_left,
        south=lat_bottom_right,
        east=lon_bottom_right,
        north=lat_top_left,
        width=src_bands.shape[1],
        height=src_bands.shape[2],
    )

    bands = []
    for i, band in enumerate(src_bands):
        band_data = io.Band(
            data=band, band_info=io.sentinel2_13_bands[i], spatial_resolution=10, transform=transform, crs="EPSG:4326"
        )
        bands.append(band_data)

    return io.Sample(bands, label=int(label), sample_name=sample_name)


def convert(max_count=None, dataset_dir=DATASET_DIR) -> None:
    """Convert BrickKiln dataset.

    Args:
        max_count: maximum number of samples
        dataset_dir: path to dataset directory
    """
    dataset_dir.mkdir(exist_ok=True, parents=True)

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(64, 64),
        n_time_steps=1,
        bands_info=io.sentinel2_13_bands,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=io.Classification(2, ["not brick kiln", "brick kiln"]),
        eval_loss=io.Accuracy,
        spatial_resolution=10,
    )
    task_specs.save(dataset_dir, overwrite=True)

    _, partition_id, _, _, _, _, id_map = read_list_eval_partition(Path(SRC_DATASET_DIR, "list_eval_partition.csv"))

    file_list = list(SRC_DATASET_DIR.iterdir())
    partition = io.Partition()
    split_map = {0: "train", 1: "valid", 2: "test"}
    sample_count = 0
    for file_idx, file_path in enumerate(tqdm(file_list)):
        if file_path.suffix != ".hdf5":
            continue

        # In this dataset, images ares stored as a batch of up to 999 samples, but sometime there are none.
        images, labels, bounds, file_id = load_examples_bloc(file_path)

        if images.shape[0] == 0:
            print("Skipping block of shape 0. Shape = %s" % (str(images.shape)))
            continue

        data = list(zip(images, labels, bounds))
        for img_idx in tqdm(range(len(data)), leave=False):
            all_bands, label, coord_box = data[img_idx]
            sample_name = f"examples_{file_id}_{img_idx}"
            split_name = split_map[partition_id[id_map[(int(file_id), img_idx)]]]
            partition.add(split_name, sample_name)
            sample = make_sample(all_bands, label, coord_box, sample_name)
            sample.write(dataset_dir)
            sample_count += 1
            # temporary for creating small datasets for development purpose
            if max_count is not None and sample_count >= max_count:
                break

        if max_count is not None and sample_count >= max_count:
            break
    partition.save(dataset_dir, "original", as_default=True)


if __name__ == "__main__":
    convert()
