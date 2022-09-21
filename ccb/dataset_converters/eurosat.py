"""Eurosat dataset."""
# EuroSat will be automatically downloaded by TorchGeo (https://github.com/microsoft/torchgeo)

import os
from pathlib import Path

import numpy as np
from torchgeo.datasets import EuroSAT
from tqdm import tqdm

from ccb import io

DATASET_NAME = "eurosat"
SRC_DATASET_DIR = Path(io.src_datasets_dir, DATASET_NAME)  # type: ignore
DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)  # type: ignore


def make_sample(images: "np.typing.NDArray[np.float_]", label, sample_name: str) -> io.Sample:
    """Create a sample from images and label.

    Args:
        images: image array to be contained in sample
        label: label to be contained in sample
        sample_name: name of sample

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
        eval_loss=io.Accuracy,
        spatial_resolution=10,
    )
    task_specs.save(dataset_dir, overwrite=True)

    sample_id = 0
    for split_name in ["train", "val", "test"]:
        eurosat_dataset = EuroSAT(
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
