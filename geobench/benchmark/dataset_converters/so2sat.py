"""So2Sat dataset."""
# So2Sat will be automatically downloaded by TorchGeo (https://github.com/microsoft/torchgeo)

import os
from pathlib import Path

import numpy as np
from torchgeo.datasets import So2Sat
from tqdm import tqdm

from geobench import io
from geobench.io.dataset import Sample
from geobench.io.task import TaskSpecifications

DATASET_NAME = "so2sat"
SRC_DATASET_DIR = io.CCB_DIR / "source" / DATASET_NAME  # type: ignore
DATASET_DIR = io.CCB_DIR / "converted" / DATASET_NAME  # type: ignore


def make_sample(
    images: "np.typing.NDArray[np.int_]", label: int, sample_name: str, task_specs: TaskSpecifications
) -> Sample:
    """Create a sample from images and label.

    Args:
        images: image array to be contained in sample
        label: label to be contained in sample
        sample_name: name of sample
        task_specs: task specifications of this datasets

    Returns:
        sample
    """
    n_bands, _height, _width = images.shape

    transform = None  # TODO can't find the GPS coordinates from torch geo.
    crs = None

    bands = []
    for band_idx in range(n_bands):
        band_data = images[band_idx, :, :]

        band_info = task_specs.bands_info[band_idx]
        band_data = band_data.astype(np.float32)
        band = io.Band(
            data=band_data,
            band_info=band_info,
            spatial_resolution=task_specs.spatial_resolution,
            transform=transform,
            crs=crs,
            convert_to_int16=False,
        )
        bands.append(band)

    return io.Sample(bands, label=label, sample_name=sample_name)


def convert(max_count: int = None, dataset_dir: Path = DATASET_DIR) -> None:
    """Convert So2Sat dataset.

    Args:
        max_count: maximum number of samples
        dataset_dir: path to dataset directory
    """
    dataset_dir.mkdir(exist_ok=True, parents=True)
    partition = io.dataset.Partition()

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(32, 32),
        n_time_steps=1,
        bands_info=io.sentinel1_8_bands + io.sentinel2_13_bands[1:9] + io.sentinel2_13_bands[-2:],  # type: ignore
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=io.Classification(17, class_names=So2Sat.classes),
        eval_loss=io.Accuracy,
        spatial_resolution=10,
    )
    task_specs.save(str(dataset_dir), overwrite=True)
    n_samples = 0
    for split_name in ["train", "validation", "test"]:
        so2sat_dataset = So2Sat(root=SRC_DATASET_DIR, split=split_name, transforms=None, checksum=True)
        for tg_sample in tqdm(so2sat_dataset):
            sample_name = f"id_{n_samples:04d}"

            images = np.array(tg_sample["image"])
            label = tg_sample["label"]

            sample = make_sample(images, int(label), sample_name, task_specs)
            sample.write(str(dataset_dir))

            partition.add(split_name.replace("validation", "valid"), sample_name)

            n_samples += 1
            if max_count is not None and n_samples >= max_count:
                break

        if max_count is not None and n_samples >= max_count:
            break

    partition.save(str(dataset_dir), "original", as_default=True)


if __name__ == "__main__":
    convert()
