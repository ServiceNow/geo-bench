"""Big Earth Net dataset."""
from pathlib import Path

import numpy as np
from torchgeo.datasets import BigEarthNet
from tqdm import tqdm

from ccb import io

DATASET_NAME = "bigearthnet"
SRC_DATASET_DIR = Path(io.src_datasets_dir, "bigearthnet")  # type: ignore
DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)  # type: ignore


def make_sample(images: np.array, label, sample_name: str, task_specs: io.TaskSpecifications) -> io.Sample:
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

        if task_specs.bands_info is not None:
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


def convert(max_count=None, dataset_dir=DATASET_DIR) -> None:
    """Convert BigEarthNet dataset.

    Args:
        max_count: maximum number of samples
        dataset_dir: path to dataset directory
    """
    dataset_dir.mkdir(exist_ok=True, parents=True)
    partition = io.dataset.Partition()

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(120, 120),
        n_time_steps=1,
        bands_info=io.sentinel2_13_bands[0:10] + io.sentinel2_13_bands[-2:],
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=io.MultiLabelClassification(43, class_names=BigEarthNet.class_sets[43]),
        eval_loss=io.MultilabelAccuracy,
        spatial_resolution=10,
    )

    task_specs.save(dataset_dir, overwrite=True)
    n_samples = 0
    for split_name in ["train", "val", "test"]:
        bigearthnet_dataset = BigEarthNet(
            root=SRC_DATASET_DIR,
            split=split_name,
            bands="s2",
            download=False,
            transforms=None,
            checksum=False,
            num_classes=43,
        )

        for i, tg_sample in enumerate(tqdm(bigearthnet_dataset)):
            sample_name = f"id_{n_samples:04d}"

            images = np.array(tg_sample["image"])
            label = np.array(tg_sample["label"])

            sample = make_sample(images, label, sample_name, task_specs)
            sample.write(dataset_dir)

            partition.add(split_name.replace("val", "valid"), sample_name)

            n_samples += 1
            if max_count is not None and n_samples >= max_count:
                break

        if max_count is not None and n_samples >= max_count:
            break

    partition.save(dataset_dir, "original", as_default=True)


if __name__ == "__main__":
    convert()
