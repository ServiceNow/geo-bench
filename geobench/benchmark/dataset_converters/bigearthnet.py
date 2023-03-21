"""Big Earth Net dataset."""
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from torch import Tensor
from torchgeo.datasets import BigEarthNet  # noqa: F811
from tqdm import tqdm

from geobench import io

DATASET_NAME = "bigearthnet"
SRC_DATASET_DIR = Path(io.src_datasets_dir, "bigearthnet")  # type: ignore
DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)  # type: ignore


class GeoBigEarthNet(BigEarthNet):
    """Wrapper for BigEarthNet to get geo information."""

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: str = "all",
        num_classes: int = 19,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new BigEarthNet dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            bands: load Sentinel-1 bands, Sentinel-2, or both. one of {s1, s2, all}
            num_classes: number of classes to load in target. one of {19, 43}
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        super().__init__(root, split, bands, num_classes, transforms, download, checksum)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image, crs, bounds = self._load_image(index)
        label = self._load_target(index)
        sample: Dict[str, Tensor] = {"image": image, "label": label, "crs": crs, "bounds": bounds}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        paths = self._load_paths(index)
        images = []
        for path in paths:
            # Bands are of different spatial resolutions
            # Resample to (120, 120)
            with rasterio.open(path) as dataset:
                array = dataset.read(
                    indexes=1,
                    out_shape=self.image_size,
                    out_dtype="int32",
                    resampling=Resampling.bilinear,
                )
                images.append(array)
        arrays: "np.typing.NDArray[np.int_]" = np.stack(images, axis=0)
        tensor = torch.from_numpy(arrays).float()
        return tensor, dataset.crs, dataset.bounds


def make_sample(
    images: "np.typing.NDArray[np.int_]", label, sample_name: str, task_specs: io.TaskSpecifications
) -> io.Sample:
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
        # eval_loss=io.MultilabelAccuracy,
        spatial_resolution=10,
    )

    task_specs.save(dataset_dir, overwrite=True)
    n_samples = 0
    for split_name in ["train", "val", "test"]:
        bigearthnet_dataset = GeoBigEarthNet(
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
