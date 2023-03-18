"""Task."""

import json
import os
import pickle
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Generator, List, Sequence, Tuple, Union

import numpy as np

from geobench import io
from geobench.io.dataset import BandInfo, GeobenchDataset, Landsat8, Sentinel1, Sentinel2, SpectralBand
from geobench.io.label import Classification


class TaskSpecifications:
    """Task Specifications define information necessary to run a training/evaluation on a dataset."""

    def __init__(
        self,
        dataset_name: str,
        bands_info: List[Any],
        spatial_resolution: float,
        benchmark_name: str = None,
        patch_size: Tuple[int, int] = None,
        n_time_steps: int = None,
        bands_stats=None,
        label_type=None,
    ) -> None:
        """Initialize a new instance of TaskSpecifications.

        Args:
            dataset_name: The name of the dataset.
            bands_info: band info
            spatial_resolution: physical distance between pixels in meters.
            benchmark_name: The name of the benchmark used. Defaults to "converted".
            patch_size: maximum image patch size across bands (width, height).
            n_time_steps: integer specifying the number of time steps for each sample.
                This should be 1 for most dataset unless it's time series.
            bands_info: list of object of type BandInfo descrbing the type of each band.
            label_type: The type of the label e.g. Classification, SegmentationClasses, Regression.
        """
        self.dataset_name = dataset_name
        self.benchmark_name = benchmark_name
        self.patch_size = patch_size
        self.n_time_steps = n_time_steps
        self.bands_info = bands_info
        self.bands_stats = bands_stats
        self.label_type = label_type
        self.spatial_resolution = spatial_resolution

    def __str__(self) -> str:
        """Return strin representation of class."""
        shape = "x".join([str(sz) for sz in self.patch_size])
        lines = [
            f"{self.benchmark_name}/{self.dataset_name}",
            f"  {len(self.bands_info)} bands, max shape {shape} @ {self.spatial_resolution}m resolution.",
        ]
        return "\n".join(lines)

    def save(self, directory: str, overwrite: bool = False) -> None:
        """Save task specs.

        Args:
            directory: directory where task_specs.pkl can be found
            overwrite: whether or not to overwrite existing task_specs

        Raises:
            Exception if task_specs already exists and overwrite is False
        """
        file_path = Path(directory, "task_specs.pkl")
        if file_path.exists() and not overwrite:
            raise Exception("task_specifications.pkl alread exists and overwrite is set to False.")
        with open(file_path, "wb") as fd:
            pickle.dump(self, fd, protocol=4)

    def get_dataset(
        self,
        benchmark_dir: str = None,
        split: Union[str, None] = None,
        partition_name: str = "default",
        transform=None,
        band_names: Sequence[str] = ("red", "green", "blue"),
        format: str = "hdf5",
    ) -> GeobenchDataset:
        """Retrieve dataset for a given split and partition with chosen transform, format and bands.

        Args:
            benchmark_dir: path to benchmark directory where dataset can be found
            split: dataset split to choose
            partition_name: name of partition, i.e. 'default' for default_partition.json
            transform: dataset transforms
            file_format: 'hdf5' or 'tif'
            band_names: band names to select from dataset
        """
        return GeobenchDataset(
            dataset_dir=self.get_dataset_dir(benchmark_dir),
            split=split,
            partition_name=partition_name,
            transform=transform,
            format=format,
            band_names=band_names,
        )

    def get_dataset_dir(self, benchmark_dir: Union[Path, str] = None):
        """Retrieve directory where dataset is read."""
        if benchmark_dir is None:
            benchmark_dir = io.CCB_DIR / self.benchmark_name  # type: ignore
        return Path(benchmark_dir) / self.dataset_name

    def get_label_map(self, benchmark_dir: str = None) -> Union[None, Dict[str, List[str]]]:
        """Retriebe the label map, a dictionary defining labels to input paths.

        Args:
            benchmark_dir: benchmark directory from which to retrieve dataset


        Returns:
            label map if present or None
        """
        label_map_path = self.get_dataset_dir(benchmark_dir=benchmark_dir) / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, "r") as fp:
                label_map: Dict[str, List[str]] = json.load(fp)
            return label_map
        else:
            return None

    def label_stats(self, benchmark_dir: str = None) -> Union[None, Dict[str, List[Any]]]:
        """Retriebe the label stats, a dictionary defining labels to statistics.

        Returns:
            label stats if present or None
        """
        label_stats_path = self.get_dataset_dir(benchmark_dir=benchmark_dir) / "label_stats.json"
        if label_stats_path.exists():
            with open(label_stats_path, "r") as fp:
                label_stats = json.load(fp)
            return label_stats
        else:
            return None


def task_iterator(benchmark_dir: str, task_filter: List[str] = None) -> Generator[TaskSpecifications, None, None]:
    """Iterate over all tasks present in a benchmark.

    Args:
        benchmark_name: name of the benchmark

    Returns:
        task specifications for the desired benchmark dataset
    """
    benchmark_dir_path = Path(benchmark_dir)

    for dataset_dir in benchmark_dir_path.iterdir():
        if not dataset_dir.is_dir():
            continue

        if task_filter is not None:
            if dataset_dir.name not in task_filter:
                continue

        yield load_task_specs(dataset_dir)


def load_task_specs(dataset_dir: Path, rename_benchmark: bool = True) -> TaskSpecifications:
    """Load task specifications from a path.

    Args:
        dataset_dir: path to dataset directory of task_specifications
        rename_benchmark: whether or not to rename benchmark with with benchmark directory name

    Returns:
        task specifications
    """
    dataset_dir = Path(dataset_dir)
    with open(dataset_dir / "task_specs.pkl", "rb") as fd:
        task_specs = pickle.load(fd)

    if rename_benchmark:
        task_specs.benchmark_name = dataset_dir.parent.name
    return task_specs
