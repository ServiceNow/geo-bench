"""Task."""

from functools import cached_property
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Generator, List, Sequence, Tuple, Union

import numpy as np

from geobench import GEO_BENCH_DIR
from geobench.dataset import GeobenchDataset, Sample, _load_band_stats


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
            raise Exception("task_specs.pkl alread exists and overwrite is set to False.")
        with open(file_path, "wb") as fd:
            pickle.dump(self, fd, protocol=4)

    def get_dataset(
        self,
        split: Union[str, None] = None,
        partition_name: str = "default",
        transform=None,
        band_names: Sequence[str] = ("red", "green", "blue"),
        format: str = "hdf5",
    ) -> GeobenchDataset:
        """Retrieve dataset for a given split and partition with chosen transform, format and bands.

        Args:
            split: dataset split to choose
            partition_name: name of partition, i.e. 'default' for default_partition.json
            transform: callable for transforming a sample after loading
            file_format: 'hdf5' or 'tif'
            band_names: band names to select from dataset
        """
        return GeobenchDataset(
            dataset_dir=self.get_dataset_dir(),
            split=split,
            partition_name=partition_name,
            transform=transform,
            format=format,
            band_names=band_names,
        )

    def get_dataset_dir(self) -> Path:
        """Retrieve directory where dataset is read."""
        return GEO_BENCH_DIR / self.benchmark_name / self.dataset_name

    def get_label_map(self) -> Union[None, Dict[str, List[str]]]:
        """Retriebe the label map, a dictionary defining labels to input paths.

        Args:
            benchmark_dir: benchmark directory from which to retrieve dataset


        Returns:
            label map if present or None
        """
        label_map_path = self.get_dataset_dir() / "label_map.json"
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
        label_stats_path = self.get_dataset_dir() / "label_stats.json"
        if label_stats_path.exists():
            with open(label_stats_path, "r") as fp:
                label_stats = json.load(fp)
            return label_stats
        else:
            return None

    @cached_property
    def band_stats(self):
        """Retrieve band stats."""
        return _load_band_stats(self.get_dataset_dir())

    def get_pytorch_data_module(
        self,
        partition_name: str = "default",
        batch_size: int = 64,
        num_workers: int = 8,
        val_batch_size: int = None,
        train_transform=None,
        eval_transform=None,
        collate_fn=None,
        band_names: Sequence[str] = ("red", "green", "blue"),
    ):
        """return pytorch data module for this dataset."""

        # import this module only on demand to avoid strict dependency on pytorch
        from geobench.torch_toolbox.dataset import DataModule

        data_module = DataModule(
            self,
            partition_name=partition_name,
            batch_size=batch_size,
            num_workers=num_workers,
            val_batch_size=val_batch_size,
            train_transform=train_transform,
            eval_transform=eval_transform,
            collate_fn=collate_fn,
            band_names=band_names,
        )
        return data_module

    def self_update_info(self, samples: List[Sample], verbose=False):
        old_bands_info = self.bands_info
        old_shapes = self.patch_size
        old_resolutions = self.spatial_resolution

        bands_info = None
        shapes = None
        for sample in samples:
            if bands_info is None:
                bands_info = [band.band_info for band in sample.bands]
                shapes = [band.data.shape for band in sample.bands]
            else:
                assert len(bands_info) == len(sample.bands)
                for i, band_info in enumerate(bands_info):
                    assert band_info == sample.bands[i].band_info
                for i, shape in enumerate(shapes):
                    assert shape == sample.bands[i].data.shape

        resolutions = [band_info.spatial_resolution for band_info in bands_info]

        self.bands_info = bands_info
        # remove None
        self.spatial_resolution = np.min([res for res in resolutions if res is not None])
        areas = [shape[0] * shape[1] for shape in shapes]
        self.patch_size = shapes[np.argmax(areas)]

        if verbose:
            print(f"Updated task specs for {self.dataset_name}")
            print(f"  patch_size: {old_shapes} -> {self.patch_size}")
            print(f"  spatial_resolution: {old_resolutions} -> {self.spatial_resolution}")

            for old_band_info, band_info in zip(old_bands_info, self.bands_info):
                print(f"  ´{old_band_info} -> {band_info}")


def task_iterator(
    benchmark_name: str = None, ignore_task: List[str] = None, benchmark_dir: str = None
) -> Generator[TaskSpecifications, None, None]:
    """Iterate over all tasks present in a benchmark.

    Args:
        benchmark_name: name of the benchmark
        ignore_task: list of task names to exclude
        benchmark_dir: override default benchmark directory. If None, will
            use $GEO_BENCH_DIR / benchmark_name

    Returns:
        task specifications for the desired benchmark dataset
    """
    if benchmark_dir is None:
        benchmark_dir_path = GEO_BENCH_DIR / benchmark_name
    else:
        benchmark_dir_path = Path(benchmark_dir)

    for dataset_dir in benchmark_dir_path.iterdir():
        if not dataset_dir.is_dir():
            continue

        if dataset_dir.name.startswith("_") or dataset_dir.name.startswith("."):
            continue

        if ignore_task is not None:
            if dataset_dir.name not in ignore_task:
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
    assert isinstance(task_specs, TaskSpecifications)

    # ensures consistency with benchmark directory name for backward compatibility
    if rename_benchmark:
        task_specs.benchmark_name = dataset_dir.parent.name
        task_specs.dataset_name = dataset_dir.name
    return task_specs


class SegmentationAccuracy:
    """For loading old pickles"""


class Accuracy:
    """For loading old pickles"""


class MultilabelAccuracy:
    """For loading old pickles"""
