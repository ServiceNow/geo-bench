"""Task."""

import json
import os
import pickle
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Generator, List, Sequence, Tuple, Union

import numpy as np

from ccb import io
from ccb.io.dataset import BandInfo, Dataset
from ccb.io.label import Classification


class TaskSpecifications:
    """Task Specifications define information necessary to run a training/evaluation on a dataset."""

    def __init__(
        self,
        dataset_name: str,
        benchmark_name: str,
        patch_size: Tuple[int, int] = None,
        n_time_steps: int = None,
        bands_info: List[BandInfo] = None,
        bands_stats=None,
        label_type=None,
        eval_loss=None,
        eval_metrics=None,
        spatial_resolution: float = None,
    ) -> None:
        """Initialize a new instance of TaskSpecifications.

        Args:
            dataset_name: The name of the dataset.
            benchmark_name: The name of the benchmark used. Defaults to "converted".
            patch_size: maximum image patch size across bands (width, height).
            n_time_steps: integer specifying the number of time steps for each sample.
                This should be 1 for most dataset unless it's time series.
            bands_info: list of object of type BandInfo descrbing the type of each band.
            label_type: The type of the label e.g. Classification, SegmentationClasses, Regression.
            eval_loss: Object of type Loss, e.g. Accuracy, SegmentationAccuracy.
            spatial_resolution: physical distance between pixels in meters.
        """
        self.dataset_name = dataset_name
        self.benchmark_name = benchmark_name
        self.patch_size = patch_size
        self.n_time_steps = n_time_steps
        self.bands_info = bands_info
        self.bands_stats = bands_stats
        self.label_type = label_type
        self.eval_loss = eval_loss
        self.eval_metrics = eval_metrics
        self.spatial_resolution = spatial_resolution

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
        benchmark_dir: Path = None,
        split: Union[str, None] = None,
        partition_name: str = "default",
        transform=None,
        band_names: Sequence[str] = ("red", "green", "blue"),
        format: str = "hdf5",
    ) -> Dataset:
        """Retrieve dataset for a given split and partition with chosen transform, format and bands.

        Args:
            benchmark_dir: path to benchmark directory where dataset can be found
            split: dataset split to choose
            partition_name: name of partition, i.e. 'default' for default_partition.json
            transform: dataset transforms
            file_format: 'hdf5' or 'tif'
            band_names: band names to select from dataset
        """
        if benchmark_dir is None:
            benchmark_dir = io.CCB_DIR / self.benchmark_name
        return Dataset(
            dataset_dir=self.get_dataset_dir(benchmark_dir),
            split=split,
            partition_name=partition_name,
            transform=transform,
            format=format,
            band_names=band_names,
        )

    def get_dataset_dir(self, benchmark_dir: str = None):
        """Retrieve directory where dataset is read."""
        # benchmark_name = self.benchmark_name or "default"
        # benchmark_dir = get_benchmark_dir(benchmark_name)
        if benchmark_dir is None:
            benchmark_dir = io.CCB_DIR / self.benchmark_name
        return Path(benchmark_dir) / self.dataset_name

    # # for backward compatibility (we'll remove soon)
    # @cached_property
    # def benchmark_name(self):
    #     """Return benchmark name."""
    #     return "default"

    def get_label_map(self) -> Union[None, Dict[str, List[str]]]:
        """Retriebe the label map, a dictionary defining labels to input paths.

        Returns:
            label map if present or None
        """
        label_map_path = self.get_dataset_dir() / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, "r") as fp:
                label_map = json.load(fp)
            return label_map
        else:
            return None

    @cached_property
    def label_stats(self) -> Union[None, Dict[str, List[Any]]]:
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


# def get_benchmark_dir(benchmark_name: str) -> Path:
#     """Retrieve benchmark directory depending on experiment or testing case.

#     Args:
#         benchmark_name: name of the benchmark

#     Returns:
#         path to benchmark directory
#     """
#     if benchmark_name in ["ccb-test-classification", "ccb-test-segmentation"]:
#         ccb_dir = Path(os.path.abspath(os.path.join("tests", "data")))
#     else:
#         ccb_dir: Path = CCB_DIR
#     return ccb_dir / benchmark_name


def task_iterator(benchmark_dir: str) -> Generator[TaskSpecifications, None, None]:
    """Iterate over all tasks present in a benchmark.

    Args:
        benchmark_name: name of the benchmark

    Returns:
        task specifications for the desired benchmark dataset
    """
    benchmark_dir = Path(benchmark_dir)

    for dataset_dir in benchmark_dir.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("_") or dataset_dir.name.startswith("."):
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


class Loss(object):
    """Loss.

    Define a general loss object that is library agnostic.
    """

    def __call__(self, label, prediction) -> float:
        """Define computation when Loss is called.

        Args:
            label: Array containing labels
            predictions: model predictions

        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        """Return string representation of loss object."""
        return self.__class__.__name__

    @property
    def name(self):
        """Return loss object name."""
        return self.__class__.__name__.lower()


class Accuracy(Loss):
    """Accuracy.

    Define Accuracy computations
    """

    def __call__(self, label, prediction) -> float:
        """Define computation when Accuracy is called.

        Args:
            label: Array containing labels
            predictions: model predictions

        Returns:
            accuracy scores
        """
        return float(label != prediction)


class MultilabelAccuracy(Loss):
    """Multilabel Accuracy.

    Define Multi-label Accuracy computations
    """

    def __call__(self, prediction, label):
        """Define computation when Accuracy is called.

        Args:
            label: Array containing labels
            predictions: model predictions

        Returns:
            mulit-label accuracy scores
        """
        return np.mean(label != prediction)


class AccuracyTop30(Loss):
    """Top 30 Accuracy."""

    # TODO: Could be integrated above or with extra argument for TopK
    pass


class CrossEntropy(Loss):
    """Cross Entropy Loss."""

    pass


class SegmentationAccuracy(Loss):
    """Segmentation Accuracy."""

    pass


mnist_task_specs = TaskSpecifications(
    dataset_name="MNIST",
    benchmark_name="test",
    patch_size=(28, 28),
    bands_info=[BandInfo("grey")],
    label_type=Classification(10),
    eval_loss=CrossEntropy(),
    eval_metrics=[Accuracy()],
)

imagenet_task_specs = TaskSpecifications(
    dataset_name="imagenet",
    benchmark_name="imagenet",
    patch_size=(256, 256),
    bands_info=[BandInfo("red"), BandInfo("green"), BandInfo("blue")],
    label_type=Classification(1000),
    eval_loss=CrossEntropy(),
    eval_metrics=[Accuracy()],
)
