from functools import cached_property
from typing import List
import pickle
from pathlib import Path
from ccb.io.label import Classification

from ccb.io.dataset import Dataset, BandInfo, CCB_DIR


class TaskSpecifications:
    """
    Attributes:
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

    def __init__(
        self,
        dataset_name=None,
        benchmark_name=None,
        patch_size=None,
        n_time_steps=None,
        bands_info=None,
        bands_stats=None,
        label_type=None,
        eval_loss=None,
        eval_metrics=None,
        spatial_resolution=None,
    ) -> None:
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

    def save(self, directory, overwrite=False):
        file_path = Path(directory, "task_specs.pkl")
        if file_path.exists() and not overwrite:
            raise Exception("task_specifications.pkl alread exists and overwrite is set to False.")
        with open(file_path, "wb") as fd:
            pickle.dump(self, fd, protocol=4)

    def get_dataset(self, split, partition="default", transform=None):
        if self.benchmark_name == "test":
            import torchvision.transforms as tt
            import torchvision

            if transform is None:
                transform = tt.ToTensor()
            return torchvision.datasets.MNIST("/tmp/mnist", train=split == "train", transform=transform, download=True)
        else:
            return Dataset(self.get_dataset_dir(), split, partition_name=partition, transform=transform)

    def get_dataset_dir(self):
        benchmark_name = self.benchmark_name or "default"
        return CCB_DIR / benchmark_name / self.dataset_name

    # for backward compatibility (we'll remove soon)
    @cached_property
    def benchmark_name(self):
        return "default"


def task_iterator(benchmark_name: str = "default") -> TaskSpecifications:

    if benchmark_name == "test":
        yield mnist_task_specs
        return

    benchmark_dir = CCB_DIR / benchmark_name

    for dataset_dir in benchmark_dir:
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("_") or dataset_dir.name.startswith("."):
            continue

        with open(dataset_dir / "task_specs.pkl", "rb") as fd:
            task_specs = pickle.load(fd)

        yield task_specs


class Loss(object):
    def __call__(self, label, prediction):
        raise NotImplementedError()

    @property
    def name(self):
        return self.__class__.__name__.lower()


class Accuracy(Loss):
    def __call__(self, prediction, label):
        return float(label != prediction)


class CrossEntropy(Loss):
    pass


class SegmentationAccuracy(Loss):
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
