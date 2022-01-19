
from typing import List
import pickle
from pathlib import Path


class TaskSpecifications:
    """
    Attributes:
        dataset_name: The name of the dataset.
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
        patch_size=None,
        n_time_steps=None,
        bands_info=None,
        bands_stats=None,
        label_type=None,
        eval_loss=None,
        spatial_resolution=None,
    ) -> None:
        self.dataset_name = dataset_name
        self.patch_size = patch_size
        self.n_time_steps = n_time_steps
        self.bands_info = bands_info
        self.bands_stats = bands_stats
        self.label_type = label_type
        self.eval_loss = eval_loss
        self.spatial_resolution = spatial_resolution

    def save(self, directory):
        file_path = Path(directory, "task_specifications.pkl")
        with open(file_path, "wb") as fd:
            pickle.dump(self, fd, protocol=4)


class Loss(object):
    def __call__(self, label, prediction):
        raise NotImplemented()

    @property
    def name(self):
        return self.__class__.__name__.lower()


class Accuracy(Loss):
    def __call__(self, prediction, label):
        return float(label != prediction)


class SegmentationAccuracy(Loss):
    pass


class LabelType(object):
    pass

    def assert_valid(self):
        raise NotImplemented()


class Classification(LabelType):
    def __init__(self, n_classes, class_names=None) -> None:
        super().__init__()
        self.n_classes = n_classes
        if class_names is not None:
            assert len(class_names) == n_classes, f"{len(class_names)} vs {n_classes}"
        self.class_name = class_names

    def assert_valid(self, value):
        assert isinstance(value, int)
        assert value >= 0, f"{value} is smaller than 0."
        assert value < self.n_classes, f"{value} is >= to {self.n_classes}."


class Regression(LabelType):
    def __init__(self, min_val=None, max_val=None) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def assert_valid(self, value):
        assert isinstance(value, float)
        if self.min_val is not None:
            assert value >= self.min_val
        if self.max_val is not None:
            assert value <= self.max_val


class Detection(LabelType):
    def __init__(self) -> None:
        super().__init__()

    def assert_valid(self, value: List[dict]):
        assert isinstance(value, (list, tuple))
        for box in value:
            assert isinstance(box, dict)
            assert len(box) == 4
            for key in ("xmin", "ymin", "xmax", "ymax"):
                assert key in box
                assert box[key] >= 0
            assert box["xmin"] < box["xmax"]
            assert box["ymin"] < box["ymax"]
