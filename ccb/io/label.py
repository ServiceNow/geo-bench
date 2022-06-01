from types import new_class
from typing import List
import numpy as np


class LabelType(object):
    pass

    def assert_valid(self):
        raise NotImplementedError()


class Classification(LabelType):
    def __init__(self, n_classes, class_names=None) -> None:
        super().__init__()
        self.n_classes = n_classes
        if class_names is not None:
            assert len(class_names) == n_classes, f"{len(class_names)} vs {n_classes}"
        self._class_names = class_names

    @property
    def class_names(self):
        if hasattr(self, "_class_names"):
            return self._class_names
        else:
            return self.class_name  # for backward compatibility with saved pickles with a typo

    def assert_valid(self, value):
        assert isinstance(value, int)
        assert value >= 0, f"{value} is smaller than 0."
        assert value < self.n_classes, f"{value} is >= to {self.n_classes}."

    def __repr__(self) -> str:
        if self.class_names is not None:
            if self.n_classes > 3:
                names = ", ".join(self.class_names[:3]) + "..."
            else:
                names = ", ".join(self.class_names) + "."
        else:
            names = "missing class names"
        return f"{self.n_classes}-classification ({names})"

    def label_stats(self, value):
        one_hot = np.zeros((self.n_classes,))
        one_hot[value] = 1
        return one_hot


class SemanticSegmentation(Classification):
    def assert_valid(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 2
        assert (value >= 0).all(), f"{value} is smaller than 0."
        assert (value < self.n_classes).all(), f"{value} is >= to {self.n_classes}."


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


class PointAnnotation(LabelType):
    def assert_valid(self, value: List[dict]):
        assert isinstance(value, (list, tuple))
        for point in value:
            assert isinstance(point, (list, tuple))
            assert len(point) == 2
            assert tuple(point) >= (0, 0)


class MultiLabelClassification(LabelType):
    def __init__(self, n_classes, class_names=None) -> None:
        super().__init__()
        self.n_classes = n_classes
        if class_names is not None:
            assert len(class_names) == n_classes, f"{len(class_names)} vs {n_classes}"
        self.class_name = class_names

    def assert_valid(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value) == self.n_classes
        assert all(np.unique(value) == [0, 1])

    def label_stats(self, value):
        return value
