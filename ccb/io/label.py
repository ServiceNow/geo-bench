from typing import List


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
