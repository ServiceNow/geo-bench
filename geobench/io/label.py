"""Label."""

from types import new_class
from typing import List, Optional

import numpy as np


class LabelType(object):
    """Label Type."""

    # TODO write docstring about Loss and evalfunction for each type

    pass

    def assert_valid(self, value) -> None:
        """Check if label type is valid."""
        raise NotImplementedError()

    def value_to_str(self, value) -> str:
        """Convert label to string for visualization."""
        return ""


class Classification(LabelType):
    """Classification label.

    A classification label is trained with CrossEntropy Loss.

    """

    def __init__(self, n_classes: int, class_names: List[str] = None) -> None:
        """Initialize new instance of classification label.

        Args:
            n_classes: number of classes
            class_names: number of classes
        """
        super().__init__()
        self.n_classes = n_classes
        if class_names is not None:
            assert len(class_names) == n_classes, f"{len(class_names)} vs {n_classes}"
        self._class_names = class_names

    @property
    def class_names(self) -> Optional[List[str]]:
        """Return class names."""
        if hasattr(self, "_class_names"):
            return self._class_names
        else:
            return None

    def assert_valid(self, value) -> None:
        """Check if classification label is valid.

        Args:
            value to check
        """
        assert isinstance(value, int)
        assert value >= 0, f"{value} is smaller than 0."
        assert value < self.n_classes, f"{value} is >= to {self.n_classes}."

    def __repr__(self) -> str:
        """Return representation of classification label.

        Returns:
            string representation
        """
        if self.class_names is not None:
            if self.n_classes > 3:
                names = ", ".join(self.class_names[:3]) + "..."
            else:
                names = ", ".join(self.class_names) + "."
        else:
            names = "missing class names"
        return f"{self.n_classes}-classification ({names})"

    def label_stats(self, value):
        """Create label stats."""
        one_hot = np.zeros((self.n_classes,))
        one_hot[value] = 1
        return one_hot

    def value_to_str(self, value):
        """Convert numeric class label to name."""
        if self.class_names is None:
            return str(value)
        return self.class_names[value]


class SemanticSegmentation(Classification):
    """Semantic Segmentation label.

    A Semantic Segmentation label is trained with CrossEntropy.
    """

    def assert_valid(self, value) -> None:
        """Check if a semantic segmentation label is valid.

        Args:
            value: value to be checked
        """
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 2
        assert (value >= 0).all(), f"{value} is smaller than 0."
        assert (value < self.n_classes).all(), f"{value} is >= to {self.n_classes}."


class MultiLabelClassification(LabelType):
    """Multi-label classification label.

    A Multi-label classification task is trained with
    balanced binary-cross-entropy.
    """

    def __init__(self, n_classes, class_names=None) -> None:
        """Initialize new instance of classification label.

        Args:
            n_classes: number of classes
            class_names: number of classes
        """
        super().__init__()
        self.n_classes = n_classes
        if class_names is not None:
            assert len(class_names) == n_classes, f"{len(class_names)} vs {n_classes}"
        self.class_name = class_names

    def assert_valid(self, value) -> None:
        """Check if a semantic segmentation label is valid.

        Args:
            value: list of dictionary containing boxes and label
        """
        assert isinstance(value, np.ndarray)
        assert len(value) == self.n_classes
        assert all(np.unique(value) == [0, 1])

    def label_stats(self, value):
        """Return label stats."""
        return value

    def value_to_str(self, value):
        """Convert numeric class label to name."""
        if self.class_name is None:
            return str(value)
        names = []
        for i, active in enumerate(value):
            if active == 1:
                names.append(self.class_name[i])
        return " &\n".join(names)
