from typing import List

from . import TaskSpecifications


class Dataset(object):
    def __init__(self, name: str, path: str, task_specs: List[TaskSpecifications]):
        self.name = name
        self.path = path
        self.task_specs = task_specs


DATASETS = [
    Dataset(
        name="dataset1",
        path="/dataset1/",
        task_specs=TaskSpecifications(
            input_shape=(1, 2, 3),
            features_shape=(4, 5, 6),
            spatial_resolution=10,
            temporal_resolution=11,
            band_names=["acdc", "queen"],
            band_wavelength=0.2,
            task_type="classification",
            n_classes=10,
        ),
    ),
    Dataset(
        name="dataset2",
        path="/dataset2/",
        task_specs=TaskSpecifications(
            input_shape=(1, 2, 3),
            features_shape=(4, 5, 6),
            spatial_resolution=2,
            temporal_resolution=2,
            band_names=["bob marley", "snoop dog"],
            band_wavelength=0.1,
            task_type="semantic segmentation",
            n_classes=10,
        ),
    ),
]


def iter_datasets():
    """
    Iterator over available datasets

    """
    for ds in DATASETS:
        yield ds
