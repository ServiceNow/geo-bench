from . import TaskSpecifications


DATASETS = [("dataset1", "dataset1/"), ("dataset2", "dataset2/")]


class Dataset(object):
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path

    @property
    def task_specs(self):
        return [
            TaskSpecifications(
                input_shape=(1, 2, 3),
                features_shape=(4, 5, 6),
                spatial_resolution=10,
                temporal_resolution=11,
                band_names=["acdc", "queen"],
                band_wavelength=0.2,
                task_type="classification",
                n_classes=10,
                dataset_name=self.name,
            ),
            TaskSpecifications(
                input_shape=(1, 2, 3),
                features_shape=(4, 5, 6),
                spatial_resolution=2,
                temporal_resolution=2,
                band_names=["bob marley", "snoop dog"],
                band_wavelength=0.1,
                task_type="semantic segmentation",
                n_classes=10,
                dataset_name=self.name,
            ),
        ]


def iter_datasets():
    for ds, path in DATASETS:
        yield Dataset(ds, path)
