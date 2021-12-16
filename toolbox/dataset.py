DATASETS = [("dataset1", "dataset1/"), ("dataset2", "dataset2/")]


class Dataset(object):
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path

    def task_specs(self):
        return {}


def iter_datasets():
    for ds, path in DATASETS:
        yield Dataset(ds, path)
